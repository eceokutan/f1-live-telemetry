"""
Push-to-Talk (PTT) Controller for F1 Telemetry Dashboard.

Monitors keyboard (V key via pynput) and optional joystick button (via pygame)
for push-to-talk activation. Works globally — captures input even when the
game window has focus, not the dashboard.

Usage:
    controller = PTTController(joystick_button_index=9, keyboard_key="v")
    controller.ptt_pressed.connect(voice_worker.start_recording)
    controller.ptt_released.connect(voice_worker.stop_recording)
    controller.start()
"""

import logging
import threading
import time
import ctypes

from PyQt5 import QtCore

logger = logging.getLogger(__name__)


class PTTController(QtCore.QObject):
    """
    Push-to-talk controller that monitors keyboard and joystick inputs.

    Emits signals when PTT is pressed/released, regardless of which window
    has focus. Keyboard uses pynput (global hook), joystick uses pygame (polling).

    Signals:
        ptt_pressed  - Emitted when any PTT input is pressed
        ptt_released - Emitted when all PTT inputs are released
        status_update(str) - Status messages for logging
    """

    ptt_pressed = QtCore.pyqtSignal()
    ptt_released = QtCore.pyqtSignal()
    status_update = QtCore.pyqtSignal(str)

    def __init__(self, joystick_button_index: int = 11, keyboard_key: str = "v", parent=None):
        """
        Initialize PTT controller.

        Args:
            joystick_button_index: Joystick button index for PTT (default 11,
                                   Thrustmaster T128X RSB). Use --ptt-button to override.
            keyboard_key: Keyboard key name captured from launcher settings
                          (e.g. "v", "space", "f1").
        """
        super().__init__(parent)

        self._joystick_button_index = joystick_button_index
        self._keyboard_key = (keyboard_key or "v").strip().lower()
        self._keyboard_vk = self._resolve_virtual_key(self._keyboard_key)

        # State (protected by _lock)
        self._lock = threading.Lock()
        self._keyboard_held = False
        self._joystick_held = False
        self._ptt_active = False

        # Control
        self._running = False
        self._keyboard_listener = None
        self._keyboard_poll_thread = None
        self._joystick_thread = None

    def start(self):
        """Start monitoring PTT inputs (keyboard + joystick)."""
        self._running = True
        self._start_keyboard_listener()
        self._start_keyboard_polling_fallback()
        self._start_joystick_polling()
        logger.info(
            "PTT controller started (keyboard=%s, joystick=button %d)",
            self._keyboard_key,
            self._joystick_button_index,
        )
        self.status_update.emit(
            f"PTT ready (hold {self._keyboard_key.upper()} key or joystick button)"
        )

    def stop(self):
        """Stop monitoring PTT inputs."""
        self._running = False

        if self._keyboard_listener:
            try:
                self._keyboard_listener.stop()
            except Exception:
                pass

        # Joystick thread exits via _running=False (daemon thread)
        logger.info("PTT controller stopped")

    def _start_keyboard_listener(self):
        """Start global keyboard listener for configured key using pynput.

        On macOS, pynput requires Accessibility permissions and may crash the
        process with SIGTRAP if not granted.  We detect macOS and skip pynput
        entirely, relying on the Qt key-event fallback instead.
        """
        import sys

        if sys.platform == "darwin":
            logger.info(
                "macOS detected — skipping pynput global keyboard hook "
                "(use Qt key events instead)"
            )
            self.status_update.emit(
                f"PTT ready (press {self._keyboard_key.upper()} key while dashboard is focused)"
            )
            return

        try:
            from pynput import keyboard

            def on_press(key):
                try:
                    if self._matches_keyboard_key(key):
                        self._update_state(keyboard_held=True)
                except Exception:
                    pass

            def on_release(key):
                try:
                    if self._matches_keyboard_key(key):
                        self._update_state(keyboard_held=False)
                except Exception:
                    pass

            self._keyboard_listener = keyboard.Listener(
                on_press=on_press,
                on_release=on_release
            )
            self._keyboard_listener.daemon = True
            self._keyboard_listener.start()
            logger.info("Keyboard PTT active (%s key)", self._keyboard_key)

        except ImportError:
            logger.error("pynput not installed — keyboard PTT disabled. "
                         "Install with: pip install pynput")
            self.status_update.emit("Keyboard PTT unavailable (pynput not installed)")
        except Exception as e:
            logger.error("Failed to start keyboard listener: %s", e)

    def _start_joystick_polling(self):
        """Start joystick polling in a daemon thread. Gracefully degrades if pygame is unavailable."""
        try:
            import pygame  # noqa: F401 — test import only
        except ImportError:
            logger.warning("pygame not installed — joystick PTT disabled. "
                           "Install with: pip install pygame")
            self.status_update.emit("Joystick PTT unavailable (install pygame)")
            return

        self._joystick_thread = threading.Thread(
            target=self._joystick_poll_loop,
            daemon=True,
            name="PTT-Joystick"
        )
        self._joystick_thread.start()

    def _start_keyboard_polling_fallback(self):
        """
        Start a Windows key-state polling fallback for the configured key.

        This improves reliability in scenarios where global keyboard hooks are
        throttled or intermittently blocked by game focus/state.
        """
        if self._keyboard_vk is None:
            logger.info(
                "Keyboard polling fallback disabled for key '%s' (no Win32 VK mapping)",
                self._keyboard_key,
            )
            return
        self._keyboard_poll_thread = threading.Thread(
            target=self._keyboard_poll_loop,
            daemon=True,
            name="PTT-KeyboardPoll"
        )
        self._keyboard_poll_thread.start()

    def _keyboard_poll_loop(self):
        """Poll configured key state via Win32 API as fallback."""
        if not hasattr(ctypes, "windll"):
            return

        while self._running:
            try:
                pressed = bool(ctypes.windll.user32.GetAsyncKeyState(self._keyboard_vk) & 0x8000)
                self._update_state(keyboard_held=pressed)
            except Exception:
                # Keep hook-based mode alive even if fallback polling fails.
                pass
            time.sleep(0.016)  # ~60Hz polling

    def _joystick_poll_loop(self):
        """Poll joystick button state in a background thread."""
        import os
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

        import pygame
        pygame.init()
        pygame.joystick.init()

        joystick = None

        logger.info("Joystick PTT polling started (looking for controllers...)")

        while self._running:
            # Try to find joystick if not connected
            if joystick is None:
                pygame.joystick.quit()
                pygame.joystick.init()

                if pygame.joystick.get_count() > 0:
                    joystick = pygame.joystick.Joystick(0)
                    joystick.init()
                    name = joystick.get_name()
                    num_buttons = joystick.get_numbuttons()
                    logger.info("Joystick connected: %s (%d buttons)", name, num_buttons)
                    self.status_update.emit(
                        f"Joystick: {name} (PTT=button {self._joystick_button_index})"
                    )

                    if self._joystick_button_index >= num_buttons:
                        logger.warning(
                            "Button index %d out of range (joystick has %d buttons). "
                            "Use --ptt-button to set correct index.",
                            self._joystick_button_index, num_buttons
                        )
                else:
                    time.sleep(2.0)  # Check for joystick every 2 seconds
                    continue

            # Poll joystick
            try:
                pygame.event.pump()

                # Log all button presses at DEBUG level for discovery
                if logger.isEnabledFor(logging.DEBUG):
                    num_buttons = joystick.get_numbuttons()
                    for i in range(num_buttons):
                        if joystick.get_button(i):
                            logger.debug("Joystick button %d pressed", i)

                # Check PTT button
                pressed = joystick.get_button(self._joystick_button_index)
                self._update_state(joystick_held=bool(pressed))

            except Exception:
                logger.warning("Joystick disconnected, will retry...")
                joystick = None
                self._update_state(joystick_held=False)

            time.sleep(0.016)  # ~60Hz polling

        # Cleanup
        try:
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass

    def _update_state(self, keyboard_held=None, joystick_held=None):
        """
        Thread-safe state update. Emits signals only on PTT state transitions.

        The composite state is keyboard_held OR joystick_held — PTT is active
        if either input is held. Signals fire only when the composite state changes.
        This handles key repeat (redundant presses) correctly.
        """
        with self._lock:
            if keyboard_held is not None:
                self._keyboard_held = keyboard_held
            if joystick_held is not None:
                self._joystick_held = joystick_held

            new_active = self._keyboard_held or self._joystick_held

            if new_active != self._ptt_active:
                self._ptt_active = new_active
                if new_active:
                    logger.info("PTT activated")
                    self.ptt_pressed.emit()
                else:
                    logger.info("PTT released")
                    self.ptt_released.emit()

    @staticmethod
    def _resolve_virtual_key(key_name: str):
        """Best-effort mapping from key name to Win32 virtual-key code."""
        if not key_name:
            return None

        if len(key_name) == 1:
            ch = key_name.upper()
            if "A" <= ch <= "Z" or "0" <= ch <= "9":
                return ord(ch)

        key_map = {
            "space": 0x20,
            "enter": 0x0D,
            "return": 0x0D,
            "tab": 0x09,
            "esc": 0x1B,
            "escape": 0x1B,
            "backspace": 0x08,
            "left": 0x25,
            "up": 0x26,
            "right": 0x27,
            "down": 0x28,
        }

        if key_name in key_map:
            return key_map[key_name]

        if key_name.startswith("f") and key_name[1:].isdigit():
            fn = int(key_name[1:])
            if 1 <= fn <= 24:
                return 0x70 + (fn - 1)

        return None

    def handle_key_press(self, qt_key_text: str):
        """Handle a key press from Qt key events (macOS fallback).

        Called by MainWindow.keyPressEvent when PTT is active on macOS.
        """
        if qt_key_text.lower() == self._keyboard_key:
            self._update_state(keyboard_held=True)

    def handle_key_release(self, qt_key_text: str):
        """Handle a key release from Qt key events (macOS fallback).

        Called by MainWindow.keyReleaseEvent when PTT is active on macOS.
        """
        if qt_key_text.lower() == self._keyboard_key:
            self._update_state(keyboard_held=False)

    def _matches_keyboard_key(self, key) -> bool:
        """Return True when pynput key object matches configured keyboard key."""
        try:
            char = getattr(key, "char", None)
            if char and char.lower() == self._keyboard_key:
                return True

            name = getattr(key, "name", None)
            if name and name.lower() == self._keyboard_key:
                return True

            vk = getattr(key, "vk", None)
            if vk is not None and self._keyboard_vk is not None and vk == self._keyboard_vk:
                return True
        except Exception:
            return False

        return False
