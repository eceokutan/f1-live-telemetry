#!/usr/bin/env python3
"""
Jarvis F1 Telemetry Suite - Main Entry Point

Unified launcher for:
- Jarvis Live: Real-time telemetry visualization and AI race engineering
- Jarvis Post: Post-race session analysis with AI coaching

Usage:
    python main.py      # Opens unified launcher
"""
import sys
import os
import logging
import threading

# When running as a PyInstaller bundle, set working directory to the exe's folder
# so that relative paths (config.json, data/, race_engineer_gguf/) resolve correctly.
if getattr(sys, "frozen", False):
    os.chdir(os.path.dirname(sys.executable))

# Pre-load native DLLs BEFORE PyQt5 to avoid DLL conflicts on Windows.
# PyQt5 changes the DLL search path, which breaks onnxruntime/ctranslate2 if loaded after.
for _mod in ("onnxruntime", "ctranslate2"):
    try:
        __import__(_mod)
    except Exception:
        pass

from PyQt5 import QtWidgets, QtCore

# Configure logging FIRST - before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

logger.info("Jarvis F1 Telemetry Suite starting")

# Module-level placeholders — populated by the loading screen
MainWindow = None
AcTelemetryWorker = None
AIRaceEngineerWorker = None
AI_AVAILABLE = False
VoiceInputWorker = None
VOICE_AVAILABLE = False
TTSOutputWorker = None
TTS_AVAILABLE = False
PTTController = None
PTT_AVAILABLE = False
SessionRecorder = None
RECORDER_AVAILABLE = False

KOKORO_VOICE_ID = "bm_lewis"
KOKORO_LANG = "en-gb"
KOKORO_SPEED = 1.3
KOKORO_USE_CUDA = False


def run_jarvis_live(settings: dict) -> bool:
    """
    Launch Jarvis Live - real-time telemetry dashboard.

    Args:
        settings: Configuration dict from the launcher.
    """
    global MainWindow, AcTelemetryWorker
    global AI_AVAILABLE, AIRaceEngineerWorker
    global VOICE_AVAILABLE, VoiceInputWorker
    global TTS_AVAILABLE, TTSOutputWorker
    global PTT_AVAILABLE, PTTController
    global RECORDER_AVAILABLE, SessionRecorder

    game = "ac"
    enable_ptt = settings.get("voice_mode") == "push_to_talk"
    ptt_key = settings.get("ptt_key", "v")
    local_model_path = settings.get(
        "local_model_path",
        settings.get("local_adapter_path", "race_engineer_gguf/granite-race-engineer-Q4_K_M.gguf"),
    )

    # Export local LLM model path so AI worker can pick it up.
    os.environ["LOCAL_MODEL_PATH"] = str(local_model_path or "race_engineer_gguf/granite-race-engineer-Q4_K_M.gguf")

    logger.info("Starting Jarvis Live for: %s", game.upper())
    logger.info("LLM settings - model=%s", local_model_path)

    window = MainWindow()

    # AC-only backend
    logger.info("Initializing %s telemetry backend", game.upper())
    telemetry_thread = AcTelemetryWorker()

    # Connect signals
    telemetry_thread.lap_completed.connect(window.handle_lap_complete)
    telemetry_thread.status_update.connect(lambda msg: logger.info("Status: %s", msg))

    if hasattr(telemetry_thread, 'session_info_update'):
        telemetry_thread.session_info_update.connect(window.update_session_info)
    if hasattr(telemetry_thread, 'live_data_update'):
        telemetry_thread.live_data_update.connect(window.update_live_data)
    if hasattr(telemetry_thread, 'realtime_sample'):
        telemetry_thread.realtime_sample.connect(window.handle_realtime_sample)
    if hasattr(telemetry_thread, 'session_reset'):
        telemetry_thread.session_reset.connect(window.reset_session)

    logger.info("Signals connected")

    # Initialize mandatory AI race engineer
    ai_thread = None
    voice_thread = None
    tts_thread = None
    tts_runtime = {"worker": None, "restarting": False}
    ptt_controller = None
    if not AI_AVAILABLE:
        logger.critical("AI Race Engineer module is required but unavailable")
        QtWidgets.QMessageBox.critical(
            None,
            "AI Required",
            "Jarvis Live requires the AI Race Engineer module, but it is unavailable.",
        )
        return False

    logger.info("Initializing AI Race Engineer")
    try:
        ai_thread = AIRaceEngineerWorker(
            track_name="Unknown Track",
            session_id="ac_session_001",
            verbosity="moderate"
        )

        ai_thread.ai_commentary.connect(window.handle_ai_commentary)
        ai_thread.driver_query_received.connect(window.handle_driver_query)
        ai_thread.status_update.connect(window.handle_ai_status_update)
        ai_thread.processing_query.connect(
            lambda busy: window.set_mic_input_disabled(busy, reason="AI processing query")
        )
        ai_thread.status_update.connect(lambda msg: logger.info("AI: %s", msg))

        ai_sample_counter = [0]
        AI_SAMPLE_RATE = 12

        def throttled_ai_telemetry(sample):
            ai_sample_counter[0] += 1
            if ai_sample_counter[0] >= AI_SAMPLE_RATE:
                ai_sample_counter[0] = 0
                try:
                    ai_thread.process_telemetry(sample)
                except Exception:
                    pass

        if hasattr(telemetry_thread, 'realtime_sample'):
            telemetry_thread.realtime_sample.connect(throttled_ai_telemetry)

        if hasattr(telemetry_thread, 'live_data_update'):
            def on_live_data_for_ai(data: dict):
                if "ac_status" in data:
                    ai_thread.update_ac_status(data.get("ac_status", 0))
                else:
                    ai_thread.update_ac_status(2)

            telemetry_thread.live_data_update.connect(on_live_data_for_ai)

        # Wire session info to AI thread so it gets real track/car names
        # and can seed fuel consumption estimate from datasheet
        if hasattr(telemetry_thread, 'session_info_update'):
            telemetry_thread.session_info_update.connect(ai_thread.update_session_info)

        ai_thread.start()
        logger.info("AI Race Engineer started")

    except Exception as e:
        logger.error("Failed to initialize AI Race Engineer: %s", e, exc_info=True)
        QtWidgets.QMessageBox.critical(
            None,
            "AI Initialization Failed",
            f"Jarvis Live requires AI, but initialization failed:\n{e}",
        )
        return False

    # Initialize voice input
    voice_mode_setting = settings.get("voice_mode", "disabled")
    if VOICE_AVAILABLE and voice_mode_setting != "disabled":
        voice_mode = "PTT" if enable_ptt else "VAD"
        logger.info("Initializing Voice Input (faster-whisper, %s mode)", voice_mode)
        try:
            voice_thread = VoiceInputWorker(
                whisper_model_size="base",
                ptt_mode=enable_ptt
            )
            if ai_thread:
                voice_thread.speech_detected.connect(ai_thread.process_driver_query)
                ai_thread.processing_query.connect(
                    lambda busy: voice_thread.pause() if busy else voice_thread.resume()
                )
            voice_thread.vad_state_changed.connect(window.handle_vad_state_change)
            voice_thread.status_update.connect(lambda msg: logger.info("Voice: %s", msg))
            voice_thread.error_occurred.connect(lambda err: logger.error("Voice: %s", err))
            voice_thread.start()
            logger.info("Voice Input started (%s mode)", voice_mode)
        except Exception as e:
            logger.error("Failed to initialize Voice Input: %s", e, exc_info=True)
            voice_thread = None

    # Initialize PTT controller
    if enable_ptt and voice_thread and PTT_AVAILABLE:
        # Build keyboard keys and joystick button lists from config slots
        keyboard_keys = []
        joystick_buttons = []
        for slot_prefix in ("ptt_slot_1", "ptt_slot_2"):
            stype = settings.get(f"{slot_prefix}_type", "disabled")
            svalue = settings.get(f"{slot_prefix}_value", "")
            if stype == "keyboard" and svalue:
                keyboard_keys.append(svalue)
            elif stype == "joystick" and svalue:
                try:
                    joystick_buttons.append(int(svalue))
                except ValueError:
                    pass
        # Fallback: if no keyboard key configured, use legacy ptt_key
        if not keyboard_keys:
            keyboard_keys.append(ptt_key)
        logger.info(
            "Initializing PTT Controller (keyboard=%s, joystick buttons=%s)",
            keyboard_keys, joystick_buttons,
        )
        try:
            ptt_controller = PTTController(
                keyboard_keys=keyboard_keys,
                joystick_button_indices=joystick_buttons,
            )
            ptt_controller.ptt_pressed.connect(voice_thread.start_recording)
            ptt_controller.ptt_released.connect(voice_thread.stop_recording)
            ptt_controller.status_update.connect(lambda msg: logger.info("PTT: %s", msg))
            ptt_controller.start()
            window.set_ptt_controller(ptt_controller)
            logger.info("PTT Controller started")
        except Exception as e:
            logger.error("Failed to initialize PTT Controller: %s", e, exc_info=True)
            ptt_controller = None

    # Initialize TTS output (Kokoro local TTS)
    if TTS_AVAILABLE and voice_mode_setting != "disabled":
        logger.info(
            "Initializing TTS Output (kokoro, voice=%s, speed=%.2f)",
            KOKORO_VOICE_ID,
            KOKORO_SPEED,
        )
        try:
            def _build_tts_worker() -> TTSOutputWorker:
                worker = TTSOutputWorker(
                    kokoro_voice_id=KOKORO_VOICE_ID,
                    kokoro_lang=KOKORO_LANG,
                    kokoro_speed=KOKORO_SPEED,
                    kokoro_use_cuda=KOKORO_USE_CUDA,
                )
                worker.status_update.connect(lambda msg: logger.info("TTS: %s", msg))
                worker.error_occurred.connect(lambda err: logger.error("TTS: %s", err))
                def _on_tts_worker_finished():
                    logger.warning("TTS worker stopped")
                    if voice_thread:
                        voice_thread.resume()
                    window.set_mic_input_disabled(False, reason="TTS playback")

                worker.finished.connect(_on_tts_worker_finished)
                if voice_thread:
                    worker.playback_started.connect(voice_thread.pause)
                    worker.playback_finished.connect(voice_thread.resume)
                    worker.playback_started.connect(
                        lambda: window.set_mic_input_disabled(True, reason="TTS playback")
                    )
                    worker.playback_finished.connect(
                        lambda: window.set_mic_input_disabled(False, reason="TTS playback")
                    )

                return worker

            def _ensure_tts_worker_running(reason: str) -> bool:
                if tts_runtime["restarting"]:
                    return False
                worker = tts_runtime["worker"]
                if worker and worker.isRunning():
                    return True

                tts_runtime["restarting"] = True
                try:
                    worker = _build_tts_worker()
                    tts_runtime["worker"] = worker
                    worker.start()
                    logger.info("TTS worker started (%s)", reason)
                    return True
                except Exception as e:
                    logger.error("Failed to start TTS worker (%s): %s", reason, e, exc_info=True)
                    tts_runtime["worker"] = None
                    return False
                finally:
                    tts_runtime["restarting"] = False

            def _speak_with_retry(message: str, priority: int = 2, retries_left: int = 6):
                worker = tts_runtime["worker"]
                if worker and worker.isRunning():
                    worker.speak(message, priority=priority)
                    return

                if not _ensure_tts_worker_running("auto-restart"):
                    if retries_left <= 0:
                        logger.error("Dropping TTS message after failed restart: %s", message[:80])
                        return
                    QtCore.QTimer.singleShot(
                        200, lambda m=message, p=priority, r=retries_left - 1: _speak_with_retry(m, p, r)
                    )
                    return

                if retries_left <= 0:
                    logger.error("Dropping TTS message; worker never became ready: %s", message[:80])
                    return

                QtCore.QTimer.singleShot(
                    200, lambda m=message, p=priority, r=retries_left - 1: _speak_with_retry(m, p, r)
                )

            def on_ai_commentary_for_tts(msg, trigger, priority):
                _speak_with_retry(msg, priority=priority)

            if ai_thread:
                ai_thread.ai_commentary.connect(on_ai_commentary_for_tts)

            if _ensure_tts_worker_running("initialization"):
                tts_thread = tts_runtime["worker"]
                logger.info("TTS Output started")
            else:
                tts_thread = None
        except Exception as e:
            logger.error("Failed to initialize TTS Output: %s", e, exc_info=True)
            tts_thread = None
    elif not TTS_AVAILABLE and voice_mode_setting != "disabled":
        logger.warning("Voice mode enabled but TTS module is unavailable")

    # Initialize session recorder (optional)
    recorder_thread = None
    current_lap_number = [0]
    session_info = {"track": "", "car": "", "player": ""}

    if RECORDER_AVAILABLE:
        logger.info("Initializing Session Recorder")
        try:
            recorder_thread = SessionRecorder(db_path="data/telemetry_sessions.db")
            recorder_thread.status_update.connect(lambda msg: logger.info("Recorder: %s", msg))
            recorder_thread.error_occurred.connect(lambda err: logger.error("Recorder: %s", err))
            recorder_thread.start()

            def on_session_reset():
                """End current recording session so the next session_info_update creates a new one."""
                if recorder_thread and recorder_thread.session_id:
                    logger.info("Session reset — ending recorder session %s", recorder_thread.session_id)
                    recorder_thread.end_session()

            if hasattr(telemetry_thread, 'session_reset'):
                telemetry_thread.session_reset.connect(on_session_reset)

            def on_session_info(info: dict):
                session_info["track"] = info.get("track", "")
                session_info["car"] = info.get("car_model", "")
                session_info["player"] = info.get("player_name", "")
                if recorder_thread and not recorder_thread.session_id:
                    recorder_thread.start_session(
                        game=game,
                        track_name=session_info["track"],
                        car_model=session_info["car"],
                        player_name=session_info["player"],
                        session_type=info.get("session_type", "")
                    )

            if hasattr(telemetry_thread, 'session_info_update'):
                telemetry_thread.session_info_update.connect(on_session_info)

            if hasattr(telemetry_thread, 'realtime_sample'):
                telemetry_thread.realtime_sample.connect(
                    lambda sample: recorder_thread.record_telemetry_sample(sample)
                )

            def on_lap_complete(lap_id: int, samples: list, lap_valid: bool = True, last_time_ms: int = 0):
                if not samples:
                    return
                current_lap_number[0] = lap_id
                lap_start_t = samples[0].get("t", 0.0)
                lap_end_t = samples[-1].get("t", 0.0)
                lap_time = max(0.0, lap_end_t - lap_start_t)
                speeds = [s.get("speed", 0.0) for s in samples]
                avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
                max_speed = max(speeds) if speeds else 0.0
                min_speed = min(speeds) if speeds else 0.0
                fuel_start = samples[0].get("fuel", 0.0)
                fuel_end = samples[-1].get("fuel", 0.0)
                resolved_lap_valid = bool(lap_valid)

                if recorder_thread:
                    recorder_thread.record_lap(
                        lap_number=lap_id,
                        lap_time=lap_time,
                        fuel_start=fuel_start,
                        fuel_end=fuel_end,
                        avg_speed=avg_speed,
                        max_speed=max_speed,
                        min_speed=min_speed,
                        valid=resolved_lap_valid
                    )

            telemetry_thread.lap_completed.connect(on_lap_complete)

            if ai_thread:
                def on_ai_commentary(message: str, trigger: str, priority: str):
                    if recorder_thread:
                        recorder_thread.record_ai_commentary(
                            message=message,
                            trigger=trigger,
                            priority=priority,
                            lap_number=current_lap_number[0]
                        )

                ai_thread.ai_commentary.connect(on_ai_commentary)

            if voice_thread and ai_thread:
                last_query = [""]

                def on_driver_query(query: str):
                    last_query[0] = query

                def on_query_response(message: str, trigger: str, priority: str):
                    if recorder_thread and last_query[0] and trigger == "driver_query":
                        recorder_thread.record_voice_query(
                            query_text=last_query[0],
                            response_text=message,
                            lap_number=current_lap_number[0]
                        )
                        last_query[0] = ""

                ai_thread.driver_query_received.connect(on_driver_query)
                ai_thread.ai_commentary.connect(on_query_response)

            logger.info("Session Recorder started")
        except Exception as e:
            logger.error("Failed to initialize Session Recorder: %s", e, exc_info=True)
            recorder_thread = None

    # Start telemetry thread
    logger.info("Starting telemetry worker thread")
    telemetry_thread.start()

    # Show window
    window.show()

    mode_parts = []
    if ai_thread:
        mode_parts.append("AI Race Engineer")
    if voice_thread:
        mode_parts.append("PTT Voice" if enable_ptt else "Voice Input")
    if tts_thread:
        mode_parts.append("TTS Output")
    if mode_parts:
        logger.info("Jarvis Live ready - %s active", " + ".join(mode_parts))
    else:
        logger.info("Jarvis Live ready - telemetry only")

    if recorder_thread:
        logger.info("Session Recording enabled - all data saved to database")

    # Run Qt event loop (blocks until window is closed)
    app = QtWidgets.QApplication.instance()
    app.exec_()

    # Clean shutdown
    logger.info("Shutting down Jarvis Live...")

    try:
        telemetry_thread.stop()
        telemetry_thread.wait(2000)
    except Exception as e:
        logger.warning("Error stopping telemetry thread: %s", e)

    if ai_thread:
        try:
            ai_thread.stop()
            ai_thread.wait(2000)
        except Exception as e:
            logger.warning("Error stopping AI thread: %s", e)

    if voice_thread:
        try:
            voice_thread.stop()
            voice_thread.wait(2000)
        except Exception as e:
            logger.warning("Error stopping voice thread: %s", e)

    if ptt_controller:
        try:
            ptt_controller.stop()
        except Exception as e:
            logger.warning("Error stopping PTT controller: %s", e)

    active_tts_worker = tts_runtime["worker"] if tts_runtime["worker"] else tts_thread
    if active_tts_worker:
        try:
            active_tts_worker.stop()
            active_tts_worker.wait(2000)
        except Exception as e:
            logger.warning("Error stopping TTS thread: %s", e)

    if recorder_thread:
        try:
            recorder_thread.stop()
            recorder_thread.wait(2000)
        except Exception as e:
            logger.warning("Error stopping recorder thread: %s", e)

    logger.info("Jarvis Live shutdown complete")
    return bool(getattr(window, "exit_application_requested", False))


def run_jarvis_post(session_id: int) -> bool:
    """
    Launch Jarvis Post - post-race telemetry analysis.

    Args:
        session_id: Session ID to export and analyze.
    """
    from data.session_exporter import SessionExporter
    from ui.post_race import LapViewerWindow

    logger.info("Starting Jarvis Post for session %d", session_id)

    # Export session from SQLite to CSV
    exporter = SessionExporter()
    try:
        export_dir = exporter.export_session(session_id)
        logger.info("Session exported to: %s", export_dir)
    except Exception as e:
        logger.error("Failed to export session: %s", e)
        QtWidgets.QMessageBox.critical(
            None, "Export Error",
            f"Failed to export session {session_id}:\n{str(e)}"
        )
        return False

    # Launch post-race viewer
    window = LapViewerWindow()
    window.show()

    # Load the exported session
    window.load_session_from_file(export_dir)

    # Run Qt event loop (blocks until window is closed)
    app = QtWidgets.QApplication.instance()
    app.exec_()

    logger.info("Jarvis Post shutdown complete")
    return bool(getattr(window, "exit_application_requested", False))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # Show loading screen immediately (~200ms from launch)
    from ui.startup_loader import LoadingScreen, StartupLoaderThread

    loading_screen = LoadingScreen()
    loading_screen.show()
    loading_screen.set_stage_status(0, "done", "Application shell ready")
    app.processEvents()

    loader_thread = StartupLoaderThread()
    loader_thread.stage_update.connect(loading_screen.set_stage_status)
    loader_thread.fatal_error.connect(loading_screen.show_fatal_error)

    startup_results = [None]

    def _on_all_done(r):
        startup_results[0] = r
        # Stage 10: Fonts & config (must run on main thread)
        loading_screen.set_stage_status(9, "running", "Loading fonts...")
        from ui.styles import load_fonts
        load_fonts()
        loading_screen.set_stage_status(9, "done", "Fonts and config loaded")

        # Stage 11: Complete
        loading_screen.set_stage_status(10, "done", "Startup complete")
        QtCore.QTimer.singleShot(400, loading_screen.close)

    loader_thread.all_done.connect(_on_all_done)
    loader_thread.start()

    # Process events until loading screen closes
    while loading_screen.isVisible():
        app.processEvents()
        QtCore.QThread.msleep(10)

    loader_thread.wait()

    # Check for fatal error (loading screen closed by Quit button → sys.exit already called)
    if startup_results[0] is None:
        sys.exit(1)

    # Populate module-level variables from loader results
    r = startup_results[0]
    MainWindow = r["MainWindow"]
    AcTelemetryWorker = r["AcTelemetryWorker"]
    AIRaceEngineerWorker = r["AIRaceEngineerWorker"]
    AI_AVAILABLE = r["AI_AVAILABLE"]
    VoiceInputWorker = r["VoiceInputWorker"]
    VOICE_AVAILABLE = r["VOICE_AVAILABLE"]
    TTSOutputWorker = r["TTSOutputWorker"]
    TTS_AVAILABLE = r["TTS_AVAILABLE"]
    SessionRecorder = r["SessionRecorder"]
    RECORDER_AVAILABLE = r["RECORDER_AVAILABLE"]
    PTTController = r["PTTController"]
    PTT_AVAILABLE = r["PTT_AVAILABLE"]

    logger.info("All core modules imported")

    from ui.unified_launcher import UnifiedLauncher
    from ui.launcher import LauncherWindow
    from ui.session_picker import SessionPickerDialog
    from ui.config_manager import load_config

    # Load saved settings once
    settings = load_config()

    while True:
        # Show unified launcher
        launcher = UnifiedLauncher()
        launcher.exec_()
        action = launcher.get_action()

        if action == UnifiedLauncher.ACTION_QUIT:
            break

        elif action == UnifiedLauncher.ACTION_LIVE:
            # Launch Jarvis Live with current settings
            logger.info("User chose: Start Jarvis Live")
            try:
                exit_requested = run_jarvis_live(settings)
                if exit_requested:
                    break
            except Exception as e:
                logger.critical("Jarvis Live error: %s", e, exc_info=True)
            # After live window closes, loop back to launcher

        elif action == UnifiedLauncher.ACTION_POST:
            # Show session picker
            logger.info("User chose: Start Jarvis Post")
            picker = SessionPickerDialog()
            picker.exec_()

            if picker.was_accepted():
                session_id = picker.get_selected_session_id()
                logger.info("Selected session: %d", session_id)
                try:
                    exit_requested = run_jarvis_post(session_id)
                    if exit_requested:
                        break
                except Exception as e:
                    logger.critical("Jarvis Post error: %s", e, exc_info=True)
            # After post window closes (or picker cancelled), loop back to launcher

        elif action == UnifiedLauncher.ACTION_SETTINGS:
            # Show settings dialog (reuse existing LauncherWindow)
            logger.info("User chose: Settings")
            settings_dialog = LauncherWindow()
            settings_dialog.exec_()

            if settings_dialog.was_accepted():
                settings = settings_dialog.get_settings()
                logger.info(
                    "Settings updated - Voice: %s",
                    settings.get("voice_mode"),
                )
            # Loop back to launcher

    logger.info("Goodbye!")
    sys.exit(0)
