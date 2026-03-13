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

# Import torch BEFORE PyQt5 to avoid DLL conflict on Windows
# (PyQt5 changes DLL search paths, breaking torch's c10.dll loading)
try:
    import torch
except ImportError:
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

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

logger.info("Jarvis F1 Telemetry Suite starting")

# Import UI
from ui.main_window import MainWindow

# Import telemetry backends
from telemetry.backends.ac_backend import AcTelemetryWorker
from telemetry.backends.acc_backend import AccTelemetryWorker

# Try to import AI worker (optional)
try:
    from ai.race_engineer import AIRaceEngineerWorker
    AI_AVAILABLE = True
    logger.info("AI Race Engineer module available")
except Exception as e:
    AI_AVAILABLE = False
    logger.warning("AI Race Engineer not available: %s", e)

# Try to import voice input worker (optional)
try:
    from ai.voice_input import VoiceInputWorker
    VOICE_AVAILABLE = True
    logger.info("Voice Input module available")
except Exception as e:
    VOICE_AVAILABLE = False
    logger.warning("Voice Input not available: %s", e)

# Try to import TTS output worker (optional)
try:
    from ai.tts_output import TTSOutputWorker
    TTS_AVAILABLE = True
    logger.info("TTS Output module available")
except Exception as e:
    TTS_AVAILABLE = False
    logger.warning("TTS Output not available: %s", e)

# Try to import PTT controller (optional)
try:
    from ai.ptt_controller import PTTController
    PTT_AVAILABLE = True
    logger.info("PTT Controller module available")
except ImportError as e:
    PTT_AVAILABLE = False
    logger.warning("PTT Controller not available: %s", e)

# Try to import session recorder (optional)
try:
    from data.session_recorder import SessionRecorder
    RECORDER_AVAILABLE = True
    logger.info("Session Recorder module available")
except ImportError as e:
    RECORDER_AVAILABLE = False
    logger.warning("Session Recorder not available: %s", e)

logger.info("All core modules imported")

_prewarm_started = False
KOKORO_VOICE_ID = "bm_lewis"
KOKORO_LANG = "en-gb"
KOKORO_SPEED = 1.3
KOKORO_USE_CUDA = False


def _should_prewarm_models() -> bool:
    """Whether to run background model prewarm at startup."""
    val = os.getenv("PREWARM_MODELS_ON_START", "1").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _start_background_model_prewarm(_settings: dict):
    """Prewarm STT/TTS model caches in background while user is in launcher."""
    global _prewarm_started
    if _prewarm_started or not _should_prewarm_models():
        return

    _prewarm_started = True

    def _worker():
        prewarm_stt = True
        prewarm_tts = True
        try:
            # Run cache checks inside the worker so launcher creation is never
            # blocked by optional model imports (e.g., faster_whisper).
            try:
                from ai.model_prewarm import needs_faster_whisper_prewarm, needs_kokoro_prewarm

                prewarm_stt = needs_faster_whisper_prewarm(model_size="base")
                prewarm_tts = needs_kokoro_prewarm()
            except Exception as e:
                logger.warning(
                    "Could not inspect model cache state, falling back to full prewarm: %s",
                    e,
                )

            if not prewarm_stt and not prewarm_tts:
                logger.info("Background model prewarm skipped (all required caches already warm)")
                return

            from ai.model_prewarm import prewarm_faster_whisper, prewarm_kokoro

            logger.info(
                "Background model prewarm started (stt=%s, tts=%s)",
                prewarm_stt,
                prewarm_tts,
            )

            if prewarm_stt:
                try:
                    prewarm_faster_whisper(model_size="base")
                    logger.info("Background prewarm: faster-whisper ready")
                except Exception as e:
                    logger.warning("Background prewarm: faster-whisper failed: %s", e)
            else:
                logger.info("Background prewarm: faster-whisper skipped (cache warm)")

            if prewarm_tts:
                try:
                    prewarm_kokoro(
                        voice_id=KOKORO_VOICE_ID,
                        lang=KOKORO_LANG,
                        speed=KOKORO_SPEED,
                        use_cuda=KOKORO_USE_CUDA,
                    )
                    logger.info("Background prewarm: Kokoro ready")
                except Exception as e:
                    logger.warning("Background prewarm: Kokoro failed: %s", e)
            else:
                logger.info("Background prewarm: Kokoro skipped (cache warm)")

            logger.info("Background model prewarm finished")
        except Exception as e:
            logger.warning("Background prewarm initialization failed: %s", e)

    threading.Thread(target=_worker, name="model-prewarm", daemon=True).start()


def run_jarvis_live(settings: dict):
    """
    Launch Jarvis Live - real-time telemetry dashboard.

    Args:
        settings: Configuration dict from the launcher.
    """
    game = "ac"
    enable_ai = settings.get("ai_enabled", False)
    enable_ptt = settings.get("voice_mode") == "push_to_talk"
    ptt_key = settings.get("ptt_key", "v")

    # Inject credentials into environment so existing code picks them up
    credential_map = {
        "HUGGINGFACE_TOKEN": "huggingface_token",
        "HUGGINGFACE_MODEL_ID": "huggingface_model_id",
    }
    for env_key, settings_key in credential_map.items():
        val = settings.get(settings_key, "")
        if val:
            os.environ[env_key] = val

    logger.info("Starting Jarvis Live for: %s", game.upper())

    window = MainWindow()

    # Choose backend
    logger.info("Initializing %s telemetry backend", game.upper())
    if game == "ac":
        telemetry_thread = AcTelemetryWorker()
    elif game == "acc":
        telemetry_thread = AccTelemetryWorker(host="127.0.0.1", port=9232, password="")
    else:
        raise ValueError(f"Unknown game '{game}'. Use 'ac' or 'acc'.")

    # Connect signals
    telemetry_thread.lap_completed.connect(window.handle_lap_complete)
    telemetry_thread.status_update.connect(lambda msg: logger.info("Status: %s", msg))

    if hasattr(telemetry_thread, 'session_info_update'):
        telemetry_thread.session_info_update.connect(window.update_session_info)
    if hasattr(telemetry_thread, 'live_data_update'):
        telemetry_thread.live_data_update.connect(window.update_live_data)
    if hasattr(telemetry_thread, 'realtime_sample'):
        telemetry_thread.realtime_sample.connect(window.handle_realtime_sample)

    logger.info("Signals connected")

    # Initialize AI race engineer (optional)
    ai_thread = None
    voice_thread = None
    tts_thread = None
    tts_runtime = {"worker": None, "restarting": False}
    ptt_controller = None
    if enable_ai and AI_AVAILABLE:
        logger.info("Initializing AI Race Engineer")

        huggingface_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_API_KEY", "")
        huggingface_model_id = os.getenv("HUGGINGFACE_MODEL_ID", "")

        if not huggingface_token or not huggingface_model_id:
            logger.warning("AI Race Engineer requires HUGGINGFACE_TOKEN and HUGGINGFACE_MODEL_ID in .env")
        else:
            try:
                ai_thread = AIRaceEngineerWorker(
                    huggingface_token=huggingface_token,
                    hf_model_id=huggingface_model_id,
                    track_name="Unknown Track",
                    session_id="ac_session_001",
                    verbosity="moderate"
                )

                ai_thread.ai_commentary.connect(window.handle_ai_commentary)
                ai_thread.driver_query_received.connect(window.handle_driver_query)
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

                ai_thread.start()
                logger.info("AI Race Engineer started")

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
                        voice_thread.speech_detected.connect(ai_thread.process_driver_query)
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
                    ptt_button_index = 11
                    logger.info("Initializing PTT Controller (button index=%d)", ptt_button_index)
                    try:
                        ptt_controller = PTTController(joystick_button_index=ptt_button_index)
                        ptt_controller.ptt_pressed.connect(voice_thread.start_recording)
                        ptt_controller.ptt_released.connect(voice_thread.stop_recording)
                        ptt_controller.status_update.connect(lambda msg: logger.info("PTT: %s", msg))
                        ptt_controller.start()
                        logger.info("PTT Controller started")
                    except Exception as e:
                        logger.error("Failed to initialize PTT Controller: %s", e, exc_info=True)
                        ptt_controller = None

                # Initialize TTS output (Kokoro local TTS)
                if TTS_AVAILABLE and voice_mode_setting != "disabled":
                    logger.info(
                        "Initializing TTS Output (kokoro, sentence pipelining mode, voice=%s, speed=%.2f)",
                        KOKORO_VOICE_ID,
                        KOKORO_SPEED,
                    )
                    try:
                        def _build_tts_worker() -> TTSOutputWorker:
                            worker = TTSOutputWorker(
                                use_sentence_pipelining=True,
                                kokoro_voice_id=KOKORO_VOICE_ID,
                                kokoro_lang=KOKORO_LANG,
                                kokoro_speed=KOKORO_SPEED,
                                kokoro_use_cuda=KOKORO_USE_CUDA,
                            )
                            worker.status_update.connect(lambda msg: logger.info("TTS: %s", msg))
                            worker.error_occurred.connect(lambda err: logger.error("TTS: %s", err))
                            worker.finished.connect(lambda: logger.warning("TTS worker stopped"))
                            if voice_thread:
                                worker.playback_started.connect(voice_thread.pause)
                                worker.playback_finished.connect(voice_thread.resume)
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

                        def _speak_with_retry(message: str, retries_left: int = 6):
                            worker = tts_runtime["worker"]
                            if worker and worker.isRunning():
                                worker.speak(message)
                                return

                            if not _ensure_tts_worker_running("auto-restart"):
                                if retries_left <= 0:
                                    logger.error("Dropping TTS message after failed restart: %s", message[:80])
                                    return
                                QtCore.QTimer.singleShot(
                                    200, lambda m=message, r=retries_left - 1: _speak_with_retry(m, r)
                                )
                                return

                            if retries_left <= 0:
                                logger.error("Dropping TTS message; worker never became ready: %s", message[:80])
                                return

                            QtCore.QTimer.singleShot(
                                200, lambda m=message, r=retries_left - 1: _speak_with_retry(m, r)
                            )

                        def on_ai_commentary_for_tts(msg, trigger, priority):
                            _speak_with_retry(msg)

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

            except Exception as e:
                logger.error("Failed to initialize AI Race Engineer: %s", e, exc_info=True)
                ai_thread = None
    elif enable_ai:
        logger.warning("AI requested but AIRaceEngineerWorker module not available")

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
                        ai_enabled=enable_ai,
                        session_type=info.get("session_type", "")
                    )

            if hasattr(telemetry_thread, 'session_info_update'):
                telemetry_thread.session_info_update.connect(on_session_info)

            if hasattr(telemetry_thread, 'realtime_sample'):
                telemetry_thread.realtime_sample.connect(
                    lambda sample: recorder_thread.record_telemetry_sample(sample)
                )

            def on_lap_complete(lap_id: int, samples: list):
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

                # NOTE: per-sample lap_valid (based on gfx.lastTimeMs) is
                # unreliable — lastTimeMs only updates AFTER a lap completes,
                # so during the first lap it's always 0 making every sample
                # "invalid".  Default to valid=True; the AC backend already
                # logs the correct lastTimeMs at the moment of completion.
                lap_valid = True

                if recorder_thread:
                    recorder_thread.record_lap(
                        lap_number=lap_id,
                        lap_time=lap_time,
                        fuel_start=fuel_start,
                        fuel_end=fuel_end,
                        avg_speed=avg_speed,
                        max_speed=max_speed,
                        min_speed=min_speed,
                        valid=lap_valid
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


def run_jarvis_post(session_id: int):
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
        return

    # Launch post-race viewer
    window = LapViewerWindow()
    window.show()

    # Load the exported session
    window.load_session_from_file(export_dir)

    # Run Qt event loop (blocks until window is closed)
    app = QtWidgets.QApplication.instance()
    app.exec_()

    logger.info("Jarvis Post shutdown complete")


if __name__ == "__main__":
    from ui.unified_launcher import UnifiedLauncher
    from ui.launcher import LauncherWindow
    from ui.session_picker import SessionPickerDialog
    from ui.config_manager import load_config

    app = QtWidgets.QApplication(sys.argv)

    # Load custom fonts (Bebas Neue, Rajdhani) before any UI is shown
    from ui.styles import load_fonts
    load_fonts()

    # Load saved settings once
    settings = load_config()
    _start_background_model_prewarm(settings)

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
                run_jarvis_live(settings)
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
                    run_jarvis_post(session_id)
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
                    "Settings updated - AI: %s | Voice: %s",
                    settings.get("ai_enabled"),
                    settings.get("voice_mode"),
                )
            # Loop back to launcher

    logger.info("Goodbye!")
    sys.exit(0)
