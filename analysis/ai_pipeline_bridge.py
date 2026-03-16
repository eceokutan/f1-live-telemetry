"""
Bridge layer between the UI and an optional external explorer AI pipeline.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple
from pathlib import Path
import asyncio
import importlib
import inspect
import logging
import os
import sys
import threading
from contextlib import contextmanager

import pandas as pd

from data.lap import Lap
from data.session import Session
from .ai_pipeline_types import (
    AIAnalysisResult,
    EXTERNAL_PIPELINE_MODULES,
)

logger = logging.getLogger(__name__)


class AIPipelineBridge:
    """
    Desktop-first adapter that runs AI pipelines in-process and falls back to
    local telemetry heuristics when external AI is unavailable.
    """

    def __init__(self, module_candidates: Optional[tuple[str, ...]] = None):
        self._module_candidates = module_candidates or EXTERNAL_PIPELINE_MODULES
        self._external_handler: Optional[Any] = None
        self._external_source = ""
        self._jarvis_llm_client: Optional[Any] = None
        self._jarvis_warmup_started = False
        self._discover_external_pipeline()

    def generate(self, session: Session, lap: Lap) -> AIAnalysisResult:
        """Generate coach + analyst outputs for a specific lap."""
        external = self._generate_from_external(session, lap)
        if external:
            return external

        from_comments = self._generate_from_commentary(session, lap)
        if from_comments:
            if from_comments.coach and from_comments.analyst:
                return from_comments
            fallback = self._generate_fallback(session, lap)
            return AIAnalysisResult(
                coach=from_comments.coach or fallback.coach,
                analyst=from_comments.analyst or fallback.analyst,
                source=f"{from_comments.source}+built_in_fallback",
            )

        return self._generate_fallback(session, lap)

    def _discover_external_pipeline(self) -> None:
        """Locate an external explorer AI pipeline if available."""
        jarvis_handler = self._discover_jarvis_post_pipeline()
        if jarvis_handler:
            self._external_handler = jarvis_handler
            self._start_jarvis_warmup(jarvis_handler)
            return

        for module_name in self._module_candidates:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue

            for class_name in ("EIMAPipeline", "AIPipeline", "Pipeline"):
                pipeline_cls = getattr(module, class_name, None)
                if callable(pipeline_cls):
                    try:
                        self._external_handler = pipeline_cls()
                        self._external_source = f"{module_name}.{class_name}"
                        return
                    except Exception:
                        continue

            for function_name in ("analyze", "run", "generate", "process"):
                fn = getattr(module, function_name, None)
                if callable(fn):
                    self._external_handler = fn
                    self._external_source = f"{module_name}.{function_name}"
                    return

    def _discover_jarvis_post_pipeline(self) -> Optional[dict[str, Any]]:
        """
        Locate Jarvis Post package and return callables for analyst/coach.

        Discovery order is deterministic:
        1) Explicit root via JARVIS_POST_ROOT (if valid)
        2) Repository root (analysis/..)
        """
        for root in self._get_jarvis_post_candidate_roots():
            if not self._is_valid_jarvis_post_root(root):
                logger.warning(
                    "Skipping invalid JARVIS_POST_ROOT candidate: %s",
                    root,
                )
                continue

            handler = self._load_jarvis_post_handler(root)
            if handler is None:
                continue

            self._external_source = f"jarvis_post:{root}"
            logger.info("Using Jarvis Post pipeline root: %s", root)
            return handler

        return None

    def _get_jarvis_post_candidate_roots(self) -> list[Path]:
        """
        Return deterministic Jarvis Post candidate roots.

        Avoids scanning unrelated directories or importing external .env files.
        """
        candidates: list[Path] = []

        env_root = (os.getenv("JARVIS_POST_ROOT") or "").strip()
        if env_root:
            try:
                candidates.append(Path(env_root).expanduser().resolve())
            except Exception:
                logger.warning("Invalid JARVIS_POST_ROOT value: %s", env_root)

        repo_root = Path(__file__).resolve().parent.parent
        if repo_root not in candidates:
            candidates.append(repo_root)

        return candidates

    def _is_valid_jarvis_post_root(self, root: Path) -> bool:
        """Validate that a root contains the expected local Jarvis Post package."""
        coaching_file = root / "jarvis_post" / "agents" / "coaching.py"
        race_analysis_file = root / "jarvis_post" / "agents" / "race_analysis.py"
        return coaching_file.exists() and race_analysis_file.exists()

    @contextmanager
    def _temporary_sys_path(self, root: Path):
        """Temporarily prepend a root to sys.path for controlled imports."""
        root_str = str(root)
        inserted = False
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
            inserted = True
        try:
            yield
        finally:
            if inserted:
                try:
                    sys.path.remove(root_str)
                except ValueError:
                    pass

    def _load_jarvis_post_handler(self, root: Path) -> Optional[dict[str, Any]]:
        """
        Import Jarvis Post agent/client classes from a specific root.

        Returns:
            Normalized handler dict or None if import failed.
        """
        with self._temporary_sys_path(root):
            importlib.invalidate_caches()
            try:
                race_module = importlib.import_module("jarvis_post.agents.race_analysis")
                coaching_module = importlib.import_module("jarvis_post.agents.coaching")
                race_agent_cls = getattr(race_module, "RaceAnalysisAgent", None)
                coach_agent_cls = getattr(coaching_module, "CoachingAgent", None)
                if not callable(race_agent_cls) or not callable(coach_agent_cls):
                    logger.warning("Jarvis Post agent classes missing under root: %s", root)
                    return None
            except Exception as exc:
                logger.warning("Failed importing Jarvis Post agents from %s: %s", root, exc)
                return None

            llm_client_cls = None
            try:
                local_client_module = importlib.import_module("jarvis_post.llm.local_client")
                llm_client_cls = getattr(local_client_module, "LocalGGUFClient", None)
            except Exception as exc:
                logger.warning("Jarvis Post local GGUF client import failed from %s: %s", root, exc)
                llm_client_cls = None

            return {
                "kind": "jarvis_post",
                "root": root,
                "race_agent_cls": race_agent_cls,
                "coach_agent_cls": coach_agent_cls,
                "llm_client_cls": llm_client_cls,
            }

    def _generate_from_external(self, session: Session, lap: Lap) -> Optional[AIAnalysisResult]:
        if self._external_handler is None:
            return None

        if isinstance(self._external_handler, dict) and self._external_handler.get("kind") == "jarvis_post":
            return self._generate_from_jarvis_post(session, lap, self._external_handler)

        context = self._build_context(session, lap)
        coach, analyst = self._call_external_combined(self._external_handler, session, lap, context)

        if not coach:
            coach = self._call_external_role(self._external_handler, "coach", session, lap, context)
        if not analyst:
            analyst = self._call_external_role(self._external_handler, "analyst", session, lap, context)

        if coach and analyst:
            return AIAnalysisResult(
                coach=coach,
                analyst=analyst,
                source=f"external:{self._external_source}",
            )

        return None

    def _generate_from_jarvis_post(
        self,
        session: Session,
        lap: Lap,
        handler: dict[str, Any],
    ) -> Optional[AIAnalysisResult]:
        """Run dual-agent post-race analysis using Jarvis Post (analyst + coach)."""
        race_agent_cls = handler.get("race_agent_cls")
        coach_agent_cls = handler.get("coach_agent_cls")
        llm_client_cls = handler.get("llm_client_cls")

        if not all((race_agent_cls, coach_agent_cls)):
            return None

        payload = self._build_jarvis_post_payload(session, lap)
        coach_payload = dict(payload)
        coach_payload.pop("options", None)
        coach_payload["driver_context"] = self._build_driver_context(session, lap)

        if not llm_client_cls:
            logger.warning("Jarvis Post LLM client class is unavailable; skipping external analysis")
            return None
        if not self._has_postrace_local_model():
            logger.warning(
                "Jarvis Post external AI disabled: GGUF model file not found"
            )
            return None

        try:
            if self._jarvis_llm_client is None:
                self._jarvis_llm_client = llm_client_cls()
            analyst_agent = race_agent_cls(self._jarvis_llm_client)
            coach_agent = coach_agent_cls(self._jarvis_llm_client)
        except Exception as exc:
            logger.error("Failed to initialize Jarvis Post agents/client: %s", exc, exc_info=True)
            return None

        try:
            analyst_raw, coach_raw = self._run_async(
                self._run_dual_agent_analysis(
                    analyst_agent=analyst_agent,
                    coach_agent=coach_agent,
                    analyst_payload=payload,
                    coach_payload=coach_payload,
                )
            )
        except Exception as exc:
            logger.error("Jarvis Post external AI request failed: %s", exc, exc_info=True)
            return None

        analyst = self._normalize_text(analyst_raw.get("content") if isinstance(analyst_raw, dict) else analyst_raw)
        coach = self._normalize_text(coach_raw.get("content") if isinstance(coach_raw, dict) else coach_raw)

        if coach and analyst:
            return AIAnalysisResult(coach=coach, analyst=analyst, source=f"external:{self._external_source}")

        return None

    async def _run_dual_agent_analysis(
        self,
        analyst_agent: Any,
        coach_agent: Any,
        analyst_payload: dict[str, Any],
        coach_payload: dict[str, Any],
    ) -> tuple[Any, Any]:
        """Run analyst + coach in parallel to reduce end-to-end latency."""
        analyst_raw, coach_raw = await asyncio.gather(
            analyst_agent.analyse(analyst_payload),
            coach_agent.analyse(coach_payload),
        )
        return analyst_raw, coach_raw

    def _has_postrace_local_model(self) -> bool:
        """Check whether the local GGUF model file exists for post-race analysis."""
        env_path = (os.getenv("POSTRACE_GGUF_MODEL_PATH") or "").strip()
        if env_path:
            return Path(env_path).exists()

        default_path = Path(__file__).resolve().parent.parent / "postrace_gguf" / "granite-postrace-analyst-Q5_K_M.gguf"
        return default_path.exists()

    def _call_external_combined(
        self,
        handler: Any,
        session: Session,
        lap: Lap,
        context: dict[str, Any],
    ) -> Tuple[str, str]:
        callables: list[Callable[..., Any]] = []
        if callable(handler):
            callables.append(handler)
        for method_name in ("analyze", "run", "generate", "process"):
            method = getattr(handler, method_name, None)
            if callable(method):
                callables.append(method)

        for fn in callables:
            try:
                response = self._invoke_callable(fn, session, lap, context)
            except Exception:
                continue

            coach, analyst = self._extract_coach_analyst(response)
            if coach or analyst:
                return coach, analyst

        return "", ""

    def _call_external_role(
        self,
        handler: Any,
        role: str,
        session: Session,
        lap: Lap,
        context: dict[str, Any],
    ) -> str:
        role_methods = ("analyze_model", "run_model", "generate_model", "analyze_role")

        for method_name in role_methods:
            method = getattr(handler, method_name, None)
            if not callable(method):
                continue
            try:
                response = self._invoke_callable(method, session, lap, context, role=role)
            except Exception:
                continue
            text = self._normalize_text(response)
            if text:
                return text

        if callable(handler):
            try:
                response = self._invoke_callable(handler, session, lap, context, role=role)
            except Exception:
                return ""
            return self._normalize_text(response)

        return ""

    def _invoke_callable(
        self,
        fn: Callable[..., Any],
        session: Session,
        lap: Lap,
        context: dict[str, Any],
        role: Optional[str] = None,
    ) -> Any:
        signature = inspect.signature(fn)
        params = signature.parameters

        known_kwargs: dict[str, Any] = {
            "session": session,
            "lap": lap,
            "context": context,
            "models": ["coach", "analyst"],
            "roles": ["coach", "analyst"],
        }
        if role:
            known_kwargs["role"] = role
            known_kwargs["model"] = role

        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in params.values()
        )

        if accepts_var_kwargs:
            return fn(**known_kwargs)

        kwargs = {name: value for name, value in known_kwargs.items() if name in params}
        if kwargs:
            return fn(**kwargs)

        required = [
            parameter
            for parameter in params.values()
            if parameter.default is inspect._empty
            and parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        required_count = len(required)

        if required_count >= 3:
            return fn(session, lap, context)
        if required_count == 2:
            return fn(session, lap)
        if required_count == 1:
            return fn(context)
        return fn()

    def _extract_coach_analyst(self, response: Any) -> Tuple[str, str]:
        if isinstance(response, dict):
            coach = self._extract_text_from_dict(response, ("coach", "coach_output", "coach_response"))
            analyst = self._extract_text_from_dict(
                response, ("analyst", "analysis", "analyst_output", "analyst_response")
            )

            nested = response.get("responses")
            if isinstance(nested, dict):
                if not coach:
                    coach = self._extract_text_from_dict(nested, ("coach",))
                if not analyst:
                    analyst = self._extract_text_from_dict(nested, ("analyst",))

            return coach, analyst

        if isinstance(response, (tuple, list)) and len(response) >= 2:
            return self._normalize_text(response[0]), self._normalize_text(response[1])

        return "", ""

    def _extract_text_from_dict(self, payload: dict[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            if key not in payload:
                continue
            text = self._normalize_text(payload.get(key))
            if text:
                return text
        return ""

    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in ("text", "message", "content", "output", "analysis"):
                text = value.get(key)
                if isinstance(text, str) and text.strip():
                    return text.strip()
            return str(value).strip()
        if isinstance(value, (tuple, list)):
            parts = [self._normalize_text(item) for item in value]
            parts = [part for part in parts if part]
            return "\n".join(parts)
        return str(value).strip()

    def _generate_from_commentary(self, session: Session, lap: Lap) -> Optional[AIAnalysisResult]:
        if not session.ai_commentary:
            return None

        lap_comments = [comment for comment in session.ai_commentary if comment.lap_number == lap.lap_number]
        if not lap_comments:
            return None

        coach_messages = [
            comment.message.strip()
            for comment in lap_comments
            if (comment.model or "").strip().lower() == "coach" and comment.message.strip()
        ]
        analyst_messages = [
            comment.message.strip()
            for comment in lap_comments
            if (comment.model or "").strip().lower() == "analyst" and comment.message.strip()
        ]
        unscoped_messages = [
            comment.message.strip()
            for comment in lap_comments
            if not (comment.model or "").strip() and comment.message.strip()
        ]

        coach = "\n\n".join(coach_messages[:4]) if coach_messages else ""
        analyst = "\n\n".join(analyst_messages[:4]) if analyst_messages else ""

        if unscoped_messages and not analyst:
            analyst = "\n\n".join(unscoped_messages[:4])

        if coach or analyst:
            return AIAnalysisResult(coach=coach, analyst=analyst, source="session_ai_commentary")

        return None

    def _generate_fallback(self, session: Session, lap: Lap) -> AIAnalysisResult:
        summary = lap.summary
        df = lap.telemetry

        throttle_mean = self._mean_percentage(df, "throttle")
        brake_mean = self._mean_percentage(df, "brake")
        heavy_brake_samples = int((df["brake"] >= 0.80).sum()) if "brake" in df.columns else 0

        avg_tire_temp = self._mean_columns(
            df,
            ("tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr"),
        )
        avg_tire_pressure = self._mean_columns(
            df,
            ("tyre_pressure_fl", "tyre_pressure_fr", "tyre_pressure_rl", "tyre_pressure_rr"),
        )

        # G-forces
        avg_g_lat = self._safe_mean(df, "g_force_lat")
        max_g_lat = self._safe_max(df, "g_force_lat")
        avg_g_lon = self._safe_mean(df, "g_force_lon")
        max_g_lon = self._safe_max(df, "g_force_lon")

        # Wheel slip — count samples where any tire exceeds threshold
        slip_cols = ("wheel_slip_fl", "wheel_slip_fr", "wheel_slip_rl", "wheel_slip_rr")
        excessive_slip_samples = 0
        max_slip = 0.0
        for col in slip_cols:
            if col in df.columns:
                excessive_slip_samples += int((df[col].abs() > 5.0).sum())
                col_max = float(df[col].abs().max())
                if col_max > max_slip:
                    max_slip = col_max

        # Suspension travel
        avg_susp = self._mean_columns(
            df,
            ("suspension_fl", "suspension_fr", "suspension_rl", "suspension_rr"),
        )

        # Ride height
        avg_ride_front = self._safe_mean(df, "ride_height_front")
        avg_ride_rear = self._safe_mean(df, "ride_height_rear")

        # Car damage
        damage_cols = ("car_damage_front", "car_damage_rear", "car_damage_left",
                       "car_damage_right", "car_damage_centre")
        max_damage = 0.0
        damage_zones = []
        for col in damage_cols:
            if col in df.columns:
                zone_max = float(df[col].max())
                if zone_max > 0:
                    zone_name = col.replace("car_damage_", "")
                    damage_zones.append(f"{zone_name}: {zone_max:.0f}%")
                if zone_max > max_damage:
                    max_damage = zone_max
        damage_str = ", ".join(damage_zones) if damage_zones else "No damage"

        fastest_lap = session.get_fastest_lap()
        if fastest_lap and fastest_lap.lap_number != lap.lap_number:
            delta = lap.lap_time - fastest_lap.lap_time
            lap_delta = f"{delta:+.3f}s vs fastest lap ({fastest_lap.lap_number})"
        else:
            lap_delta = "Current lap is session fastest or only valid lap."

        # Build coaching lines
        coach_lines = [
            f"Lap {lap.lap_number} coaching focus:",
            f"- Smooth brake release into apex; heavy-brake samples: {heavy_brake_samples}.",
            f"- Balance throttle pickup after apex; mean throttle is {throttle_mean:.1f}%.",
            f"- Keep tire temps in working range (avg {avg_tire_temp:.1f} C).",
            f"- Fuel used this lap: {summary.fuel_used:.2f} L.",
        ]
        if excessive_slip_samples > 0:
            coach_lines.append(
                f"- Excessive wheel slip detected in {excessive_slip_samples} samples "
                f"(max slip: {max_slip:.1f}). Smooth inputs on throttle/brake to reduce wheelspin."
            )
        if max_damage > 0:
            coach_lines.append(f"- Car damage present ({damage_str}). Drive cautiously.")
        coach = "\n".join(coach_lines)

        # Build analyst lines
        analyst_lines = [
            f"Lap {lap.lap_number} telemetry summary:",
            f"- Lap time: {lap.lap_time:.3f}s ({lap_delta})",
            f"- Speed: avg {summary.avg_speed:.1f} km/h, max {summary.max_speed:.1f} km/h, min {summary.min_speed:.1f} km/h.",
            f"- Inputs: mean throttle {throttle_mean:.1f}%, mean brake {brake_mean:.1f}%.",
            f"- Tires: avg temperature {avg_tire_temp:.1f} C, avg pressure {avg_tire_pressure:.2f} PSI.",
        ]
        if avg_g_lat is not None:
            analyst_lines.append(
                f"- G-Forces: lateral avg {avg_g_lat:.2f}G (max {max_g_lat:.2f}G), "
                f"longitudinal avg {avg_g_lon:.2f}G (max {max_g_lon:.2f}G)."
            )
        if max_slip > 0:
            analyst_lines.append(
                f"- Wheel slip: max {max_slip:.1f}, excessive slip samples: {excessive_slip_samples}."
            )
        if avg_susp > 0:
            analyst_lines.append(f"- Suspension travel: avg {avg_susp:.4f} m.")
        if avg_ride_front is not None:
            analyst_lines.append(
                f"- Ride height: front avg {avg_ride_front:.4f} m, rear avg {avg_ride_rear:.4f} m."
            )
        if max_damage > 0:
            analyst_lines.append(f"- Car damage: {damage_str}.")
        analyst = "\n".join(analyst_lines)

        return AIAnalysisResult(
            coach=coach,
            analyst=analyst,
            source="built_in_fallback",
        )

    def _build_context(self, session: Session, lap: Lap) -> dict[str, Any]:
        summary = lap.summary
        return {
            "session": {
                "session_id": session.metadata.session_id,
                "track": session.metadata.track_name,
                "car": session.metadata.car_model,
                "player": session.metadata.player_name,
                "total_laps": len(session.laps),
            },
            "lap": {
                "lap_number": lap.lap_number,
                "lap_time": lap.lap_time,
                "avg_speed": summary.avg_speed,
                "max_speed": summary.max_speed,
                "min_speed": summary.min_speed,
                "fuel_used": summary.fuel_used,
            },
        }

    def _build_jarvis_post_payload(self, session: Session, lap: Lap) -> dict[str, Any]:
        """Convert local session structures to Jarvis Post request shape."""
        fastest = session.get_fastest_lap()
        telemetry_columns = [
            "lap_number", "elapsed_time", "pos_x", "pos_z",
            "speed", "gear", "rpm", "throttle", "brake", "drs", "fuel",
            "tyre_pressure_fl", "tyre_pressure_fr", "tyre_pressure_rl", "tyre_pressure_rr",
            "tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr",
            "g_force_lat", "g_force_lon", "steer_angle",
            "wheel_slip_fl", "wheel_slip_fr", "wheel_slip_rl", "wheel_slip_rr",
            "suspension_fl", "suspension_fr", "suspension_rl", "suspension_rr",
            "ride_height_front", "ride_height_rear",
            "car_damage_front", "car_damage_rear", "car_damage_left",
            "car_damage_right", "car_damage_centre",
        ]

        lap_df = lap.telemetry
        available_columns = [column for column in telemetry_columns if column in lap_df.columns]
        telemetry_records = lap_df[available_columns].to_dict("records")

        laps_payload = []
        for lap_item in session.laps:
            laps_payload.append({
                "lap_number": lap_item.lap_number,
                "lap_time": float(lap_item.lap_time),
                "fuel_start": float(lap_item.summary.fuel_start),
                "fuel_end": float(lap_item.summary.fuel_end),
                "avg_speed": float(lap_item.summary.avg_speed),
                "max_speed": float(lap_item.summary.max_speed),
                "valid": bool(lap_item.summary.valid),
            })

        metadata = {
            "game": session.metadata.game,
            "track_name": session.metadata.track_name,
            "car_model": session.metadata.car_model,
            "player_name": session.metadata.player_name,
            "total_laps": len(session.laps),
            "best_lap_time": float(fastest.lap_time) if fastest else None,
        }

        return {
            "session_id": f"session_{session.metadata.session_id}",
            "session_metadata": metadata,
            "laps": laps_payload,
            "telemetry": telemetry_records,
            "options": {
                "focus_lap": lap.lap_number,
                "include_lap_breakdown": True,
                "include_tyre_analysis": True,
                "include_fuel_analysis": True,
            },
        }

    def _build_driver_context(self, session: Session, lap: Lap) -> dict[str, Any]:
        """Build lightweight driver context for coaching prompt tuning."""
        fastest = session.get_fastest_lap()
        is_fastest_lap = bool(fastest and fastest.lap_number == lap.lap_number)
        focus_areas = []

        if lap.summary.avg_speed < 120:
            focus_areas.append("minimum corner speed")
        if lap.summary.max_speed < 220:
            focus_areas.append("straight-line speed")
        if not focus_areas:
            focus_areas = ["corner exits", "consistency"]

        return {
            "experience_level": "intermediate",
            "focus_areas": focus_areas,
            "recent_sessions": 1,
            "current_lap_is_fastest": is_fastest_lap,
        }

    def _run_async(self, coroutine: Any) -> Any:
        """Run a coroutine from synchronous UI context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            result_box: dict[str, Any] = {"value": None, "error": None}

            def _runner() -> None:
                try:
                    result_box["value"] = asyncio.run(coroutine)
                except Exception as exc:
                    result_box["error"] = exc

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            thread.join()

            if result_box["error"] is not None:
                raise result_box["error"]
            return result_box["value"]

        return asyncio.run(coroutine)

    def _start_jarvis_warmup(self, handler: dict[str, Any]) -> None:
        """Best-effort background warmup so first interactive request is faster."""
        if self._jarvis_warmup_started:
            return
        if not self._has_postrace_local_model():
            return

        llm_client_cls = handler.get("llm_client_cls")
        if not llm_client_cls:
            return

        self._jarvis_warmup_started = True

        def _warmup() -> None:
            try:
                if self._jarvis_llm_client is None:
                    self._jarvis_llm_client = llm_client_cls()
                warmup_fn = getattr(self._jarvis_llm_client, "warmup_sync", None)
                if callable(warmup_fn):
                    warmup_fn()
            except Exception as exc:
                logger.warning("Jarvis Post warmup failed: %s", exc)

        threading.Thread(target=_warmup, daemon=True).start()

    def _mean_columns(self, df: pd.DataFrame, columns: tuple[str, ...]) -> float:
        values = []
        for column in columns:
            if column in df.columns:
                values.append(float(df[column].mean()))
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _mean_percentage(self, df: pd.DataFrame, column: str) -> float:
        if column not in df.columns:
            return 0.0
        return float(df[column].mean()) * 100.0

    def _safe_mean(self, df: pd.DataFrame, column: str) -> Optional[float]:
        if column not in df.columns:
            return None
        return float(df[column].mean())

    def _safe_max(self, df: pd.DataFrame, column: str) -> Optional[float]:
        if column not in df.columns:
            return None
        return float(df[column].abs().max())
