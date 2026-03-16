"""Race analysis agent for Jarvis Post."""

import json

from .base import BaseAgent
from ..config.prompts import RACE_ANALYSIS_SYSTEM_PROMPT


class RaceAnalysisAgent(BaseAgent):
    """Agent for comprehensive technical race analysis."""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.max_tokens = 2000
        self.temperature = 0.4

    async def analyse(self, session_data: dict) -> dict:
        """Analyse session data and return technical breakdown.

        Args:
            session_data: Dictionary containing session_metadata, laps, and telemetry.

        Returns:
            Dictionary with 'content' (analysis text) and 'tokens_used'.
        """
        prompt = self._build_prompt(session_data)

        result = await self.llm.generate(
            prompt=prompt,
            system_prompt=RACE_ANALYSIS_SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return self._parse_response(result)

    def analyse_stream(self, session_data: dict):
        """Streaming variant — yields str chunks, then a final StreamComplete."""
        prompt = self._build_prompt(session_data)
        yield from self.llm.generate_stream(
            prompt=prompt,
            system_prompt=RACE_ANALYSIS_SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def _build_prompt(self, session_data: dict) -> str:
        """Build a JSON prompt matching the model's training data format.

        The fine-tuned model was trained on JSON inputs with keys:
        result, lap_summary, laps, stints, pit_stops, car_data_cols, car_data.
        """
        laps = session_data.get("laps", [])
        telemetry = session_data.get("telemetry", [])
        metadata = session_data.get("session_metadata", {})

        # --- lap times list ---
        lap_times = [round(lap.get("lap_time", 0), 3) for lap in laps]
        valid_times = [t for t in lap_times if t > 0]

        # --- lap_summary ---
        fastest_time = min(valid_times) if valid_times else 0
        slowest_time = max(valid_times) if valid_times else 0
        fastest_lap = lap_times.index(fastest_time) + 1 if fastest_time else 1
        slowest_lap = lap_times.index(slowest_time) + 1 if slowest_time else 1
        avg_time = round(sum(valid_times) / len(valid_times), 3) if valid_times else 0

        lap_summary = {
            "total": len(laps),
            "avg": avg_time,
            "fastest": {"lap": fastest_lap, "time": fastest_time},
            "slowest": {"lap": slowest_lap, "time": slowest_time},
        }

        # --- result (position not tracked, use metadata if available) ---
        result = {
            "start": metadata.get("start_position", 1),
            "finish": metadata.get("finish_position", 1),
            "changes": metadata.get("position_changes", 0),
        }

        # --- stints: infer from fuel resets (pit stops) ---
        # Detect pit laps as laps where fuel_start > previous lap fuel_end significantly
        stints = []
        pit_stops = []
        stint_start = 1
        stint_num = 1
        prev_fuel_end = None
        for i, lap in enumerate(laps):
            lap_num = lap.get("lap_number", i + 1)
            fuel_start = lap.get("fuel_start")
            fuel_end = lap.get("fuel_end")
            is_pit = (
                prev_fuel_end is not None
                and fuel_start is not None
                and fuel_start > prev_fuel_end + 2.0  # >2L increase = refuel
            )
            if is_pit and i > 0:
                stint_laps = laps[stint_start - 1: i]
                stint_times = [l.get("lap_time", 0) for l in stint_laps if l.get("lap_time", 0) > 0]
                stints.append({
                    "stint": stint_num,
                    "compound": "UNKNOWN",
                    "laps": f"{stint_start}-{lap_num - 1}",
                    "length": len(stint_laps),
                    "avg_time": round(sum(stint_times) / len(stint_times), 3) if stint_times else 0,
                    "fastest_time": round(min(stint_times), 3) if stint_times else 0,
                })
                pit_stops.append({"lap": lap_num - 1, "duration": None, "inferred": True})
                stint_start = lap_num
                stint_num += 1
            if fuel_end is not None:
                prev_fuel_end = fuel_end

        # Final stint
        final_laps = laps[stint_start - 1:]
        final_times = [l.get("lap_time", 0) for l in final_laps if l.get("lap_time", 0) > 0]
        if final_laps:
            stints.append({
                "stint": stint_num,
                "compound": "UNKNOWN",
                "laps": f"{stint_start}-{len(laps)}",
                "length": len(final_laps),
                "avg_time": round(sum(final_times) / len(final_times), 3) if final_times else 0,
                "fastest_time": round(min(final_times), 3) if final_times else 0,
            })

        # --- car_data: sample up to 20 rows from telemetry ---
        car_data_cols = [
            "t_sec", "speed_kmh", "rpm", "gear", "throttle_pct", "brake", "drs",
            "steer_angle",
            "tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr",
            "tyre_pres_fl", "tyre_pres_fr", "tyre_pres_rl", "tyre_pres_rr",
            "slip_fl", "slip_fr", "slip_rl", "slip_rr",
            "dmg_front", "dmg_rear", "dmg_left", "dmg_right", "dmg_centre",
        ]
        car_data = []
        if telemetry:
            step = max(1, len(telemetry) // 20)
            for row in telemetry[::step][:20]:
                throttle_raw = row.get("throttle", 0) or 0
                brake_raw = row.get("brake", 0) or 0
                throttle_pct = throttle_raw * 100 if throttle_raw <= 1 else throttle_raw
                brake_pct = brake_raw * 100 if brake_raw <= 1 else brake_raw
                car_data.append([
                    round(row.get("elapsed_time", 0), 1),
                    round(row.get("speed", 0), 0),
                    int(row.get("rpm", 0)),
                    int(row.get("gear", 0)),
                    round(throttle_pct, 0),
                    round(brake_pct, 0),
                    int(row.get("drs", 0)),
                    round(row.get("steer_angle", 0), 1),
                    round(row.get("tyre_temp_fl", 0), 1),
                    round(row.get("tyre_temp_fr", 0), 1),
                    round(row.get("tyre_temp_rl", 0), 1),
                    round(row.get("tyre_temp_rr", 0), 1),
                    round(row.get("tyre_pressure_fl", 0), 1),
                    round(row.get("tyre_pressure_fr", 0), 1),
                    round(row.get("tyre_pressure_rl", 0), 1),
                    round(row.get("tyre_pressure_rr", 0), 1),
                    round(row.get("wheel_slip_fl", 0), 1),
                    round(row.get("wheel_slip_fr", 0), 1),
                    round(row.get("wheel_slip_rl", 0), 1),
                    round(row.get("wheel_slip_rr", 0), 1),
                    round(row.get("car_damage_front", 0), 0),
                    round(row.get("car_damage_rear", 0), 0),
                    round(row.get("car_damage_left", 0), 0),
                    round(row.get("car_damage_right", 0), 0),
                    round(row.get("car_damage_centre", 0), 0),
                ])

        payload = {
            "result": result,
            "lap_summary": lap_summary,
            "laps": lap_times,
            "stints": stints,
            "pit_stops": pit_stops,
            "car_data_cols": car_data_cols,
            "car_data": car_data,
        }

        return json.dumps(payload, separators=(",", ":"))

    def _parse_response(self, result: dict) -> dict:
        return {
            "content": result.get("content", ""),
            "tokens_used": result.get("tokens_used", 0),
        }
