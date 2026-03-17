"""Race analysis agent for Jarvis Post."""

import json
from collections import defaultdict

from .base import BaseAgent
from ..config.prompts import RACE_ANALYSIS_SYSTEM_PROMPT


class RaceAnalysisAgent(BaseAgent):
    """Agent for comprehensive technical race analysis."""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.max_tokens = 2000
        self.temperature = 0.3

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

        # --- car_data: fixed-size lap aggregates (max 20 rows) ---
        # Keep prompt size bounded for latency while improving representativeness
        # versus single-point telemetry sampling.
        car_data_cols = [
            "lap",
            "samples",
            "t_start_sec",
            "t_end_sec",
            "avg_speed_kmh",
            "max_speed_kmh",
            "avg_throttle_pct",
            "avg_brake_pct",
            "avg_steer_angle",
            "avg_tyre_temp_c",
            "avg_tyre_pressure_psi",
            "max_damage_sum",
        ]
        car_data = self._build_lap_aggregate_rows(telemetry, max_rows=60)

        # --- session identity (so the model knows what it's analysing) ---
        session_context = {}
        if metadata.get("track"):
            session_context["track"] = metadata["track"]
        if metadata.get("car"):
            session_context["car"] = metadata["car"]
        if metadata.get("player_name"):
            session_context["driver"] = metadata["player_name"]

        payload = {
            "session": session_context,
            "result": result,
            "lap_summary": lap_summary,
            "laps": lap_times,
            "stints": stints,
            "pit_stops": pit_stops,
            "car_data_mode": "lap_aggregate",
            "telemetry_total_rows": len(telemetry),
            "car_data_cols": car_data_cols,
            "car_data": car_data,
        }

        return json.dumps(payload, separators=(",", ":"))

    def _build_lap_aggregate_rows(self, telemetry: list[dict], max_rows: int = 20) -> list[list]:
        """Build compact, representative lap-level telemetry rows."""
        if not telemetry:
            return []

        by_lap: dict[int, list[dict]] = defaultdict(list)
        for sample in telemetry:
            lap = sample.get("lap_number")
            if lap is None:
                continue
            try:
                by_lap[int(lap)].append(sample)
            except Exception:
                continue

        if not by_lap:
            return []

        lap_numbers = sorted(by_lap.keys())
        selected_laps = self._select_evenly_spaced(lap_numbers, max_rows)

        rows: list[list] = []
        for lap in selected_laps:
            samples = by_lap.get(lap, [])
            if not samples:
                continue
            rows.append(self._aggregate_lap_row(lap, samples))

        return rows

    @staticmethod
    def _select_evenly_spaced(items: list[int], max_items: int) -> list[int]:
        """Return up to max_items values spread across the full list."""
        if len(items) <= max_items:
            return items

        last = len(items) - 1
        idxs = {round(i * last / (max_items - 1)) for i in range(max_items)}
        return [items[i] for i in sorted(idxs)]

    @staticmethod
    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _aggregate_lap_row(self, lap: int, samples: list[dict]) -> list:
        """Compute stable per-lap metrics from raw sample points."""
        elapsed = [float(s.get("elapsed_time", 0) or 0) for s in samples]
        speeds = [float(s.get("speed", 0) or 0) for s in samples]
        steer = [float(s.get("steer_angle", 0) or 0) for s in samples]

        throttle_pct = []
        brake_pct = []
        for s in samples:
            thr = float(s.get("throttle", 0) or 0)
            brk = float(s.get("brake", 0) or 0)
            throttle_pct.append(thr * 100.0 if thr <= 1.0 else thr)
            brake_pct.append(brk * 100.0 if brk <= 1.0 else brk)

        tyre_temp_vals = []
        tyre_pressure_vals = []
        for s in samples:
            for key in ("tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr"):
                val = s.get(key)
                if val is not None:
                    tyre_temp_vals.append(float(val))
            for key in ("tyre_pressure_fl", "tyre_pressure_fr", "tyre_pressure_rl", "tyre_pressure_rr"):
                val = s.get(key)
                if val is not None:
                    tyre_pressure_vals.append(float(val))

        damage_sums = []
        for s in samples:
            damage_total = 0.0
            for key in (
                "car_damage_front",
                "car_damage_rear",
                "car_damage_left",
                "car_damage_right",
                "car_damage_centre",
            ):
                damage_total += float(s.get(key, 0) or 0)
            damage_sums.append(damage_total)

        return [
            int(lap),
            len(samples),
            round(min(elapsed), 1) if elapsed else 0.0,
            round(max(elapsed), 1) if elapsed else 0.0,
            round(self._avg(speeds), 1),
            round(max(speeds), 1) if speeds else 0.0,
            round(self._avg(throttle_pct), 1),
            round(self._avg(brake_pct), 1),
            round(self._avg(steer), 3),
            round(self._avg(tyre_temp_vals), 1),
            round(self._avg(tyre_pressure_vals), 2),
            round(max(damage_sums), 1) if damage_sums else 0.0,
        ]

    def _parse_response(self, result: dict) -> dict:
        return {
            "content": result.get("content", ""),
            "tokens_used": result.get("tokens_used", 0),
        }
