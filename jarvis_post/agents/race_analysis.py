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

        # --- per-lap data (times + fuel) ---
        lap_times = [round(lap.get("lap_time", 0), 3) for lap in laps]
        valid_flags = [lap.get("valid", True) for lap in laps]
        incomplete_laps = [
            laps[i].get("lap_number", i + 1)
            for i, v in enumerate(valid_flags)
            if not v
        ]
        # Exclude incomplete laps from summary statistics
        valid_times = [t for t, v in zip(lap_times, valid_flags) if t > 0 and v]

        # Build enriched per-lap list with fuel data
        laps_with_fuel = []
        for i, lap in enumerate(laps):
            lap_entry = {
                "lap": lap.get("lap_number", i + 1),
                "lap_time": round(lap.get("lap_time", 0), 3),
                "fuel_start": round(lap.get("fuel_start", 0) or 0, 2),
                "fuel_end": round(lap.get("fuel_end", 0) or 0, 2),
            }
            fuel_used = lap_entry["fuel_start"] - lap_entry["fuel_end"]
            if fuel_used > 0:
                lap_entry["fuel_used"] = round(fuel_used, 2)
            laps_with_fuel.append(lap_entry)

        # --- lap_summary ---
        fastest_time = min(valid_times) if valid_times else 0
        slowest_time = max(valid_times) if valid_times else 0
        fastest_lap = 1
        slowest_lap = 1
        if valid_times:
            for i, (lap, t, v) in enumerate(zip(laps, lap_times, valid_flags)):
                if t > 0 and v and t == fastest_time:
                    fastest_lap = lap.get("lap_number", i + 1)
                    break
            for i, (lap, t, v) in enumerate(zip(laps, lap_times, valid_flags)):
                if t > 0 and v and t == slowest_time:
                    slowest_lap = lap.get("lap_number", i + 1)
                    break
        avg_time = round(sum(valid_times) / len(valid_times), 3) if valid_times else 0

        lap_summary = {
            "total": len(laps),
            "completed": len(valid_times),
            "incomplete": len(incomplete_laps),
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
        stint_start_idx = 0
        stint_start_lap = laps[0].get("lap_number", 1) if laps else 1
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
                stint_laps = laps[stint_start_idx: i]
                stint_times = [l.get("lap_time", 0) for l in stint_laps if l.get("lap_time", 0) > 0]
                prev_lap_num = laps[i - 1].get("lap_number", i)
                stints.append({
                    "stint": stint_num,
                    "compound": "UNKNOWN",
                    "laps": f"{stint_start_lap}-{prev_lap_num}",
                    "length": len(stint_laps),
                    "avg_time": round(sum(stint_times) / len(stint_times), 3) if stint_times else 0,
                    "fastest_time": round(min(stint_times), 3) if stint_times else 0,
                })
                pit_stops.append({"lap": prev_lap_num, "duration": None, "inferred": True})
                stint_start_idx = i
                stint_start_lap = lap_num
                stint_num += 1
            if fuel_end is not None:
                prev_fuel_end = fuel_end

        # Final stint
        last_lap_num = laps[-1].get("lap_number", len(laps)) if laps else 0
        final_laps = laps[stint_start_idx:]
        final_times = [l.get("lap_time", 0) for l in final_laps if l.get("lap_time", 0) > 0]
        if final_laps:
            stints.append({
                "stint": stint_num,
                "compound": "UNKNOWN",
                "laps": f"{stint_start_lap}-{last_lap_num}",
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
            "fuel_start",
            "fuel_end",
            "max_damage_sum",
        ]
        car_data = self._build_lap_aggregate_rows(telemetry, max_rows=60)

        # --- session identity (so the model knows what it's analysing) ---
        session_context = {}
        if metadata.get("track") or metadata.get("track_name"):
            session_context["track"] = metadata.get("track") or metadata["track_name"]
        if metadata.get("car") or metadata.get("car_model"):
            session_context["car"] = metadata.get("car") or metadata["car_model"]
        if metadata.get("player_name"):
            session_context["driver"] = metadata["player_name"]

        payload = {
            "session": session_context,
            "result": result,
            "lap_summary": lap_summary,
            "laps": laps_with_fuel,
            "incomplete_laps": incomplete_laps,
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

        # Fuel: first and last sample values for this lap
        fuel_vals = [float(s.get("fuel", 0) or 0) for s in samples]
        fuel_start = round(fuel_vals[0], 2) if fuel_vals else 0.0
        fuel_end = round(fuel_vals[-1], 2) if fuel_vals else 0.0

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
            fuel_start,
            fuel_end,
            round(max(damage_sums), 1) if damage_sums else 0.0,
        ]

    def _parse_response(self, result: dict) -> dict:
        return {
            "content": result.get("content", ""),
            "tokens_used": result.get("tokens_used", 0),
        }
