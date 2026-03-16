"""Coaching agent for Jarvis Post."""

import re

from .base import BaseAgent
from ..llm.local_client import StreamComplete
from ..preprocessing.telemetry import preprocess_for_analysis
from ..config.prompts import COACHING_SYSTEM_PROMPT

# Sentinel string emitted to tell the UI to clear the widget and start fresh.
STREAM_RESET = "\x00RESET\x00"


class CoachingAgent(BaseAgent):
    """Agent for actionable coaching feedback."""

    def __init__(self, llm_client):
        self.llm = llm_client
        # Keep coaching concise to match UX intent and avoid truncation.
        self.max_tokens = 500
        self.retry_max_tokens = 320
        self.temperature = 0.6

    async def analyse(self, session_data: dict) -> dict:
        """Analyse session data and return coaching tips.

        Args:
            session_data: Dictionary containing session_metadata, laps, telemetry,
                         and optionally driver_context.

        Returns:
            Dictionary with 'content' (coaching tips) and 'tokens_used'.
        """
        # Preprocess telemetry into statistical summaries
        summary = preprocess_for_analysis(
            session_data.get("telemetry", []),
            session_data.get("laps", []),
        )

        # Build the prompt for the LLM
        prompt = self._build_prompt(session_data, summary)

        # Generate coaching feedback using the LLM
        result = await self.llm.generate(
            prompt=prompt,
            system_prompt=COACHING_SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        content = str(result.get("content", "")).strip()
        tokens_used = int(result.get("tokens_used", 0) or 0)
        finish_reason = result.get("finish_reason")

        if self._is_likely_truncated(content, tokens_used, self.max_tokens, finish_reason):
            concise_prompt = (
                prompt
                + "\n\nMANDATORY LENGTH LIMITS:\n"
                + "- Keep total response under 160 words.\n"
                + "- Give 3-5 tips maximum.\n"
                + "- Each tip must be one sentence.\n"
                + "- End with one short closing sentence.\n"
                + "- If space is tight, shorten tips instead of cutting off mid-thought.\n"
            )
            retry = await self.llm.generate(
                prompt=concise_prompt,
                system_prompt=COACHING_SYSTEM_PROMPT,
                max_tokens=self.retry_max_tokens,
                temperature=0.4,
            )

            retry_content = str(retry.get("content", "")).strip()
            retry_tokens = int(retry.get("tokens_used", 0) or 0)
            retry_finish = retry.get("finish_reason")
            if not self._is_likely_truncated(
                retry_content,
                retry_tokens,
                self.retry_max_tokens,
                retry_finish,
            ):
                return {
                    "content": self._finalize_text(retry_content),
                    "tokens_used": retry_tokens,
                }

            # Deterministic last resort: produce complete brief coaching text.
            fallback_content = self._build_compact_fallback(session_data, summary)
            return {"content": fallback_content, "tokens_used": 0}

        return {
            "content": self._finalize_text(content),
            "tokens_used": tokens_used,
        }

    def analyse_stream(self, session_data: dict):
        """Streaming variant with truncation-retry.

        Yields str chunks progressively.  Special sentinels:
        - ``STREAM_RESET`` — tells the UI to clear and start fresh (fallback).
        - ``StreamComplete`` — final item (end of generation).
        """
        summary = preprocess_for_analysis(
            session_data.get("telemetry", []),
            session_data.get("laps", []),
        )
        prompt = self._build_prompt(session_data, summary)

        # --- First attempt ---
        accumulated = []
        first_complete = None
        for item in self.llm.generate_stream(
            prompt=prompt,
            system_prompt=COACHING_SYSTEM_PROMPT,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        ):
            if isinstance(item, StreamComplete):
                first_complete = item
            else:
                accumulated.append(item)
                yield item

        first_text = "".join(accumulated).strip()
        tokens_used = first_complete.tokens_used if first_complete else 0
        finish_reason = first_complete.finish_reason if first_complete else "stop"

        if not self._is_likely_truncated(first_text, tokens_used, self.max_tokens, finish_reason):
            yield StreamComplete(tokens_used=tokens_used, finish_reason=finish_reason)
            return

        # --- Retry with shorter constraints ---
        yield "\n\n(Regenerating with shorter constraints...)\n\n"

        concise_prompt = (
            prompt
            + "\n\nMANDATORY LENGTH LIMITS:\n"
            + "- Keep total response under 160 words.\n"
            + "- Give 3-5 tips maximum.\n"
            + "- Each tip must be one sentence.\n"
            + "- End with one short closing sentence.\n"
            + "- If space is tight, shorten tips instead of cutting off mid-thought.\n"
        )

        retry_accumulated = []
        retry_complete = None
        for item in self.llm.generate_stream(
            prompt=concise_prompt,
            system_prompt=COACHING_SYSTEM_PROMPT,
            max_tokens=self.retry_max_tokens,
            temperature=0.4,
        ):
            if isinstance(item, StreamComplete):
                retry_complete = item
            else:
                retry_accumulated.append(item)
                yield item

        retry_text = "".join(retry_accumulated).strip()
        retry_tokens = retry_complete.tokens_used if retry_complete else 0
        retry_finish = retry_complete.finish_reason if retry_complete else "stop"

        if not self._is_likely_truncated(retry_text, retry_tokens, self.retry_max_tokens, retry_finish):
            yield StreamComplete(tokens_used=retry_tokens, finish_reason=retry_finish)
            return

        # --- Deterministic fallback: reset widget, emit full fallback ---
        yield STREAM_RESET
        fallback_content = self._build_compact_fallback(session_data, summary)
        yield fallback_content
        yield StreamComplete(tokens_used=0, finish_reason="fallback")

    def _build_prompt(self, session_data: dict, summary: dict) -> str:
        """Build the prompt for the LLM from session data and telemetry summary.

        Args:
            session_data: The full session data dictionary.
            summary: Preprocessed telemetry summary.

        Returns:
            Formatted prompt string for the LLM.
        """
        metadata = session_data.get("session_metadata", {})
        laps = session_data.get("laps", [])
        driver_context = session_data.get("driver_context", {})

        # Build session context section (more concise for coaching)
        session_context = f"""SESSION OVERVIEW:
- Track: {metadata.get('track_name', 'Unknown')}
- Car: {metadata.get('car_model', 'Unknown')}
- Total Laps: {metadata.get('total_laps', 0)}
- Best Lap: {metadata.get('best_lap_time', 'N/A')}s"""

        # Build lap summary (focus on improvement)
        if laps:
            lap_times = [lap.get('lap_time') for lap in laps if lap.get('lap_time')]
            if lap_times:
                first_lap = lap_times[0]
                best_lap = min(lap_times)
                improvement = first_lap - best_lap
                lap_section = f"""LAP PERFORMANCE:
- First Lap: {first_lap:.3f}s
- Best Lap: {best_lap:.3f}s
- Improvement: {improvement:.3f}s
- Lap Count: {len(lap_times)}"""
            else:
                lap_section = "LAP PERFORMANCE: No lap time data available"
        else:
            lap_section = "LAP PERFORMANCE: No lap data available"

        # Build key metrics section
        throttle = summary.get('throttle_patterns', {})
        braking = summary.get('braking_patterns', {})
        tyres = summary.get('tyre_evolution', {})

        metrics_section = f"""KEY METRICS:
- Average Throttle: {throttle.get('average_throttle', 0):.0%}
- Full Throttle Time: {throttle.get('full_throttle_pct', 0):.1f}%
- Throttle Hesitations: {throttle.get('hesitation_count', 0)}
- Braking Zones: {braking.get('braking_zone_count', 0)}
- Tyre Balance (Front-Rear): {tyres.get('front_rear_temp_diff', 0):+.1f}C
- Tyre Balance (Left-Right): {tyres.get('left_right_temp_diff', 0):+.1f}C"""

        # Build driver context section if provided
        driver_section = ""
        if driver_context:
            driver_lines = []
            if driver_context.get('experience_level'):
                driver_lines.append(f"- Experience Level: {driver_context.get('experience_level')}")
            if driver_context.get('focus_areas'):
                focus = ', '.join(driver_context.get('focus_areas', []))
                driver_lines.append(f"- Focus Areas: {focus}")
            if driver_context.get('recent_sessions'):
                driver_lines.append(f"- Recent Sessions: {driver_context.get('recent_sessions')}")
            if driver_lines:
                driver_section = "\nDRIVER CONTEXT:\n" + "\n".join(driver_lines)

        # Build areas for improvement based on telemetry
        areas_section = self._identify_improvement_areas(summary)

        prompt = f"""{session_context}

{lap_section}

{metrics_section}
{driver_section}

{areas_section}

Please provide 3-5 actionable coaching tips based on this session data. Focus on the most impactful improvements for the next session.

MANDATORY OUTPUT RULES:
- Keep total response between 120 and 180 words.
- Use 1 brief opening sentence.
- Then list 3-5 numbered tips.
- Each tip must be one sentence with a clear action.
- End with 1 brief closing sentence."""

        return prompt

    def _identify_improvement_areas(self, summary: dict) -> str:
        """Identify potential areas for improvement from telemetry summary.

        Args:
            summary: Preprocessed telemetry summary.

        Returns:
            String describing potential improvement areas.
        """
        areas = []

        throttle = summary.get('throttle_patterns', {})
        braking = summary.get('braking_patterns', {})
        tyres = summary.get('tyre_evolution', {})

        # Check for throttle hesitations
        if throttle.get('hesitation_count', 0) > 2:
            areas.append("Multiple throttle hesitations detected - possible confidence issue on corner exit")

        # Check for low full throttle percentage
        if throttle.get('full_throttle_pct', 0) < 30:
            areas.append("Low full throttle percentage - may be leaving time on straights")

        # Check for tyre imbalance
        front_rear_diff = abs(tyres.get('front_rear_temp_diff', 0))
        if front_rear_diff > 5:
            areas.append(f"Significant front-rear tyre temperature difference ({front_rear_diff:.1f}C)")

        left_right_diff = abs(tyres.get('left_right_temp_diff', 0))
        if left_right_diff > 3:
            areas.append(f"Left-right tyre temperature imbalance ({left_right_diff:.1f}C)")

        if areas:
            return "POTENTIAL IMPROVEMENT AREAS:\n" + "\n".join(f"- {area}" for area in areas)
        else:
            return "POTENTIAL IMPROVEMENT AREAS: Session data looks solid - focus on consistency"

    def _parse_response(self, result: dict) -> dict:
        """Parse and structure the LLM response.

        Args:
            result: Raw response from the LLM.

        Returns:
            Dictionary with 'content' and 'tokens_used'.
        """
        return {
            "content": result.get("content", ""),
            "tokens_used": result.get("tokens_used", 0),
        }

    def _is_likely_truncated(
        self,
        content: str,
        tokens_used: int,
        max_tokens: int,
        finish_reason: str | None = None,
    ) -> bool:
        """
        Infer truncation when usage hits cap or output ends mid-thought.
        """
        text = (content or "").strip()
        if not text:
            return False

        if finish_reason == "length":
            return True
        if tokens_used >= max_tokens and max_tokens > 0:
            return True

        # Guard against obvious mid-word or mid-sentence cutoffs.
        if len(text) > 200 and text[-1] not in ".!?":
            if re.search(r"[A-Za-z0-9]$", text):
                return True

        return False

    def _finalize_text(self, text: str) -> str:
        """
        Ensure returned text ends cleanly on sentence boundaries.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return ""

        if cleaned[-1] in ".!?":
            return cleaned

        last_sentence_end = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"))
        if last_sentence_end >= int(len(cleaned) * 0.6):
            return cleaned[: last_sentence_end + 1].rstrip()
        return f"{cleaned}."

    def _build_compact_fallback(self, session_data: dict, summary: dict) -> str:
        """
        Final fallback used only if model responses are repeatedly truncated.
        """
        throttle = summary.get("throttle_patterns", {})
        braking = summary.get("braking_patterns", {})
        tyres = summary.get("tyre_evolution", {})

        avg_throttle = throttle.get("average_throttle", 0.0) * 100.0
        full_throttle = throttle.get("full_throttle_pct", 0.0)
        hesitations = int(throttle.get("hesitation_count", 0) or 0)
        brake_zones = int(braking.get("braking_zone_count", 0) or 0)
        front_rear = float(tyres.get("front_rear_temp_diff", 0.0) or 0.0)

        tips = [
            f"1. Smooth throttle pickup out of corners; hesitations were {hesitations}, so focus on one clean squeeze per exit.",
            f"2. Build straight-line commitment; full-throttle time was {full_throttle:.1f}%, so target earlier confident acceleration.",
            f"3. Stabilize braking phases; you had {brake_zones} braking zones, so prioritize repeatable initial pressure and release.",
            f"4. Manage tyre balance; front-rear temperature delta was {front_rear:+.1f}C, so adjust driving inputs to reduce imbalance.",
        ]

        if avg_throttle < 45:
            opening = "You have a solid baseline, and the biggest gains now come from cleaner exits and commitment."
        else:
            opening = "Strong baseline pace; the next gains come from precision and consistency in key phases."

        closing = "Apply these four focuses next run and keep each lap execution deliberate and repeatable."
        return "\n".join([opening, *tips, closing])
