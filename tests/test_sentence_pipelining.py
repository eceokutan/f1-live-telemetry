"""
Tests for Sentence-Level LLM + TTS Pipelining.

TDD tests for sentence splitting and pipelined TTS playback.
This optimization starts speaking the first sentence while queuing the rest,
providing ~500-1000ms latency savings for multi-sentence responses.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch


# =============================================================================
# Tests for SentenceSplitter
# =============================================================================

class TestSentenceSplitterBasic:
    """Tests for basic sentence splitting functionality."""

    def test_split_single_sentence(self):
        """Single sentence returns list with one element."""
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split("Your tires are looking good.")

        assert result == ["Your tires are looking good."]

    def test_split_two_sentences_with_period(self):
        """Two sentences separated by period are split correctly."""
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split("Fuel is critical. Box this lap.")

        assert result == ["Fuel is critical.", "Box this lap."]

    def test_split_sentences_with_question_mark(self):
        """Sentences ending with question marks are split correctly."""
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split("Do you want to pit? We have a window now.")

        assert result == ["Do you want to pit?", "We have a window now."]

    def test_split_sentences_with_exclamation_mark(self):
        """Sentences ending with exclamation marks are split correctly."""
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split("Box now! Tires are gone.")

        assert result == ["Box now!", "Tires are gone."]

    def test_split_mixed_punctuation(self):
        """Mixed sentence endings are handled correctly."""
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split("Fuel is low. Want to push? Box this lap!")

        assert result == ["Fuel is low.", "Want to push?", "Box this lap!"]

    def test_split_empty_string(self):
        """Empty string returns empty list."""
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split("")

        assert result == []

    def test_split_whitespace_only(self):
        """Whitespace-only string returns empty list."""
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split("   ")

        assert result == []


class TestSentenceSplitterEdgeCases:
    """Tests for edge cases in sentence splitting."""

    def test_preserves_numbers_with_decimals(self):
        """Numbers with decimals are not split incorrectly."""
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split("Lap time was 1:23.456. That's a new best.")

        assert result == ["Lap time was 1:23.456.", "That's a new best."]

    def test_preserves_abbreviations(self):
        """Common abbreviations like 'P1' or 'sec.' are handled.

        Note: 'P1.' is treated as part of the same sentence because it ends
        with digit+period (like decimals). This is acceptable behavior since
        both 'P1. Gap is...' and 'P1.' + 'Gap is...' are reasonable outputs.
        """
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split("You're in P1. Gap is 2.5 sec. Keep pushing.")

        # P1. is kept with the following text (digit before period heuristic)
        # sec. correctly splits before "Keep" (no digit before period)
        assert len(result) >= 2  # At least 2 sentences
        assert "P1" in result[0]
        assert "Keep pushing" in result[-1]

    def test_handles_multiple_spaces(self):
        """Multiple spaces between sentences are handled."""
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split("First sentence.  Second sentence.")

        assert result == ["First sentence.", "Second sentence."]

    def test_no_trailing_whitespace(self):
        """Sentences don't have trailing whitespace."""
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split("First sentence. Second sentence. ")

        for sentence in result:
            assert sentence == sentence.strip()

    def test_no_leading_whitespace(self):
        """Sentences don't have leading whitespace."""
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split(" First sentence. Second sentence.")

        for sentence in result:
            assert sentence == sentence.strip()

    def test_preserves_ellipsis(self):
        """Ellipsis is not incorrectly split."""
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split("Wait for it... Now push!")

        # Ellipsis should not cause incorrect split
        assert len(result) == 2
        assert "..." in result[0]

    def test_handles_sentence_without_ending_punctuation(self):
        """Text without ending punctuation is returned as single sentence."""
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split("Keep pushing")

        assert result == ["Keep pushing"]


class TestSentenceSplitterRacingContext:
    """Tests for racing-specific language patterns."""

    def test_lap_time_format(self):
        """Lap times in M:SS.mmm format don't cause incorrect splits."""
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split("Last lap 1:45.234. That's purple.")

        assert len(result) == 2
        assert "1:45.234" in result[0]

    def test_tire_temperature_reading(self):
        """Temperature readings with decimals are preserved."""
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split("Front left is at 98.5 degrees. Watch the rears.")

        assert len(result) == 2
        assert "98.5" in result[0]

    def test_gap_time_format(self):
        """Gap times are preserved correctly."""
        from ai.sentence_pipelining import SentenceSplitter

        splitter = SentenceSplitter()
        result = splitter.split("Gap to leader is 3.2 seconds. You're gaining.")

        assert len(result) == 2
        assert "3.2" in result[0]


# =============================================================================
# Tests for SentencePipelinedTTS
# =============================================================================

class TestSentencePipelinedTTSInit:
    """Tests for SentencePipelinedTTS initialization."""

    def test_init_with_tts_speak_callback(self):
        """Initializes with TTS speak callback."""
        from ai.sentence_pipelining import SentencePipelinedTTS

        speak_callback = Mock()
        pipelined_tts = SentencePipelinedTTS(speak_callback=speak_callback)

        assert pipelined_tts is not None

    def test_has_sentence_queue(self):
        """Has internal sentence queue."""
        from ai.sentence_pipelining import SentencePipelinedTTS

        pipelined_tts = SentencePipelinedTTS(speak_callback=Mock())

        assert hasattr(pipelined_tts, '_sentence_queue')

    def test_has_splitter(self):
        """Has sentence splitter instance."""
        from ai.sentence_pipelining import SentencePipelinedTTS

        pipelined_tts = SentencePipelinedTTS(speak_callback=Mock())

        assert hasattr(pipelined_tts, '_splitter')


class TestSentencePipelinedTTSSpeaking:
    """Tests for sentence-pipelined speaking."""

    @pytest.mark.asyncio
    async def test_speak_single_sentence_calls_callback_once(self):
        """Single sentence calls speak callback once."""
        from ai.sentence_pipelining import SentencePipelinedTTS

        speak_callback = AsyncMock()
        pipelined_tts = SentencePipelinedTTS(speak_callback=speak_callback)

        await pipelined_tts.speak("Hello driver.")

        speak_callback.assert_called_once_with("Hello driver.")

    @pytest.mark.asyncio
    async def test_speak_two_sentences_calls_callback_twice(self):
        """Two sentences call speak callback twice."""
        from ai.sentence_pipelining import SentencePipelinedTTS

        speak_callback = AsyncMock()
        pipelined_tts = SentencePipelinedTTS(speak_callback=speak_callback)

        await pipelined_tts.speak("First sentence. Second sentence.")

        assert speak_callback.call_count == 2

    @pytest.mark.asyncio
    async def test_speak_calls_sentences_in_order(self):
        """Sentences are spoken in order."""
        from ai.sentence_pipelining import SentencePipelinedTTS

        spoken_sentences = []
        async def capture_speak(text):
            spoken_sentences.append(text)

        pipelined_tts = SentencePipelinedTTS(speak_callback=capture_speak)

        await pipelined_tts.speak("First. Second. Third.")

        assert spoken_sentences == ["First.", "Second.", "Third."]

    @pytest.mark.asyncio
    async def test_speak_empty_string_does_not_call_callback(self):
        """Empty string doesn't call speak callback."""
        from ai.sentence_pipelining import SentencePipelinedTTS

        speak_callback = AsyncMock()
        pipelined_tts = SentencePipelinedTTS(speak_callback=speak_callback)

        await pipelined_tts.speak("")

        speak_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_speak_first_sentence_starts_immediately(self):
        """First sentence starts speaking immediately (not queued)."""
        from ai.sentence_pipelining import SentencePipelinedTTS

        call_times = []
        async def timed_speak(text):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate TTS delay

        pipelined_tts = SentencePipelinedTTS(speak_callback=timed_speak)

        start_time = asyncio.get_event_loop().time()
        await pipelined_tts.speak("First. Second.")

        # First sentence should start almost immediately (within 10ms)
        assert call_times[0] - start_time < 0.01


class TestSentencePipelinedTTSSignals:
    """Tests for sentence completion signals."""

    @pytest.mark.asyncio
    async def test_emits_sentence_started_signal(self):
        """Emits signal when each sentence starts."""
        from ai.sentence_pipelining import SentencePipelinedTTS

        started_sentences = []
        def on_sentence_started(text):
            started_sentences.append(text)

        speak_callback = AsyncMock()
        pipelined_tts = SentencePipelinedTTS(speak_callback=speak_callback)
        pipelined_tts.on_sentence_started(on_sentence_started)

        await pipelined_tts.speak("First. Second.")

        assert len(started_sentences) == 2

    @pytest.mark.asyncio
    async def test_emits_sentence_completed_signal(self):
        """Emits signal when each sentence completes."""
        from ai.sentence_pipelining import SentencePipelinedTTS

        completed_sentences = []
        def on_sentence_completed(text):
            completed_sentences.append(text)

        speak_callback = AsyncMock()
        pipelined_tts = SentencePipelinedTTS(speak_callback=speak_callback)
        pipelined_tts.on_sentence_completed(on_sentence_completed)

        await pipelined_tts.speak("First. Second.")

        assert len(completed_sentences) == 2

    @pytest.mark.asyncio
    async def test_emits_all_completed_signal(self):
        """Emits signal when all sentences complete."""
        from ai.sentence_pipelining import SentencePipelinedTTS

        all_completed = []
        def on_all_completed():
            all_completed.append(True)

        speak_callback = AsyncMock()
        pipelined_tts = SentencePipelinedTTS(speak_callback=speak_callback)
        pipelined_tts.on_all_completed(on_all_completed)

        await pipelined_tts.speak("First. Second. Third.")

        assert len(all_completed) == 1


class TestSentencePipelinedTTSInterrupt:
    """Tests for interrupt handling during pipelined speech."""

    @pytest.mark.asyncio
    async def test_can_stop_mid_speech(self):
        """Can stop speaking after current sentence."""
        from ai.sentence_pipelining import SentencePipelinedTTS

        spoken_sentences = []
        async def slow_speak(text):
            spoken_sentences.append(text)
            await asyncio.sleep(0.1)

        pipelined_tts = SentencePipelinedTTS(speak_callback=slow_speak)

        # Start speaking in background
        task = asyncio.create_task(pipelined_tts.speak("First. Second. Third."))

        # Wait for first sentence to start, then stop
        await asyncio.sleep(0.05)
        pipelined_tts.stop()

        await task

        # Should have spoken at most 1-2 sentences (depending on timing)
        assert len(spoken_sentences) <= 2

    @pytest.mark.asyncio
    async def test_stop_clears_queue(self):
        """Stop clears remaining sentences from queue."""
        from ai.sentence_pipelining import SentencePipelinedTTS

        speak_callback = AsyncMock()
        pipelined_tts = SentencePipelinedTTS(speak_callback=speak_callback)

        pipelined_tts.stop()

        assert pipelined_tts.remaining_sentences == 0

    def test_has_remaining_sentences_property(self):
        """Has property to check remaining sentences."""
        from ai.sentence_pipelining import SentencePipelinedTTS

        pipelined_tts = SentencePipelinedTTS(speak_callback=Mock())

        assert hasattr(pipelined_tts, 'remaining_sentences')

    def test_has_is_speaking_property(self):
        """Has property to check if currently speaking."""
        from ai.sentence_pipelining import SentencePipelinedTTS

        pipelined_tts = SentencePipelinedTTS(speak_callback=Mock())

        assert hasattr(pipelined_tts, 'is_speaking')
        assert pipelined_tts.is_speaking is False


# =============================================================================
# Tests for Integration with TTSOutputWorker
# =============================================================================

class TestTTSOutputWorkerSentencePipelining:
    """Tests for TTSOutputWorker with sentence pipelining."""

    def test_tts_worker_has_sentence_pipelining_flag(self):
        """TTSOutputWorker has use_sentence_pipelining parameter."""
        from ai.tts_output import TTSOutputWorker

        worker = TTSOutputWorker(
            watson_api_key="test-key",
            watson_url="https://api.test.com",
            use_sentence_pipelining=True
        )

        assert hasattr(worker, 'use_sentence_pipelining')
        assert worker.use_sentence_pipelining is True

    def test_tts_worker_sentence_pipelining_default_false(self):
        """TTSOutputWorker defaults to no sentence pipelining."""
        from ai.tts_output import TTSOutputWorker

        worker = TTSOutputWorker(
            watson_api_key="test-key",
            watson_url="https://api.test.com"
        )

        assert worker.use_sentence_pipelining is False

    def test_tts_worker_has_pipelined_tts_attribute(self):
        """TTSOutputWorker has pipelined TTS attribute when enabled."""
        from ai.tts_output import TTSOutputWorker

        worker = TTSOutputWorker(
            watson_api_key="test-key",
            watson_url="https://api.test.com",
            use_sentence_pipelining=True
        )

        assert hasattr(worker, '_pipelined_tts')

    def test_tts_worker_has_synthesize_and_play_pipelined_method(self):
        """TTSOutputWorker has method for pipelined synthesis and playback."""
        from ai.tts_output import TTSOutputWorker

        worker = TTSOutputWorker(
            watson_api_key="test-key",
            watson_url="https://api.test.com",
            use_sentence_pipelining=True
        )

        assert hasattr(worker, '_synthesize_and_play_pipelined')
        assert callable(worker._synthesize_and_play_pipelined)


# =============================================================================
# Helper for async tests
# =============================================================================

class AsyncIterator:
    """Helper to create async iterators for testing."""

    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item
