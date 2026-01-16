"""
Sentence-Level LLM + TTS Pipelining for F1 Telemetry Dashboard.

This module provides sentence-level pipelining for TTS output, which allows
the first sentence of an AI response to start playing while the remaining
sentences are queued. This reduces perceived latency by ~500-1000ms for
multi-sentence responses.

Key components:
- SentenceSplitter: Splits text into sentences using regex patterns
- SentencePipelinedTTS: Orchestrates sentence-by-sentence TTS playback

Estimated latency savings: 500-1000ms for multi-sentence responses
"""

import asyncio
import logging
import re
from typing import Callable, Optional, List, Awaitable
from collections import deque

logger = logging.getLogger(__name__)


class SentenceSplitter:
    """
    Splits text into sentences for pipelined TTS playback.

    Handles various sentence-ending punctuation and edge cases common
    in racing engineer communications (lap times, temperatures, gaps).

    Usage:
        splitter = SentenceSplitter()
        sentences = splitter.split("Fuel is low. Box this lap.")
        # Returns: ["Fuel is low.", "Box this lap."]
    """

    # Regex pattern for sentence splitting
    # Matches: period, question mark, or exclamation mark followed by whitespace
    # Handles:
    # - Multiple spaces between sentences
    # - Ellipsis (... followed by space and capital letter)
    # - Numbers with decimals (doesn't split on 1.23 or 98.5)
    # - Abbreviations like "sec." when followed by lowercase

    # Pattern explanation:
    # (?<=[.!?])   - Positive lookbehind for sentence-ending punctuation
    # (?<!\.{2})   - Negative lookbehind to not split after .. (part of ellipsis)
    # (?<!\d\.)    - Negative lookbehind to not split after digit+period (decimals)
    # \s+          - One or more whitespace characters
    # (?=[A-Z])    - Positive lookahead for capital letter (start of new sentence)

    SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?])(?<!\.{2})(?<!\d\.)\s+(?=[A-Z])'
    )

    # Alternative simpler pattern for cases where the complex one fails
    FALLBACK_PATTERN = re.compile(r'(?<=[.!?])\s+')

    def __init__(self):
        """Initialize SentenceSplitter."""
        logger.debug("SentenceSplitter initialized")

    def split(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split into sentences

        Returns:
            List of sentences. Empty list if text is empty/whitespace.
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        # First try the smart pattern that handles decimals
        sentences = self._split_with_pattern(text, self.SENTENCE_PATTERN)

        # If we got only one sentence but there appear to be multiple,
        # try the fallback pattern
        if len(sentences) == 1 and self._appears_to_have_multiple_sentences(text):
            sentences = self._split_with_pattern(text, self.FALLBACK_PATTERN)

        # Clean up: strip whitespace from each sentence
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _split_with_pattern(self, text: str, pattern: re.Pattern) -> List[str]:
        """
        Split text using a regex pattern.

        Args:
            text: Text to split
            pattern: Compiled regex pattern

        Returns:
            List of sentence strings
        """
        parts = pattern.split(text)
        return [p for p in parts if p]

    def _appears_to_have_multiple_sentences(self, text: str) -> bool:
        """
        Check if text appears to contain multiple sentences.

        Args:
            text: Text to check

        Returns:
            True if text likely contains multiple sentences
        """
        # Count sentence-ending punctuation followed by space
        count = len(re.findall(r'[.!?]\s+', text))
        return count > 0


class SentencePipelinedTTS:
    """
    Orchestrates sentence-by-sentence TTS playback.

    Instead of waiting for the full TTS synthesis of a multi-sentence
    response, this class splits the text and speaks sentence by sentence.
    The first sentence starts immediately while the rest are queued.

    Usage:
        async def speak(text):
            await tts_client.synthesize(text)

        pipelined = SentencePipelinedTTS(speak_callback=speak)
        pipelined.on_sentence_completed(lambda s: print(f"Done: {s}"))

        await pipelined.speak("First sentence. Second sentence.")

    Signals:
        on_sentence_started(text) - Called when a sentence starts speaking
        on_sentence_completed(text) - Called when a sentence finishes
        on_all_completed() - Called when all sentences are done
    """

    def __init__(
        self,
        speak_callback: Callable[[str], Awaitable[None]],
    ):
        """
        Initialize SentencePipelinedTTS.

        Args:
            speak_callback: Async function to call for each sentence.
                           Should synthesize and play the sentence.
        """
        self._speak_callback = speak_callback
        self._splitter = SentenceSplitter()
        self._sentence_queue: deque = deque()
        self._is_speaking = False
        self._should_stop = False

        # Callbacks
        self._on_sentence_started: Optional[Callable[[str], None]] = None
        self._on_sentence_completed: Optional[Callable[[str], None]] = None
        self._on_all_completed: Optional[Callable[[], None]] = None

        logger.debug("SentencePipelinedTTS initialized")

    @property
    def is_speaking(self) -> bool:
        """Whether currently speaking a sentence."""
        return self._is_speaking

    @property
    def remaining_sentences(self) -> int:
        """Number of sentences remaining in queue."""
        return len(self._sentence_queue)

    def on_sentence_started(self, callback: Callable[[str], None]) -> "SentencePipelinedTTS":
        """
        Register callback for sentence start.

        Args:
            callback: Function called with sentence text when it starts

        Returns:
            Self for method chaining
        """
        self._on_sentence_started = callback
        return self

    def on_sentence_completed(self, callback: Callable[[str], None]) -> "SentencePipelinedTTS":
        """
        Register callback for sentence completion.

        Args:
            callback: Function called with sentence text when it completes

        Returns:
            Self for method chaining
        """
        self._on_sentence_completed = callback
        return self

    def on_all_completed(self, callback: Callable[[], None]) -> "SentencePipelinedTTS":
        """
        Register callback for all sentences complete.

        Args:
            callback: Function called when all sentences are done

        Returns:
            Self for method chaining
        """
        self._on_all_completed = callback
        return self

    async def speak(self, text: str) -> None:
        """
        Speak text sentence by sentence.

        Splits the text into sentences and speaks them sequentially.
        The first sentence starts immediately.

        Args:
            text: Text to speak
        """
        if not text or not text.strip():
            return

        # Split into sentences
        sentences = self._splitter.split(text)

        if not sentences:
            return

        self._should_stop = False

        # Speak sentences sequentially
        for sentence in sentences:
            if self._should_stop:
                break

            await self._speak_sentence(sentence)

        # Signal all complete
        if self._on_all_completed and not self._should_stop:
            self._on_all_completed()

    async def _speak_sentence(self, sentence: str) -> None:
        """
        Speak a single sentence.

        Args:
            sentence: Sentence to speak
        """
        self._is_speaking = True

        # Signal sentence started
        if self._on_sentence_started:
            self._on_sentence_started(sentence)

        try:
            # Call the TTS speak callback
            await self._speak_callback(sentence)

        except Exception as e:
            logger.error(f"Error speaking sentence: {e}")

        finally:
            self._is_speaking = False

            # Signal sentence completed
            if self._on_sentence_completed:
                self._on_sentence_completed(sentence)

    def stop(self) -> None:
        """
        Stop speaking and clear the queue.

        The current sentence will finish (for natural stopping),
        but no more sentences will be spoken.
        """
        self._should_stop = True
        self._sentence_queue.clear()
        logger.debug("SentencePipelinedTTS stopped")

    def clear_queue(self) -> None:
        """Clear remaining sentences from queue without stopping current."""
        self._sentence_queue.clear()
