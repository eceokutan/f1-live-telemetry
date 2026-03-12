# Voice Pipeline Latency Reduction Guide

**Purpose:** Strategies to reduce end-to-end latency in the desktop app voice pipeline
**Target:** <3000ms (from docs.md specification)

---

## Current Pipeline Latency Breakdown

| Stage | Current Implementation | Estimated Latency | Target (docs.md) |
|-------|----------------------|-------------------|------------------|
| VAD silence detection | 300ms padding + 500ms min duration | ~800ms | N/A |
| Watson STT | Batch (full utterance) | ~500-1000ms | N/A |
| Queue processing | asyncio with 0.1s timeouts | ~100-200ms | <50ms |
| LLM (WatsonX) | Full response generation | ~1500-3000ms | <2000ms |
| Watson TTS | Full synthesis before playback | ~500-1000ms | <500ms |
| **Total** | | **~3500-6000ms** | **<3000ms** |

---

## High-Impact Optimizations

### 1. Switch to Streaming STT (Biggest Win)

Currently `ai/voice_input.py` batches the entire utterance before sending to Watson:

```python
# Current: Batch approach (voice_input.py:321-339)
audio_data = np.concatenate(self._speech_buffer)
response = self.stt_client.recognize(audio=audio_bytes, ...)
```

**Change to Watson STT WebSocket streaming:**
- Transcribe while the user is still speaking
- Get partial results in real-time
- Saves 500-1000ms

```python
# Streaming approach using Watson's WebSocket API
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback

class StreamingCallback(RecognizeCallback):
    def on_transcription(self, transcript):
        if transcript['results'][0]['final']:
            self.on_final_transcript(transcript)

    def on_final_transcript(self, transcript):
        # Immediately send to AI pipeline
        self.speech_detected.emit(transcript['results'][0]['alternatives'][0]['transcript'])
```

**Estimated Savings:** 500-1000ms

---

### 2. Reduce VAD Padding and Minimum Duration

**Status:** ✅ IMPLEMENTED (optimized values enabled)

#### Current Configuration

In `ai/voice_input.py:64-65`:

```python
# VAD configuration (CURRENT - optimized values)
SPEECH_PAD_MS = 150  # Padding before/after speech (ms)
MIN_SPEECH_DURATION_MS = 300  # Minimum speech duration to process
```

#### How to Adjust

Edit `ai/voice_input.py` and modify the class constants in `VoiceInputWorker`:

```python
class VoiceInputWorker(QtCore.QThread):
    # ...

    # VAD configuration - ADJUST THESE VALUES
    VAD_THRESHOLD = 0.3  # Speech probability threshold (0.0-1.0)
    SPEECH_PAD_MS = 150  # ← Adjust: ms of silence before ending speech
    MIN_SPEECH_DURATION_MS = 300  # ← Adjust: minimum ms to process
```

#### Tuning Guide

| Setting | Conservative | Balanced (Current) | Aggressive |
|---------|--------------|-------------------|------------|
| `SPEECH_PAD_MS` | 300ms | **150ms** | 100ms |
| `MIN_SPEECH_DURATION_MS` | 500ms | **300ms** | 200ms |
| `VAD_THRESHOLD` | 0.5 | **0.3** | 0.2 |

**Trade-offs:**

| Direction | Effect | Risk |
|-----------|--------|------|
| Lower `SPEECH_PAD_MS` | Faster response | May cut off speech mid-sentence |
| Higher `SPEECH_PAD_MS` | More natural pauses | Slower response time |
| Lower `MIN_SPEECH_DURATION_MS` | Catches short commands ("fuel?") | May trigger on noise/coughs |
| Higher `MIN_SPEECH_DURATION_MS` | Ignores noise | Misses short queries |
| Lower `VAD_THRESHOLD` | More sensitive | More false positives |
| Higher `VAD_THRESHOLD` | Fewer false positives | May miss quiet speech |

**Recommended values for racing:**
- Fast-paced racing: `SPEECH_PAD_MS=100`, `MIN_SPEECH_DURATION_MS=200`
- Normal racing: `SPEECH_PAD_MS=150`, `MIN_SPEECH_DURATION_MS=300` (current)
- Noisy environment: `SPEECH_PAD_MS=200`, `MIN_SPEECH_DURATION_MS=400`, `VAD_THRESHOLD=0.4`

**Estimated Savings:** ~350ms (from conservative to balanced)

---

### 3. Stream TTS Playback (Start Playing Before Full Synthesis)

Currently `ai/tts_output.py` waits for complete synthesis:

```python
# Current: Full synthesis then play (tts_output.py:202-223)
audio_bytes = await self.tts_client.synthesize(text)  # Wait for ALL audio
await self._play_audio_async(audio_bytes)  # Then play
```

**Change to chunked streaming:**

```python
async def _synthesize_and_play_streaming(self, text: str):
    """Stream TTS - start playing while still synthesizing."""
    url = f"{self.service_url}/v1/synthesize"

    async with aiohttp.ClientSession() as session:
        async with session.post(url, ...) as response:
            # Start playback immediately with first chunk
            stream = self.audio.open(format=..., output=True)

            async for chunk in response.content.iter_chunked(4096):
                stream.write(chunk)  # Play as we receive

            stream.close()
```

**Estimated Savings:** 300-500ms (first audio plays while rest synthesizes)

---

### 4. Sentence-Level LLM + TTS Pipelining

The docs.md describes this in Section 5.3 (`_split_into_sentences`). Generate and speak sentence-by-sentence:

```python
async def generate_and_speak_streaming(self, prompt: str):
    """Generate response and speak first sentence while generating more."""

    # Get full response (or use streaming LLM if available)
    response = await self.llm_client.invoke(prompt)

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', response)

    # Speak first sentence immediately
    first_sentence_task = asyncio.create_task(
        self.tts_worker.speak(sentences[0])
    )

    # Queue remaining sentences
    for sentence in sentences[1:]:
        await self.tts_queue.put(sentence)

    await first_sentence_task
```

**Estimated Savings:** 500-1000ms for multi-sentence responses

---

### 5. Local TTS Alternative (Lowest Latency)

Replace Watson TTS with local synthesis for <100ms latency:

```python
# Option A: pyttsx3 (built-in, ~50ms)
import pyttsx3
engine = pyttsx3.init()
engine.say(text)
engine.runAndWait()

# Option B: Coqui TTS (better quality, ~200ms)
from TTS.api import TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
tts.tts_to_file(text=text, file_path="output.wav")
```

**Trade-off:** Lower quality voice vs. ~400-800ms savings

**Estimated Savings:** 400-800ms

---

### 6. Reduce LLM Response Length

**Status:** ✅ IMPLEMENTED (75 tokens enabled)

#### Current Configuration

**Default value** in `ai/race_engineer_core/llm_client.py:58`:

```python
max_tokens: int = 75,  # Reduced from 150 for faster responses (~200-400ms savings)
```

**Explicit override** in `ai/race_engineer.py:195`:

```python
llm_client = LLMClient(
    # ...
    max_tokens=75,  # Latency optimization: shorter responses
    max_retries=2
)
```

#### How to Adjust

**Option 1: Change the default** (affects all LLM clients)

Edit `ai/race_engineer_core/llm_client.py`:

```python
def __init__(
    self,
    # ...
    max_tokens: int = 75,  # ← Adjust this default value
    # ...
):
```

**Option 2: Change per-instance** (recommended for testing)

Edit `ai/race_engineer.py` in the `_initialize_agents()` method:

```python
llm_client = LLMClient(
    huggingface_token=self.huggingface_token,
    model_id=self.hf_model_id,
    max_tokens=75,  # ← Adjust this value
    max_retries=2
)
```

#### Tuning Guide

| max_tokens | Response Style | Latency | Use Case |
|------------|---------------|---------|----------|
| 50 | Ultra-brief ("Box now.") | ~800ms | Critical alerts only |
| **75** | Brief (1-2 sentences) | **~1200ms** | **Racing (current)** |
| 100 | Short (2-3 sentences) | ~1500ms | Detailed updates |
| 150 | Medium (3-4 sentences) | ~2000ms | Coaching mode |
| 200 | Verbose | ~2500ms | Post-race analysis |

**Trade-offs:**

| Direction | Effect | Risk |
|-----------|--------|------|
| Lower `max_tokens` | Faster responses | May truncate important info |
| Higher `max_tokens` | More complete responses | Slower, driver may lose focus |

**Verbosity interaction:**

The `verbosity` setting in `AIRaceEngineerWorker` also affects response length:

```python
# In main.py or when creating AIRaceEngineerWorker:
ai_thread = AIRaceEngineerWorker(
    # ...
    verbosity="minimal"  # Options: "minimal", "moderate", "verbose"
)
```

| Verbosity | Typical Response | Best paired with |
|-----------|-----------------|------------------|
| `"minimal"` | 1 sentence | `max_tokens=50-75` |
| `"moderate"` | 2-3 sentences | `max_tokens=75-100` |
| `"verbose"` | 3-4 sentences | `max_tokens=100-150` |

**Recommended combinations:**
- Fast racing: `max_tokens=50`, `verbosity="minimal"`
- Normal racing: `max_tokens=75`, `verbosity="minimal"` (current)
- Practice/learning: `max_tokens=100`, `verbosity="moderate"`

**Estimated Savings:** ~200-400ms (from 150 to 75 tokens)

---

## Implementation Priority

| Optimization | Effort | Latency Saved | Recommendation |
|--------------|--------|---------------|----------------|
| Streaming STT | Medium | 500-1000ms | **Do first** |
| Reduce VAD padding | Easy | 350ms | **Do first** |
| Streaming TTS | Medium | 300-500ms | High priority |
| Sentence pipelining | Medium | 500-1000ms | High priority |
| Local TTS | Easy | 400-800ms | Consider if quality acceptable |
| Reduce max_tokens | Easy | 200-400ms | Quick win |

---

## Quick Wins (Easy Changes)

### Immediate changes requiring minimal code:

1. **Reduce VAD padding** (`ai/voice_input.py:55-57`)
   ```python
   SPEECH_PAD_MS = 150  # was 300
   MIN_SPEECH_DURATION_MS = 300  # was 500
   ```

2. **Reduce max tokens** (`ai/race_engineer_core/llm_client.py:58`)
   ```python
   max_tokens: int = 75  # was 150
   ```

3. **Use minimal verbosity** (when initializing AI worker)
   ```python
   ai_worker = AIRaceEngineerWorker(
       ...,
       verbosity="minimal"  # instead of "moderate"
   )
   ```

**Combined quick wins savings:** ~600-900ms with 5 minutes of changes

---

## Realistic Target After Full Optimization

| Stage | Optimized | Savings |
|-------|-----------|---------|
| VAD + Streaming STT | ~300ms | -1000ms |
| Queue processing | ~50ms | -100ms |
| LLM (shorter responses) | ~1200ms | -500ms |
| Streaming TTS | ~200ms (time to first audio) | -500ms |
| **Total** | **~1750ms** | **-2100ms** |

This would bring you close to the docs.md target of <3000ms end-to-end, with first audio playing around 1.7-2 seconds after the driver finishes speaking.

---

## Files to Modify

| Optimization | Primary File | Secondary Files |
|--------------|--------------|-----------------|
| Streaming STT | `ai/voice_input.py` | - |
| VAD tuning | `ai/voice_input.py` | - |
| Streaming TTS | `ai/tts_output.py` | - |
| Sentence pipelining | `ai/race_engineer.py` | `ai/tts_output.py` |
| Local TTS | `ai/tts_output.py` | `requirements.txt` |
| Max tokens | `ai/race_engineer_core/llm_client.py` | - |

---

## Testing Latency

Add timing instrumentation to measure improvements:

```python
import time

class LatencyTracker:
    def __init__(self):
        self.timestamps = {}

    def mark(self, stage: str):
        self.timestamps[stage] = time.time()

    def report(self):
        stages = list(self.timestamps.keys())
        for i in range(1, len(stages)):
            delta = self.timestamps[stages[i]] - self.timestamps[stages[i-1]]
            print(f"{stages[i-1]} -> {stages[i]}: {delta*1000:.0f}ms")

# Usage in voice pipeline:
tracker = LatencyTracker()
tracker.mark("speech_end")
# ... STT ...
tracker.mark("stt_complete")
# ... LLM ...
tracker.mark("llm_complete")
# ... TTS ...
tracker.mark("first_audio")
tracker.report()
```

---

## Summary

| Category | Action | Impact |
|----------|--------|--------|
| **Do First** | Streaming STT + VAD tuning | -1350ms |
| **High Priority** | Streaming TTS + Sentence pipelining | -800-1500ms |
| **Quick Wins** | Reduce max_tokens + minimal verbosity | -400-600ms |
| **Consider** | Local TTS (quality trade-off) | -400-800ms |

**Expected result:** Reduce end-to-end latency from ~4500ms to ~1750-2200ms
