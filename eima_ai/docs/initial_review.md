# Jarvis-Granite Project Viability Analysis

**Project:** F1 Jarvis TORCS - AI Race Engineer Backend  
**Document:** Viability Assessment  
**Date:** December 2025

---

## Executive Summary

**Verdict: VIABLE — With Caveats**

The proposed architecture is sound and implementable. The technology choices are mature, the layer separation is clean, and the Live/Analysis mode split is well-reasoned. However, the <2s latency target for Live Mode is aggressive and represents the primary risk to project success.

---

## What Works Well

### 1. Architecture Split (Live/Analysis)

Sound decision. Different latency budgets genuinely warrant different approaches:
- **Live Mode:** Custom orchestrator for <2s is correct — framework overhead would kill performance
- **Analysis Mode:** LangGraph for <15s is reasonable — complexity benefits from structure

### 2. Technology Choices

| Technology | Assessment |
|------------|------------|
| IBM Granite + watsonx.ai | Production-ready, enterprise support |
| LangChain | Mature, well-documented, good IBM integration |
| Tenacity | Battle-tested retry library, minimal overhead |
| Pydantic | Industry standard for validation |
| FastAPI | Proven for WebSocket + REST hybrid APIs |

None of these are experimental — all have production track records.

### 3. LiveKit Addition

Fills a real gap in the architecture. Building WebRTC from scratch would be a project-killer:
- Connection negotiation is complex
- Jitter buffering requires expertise
- Packet loss recovery is non-trivial
- Network adaptation needs constant tuning

LiveKit handles these hard transport problems, letting the team focus on racing logic.

### 4. Layer Separation

The four-layer architecture is clean and testable:

```
Transport (LiveKit)
    ↓
Speech AI (Watson STT/TTS)
    ↓
Orchestration (Custom / LangGraph)
    ↓
Intelligence (Granite LLM)
```

Each layer can be mocked, tested, and replaced independently.

---

## Risks & Concerns

| Risk | Severity | Description |
|------|----------|-------------|
| **<2s latency target** | 🔴 **High** | Theoretical minimum is ~1.4-1.8s with no margin |
| **LiveKit + Watson integration** | 🟡 Medium | No documented examples of this combination |
| **LangGraph stability** | 🟡 Medium | Pre-1.0 library, API may change |
| **Granite racing domain knowledge** | 🟡 Medium | Not fine-tuned for motorsport terminology |
| **VR platform latency** | 🟢 Low | Analysis mode only, 15s budget is comfortable |

### Risk 1: <2s Latency Target (HIGH)

**The Math:**

| Component | Estimated Latency |
|-----------|-------------------|
| LiveKit audio capture | ~50ms |
| Watson STT | ~300-400ms |
| Granite LLM (8B, 150 tokens) | ~800-1200ms |
| Watson TTS | ~300-400ms |
| LiveKit audio delivery | ~50ms |
| **Total** | **~1500-2100ms** |

**Problem:** Best case is ~1.5s, worst case exceeds 2s. Network variance, cold starts, or API congestion could easily break the target.

**Mitigation Options:**
- Stream TTS output (start playing while still generating)
- Use smaller Granite variant if available
- Implement response caching for common scenarios
- Adjust target to <3s if stakeholders accept it

### Risk 2: LiveKit + Watson Integration (MEDIUM)

No documented production examples of LiveKit → Watson STT → Watson TTS pipeline exist. The team will need to:
- Write custom audio format conversion (LiveKit frames → Watson expected format)
- Handle streaming vs. batch transcription tradeoffs
- Manage connection lifecycle between services

**Mitigation:** Allocate dedicated spike time for integration proof-of-concept.

### Risk 3: LangGraph Stability (MEDIUM)

The specified version `^0.0.20` indicates a pre-1.0 library:
- API may change between minor versions
- Documentation may lag behind implementation
- Edge cases may not be well-handled

**Mitigation:** 
- Pin exact versions in production
- Monitor LangGraph changelog closely
- Have fallback plan to custom orchestration if needed

### Risk 4: Granite Racing Domain Knowledge (MEDIUM)

IBM Granite is a general-purpose LLM, not fine-tuned for motorsport:
- May not know specific racing terminology
- Could hallucinate incorrect strategy advice
- Tire compound names, track-specific nuances may be wrong

**Mitigation:**
- Extensive prompt engineering with racing context
- Include glossary/definitions in system prompts
- Plan for future fine-tuning once interaction logs accumulate
- Human review of AI outputs during early deployment

---

## Critical Path Items

### 1. Latency Proof-of-Concept (MUST DO FIRST)

Build a minimal end-to-end voice loop before committing to full implementation:

```
Driver speaks → LiveKit → Watson STT → Granite → Watson TTS → LiveKit → Driver hears
```

**Success Criteria:**
- P50 latency < 1.8s
- P95 latency < 2.5s
- No dropped audio frames

**Timeline:** Sprint 1, first week

### 2. Interrupt Handling Validation

The sentence-completion logic during driver speech is complex:
- Driver speaks mid-AI-response
- AI finishes current sentence
- AI processes driver query
- AI responds to new query

**Edge Cases to Test:**
- Driver interrupts at sentence boundary
- Multiple rapid interruptions
- Critical alert during driver speech
- Network latency spike during interrupt

### 3. Granite Prompt Tuning

System prompts look good on paper, but real-world validation needed:
- Test with actual telemetry data
- Verify racing terminology accuracy
- Check for hallucinated advice
- Tune verbosity levels

**Budget:** 2-3 sprints of iteration expected

---

## Recommendations

### Immediate Actions

1. **Build latency PoC in Sprint 1** — This is the make-or-break validation
2. **Document LiveKit-Watson integration** — Create internal runbook as you go
3. **Pin all dependency versions** — Especially LangGraph
4. **Create racing terminology test set** — Validate Granite's domain knowledge

### Contingency Plans

| If This Happens... | Then Do This... |
|--------------------|-----------------|
| Latency consistently >2s | Negotiate 3s target OR implement streaming TTS |
| LangGraph API breaks | Fall back to custom orchestration for Analysis Mode |
| Granite hallucinates racing terms | Add retrieval-augmented generation (RAG) with racing knowledge base |
| LiveKit-Watson integration fails | Evaluate alternatives: Twilio, Daily.co, or custom WebSocket |

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Live Mode E2E Latency | P95 < 2.5s | Structured logging |
| Voice Recognition Accuracy | >95% | Manual review sample |
| LLM Response Relevance | >90% helpful | User feedback + logging |
| System Uptime | 99.5% | Monitoring |
| Analysis Mode Response Time | P95 < 15s | API metrics |

---

## Conclusion

The jarvis-granite architecture is **well-designed and implementable**. The team has made sensible technology choices, and the separation between Live and Analysis modes is appropriate.

The primary risk is the **<2s latency target for Live Mode**. This should be validated with a proof-of-concept before full implementation begins. If latency proves problematic, the options are:

1. Streaming TTS output
2. Smaller/faster Granite variant
3. Response caching for common scenarios
4. Adjusting the target to <3s

**Recommendation:** Proceed with implementation, but gate major development work on successful latency PoC completion in Sprint 1.

---

*Document prepared for Team 17 — F1 Jarvis TORCS Project*