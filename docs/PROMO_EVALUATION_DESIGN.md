# PromoPilot Milestone 4 — Promo Evaluator

## Goal
Evaluate each timestamp-validated promo candidate for editorial quality. The evaluator does not edit video and does not claim screenplay-only inputs prove visual or audio impact.

## Scoring rubric

| Criterion | Weight | Question |
|---|---:|---|
| Hook strength | 25 | Does the first three seconds create a strong reason to stop scrolling? |
| Cliffhanger strength | 25 | Does the final beat leave a meaningful outcome unresolved? |
| Standalone clarity | 20 | Can a new viewer understand the basic situation without prior episodes? |
| Emotional engagement | 15 | Does the clip create a compelling emotional response? |
| Pacing | 10 | Does the clip sustain momentum without excessive setup or dead time? |
| Clip integrity | 5 | Are the events coherent, in order, and cut at usable boundaries? |

Each criterion is rated 0–5 by the LLM. Python calculates `sum((rating / 5) * weight)` for a score out of 100.

## Data flow

1. Load the parsed episode, candidate pack, and optional normalized story analysis.
2. Independently validate each candidate's source event IDs, timestamps, duration, and episode identity.
3. Reconstruct all timeline events from the parsed screenplay, including overlapping camera/action/dialogue events.
4. Add up to three post-cut events as context only; explicitly exclude them from the clip.
5. Send one compact candidate at a time to the configured Groq client.
6. Save the original model response, validate returned JSON and evidence IDs, and calculate the weighted score in Python.
7. Save a checkpoint per candidate so processing can resume without repeating successful API calls.

## Recommendation labels

Labels are triage hints, not publishing approval:
- `strong_candidate_for_editor_review`: score >= 80 and hook/cliffhanger both >= 4.
- `promising_candidate_for_editor_review`: score >= 65 and hook/cliffhanger both >= 3.
- `needs_editor_review`: hook or cliffhanger below 3.
- `low_priority_candidate`: otherwise.
- `rejected_or_fix_source`: deterministic validation failed.

All evaluated candidates require human review.

## Free-tier and privacy controls

- Uses the existing `src.llm_client.request_json_object` and `.env` Groq configuration.
- No paid APIs or services are introduced.
- Default completion budget is 900 tokens per candidate.
- Default delay is 20 seconds between calls; this is a conservative pacing control, not a guarantee against provider limits.
- Raw model outputs and per-candidate checkpoints are saved locally.
- Do not upload Kuku's confidential screenplay or model outputs to public GitHub. Check provider data controls before sending confidential materials to any hosted model.

## Known limits

- This milestone supports continuous single-episode candidates only.
- Script analysis cannot evaluate actual acting, rendered framing, sound/music, editing quality, or audience retention.
- LLM scores are subjective estimates and must be calibrated against editor feedback.
- Candidate generator's heuristic priority is not used as the final promo score.
