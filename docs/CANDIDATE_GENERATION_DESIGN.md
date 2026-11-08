# Milestone 3 design: source-grounded promo candidate generation

## Goal

Generate a manageable set of possible 20–60 second promo clips from one episode, using Story Intelligence moments as anchors and the original parsed timeline as the source of truth.

## Inputs

- Parsed episode JSON from `src.parser`: episode duration, scene metadata, and all original timestamped events.
- Normalized Story Intelligence JSON from `src.story_normalizer`: validated dramatic moments, source-event IDs, hook/cliffhanger potential, review flags, and episode-ending context.

The episode IDs must match.

## Candidate construction

### Hook anchors

- Consider moments labelled `hook` or meeting the configured hook-potential threshold.
- Resolve the moment's source IDs against the parsed episode.
- Prefer a dialogue event when the moment contains one; otherwise use the earliest evidenced event.
- Preserve visual-only/context-dependent warnings for downstream review.

### Ending anchors

The generator considers three kinds of possible ending:

1. A model-scored dramatic moment meeting the cliffhanger-potential threshold.
2. A question in dialogue, cut at the end of that question before the next story beat. This is a heuristic and is always flagged for review.
3. The actual episode end, derived from the source episode duration and ending context. This is an episode-end candidate, **not** automatically a confirmed cliffhanger.

### Window pairing

Pair each eligible hook anchor with later ending anchors where:

- total duration is 20–60 seconds by default;
- the selected hook event completes before the clip ends;
- both anchors resolve to original source event IDs;
- start and end remain within the episode.

### Source timeline preservation

For each candidate, include all parsed events that overlap the proposed window, including simultaneous camera, action, dialogue, and sound directions. Each timeline event retains its original source timestamps and receives separate clip-overlap fields showing which portion falls within the candidate.

## Outputs

Each candidate includes:

- stable candidate ID;
- one continuous segment with exact numeric and display timestamps;
- source start/end event IDs;
- hook and ending evidence;
- all overlapping timeline events;
- source event ID list;
- candidate-generation priority;
- editorial-review flags;
- deterministic validation result.

## What this module does not claim

- It does not confirm that every ending is a true cliffhanger.
- It does not provide the final promo quality score.
- It does not analyze rendered video, acting, audio, or camera execution.
- It does not yet assemble multi-segment or cross-episode promos.
- It does not call an LLM. Story interpretation happened in Milestone 2; this module uses those outputs and performs timestamp-sensitive operations deterministically.

The next milestone will evaluate the generated candidates for hook, cliffhanger, standalone clarity, emotional engagement, pacing, and clip integrity.
