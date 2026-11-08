# Universal Story Normalizer — Design

## Input contracts

### Parsed episode

Requires `episode_id` and `events`. Events should include `event_id`, `speaker` or `label`, `event_type`, source timestamps, numeric timeline values, and original `content`. Optional fields include `episode_number`, `scene_id`, `scene_heading`, `duration_seconds`, `logline`, and `continuity_notes`.

### Story Intelligence analysis

Requires `episode_id`. Optional `characters` is an array of objects containing `canonical_name`, `source_labels`, `description`, and `relationships`. Optional `dramatic_moments` contains moment objects with `event_ids` and editorial scores/descriptions.

## Identity policy

1. Parser speaker labels are the source of truth for who spoke in that episode.
2. Parenthetical/bracketed delivery notes and production cues are removed before comparison.
3. A model profile attaches to a speaker through exact normalized `source_labels`, or exact canonical-name matching if it did not provide source labels.
4. Profiles attached to the same source speaker are consolidated.
5. Different source speaker labels are not fuzzy-merged. Potential aliases are surfaced for human review.
6. Approved aliases in the show registry can map multiple labels to a stable character ID across episodes.

## Relationship policy

Model-generated relationship strings are claims, not proof. They remain unverified unless a human-approved record is present in the show's registry, including source event IDs.

## Moment evidence policy

`dramatic_moments[].event_ids` must all exist in the parsed episode. Evidence is reconstructed from the parser records, preventing the model from changing dialogue or timestamps in the copied evidence. Invalid moments are reported and omitted.

## Show registry policy

Each show has a separate JSON registry. Automatic updates add only observed speaker-label history. Approved aliases and relationships require explicit human edits/approval.

## Extension points

- Add a review UI for accepting/rejecting `identity_resolution_candidates`.
- Add evidence-aware relationship candidate extraction.
- Add configurable ending-context duration.
- Add series-level event/character graph after the multi-episode registry has been validated.
