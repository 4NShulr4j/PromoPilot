"""Reusable, evidence-first story normalizer for PromoPilot.

The normalizer is show-agnostic: it contains no episode IDs, character names,
fixed character counts, or episode-specific editorial overrides. It uses parsed
screenplay events as the source of truth, treats model-generated identity and
relationship assertions as proposals, and persists only observed labels plus
human-approved aliases/relationships in a show-scoped registry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DIRECTION_LABELS = {"CAMERA", "ACTION", "SOUND FX", "SOUND EFFECT", "SOUND EFFECTS"}


def _clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_speaker_label(value: Any) -> str:
    """Normalize parser/LLM speaker labels, stripping delivery annotations.

    Examples:
      'LADY JULIANA (Heartbroken)' -> 'LADY JULIANA'
      'MATRIARCH (O.S.) (Firm)' -> 'MATRIARCH'
      'GERALD [whispering]' -> 'GERALD'
    """
    text = unicodedata.normalize("NFKC", _clean_space(value))
    # Remove repeated parenthetical and square-bracketed performance notes.
    previous = None
    while text != previous:
        previous = text
        text = re.sub(r"\s*\([^()]*\)", "", text)
        text = re.sub(r"\s*\[[^\[\]]*\]", "", text)
    # Remove standalone production cues if they occur outside brackets.
    text = re.sub(r"(?i)\b(?:O\.S\.?|V\.O\.?|OFF SCREEN|VOICE OVER)\b", "", text)
    text = re.sub(r"[\s:–—-]+$", "", text)
    return _clean_space(text).upper()


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _clean_space(value)).casefold()
    text = re.sub(r"[^\w\s'-]", " ", text, flags=re.UNICODE)
    return _clean_space(text)


def _stable_id(prefix: str, namespace: str, value: str) -> str:
    digest = hashlib.sha256(f"{namespace}|{value}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _clean_space(value)
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _speaker_from_event(event: dict[str, Any]) -> str:
    speaker = normalize_speaker_label(event.get("speaker"))
    if speaker:
        return speaker
    label = normalize_speaker_label(event.get("label"))
    if label in DIRECTION_LABELS:
        return ""
    return label


def empty_series_registry(show_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "show_id": show_id,
        "observed_speaker_labels": {},
        "approved_aliases": {},
        "approved_relationships": [],
        "notes": "Only observed labels are added automatically. Approve cross-label identity aliases and relationships through human review.",
    }


def _validate_registry(registry: dict[str, Any], show_id: str) -> dict[str, Any]:
    registry = _json_copy(registry)
    if registry.get("show_id") != show_id:
        raise ValueError(
            f"Registry show_id {registry.get('show_id')!r} does not match requested show_id {show_id!r}. "
            "Use a separate registry for each show."
        )
    registry.setdefault("schema_version", SCHEMA_VERSION)
    registry.setdefault("observed_speaker_labels", {})
    registry.setdefault("approved_aliases", {})
    registry.setdefault("approved_relationships", [])
    if not isinstance(registry["observed_speaker_labels"], dict):
        raise ValueError("Registry observed_speaker_labels must be an object")
    if not isinstance(registry["approved_aliases"], dict):
        raise ValueError("Registry approved_aliases must be an object")
    if not isinstance(registry["approved_relationships"], list):
        raise ValueError("Registry approved_relationships must be a list")
    return registry


def _approved_aliases(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return normalized label -> approved target entries.

    Registry format:
      "LADY IN GOLD GOWN": {
        "character_id": "char_clarissa",
        "canonical_name": "Clarissa",
        "approved": true,
        "evidence_event_ids": ["..."],
        "approved_by": "human reviewer"
      }
    """
    result: dict[str, dict[str, Any]] = {}
    for raw_label, target in registry.get("approved_aliases", {}).items():
        if not isinstance(target, dict) or target.get("approved") is not True:
            continue
        norm = normalize_speaker_label(raw_label)
        if not norm:
            continue
        if not _clean_space(target.get("character_id")) or not _clean_space(target.get("canonical_name")):
            raise ValueError(f"Approved alias {raw_label!r} needs character_id and canonical_name")
        result[norm] = target
    return result


def _event_evidence(event: dict[str, Any]) -> dict[str, Any]:
    """Rebuild moment evidence from parser data; never trust model-copied fields."""
    return {
        "event_id": event.get("event_id"),
        "episode_number": event.get("episode_number"),
        "scene_id": event.get("scene_id"),
        "scene_heading": event.get("scene_heading"),
        "label": event.get("label"),
        "speaker": event.get("speaker"),
        "normalized_speaker": _speaker_from_event(event) or None,
        "event_type": event.get("event_type"),
        "start_time": event.get("start_time"),
        "end_time": event.get("end_time"),
        "start_seconds": event.get("start_seconds"),
        "end_seconds": event.get("end_seconds"),
        "content": event.get("content"),
    }


def _canonical_display_name(
    profiles: list[dict[str, Any]],
    source_labels: list[str],
    approved_name: str | None = None,
) -> str:
    if approved_name:
        return approved_name
    names = _unique([_clean_space(p.get("canonical_name")) for p in profiles])
    if not names:
        return source_labels[0].title() if source_labels else "Unresolved character"
    source_norms = {normalize_name(x) for x in source_labels}
    # Prefer a model's proper-name form when another profile merely repeats the
    # screenplay role label (e.g. 'Gerald' over 'The Former Groom'). This is a
    # display choice only; it does not establish identity across distinct labels.
    non_role = [n for n in names if normalize_name(n) not in source_norms]
    if non_role:
        selected = sorted(non_role, key=lambda n: (len(n.split()), len(n), n.casefold()))[0]
    else:
        selected = sorted(names, key=lambda n: (len(n), n.casefold()))[0]
    # Normalize shout-case speaker/profile labels for a readable display name.
    if selected.isupper() and any(ch.isalpha() for ch in selected):
        return selected.title()
    return selected


def _profile_claims(profile: dict[str, Any]) -> list[str]:
    raw = profile.get("relationships", [])
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    return _unique([str(x) for x in raw if _clean_space(x)])


def _semantic_tokens(value: Any) -> set[str]:
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "at",
        "with", "from", "by", "is", "are", "was", "were", "be", "her",
        "his", "their", "she", "he", "they", "who", "that", "this",
        "wearing", "reveals", "revealing", "character", "person",
    }
    text = normalize_name(value)
    return {token for token in text.split() if len(token) > 2 and token not in stopwords}


def _suggest_unlinked_identity_candidates(
    unlinked_profiles: list[dict[str, Any]],
    speaker_groups: dict[str, dict[str, Any]],
    observed: dict[str, str],
    normalized_label_to_character: dict[str, str],
) -> list[dict[str, Any]]:
    """Suggest possible profile-to-speaker links from descriptive overlap.

    Suggestions are never auto-merged. They are presented for human review.
    """
    proposals: list[dict[str, Any]] = []
    for profile in unlinked_profiles:
        profile_name = _clean_space(profile.get("canonical_name")) or "Unresolved character"
        description = _clean_space(profile.get("description"))
        profile_tokens = _semantic_tokens(description) | _semantic_tokens(profile_name)
        ranked: list[tuple[int, float, str, str]] = []
        for normalized_label, display_label in observed.items():
            group_id = normalized_label_to_character.get(normalized_label)
            if not group_id:
                continue
            group = speaker_groups[group_id]
            linked_profiles = group.get("profiles", [])
            linked_text = " ".join(
                [display_label] +
                [_clean_space(p.get("canonical_name")) + " " + _clean_space(p.get("description"))
                 for p in linked_profiles]
            )
            label_tokens = _semantic_tokens(display_label)
            overlap = profile_tokens & label_tokens
            if len(overlap) >= 2:
                denominator = max(1, len(profile_tokens | label_tokens))
                score = len(overlap) / denominator
                ranked.append((len(overlap), score, normalized_label, group_id))
        if ranked:
            ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
            best = ranked[:3]
            proposals.append({
                "profile_name": profile_name,
                "candidate_character_ids": list(dict.fromkeys(item[3] for item in best)),
                "candidate_source_labels": [observed[item[2]] for item in best],
                "status": "requires_human_review",
                "reason": "Profile description shares distinctive terms with an observed screenplay speaker label; this is a suggestion, not proof of identity.",
                "matching_terms": sorted(profile_tokens & _semantic_tokens(" ".join(observed[item[2]] for item in best))),
                "overlap_scores": [round(item[1], 3) for item in best],
            })
    return proposals


def normalize_story(
    episode: dict[str, Any],
    analysis: dict[str, Any],
    registry: dict[str, Any] | None = None,
    show_id: str | None = None,
    source_validation_report: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Create an evidence-grounded normalized analysis and updated show registry."""
    episode_id = _clean_space(episode.get("episode_id") or analysis.get("episode_id"))
    analysis_episode_id = _clean_space(analysis.get("episode_id"))
    if not episode_id:
        raise ValueError("Parsed episode is missing episode_id")
    if analysis_episode_id and analysis_episode_id != episode_id:
        raise ValueError(
            f"Parsed episode {episode_id!r} does not match analysis {analysis_episode_id!r}"
        )
    show_id = _clean_space(show_id or (registry or {}).get("show_id") or "default_show")
    registry = _validate_registry(registry or empty_series_registry(show_id), show_id)

    events = episode.get("events") or []
    event_lookup: dict[str, dict[str, Any]] = {}
    for event in events:
        event_id = _clean_space(event.get("event_id"))
        if not event_id:
            continue
        if event_id in event_lookup:
            raise ValueError(f"Duplicate event_id in parsed episode: {event_id}")
        event_lookup[event_id] = event

    # Collect the script's actual speaker labels. The screenplay parser is the
    # authoritative source; LLM labels can only link to these labels.
    observed: dict[str, str] = {}
    for event in events:
        norm = _speaker_from_event(event)
        if norm:
            display = _clean_space(event.get("speaker") or event.get("label"))
            observed.setdefault(norm, display)

    aliases = _approved_aliases(registry)
    speaker_groups: dict[str, dict[str, Any]] = {}
    normalized_label_to_character: dict[str, str] = {}

    # Build stable show-scoped speaker identities. Different observed labels
    # remain separate unless a human-approved alias exists in the registry.
    for norm_label, display_label in sorted(observed.items()):
        approved = aliases.get(norm_label)
        if approved:
            character_id = _clean_space(approved["character_id"])
            canonical_override = _clean_space(approved["canonical_name"])
            identity_status = "human_approved_alias"
        else:
            character_id = _stable_id("char", show_id, norm_label)
            canonical_override = ""
            identity_status = "speaker_label_identity"
        key = character_id
        if key not in speaker_groups:
            speaker_groups[key] = {
                "character_id": character_id,
                "canonical_override": canonical_override,
                "identity_status": identity_status,
                "source_labels": [],
                "normalized_labels": [],
                "profiles": [],
            }
        speaker_groups[key]["source_labels"].append(display_label)
        speaker_groups[key]["normalized_labels"].append(norm_label)
        normalized_label_to_character[norm_label] = character_id

    # Persist observed labels only. No model inference is silently promoted to
    # an approved alias or relationship.
    observed_registry = registry["observed_speaker_labels"]
    for norm_label, display_label in observed.items():
        entry = observed_registry.setdefault(norm_label, {"display_label": display_label, "episodes_seen": []})
        if episode_id not in entry.setdefault("episodes_seen", []):
            entry["episodes_seen"].append(episode_id)
        if not entry.get("display_label"):
            entry["display_label"] = display_label

    unresolved_profile_labels: list[dict[str, Any]] = []
    unlinked_profiles: list[dict[str, Any]] = []
    alias_proposals: list[dict[str, Any]] = []
    relationship_claims: list[dict[str, Any]] = []

    for profile_index, profile in enumerate(analysis.get("characters", []) or [], start=1):
        raw_name = _clean_space(profile.get("canonical_name")) or f"Unlabelled profile {profile_index}"
        raw_labels = profile.get("source_labels", []) or []
        if isinstance(raw_labels, str):
            raw_labels = [raw_labels]
        matched_norms: list[str] = []
        unmatched_labels: list[str] = []
        for raw_label in raw_labels:
            norm = normalize_speaker_label(raw_label)
            if norm in observed:
                matched_norms.append(norm)
            elif norm:
                unmatched_labels.append(str(raw_label))
                unresolved_profile_labels.append({
                    "profile_name": raw_name,
                    "source_label": str(raw_label),
                    "normalized_label": norm,
                    "status": "not_found_in_parsed_screenplay",
                })

        # If the model didn't provide source_labels, allow exact canonical-name
        # matching to a real screenplay speaker label. Do not fuzzy-merge names.
        if not matched_norms:
            canonical_norm = normalize_speaker_label(raw_name)
            if canonical_norm in observed:
                matched_norms.append(canonical_norm)

        matched_norms = list(dict.fromkeys(matched_norms))
        matched_character_ids = list(dict.fromkeys(
            normalized_label_to_character[n] for n in matched_norms
        ))

        claims = _profile_claims(profile)
        if len(matched_character_ids) == 1:
            speaker_groups[matched_character_ids[0]]["profiles"].append(profile)
        elif len(matched_character_ids) > 1:
            # One model profile mentions multiple speakers. That's a candidate
            # identity assertion, not sufficient evidence to merge them.
            alias_proposals.append({
                "profile_name": raw_name,
                "candidate_character_ids": matched_character_ids,
                "candidate_source_labels": [observed[n] for n in matched_norms],
                "status": "requires_human_review",
                "reason": "One model profile references multiple distinct screenplay speaker labels.",
            })
            unlinked_profiles.append(profile)
        else:
            # Keep profiles that have no direct speaker link. They may be silent
            # characters, descriptions, or model-inferred identities.
            unlinked_profiles.append(profile)

        for claim in claims:
            relationship_claims.append({
                "subject_profile_name": raw_name,
                "claim": claim,
                "status": "unverified_model_claim",
                "evidence_event_ids": [],
                "note": "Story-analysis relationship strings are not source evidence by themselves.",
            })

    # Build normalized characters from actual speaker groups.
    normalized_characters: list[dict[str, Any]] = []
    for group in speaker_groups.values():
        labels = _unique(group["source_labels"])
        profiles = group["profiles"]
        name = _canonical_display_name(profiles, labels, group.get("canonical_override") or None)
        model_names = _unique([_clean_space(p.get("canonical_name")) for p in profiles])
        descriptions = _unique([_clean_space(p.get("description")) for p in profiles])
        aliases_found = _unique([x for x in model_names if normalize_name(x) != normalize_name(name)])
        all_aliases = _unique(labels + aliases_found)
        confirmed_relationships = [
            _json_copy(rel) for rel in registry.get("approved_relationships", [])
            if rel.get("approved") is True and rel.get("subject_character_id") == group["character_id"]
        ]
        normalized_characters.append({
            "character_id": group["character_id"],
            "canonical_name": name,
            "source_labels": labels,
            "aliases": all_aliases,
            "model_names": model_names,
            "descriptions": descriptions,
            "identity_status": group["identity_status"],
            "identity_evidence": {
                "method": "exact_normalized_speaker_label_match",
                "normalized_speaker_labels": _unique(group["normalized_labels"]),
                "profile_count": len(profiles),
            },
            "confirmed_relationships": confirmed_relationships,
            "relationships": [
                {
                    "claim": rel.get("claim", ""),
                    "status": "confirmed_human_approved",
                    "evidence_event_ids": rel.get("evidence_event_ids", []),
                }
                for rel in confirmed_relationships
            ],
            "unverified_relationship_claims": [
                claim for claim in relationship_claims
                if any(normalize_name(claim["subject_profile_name"]) == normalize_name(n)
                       for n in model_names)
            ],
        })

    # Add unlinked model profiles without pretending they are confirmed speakers.
    seen_unlinked: set[str] = set()
    for profile in unlinked_profiles:
        name = _clean_space(profile.get("canonical_name")) or "Unresolved character"
        key = normalize_name(name)
        if key in seen_unlinked:
            continue
        seen_unlinked.add(key)
        normalized_characters.append({
            "character_id": _stable_id("profile", show_id, key),
            "canonical_name": name,
            "source_labels": [],
            "aliases": [],
            "model_names": [name],
            "descriptions": _unique([_clean_space(profile.get("description"))]),
            "identity_status": "unlinked_model_profile",
            "identity_evidence": {
                "method": "no_exact_source_speaker_match",
                "note": "Kept as a separate profile until a source-backed or human-approved identity link exists.",
            },
            "confirmed_relationships": [],
            "relationships": [],
            "unverified_relationship_claims": [
                c for c in relationship_claims
                if normalize_name(c["subject_profile_name"]) == key
            ],
        })

    # Suggest likely profile-to-speaker links from descriptive overlap.
    # These remain review items; the engine never silently merges distinct labels.
    semantic_suggestions = _suggest_unlinked_identity_candidates(
        unlinked_profiles, speaker_groups, observed, normalized_label_to_character
    )
    alias_proposals.extend(semantic_suggestions)

    searchable_source = " ".join([
        _clean_space(episode.get("logline")),
        _clean_space(episode.get("continuity_notes")),
        *[_clean_space(e.get("content")) for e in events],
    ]).casefold()
    for profile in unlinked_profiles:
        name = _clean_space(profile.get("canonical_name"))
        if not name:
            continue
        pattern = rf"(?<!\w){re.escape(name.casefold())}(?!\w)"
        source_match = bool(re.search(pattern, searchable_source))
        existing = next(
            (p for p in alias_proposals if normalize_name(p.get("profile_name")) == normalize_name(name)),
            None,
        )
        if existing:
            existing["source_text_match"] = source_match
        elif source_match:
            alias_proposals.append({
                "profile_name": name,
                "candidate_character_ids": [],
                "candidate_source_labels": [],
                "status": "requires_human_review",
                "reason": "The name occurs in source text but no exact speaker-label identity link was established.",
                "source_text_match": True,
            })

    # Preserve source event IDs, and reconstruct all evidence from parser data.
    normalized_moments: list[dict[str, Any]] = []
    rejected_moments: list[dict[str, Any]] = []
    moment_warnings: list[dict[str, Any]] = []
    for index, original in enumerate(analysis.get("dramatic_moments", []) or [], start=1):
        moment = _json_copy(original)
        ids = list(dict.fromkeys(str(x) for x in moment.get("event_ids", []) if str(x)))
        unknown = [event_id for event_id in ids if event_id not in event_lookup]
        if not ids or unknown:
            rejected_moments.append({
                "moment_index": index,
                "description": moment.get("description", ""),
                "reason": "missing_event_ids" if not ids else "unknown_source_event_ids",
                "unknown_event_ids": unknown,
            })
            continue
        evidence = sorted(
            [_event_evidence(event_lookup[event_id]) for event_id in ids],
            key=lambda item: (item.get("start_seconds") or 0, item.get("end_seconds") or 0, item.get("event_id") or ""),
        )
        moment["source_evidence"] = evidence
        moment["evidence_validation"] = "all_event_ids_verified"
        moment["source_event_ids"] = ids
        # Flag visual-only hook proposals for review. This is deliberately an
        # advisory rule, not an episode-specific forced reclassification.
        if str(moment.get("moment_type", "")).casefold() == "hook":
            has_dialogue = any(item.get("event_type") == "dialogue" for item in evidence)
            if not has_dialogue:
                flags = list(moment.get("editorial_review_flags", []))
                flags.append({
                    "code": "visual_only_hook_review",
                    "severity": "advisory",
                    "message": "Hook is supported only by non-dialogue events; verify that the visual beat is attention-grabbing in the rendered video.",
                })
                moment["editorial_review_flags"] = flags
                moment_warnings.append({
                    "moment_index": index,
                    "code": "visual_only_hook_review",
                    "event_ids": ids,
                })
        normalized_moments.append(moment)

    # Episode ending context is derived from the actual timeline, not injected as
    # a hardcoded editorial moment. This can help the next promo-candidate stage.
    ending_context: dict[str, Any] = {"status": "unavailable", "events": []}
    duration = episode.get("duration_seconds")
    if events and isinstance(duration, (int, float)):
        window_start = max(0.0, float(duration) - 15.0)
        final_events = [e for e in events if float(e.get("end_seconds", 0) or 0) >= window_start]
        final_events.sort(key=lambda e: (float(e.get("start_seconds", 0) or 0), float(e.get("end_seconds", 0) or 0)))
        ending_context = {
            "status": "derived_from_source_timeline_requires_editorial_interpretation",
            "window_start_seconds": window_start,
            "episode_duration_seconds": float(duration),
            "events": [_event_evidence(e) for e in final_events],
        }

    normalized = {
        "schema_version": SCHEMA_VERSION,
        "show_id": show_id,
        "episode_id": episode_id,
        "episode_summary": analysis.get("episode_summary", ""),
        "chunk_summaries": analysis.get("chunk_summaries", []),
        "characters": normalized_characters,
        "identity_resolution_candidates": alias_proposals,
        "dramatic_moments": normalized_moments,
        "episode_ending_context": ending_context,
        "source_event_count": len(events),
        "story_input_event_count": analysis.get("story_input_event_count"),
        "model_moment_count": len(analysis.get("dramatic_moments", []) or []),
        "normalized_moment_count": len(normalized_moments),
        "model_metadata": analysis.get("model_metadata", {}),
        "normalization_metadata": {
            "engine": "PromoPilot Universal Story Normalizer",
            "schema_version": SCHEMA_VERSION,
            "identity_policy": "exact source-label matches plus human-approved show-scoped aliases",
            "relationship_policy": "model claims remain unverified unless present in the human-approved series registry",
            "episode_specific_overrides": False,
        },
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "show_id": show_id,
        "episode_id": episode_id,
        "status": "success_with_review_items" if (unresolved_profile_labels or alias_proposals or rejected_moments or moment_warnings) else "success",
        "input_event_count": len(events),
        "observed_speaker_label_count": len(observed),
        "input_character_profile_count": len(analysis.get("characters", []) or []),
        "normalized_character_count": len(normalized_characters),
        "identity_resolution_candidate_count": len(alias_proposals),
        "unresolved_profile_label_count": len(unresolved_profile_labels),
        "model_relationship_claim_count": len(relationship_claims),
        "rejected_moment_count": len(rejected_moments),
        "accepted_moment_count": len(normalized_moments),
        "moment_review_flag_count": len(moment_warnings),
        "unresolved_profile_labels": unresolved_profile_labels,
        "identity_resolution_candidates": alias_proposals,
        "unverified_relationship_claims": relationship_claims,
        "rejected_moments": rejected_moments,
        "moment_warnings": moment_warnings,
        "source_validation_report_status": (source_validation_report or {}).get("status"),
        "source_validation_warnings": (source_validation_report or {}).get("warnings", []),
        "notes": [
            "No model-generated relationship was promoted to confirmed status without an approved registry entry.",
            "Different speaker labels remain distinct unless a human-approved show-scoped alias merges them.",
            "Moments with invalid source event IDs are excluded from normalized output.",
        ],
    }
    return normalized, report, registry


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return slug or "default_show"


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def save_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temp.replace(target)


def run_normalization(
    episode_path: str | Path,
    analysis_path: str | Path,
    show_id: str,
    report_path: str | Path | None = None,
    registry_path: str | Path | None = None,
    output_dir: str | Path = "outputs/story_analysis/normalized",
) -> tuple[Path, Path, Path]:
    episode = load_json(episode_path)
    analysis = load_json(analysis_path)
    source_report = load_json(report_path) if report_path and Path(report_path).exists() else None
    if registry_path:
        registry_file = Path(registry_path)
    else:
        registry_file = Path("data/series") / _safe_slug(show_id) / "registry.json"
    registry = load_json(registry_file) if registry_file.exists() else empty_series_registry(show_id)

    normalized, report, updated_registry = normalize_story(
        episode=episode,
        analysis=analysis,
        registry=registry,
        show_id=show_id,
        source_validation_report=source_report,
    )
    output_path = Path(output_dir)
    episode_id = _safe_slug(normalized["episode_id"].lower())
    normalized_path = output_path / f"{episode_id}_story_analysis_normalized.json"
    report_out_path = output_path / f"{episode_id}_validation_report_normalized.json"
    save_json(normalized_path, normalized)
    save_json(report_out_path, report)
    save_json(registry_file, updated_registry)
    return normalized_path, report_out_path, registry_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize PromoPilot story analysis generically for any show.")
    parser.add_argument("--episode", required=True, help="Parsed screenplay episode JSON")
    parser.add_argument("--analysis", required=True, help="Merged Story Intelligence JSON")
    parser.add_argument("--report", help="Optional source Story Intelligence validation report")
    parser.add_argument("--show-id", required=True, help="Stable identifier shared by episodes from the same show")
    parser.add_argument("--registry", help="Optional path to a show-scoped registry JSON")
    parser.add_argument("--output-dir", default="outputs/story_analysis/normalized")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    normalized_path, report_path, registry_path = run_normalization(
        episode_path=args.episode,
        analysis_path=args.analysis,
        show_id=args.show_id,
        report_path=args.report,
        registry_path=args.registry,
        output_dir=args.output_dir,
    )
    print("Story normalization completed.")
    print(f"Normalized analysis: {normalized_path}")
    print(f"Validation report:   {report_path}")
    print(f"Show registry:       {registry_path}")
    report = load_json(report_path)
    print(
        f"Characters: {report['normalized_character_count']} | "
        f"Moments accepted: {report['accepted_moment_count']} | "
        f"Review items: {report['identity_resolution_candidate_count'] + report['unresolved_profile_label_count'] + report['moment_review_flag_count']}"
    )


if __name__ == "__main__":
    main()
