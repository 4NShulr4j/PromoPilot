"""End-to-end orchestration for the PromoPilot TXT-to-ranked-promos pipeline.

All stages call the project's existing CLI modules. LLM usage is limited to
Story Intelligence and Promo Evaluation via the configured Groq client.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from queue import Empty, Queue
from threading import Thread
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class PipelineConfig:
    story_max_completion_tokens: int = 1400
    story_chunk_size: int = 6
    story_chunk_overlap: int = 1
    story_delay_between_chunks: float = 18.0
    max_candidates: int = 10
    evaluator_max_completion_tokens: int = 900
    evaluator_delay_between_calls: float = 18.0
    min_shortlist_score: float = 40.0
    top_n: int = 5
    overlap_threshold: float = 0.80

    def validate(self) -> None:
        if self.story_max_completion_tokens < 200:
            raise ValueError("Story completion token limit must be at least 200.")
        if not 2 <= self.story_chunk_size <= 20:
            raise ValueError("Story chunk size must be between 2 and 20.")
        if not 0 <= self.story_chunk_overlap < self.story_chunk_size:
            raise ValueError("Story chunk overlap must be smaller than chunk size.")
        if self.story_delay_between_chunks < 0 or self.evaluator_delay_between_calls < 0:
            raise ValueError("API delay values cannot be negative.")
        if not 1 <= self.max_candidates <= 40:
            raise ValueError("Maximum candidates must be between 1 and 40.")
        if self.evaluator_max_completion_tokens < 200:
            raise ValueError("Evaluator completion token limit must be at least 200.")
        if not 0 <= self.min_shortlist_score <= 100:
            raise ValueError("Minimum shortlist score must be between 0 and 100.")
        if not 1 <= self.top_n <= 20:
            raise ValueError("Top-N must be between 1 and 20.")
        if not 0.0 <= self.overlap_threshold <= 1.0:
            raise ValueError("Overlap threshold must be between 0 and 1.")


class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.message = message


def safe_slug(value: str, *, fallback: str = "run") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip()).strip("_")
    if not cleaned:
        return fallback
    return cleaned[:80]


def _emit(callback: Callable[[str, str, str], None] | None, kind: str, stage: str, message: str) -> None:
    if callback:
        callback(kind, stage, message)


def run_module(
    module: str,
    args: list[str],
    *,
    root: Path,
    stage: str,
    on_event: Callable[[str, str, str], None] | None = None,
    timeout_seconds: int = 7200,
) -> str:
    command = [sys.executable, "-u", "-m", module, *[str(x) for x in args]]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    _emit(on_event, "start", stage, f"Running {module}")
    try:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise PipelineStageError(stage, f"Could not start {module}: {exc}") from exc

    lines: list[str] = []
    output_queue: Queue[str | None] = Queue()

    def read_output() -> None:
        assert process.stdout is not None
        try:
            for output_line in process.stdout:
                output_queue.put(output_line)
        finally:
            output_queue.put(None)

    reader = Thread(target=read_output, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout_seconds
    reader_finished = False
    try:
        while not reader_finished:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                process.wait()
                raise PipelineStageError(stage, f"{module} exceeded the {timeout_seconds // 60}-minute timeout.")
            try:
                line = output_queue.get(timeout=min(0.5, remaining))
            except Empty:
                continue
            if line is None:
                reader_finished = True
                continue
            clean = line.rstrip("\r\n")
            lines.append(clean)
            _emit(on_event, "log", stage, clean)
        return_code = process.wait(timeout=max(1.0, deadline - time.monotonic()))
    except Exception:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        reader.join(timeout=1.0)

    output = "\n".join(lines).strip()
    if return_code != 0:
        tail = "\n".join(lines[-35:]) or f"Process exited with code {return_code}."
        raise PipelineStageError(stage, f"{module} failed (exit {return_code}).\n{tail}")
    _emit(on_event, "success", stage, f"Completed {module}")
    return output


def _write_uploads(uploaded_files: Iterable[tuple[str, bytes]], input_dir: Path) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    used: set[str] = set()
    for original_name, data in uploaded_files:
        basename = Path(original_name).name
        if not basename.lower().endswith(".txt"):
            raise ValueError(f"Only .txt screenplay files are supported: {basename}")
        if not data:
            raise ValueError(f"The uploaded file is empty: {basename}")
        try:
            data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{basename} is not UTF-8 encoded. Save it as UTF-8 and upload again.") from exc
        stem = safe_slug(Path(basename).stem, fallback="episode")
        name = f"{stem}.txt"
        counter = 2
        while name.lower() in used:
            name = f"{stem}_{counter}.txt"
            counter += 1
        used.add(name.lower())
        target = input_dir / name
        target.write_bytes(data)
        saved.append(target)
    if not saved:
        raise ValueError("Upload at least one screenplay TXT file.")
    return saved


def run_full_pipeline(
    uploaded_files: Iterable[tuple[str, bytes]],
    *,
    root: str | Path,
    show_id: str,
    run_id: str,
    config: PipelineConfig | None = None,
    on_event: Callable[[str, str, str], None] | None = None,
) -> list[dict]:
    """Run all PromoPilot stages for one or more uploaded episodes.

    A stable run_id allows retry after a failed LLM stage: story chunks and
    candidate evaluation checkpoints are retained and reused by the existing
    modules when their source fingerprints still match.
    """
    cfg = config or PipelineConfig()
    cfg.validate()
    safe_show = safe_slug(show_id, fallback="show")
    safe_run = safe_slug(run_id, fallback="run")
    project_root = Path(root).resolve()
    run_root = project_root / "data" / "runs" / safe_run
    input_dir = run_root / "input"
    parsed_dir = run_root / "parsed"
    story_dir = run_root / "story_analysis"
    raw_llm_dir = run_root / "raw_llm"
    normalized_dir = run_root / "normalized"
    candidate_dir = run_root / "candidates"
    evaluation_dir = run_root / "evaluations"
    ranking_dir = run_root / "ranking"
    series_registry = project_root / "data" / "series" / safe_show / "registry.json"

    for folder in (parsed_dir, story_dir, raw_llm_dir, normalized_dir, candidate_dir, evaluation_dir, ranking_dir):
        folder.mkdir(parents=True, exist_ok=True)

    saved_inputs = _write_uploads(uploaded_files, input_dir)

    # Preflight parse validates the entire batch and prevents parser output collisions
    # when two uploaded scripts claim the same episode index.
    from src.parser import parse_screenplay_file
    preflight = []
    for path in saved_inputs:
        try:
            parsed = parse_screenplay_file(path)
        except Exception as exc:
            raise ValueError(f"Could not parse {path.name}: {type(exc).__name__}: {exc}") from exc
        preflight.append((path.name, parsed["episode_id"]))
    seen_ids: dict[str, str] = {}
    for filename, episode_id in preflight:
        if episode_id in seen_ids:
            raise ValueError(
                f"Two uploaded files resolve to {episode_id}: {seen_ids[episode_id]} and {filename}. "
                "Upload only one file per episode ID in a batch."
            )
        seen_ids[episode_id] = filename

    run_module(
        "src.parser",
        ["--input", str(input_dir), "--output", str(parsed_dir)],
        root=project_root, stage="Screenplay Parser", on_event=on_event,
    )
    parsed_files = sorted(parsed_dir.glob("ep*.json"))
    if len(parsed_files) != len(saved_inputs):
        raise PipelineStageError(
            "Screenplay Parser",
            f"Expected {len(saved_inputs)} parsed episode file(s), found {len(parsed_files)} under {parsed_dir}.",
        )

    results: list[dict] = []
    for parsed_path in parsed_files:
        with parsed_path.open("r", encoding="utf-8") as handle:
            episode = json.load(handle)
        episode_id = str(episode.get("episode_id") or parsed_path.stem.upper())
        ep_slug = safe_slug(episode_id.lower(), fallback="episode")
        story_output = story_dir / f"{ep_slug}_story_analysis.json"
        story_report = story_dir / f"{ep_slug}_validation_report.json"
        normalized_analysis = normalized_dir / f"{ep_slug}_story_analysis_normalized.json"
        candidate_path = candidate_dir / f"{ep_slug}_candidates.json"
        evaluation_path = evaluation_dir / f"{ep_slug}_evaluations.json"
        quality_path = evaluation_dir / f"{ep_slug}_quality_gate.json"
        ranking_path = ranking_dir / f"{ep_slug}_ranking.json"
        final_path = ranking_dir / f"{ep_slug}_ranking_final.json"

        run_module(
            "src.story_analyzer",
            [
                "--input", str(parsed_path),
                "--output-dir", str(story_dir),
                "--raw-dir", str(raw_llm_dir),
                "--max-completion-tokens", str(cfg.story_max_completion_tokens),
                "--chunk-size", str(cfg.story_chunk_size),
                "--chunk-overlap", str(cfg.story_chunk_overlap),
                "--delay-between-chunks", str(cfg.story_delay_between_chunks),
            ],
            root=project_root, stage=f"{episode_id} · Story Intelligence", on_event=on_event,
        )
        if not story_output.exists() or not story_report.exists():
            raise PipelineStageError(f"{episode_id} · Story Intelligence", "Story analysis outputs were not created.")

        run_module(
            "src.story_normalizer",
            [
                "--episode", str(parsed_path),
                "--analysis", str(story_output),
                "--report", str(story_report),
                "--show-id", safe_show,
                "--registry", str(series_registry),
                "--output-dir", str(normalized_dir),
            ],
            root=project_root, stage=f"{episode_id} · Story Normalizer", on_event=on_event,
        )
        if not normalized_analysis.exists():
            raise PipelineStageError(f"{episode_id} · Story Normalizer", "Normalized story analysis was not created.")

        run_module(
            "src.candidate_generator",
            [
                "--episode", str(parsed_path),
                "--analysis", str(normalized_analysis),
                "--output", str(candidate_path),
                "--max-candidates", str(cfg.max_candidates),
            ],
            root=project_root, stage=f"{episode_id} · Candidate Generator", on_event=on_event,
        )

        run_module(
            "src.promo_evaluator",
            [
                "--episode", str(parsed_path),
                "--candidates", str(candidate_path),
                "--analysis", str(normalized_analysis),
                "--output", str(evaluation_path),
                "--output-dir", str(evaluation_dir / f"{ep_slug}_checkpoints"),
                "--max-completion-tokens", str(cfg.evaluator_max_completion_tokens),
                "--delay-between-calls", str(cfg.evaluator_delay_between_calls),
                "--limit", str(cfg.max_candidates),
            ],
            root=project_root, stage=f"{episode_id} · Promo Evaluator", on_event=on_event,
        )

        run_module(
            "src.evaluation_quality_gate",
            [
                "--episode", str(parsed_path),
                "--candidates", str(candidate_path),
                "--evaluations", str(evaluation_path),
                "--out", str(quality_path),
            ],
            root=project_root, stage=f"{episode_id} · Evaluation Quality Gate", on_event=on_event,
        )
        run_module(
            "src.ranking_engine",
            [
                "--quality-gate", str(quality_path),
                "--candidates", str(candidate_path),
                "--out", str(ranking_path),
                "--top-n", str(cfg.top_n),
                "--overlap-threshold", str(cfg.overlap_threshold),
            ],
            root=project_root, stage=f"{episode_id} · Ranking Engine", on_event=on_event,
        )
        run_module(
            "src.shortlist_quality_policy",
            [
                "--ranking", str(ranking_path),
                "--out", str(final_path),
                "--min-score", str(cfg.min_shortlist_score),
            ],
            root=project_root, stage=f"{episode_id} · Shortlist Policy", on_event=on_event,
        )
        if not final_path.exists():
            raise PipelineStageError(f"{episode_id} · Shortlist Policy", "Final ranking output was not created.")
        with final_path.open("r", encoding="utf-8") as handle:
            final_ranking = json.load(handle)

        artifacts = {
            "parsed_episode": parsed_path,
            "story_analysis": story_output,
            "story_validation_report": story_report,
            "normalized_analysis": normalized_analysis,
            "normalized_validation_report": normalized_dir / f"{ep_slug}_validation_report_normalized.json",
            "candidates": candidate_path,
            "evaluations": evaluation_path,
            "quality_gate": quality_path,
            "ranking": ranking_path,
            "final_ranking": final_path,
        }
        results.append({
            "episode_id": episode_id,
            "run_id": safe_run,
            "show_id": safe_show,
            "run_dir": run_root,
            "final_ranking": final_ranking,
            "artifacts": artifacts,
        })
    return results
