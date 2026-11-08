from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from src.llm_client import get_groq_api_key, get_model_name
from src.pipeline_runner import PipelineConfig, PipelineStageError, run_full_pipeline
from src.ui_helpers import (
    candidate_table_rows,
    flag_counts,
    format_timecode,
    index_candidate_details,
    load_json,
    select_best_promo,
    validate_ranking,
)

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "data" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="PromoPilot · Promo Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

if get_groq_api_key():
    os.environ.setdefault("GROQ_API_KEY", get_groq_api_key())
os.environ.setdefault("GROQ_MODEL", get_model_name())

# Premium editorial dark UI: deep ink surfaces, mint accents and restrained gold.
st.markdown(
    r"""
    <style>
      :root { --pp-bg:#0b1115; --pp-panel:#111b20; --pp-panel-2:#17242a; --pp-line:rgba(210,235,226,.12);
             --pp-text:#eff7f3; --pp-muted:#9aaca6; --pp-mint:#b8f36b; --pp-gold:#f1c66d; }
      [data-testid="stAppViewContainer"] { background: radial-gradient(ellipse at 80% -10%, rgba(66,112,89,.22), transparent 38%), var(--pp-bg); }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stSidebar"] { background: linear-gradient(180deg,#101a1f,#0b1115); border-right:1px solid var(--pp-line); }
      .block-container { max-width: 1460px; padding-top: 1.25rem; padding-bottom: 3rem; }
      h1,h2,h3,h4 { color:var(--pp-text)!important; letter-spacing:-.025em; }
      p, label, [data-testid="stMarkdownContainer"] { color:var(--pp-text); }
      [data-testid="stCaptionContainer"], .pp-muted { color:var(--pp-muted)!important; }
      [data-testid="stMetric"] { background:linear-gradient(145deg,rgba(26,42,47,.92),rgba(16,26,31,.92)); border:1px solid var(--pp-line); padding:1rem 1.05rem; border-radius:16px; }
      [data-testid="stMetricLabel"] { color:var(--pp-muted)!important; }
      [data-testid="stMetricValue"] { color:var(--pp-text)!important; font-weight:750; }
      .stButton>button, .stDownloadButton>button { border-radius:12px; min-height:2.7rem; border:1px solid rgba(184,243,107,.32); font-weight:700; }
      .stButton>button[kind="primary"] { background:linear-gradient(110deg,#b8f36b,#8bdc83); color:#102018; border:0; }
      .stButton>button:hover, .stDownloadButton>button:hover { border-color:var(--pp-mint); color:var(--pp-mint); }
      [data-testid="stFileUploader"] { border:1px dashed rgba(184,243,107,.38); border-radius:16px; background:rgba(184,243,107,.035); padding:.6rem; }
      [data-testid="stTabs"] button { font-weight:650; }
      [data-testid="stTabs"] button[aria-selected="true"] { color:var(--pp-mint)!important; }
      div[data-baseweb="select"]>div, div[data-baseweb="input"]>div, textarea { background:#121d22!important; border-color:var(--pp-line)!important; border-radius:10px!important; }
      .pp-brand { display:flex; align-items:center; gap:13px; margin:.2rem 0 .2rem; }
      .pp-brand-mark { width:48px;height:48px;display:flex;align-items:center;justify-content:center;border-radius:15px;background:linear-gradient(145deg,#c4f784,#75c77d);color:#112018;font-size:25px;box-shadow:0 10px 32px rgba(184,243,107,.12); }
      .pp-brand-title { font-size:2rem;line-height:1.05;font-weight:850;letter-spacing:-.05em;color:#f2f8f4; }
      .pp-brand-sub { font-size:.88rem;color:#9aaca6;margin-top:5px; }
      .pp-hero { padding:24px 27px; border:1px solid var(--pp-line); border-radius:22px; background:linear-gradient(115deg,rgba(30,54,48,.95),rgba(18,31,36,.98) 60%,rgba(27,37,35,.96)); position:relative; overflow:hidden; }
      .pp-hero:after { content:'✦'; position:absolute; right:34px;top:10px;font-size:100px;color:rgba(184,243,107,.07); }
      .pp-eyebrow { color:var(--pp-mint);text-transform:uppercase;letter-spacing:.16em;font-size:.7rem;font-weight:800; }
      .pp-hero h2 { font-size:1.75rem;margin:.45rem 0 .35rem; }
      .pp-hero p { color:#b8c9c2;margin:0;max-width:780px; }
      .pp-section-kicker { color:var(--pp-mint);text-transform:uppercase;letter-spacing:.14em;font-size:.68rem;font-weight:800;margin-bottom:.35rem; }
      .pp-winner { border:1px solid rgba(184,243,107,.34); border-radius:22px; padding:23px 25px; background:radial-gradient(circle at 90% 0%,rgba(184,243,107,.13),transparent 28%),linear-gradient(135deg,#172722,#111d22 66%,#19251f); box-shadow:0 18px 50px rgba(0,0,0,.18); }
      .pp-winner-top {display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;}
      .pp-winner-label {color:var(--pp-mint);font-size:.72rem;font-weight:850;letter-spacing:.16em;text-transform:uppercase;}
      .pp-winner-id {font-size:1.12rem;font-weight:800;color:#f2f8f4;margin-top:6px;}
      .pp-status {display:inline-block;border:1px solid rgba(241,198,109,.4);background:rgba(241,198,109,.1);color:#f1d28e;padding:6px 10px;border-radius:999px;font-size:.72rem;font-weight:800;letter-spacing:.03em;}
      .pp-status.good {border-color:rgba(184,243,107,.35);background:rgba(184,243,107,.09);color:var(--pp-mint);}
      .pp-winner-grid {display:grid;grid-template-columns:minmax(0,1fr) 155px;gap:24px;align-items:center;margin-top:20px;}
      .pp-score-ring {width:142px;height:142px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:conic-gradient(#b8f36b var(--score),rgba(255,255,255,.1) 0);position:relative;margin:auto;}
      .pp-score-ring:before {content:'';position:absolute;inset:9px;border-radius:50%;background:#14211d;border:1px solid rgba(255,255,255,.08);}
      .pp-score-inner {position:relative;text-align:center;z-index:1;}
      .pp-score-value {font-size:2.15rem;line-height:1;font-weight:850;color:#f2f8f4;}
      .pp-score-unit {font-size:.68rem;color:#a7bbb1;margin-top:5px;letter-spacing:.12em;text-transform:uppercase;}
      .pp-meta-row {display:flex;gap:9px;flex-wrap:wrap;margin-top:13px;}
      .pp-meta-chip {border:1px solid var(--pp-line);background:rgba(255,255,255,.035);padding:7px 10px;border-radius:10px;color:#dce9e2;font-size:.82rem;}
      .pp-quote-grid {display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:15px;}
      .pp-quote {background:rgba(3,10,10,.25);border:1px solid rgba(255,255,255,.08);border-radius:13px;padding:13px 15px;min-height:100px;}
      .pp-quote-label {color:#a9c4b4;text-transform:uppercase;letter-spacing:.12em;font-size:.65rem;font-weight:800;margin-bottom:7px;}
      .pp-quote-text {color:#f1f6f2;font-size:.96rem;line-height:1.5;font-weight:600;}
      .pp-quote-desc {color:#9eb2a8;font-size:.79rem;margin-top:6px;line-height:1.4;}
      .pp-mini-card {border:1px solid var(--pp-line);background:linear-gradient(145deg,rgba(23,36,42,.95),rgba(15,24,29,.95));border-radius:16px;padding:15px;height:100%;}
      .pp-mini-card .pp-rank {color:var(--pp-mint);font-size:.69rem;font-weight:850;letter-spacing:.14em;text-transform:uppercase;}
      .pp-mini-card .pp-mini-score {font-size:1.65rem;font-weight:850;letter-spacing:-.04em;margin:.35rem 0;color:#f4f8f5;}
      .pp-flag {display:inline-block;margin:4px 4px 0 0;padding:5px 8px;border-radius:8px;font-size:.72rem;color:#f2d48d;background:rgba(241,198,109,.1);border:1px solid rgba(241,198,109,.18);}
      .pp-timeline-item {border-left:2px solid rgba(184,243,107,.38);padding:0 0 14px 14px;margin-left:6px;position:relative;}
      .pp-timeline-item:before {content:'';position:absolute;left:-5px;top:4px;width:8px;height:8px;border-radius:50%;background:#b8f36b;}
      .pp-timeline-time {font-size:.7rem;color:var(--pp-mint);font-weight:800;letter-spacing:.08em;}
      .pp-timeline-text {font-size:.86rem;color:#dce8e1;margin-top:3px;}
      .pp-step {display:flex;gap:12px;align-items:center;padding:11px 13px;border:1px solid var(--pp-line);background:rgba(255,255,255,.025);border-radius:13px;margin:7px 0;}
      .pp-step-n {width:27px;height:27px;border-radius:9px;display:flex;align-items:center;justify-content:center;background:rgba(184,243,107,.12);color:var(--pp-mint);font-weight:850;font-size:.78rem;}
      .pp-step-title {font-weight:750;color:#e9f2ed;font-size:.9rem;}
      .pp-step-sub {color:#8fa49a;font-size:.75rem;margin-top:2px;}
      .pp-footer-note {border-top:1px solid var(--pp-line);padding-top:12px;color:#8fa49a;font-size:.76rem;}
      @media(max-width:800px){.pp-winner-grid{grid-template-columns:1fr}.pp-quote-grid{grid-template-columns:1fr}.pp-score-ring{width:120px;height:120px}}
    </style>
    """,
    unsafe_allow_html=True,
)


def discover_rankings() -> list[Path]:
    files = list(RUNS_DIR.glob("**/*_ranking_final.json"))
    sample = ROOT / "sample_data" / "ep01_ranking_final.json"
    if sample.exists():
        files.append(sample)
    return sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)


def load_candidate_payload(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    try:
        return load_json(path)
    except ValueError:
        return None


def candidate_payload_for_ranking(path: Path, ranking: dict[str, Any]) -> dict[str, Any] | None:
    """Find candidate JSON beside a run's ranking artifact when available."""
    episode = str(ranking.get("episode_id", "episode")).lower()
    possible = [
        path.parent.parent / "candidates" / f"{episode}_candidates.json",
        ROOT / "sample_data" / f"{episode}_candidates.json",
    ]
    for candidate_path in possible:
        payload = load_candidate_payload(candidate_path)
        if payload is not None:
            return payload
    return None


def _safe(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _score(candidate: dict[str, Any]) -> float:
    try:
        return float(candidate.get("ranking_score", candidate.get("adjusted_score", 0)) or 0)
    except (TypeError, ValueError):
        return 0.0


def render_winner(ranking: dict[str, Any], candidate_index: dict[str, dict[str, Any]], *, key_prefix: str) -> None:
    winner = select_best_promo(ranking)
    if winner is None:
        st.warning("No non-blocked promo candidate is available to recommend. Review the full ranking and quality-gate exclusions.")
        return

    candidate_id = str(winner.get("candidate_id", ""))
    source = candidate_index.get(candidate_id, {})
    hook = source.get("hook") or {}
    ending = source.get("ending") or {}
    segments = source.get("segments") or [{}]
    segment = segments[0] if segments else {}
    start = winner.get("start_seconds", segment.get("start_seconds"))
    end = winner.get("end_seconds", segment.get("end_seconds"))
    duration = winner.get("duration_seconds", source.get("duration_seconds"))
    score = _score(winner)
    rank = winner.get("overall_rank", 1)
    shortlist_member = candidate_id in {str(item.get("candidate_id")) for item in ranking.get("shortlist", []) if isinstance(item, dict)}
    is_recommended = shortlist_member and winner.get("shortlist_selected", True) is not False and winner.get("shortlist_eligible", True) is not False
    flags = winner.get("quality_flags") or []
    status = "PROVISIONAL · EDITORIAL REVIEW" if flags or winner.get("editorial_review_required") else "TOP RECOMMENDATION"
    status_class = "" if flags or winner.get("editorial_review_required") else "good"
    if not is_recommended:
        status = "TOP SCORING · BELOW SHORTLIST POLICY"
        status_class = ""

    st.markdown(
        f"""
        <div class="pp-winner">
          <div class="pp-winner-top">
            <div><div class="pp-winner-label">✦ Best ranked promo · Episode {_safe(ranking.get('episode_id',''))}</div>
              <div class="pp-winner-id">{_safe(candidate_id)}</div></div>
            <span class="pp-status {status_class}">{_safe(status)}</span>
          </div>
          <div class="pp-winner-grid">
            <div>
              <div style="font-size:.76rem;color:#a9bdb3;letter-spacing:.12em;text-transform:uppercase;font-weight:800">Recommended clip window</div>
              <div style="font-size:1.55rem;font-weight:850;color:#f4f8f5;margin-top:5px">{_safe(format_timecode(start))} <span style="color:#7f988b">→</span> {_safe(format_timecode(end))}</div>
              <div class="pp-meta-row">
                <span class="pp-meta-chip">🏆 Rank #{_safe(rank)}</span>
                <span class="pp-meta-chip">⏱ {_safe(f'{float(duration):.1f}' if duration is not None else '—')} sec</span>
                <span class="pp-meta-chip">🎞 Continuous single-episode clip</span>
              </div>
            </div>
            <div class="pp-score-ring" style="--score:{max(0,min(100,score))}%">
              <div class="pp-score-inner"><div class="pp-score-value">{score:.0f}</div><div class="pp-score-unit">Score / 100</div></div>
            </div>
          </div>
          <div class="pp-quote-grid">
            <div class="pp-quote"><div class="pp-quote-label">Opening hook</div>
              <div class="pp-quote-text">{_safe(hook.get('text') or 'Hook text unavailable in candidate artifact.')}</div>
              <div class="pp-quote-desc">{_safe(hook.get('description') or '')}</div></div>
            <div class="pp-quote"><div class="pp-quote-label">Promo ending</div>
              <div class="pp-quote-text">{_safe(ending.get('text') or 'Ending text unavailable in candidate artifact.')}</div>
              <div class="pp-quote-desc">{_safe(ending.get('description') or ending.get('ending_basis') or '')}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if flags:
        st.markdown("<div style='margin-top:12px;color:#b7c8bf;font-size:.8rem;font-weight:750'>EDITORIAL CHECKS</div>", unsafe_allow_html=True)
        st.markdown("".join(f"<span class='pp-flag'>⚑ {_safe(str(flag).replace('_',' ').title())}</span>" for flag in flags), unsafe_allow_html=True)
        st.caption("This is the highest-ranked eligible recommendation, but automated flags mean a human editor should review it before publishing.")
    elif not is_recommended:
        st.caption("No candidate cleared the configured shortlist threshold. This is the highest-scoring non-blocked candidate, shown for review rather than as an approved recommendation.")

    # A one-click handoff: the selected ranked record and source-backed hook/ending details.
    winner_export = {"episode_id": ranking.get("episode_id"), "winner": winner, "source_candidate": source}
    col_a, col_b, col_c = st.columns([1, 1, 1.35])
    with col_a:
        st.download_button(
            "⬇ Download best promo JSON",
            data=json.dumps(winner_export, ensure_ascii=False, indent=2),
            file_name=f"{str(ranking.get('episode_id','episode')).lower()}_best_promo.json",
            mime="application/json",
            key=f"{key_prefix}_winner_json_{candidate_id}",
            use_container_width=True,
        )
    with col_b:
        st.download_button(
            "⬇ Download clip brief",
            data=(
                f"PromoPilot — Best ranked promo\nEpisode: {ranking.get('episode_id')}\nCandidate: {candidate_id}\n"
                f"Rank: {rank}\nScore: {score:.1f}/100\nStart: {format_timecode(start)}\nEnd: {format_timecode(end)}\n"
                f"Duration: {duration} seconds\n\nHook:\n{hook.get('text','')}\n{hook.get('description','')}\n\n"
                f"Ending:\n{ending.get('text','')}\n{ending.get('description','')}\n\n"
                f"Editorial flags: {', '.join(flags) if flags else 'None'}\n\n"
                "Note: PromoPilot recommends a timestamped clip; it does not render an MP4 video.\n"
            ),
            file_name=f"{str(ranking.get('episode_id','episode')).lower()}_best_promo_brief.txt",
            mime="text/plain",
            key=f"{key_prefix}_winner_brief_{candidate_id}",
            use_container_width=True,
        )
    with col_c:
        st.caption("PromoPilot outputs an editor-ready clip recommendation and cut specification. It does not render or export the finished video.")

    timeline = source.get("timeline_events") or []
    if timeline:
        with st.expander(f"Preview source timeline · {len(timeline)} events", expanded=False):
            for event in timeline:
                when = event.get("start_time") or format_timecode(event.get("start_seconds"))
                speaker = event.get("speaker") or event.get("label") or event.get("event_type") or "Source event"
                content = event.get("content") or ""
                st.markdown(
                    f"<div class='pp-timeline-item'><div class='pp-timeline-time'>{_safe(when)} · {_safe(speaker)}</div>"
                    f"<div class='pp-timeline-text'>{_safe(content)}</div></div>",
                    unsafe_allow_html=True,
                )


def display_ranking(data: dict[str, Any], *, candidate_payload: dict[str, Any] | None = None, key_prefix: str = "ranking") -> None:
    errors = validate_ranking(data)
    if errors:
        st.error("This file is not a valid PromoPilot final ranking.")
        for error in errors:
            st.write(f"- {error}")
        return

    full = data.get("full_ranking", [])
    shortlist = data.get("shortlist", [])
    summary = data.get("summary", {})
    candidate_index = index_candidate_details(candidate_payload)

    st.markdown(f"<div class='pp-section-kicker'>Your episode result</div><h2 style='margin-top:0'>Episode {_safe(data.get('episode_id',''))} · Promo shortlist</h2>", unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Candidates evaluated", len(full))
    cols[1].metric("Recommended promos", len(shortlist))
    cols[2].metric("Quality-gate blocks", summary.get("blocked_mid_thought_count", 0))
    cols[3].metric("Overlap clusters", summary.get("overlap_cluster_count", 0))

    st.markdown("<div class='pp-section-kicker' style='margin-top:18px'>The output</div>", unsafe_allow_html=True)
    render_winner(data, candidate_index, key_prefix=key_prefix)

    if len(shortlist) > 1:
        st.markdown("### Other shortlisted promos")
        runner_ups = [item for item in shortlist if item.get("candidate_id") != (select_best_promo(data) or {}).get("candidate_id")]
        cards = st.columns(min(3, len(runner_ups)))
        for index, item in enumerate(runner_ups):
            with cards[index % len(cards)]:
                flags = item.get("quality_flags") or []
                flag_html = "".join(f"<span class='pp-flag'>{_safe(str(flag).replace('_',' ').title())}</span>" for flag in flags)
                st.markdown(
                    f"<div class='pp-mini-card'><div class='pp-rank'>Rank #{_safe(item.get('overall_rank'))}</div>"
                    f"<div class='pp-mini-score'>{_score(item):.0f}<span style='font-size:.8rem;color:#9aaca6'> / 100</span></div>"
                    f"<div style='font-size:.76rem;color:#a5b8af'>{_safe(item.get('candidate_id'))}</div>"
                    f"<div style='margin-top:9px;color:#b7c8bf;font-size:.8rem'>⏱ {_safe(item.get('duration_seconds','—'))} sec</div>"
                    f"{flag_html}</div>", unsafe_allow_html=True,
                )

    left, right = st.columns([1.3, .9])
    with left:
        st.markdown("### Complete ranking")
        rows = candidate_table_rows(full)
        st.dataframe(rows, use_container_width=True, hide_index=True)
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0].keys()) if rows else ["Rank", "Candidate", "Score"])
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("⬇ Export full ranking (CSV)", data=csv_buffer.getvalue(), file_name=f"{str(data.get('episode_id','episode')).lower()}_ranking.csv", mime="text/csv", key=f"{key_prefix}_csv")
    with right:
        st.markdown("### Editorial review queue")
        counts = flag_counts(full)
        if counts:
            st.dataframe([{"Review flag": k.replace("_", " ").title(), "Candidates": v} for k, v in counts.items()], use_container_width=True, hide_index=True)
        else:
            st.success("No automated review flags.")
        excluded = data.get("editorial_review_excluded", [])
        if excluded:
            with st.expander(f"Below-threshold candidates ({len(excluded)})"):
                st.dataframe(candidate_table_rows(excluded), use_container_width=True, hide_index=True)

    with st.expander("Inspect complete ranking data"):
        st.json(data)
    st.download_button(
        "⬇ Download final ranking JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name=f"{str(data.get('episode_id','episode')).lower()}_ranking_final.json",
        mime="application/json",
        key=f"{key_prefix}_ranking_json",
    )


def bundle_run(results: list[dict[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for result in results:
            episode_id = result["episode_id"]
            for label, path in result["artifacts"].items():
                path = Path(path)
                if path.exists() and path.is_file():
                    archive.write(path, arcname=f"{episode_id}/{label}_{path.name}")
    buffer.seek(0)
    return buffer.getvalue()


def render_run_results(results: list[dict[str, Any]], *, key_prefix: str) -> None:
    st.markdown("<div class='pp-section-kicker'>Processing complete</div><h2 style='margin-top:0'>Your promo recommendations</h2>", unsafe_allow_html=True)
    st.success(f"Completed processing for {len(results)} episode(s). The best-ranked promo is highlighted below.")
    for result in results:
        ranking = result["final_ranking"]
        candidate_path = Path(result.get("artifacts", {}).get("candidates", ""))
        payload = load_candidate_payload(candidate_path)
        display_ranking(ranking, candidate_payload=payload, key_prefix=f"{key_prefix}_{result['episode_id']}")
    st.download_button(
        "⬇ Download all episode reports (ZIP)",
        data=bundle_run(results),
        file_name=f"promopilot_{results[0]['run_id']}_reports.zip",
        mime="application/zip",
        key=f"{key_prefix}_zip",
    )


if "active_run_id" not in st.session_state:
    st.session_state.active_run_id = None
if "active_upload_signature" not in st.session_state:
    st.session_state.active_upload_signature = None
if "latest_results" not in st.session_state:
    st.session_state.latest_results = None

st.markdown(
    "<div class='pp-brand'><div class='pp-brand-mark'>🎬</div><div><div class='pp-brand-title'>PromoPilot</div>"
    "<div class='pp-brand-sub'>Microdrama Promo Studio · From screenplay to scroll-stopping story moments</div></div></div>",
    unsafe_allow_html=True,
)
st.markdown("""
<div class="pp-hero">
  <div class="pp-eyebrow">AI-assisted promo discovery</div>
  <h2>Find the moment. Build the hook. Rank the promo.</h2>
  <p>Upload your episode screenplay and let PromoPilot analyze the story, generate promo cuts, score every candidate and surface the strongest recommendation.</p>
</div>
""", unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.markdown("### Workspace status")
    st.success("All 7 processing stages connected")
    st.caption(f"AI provider · Groq / {get_model_name()}")
    if get_groq_api_key():
        st.markdown("<span class='pp-status good'>API KEY CONNECTED</span>", unsafe_allow_html=True)
    else:
        st.error("Groq API key not configured")
    st.divider()
    st.markdown("### Pipeline")
    for number, (title, subtitle) in enumerate([
        ("Parse screenplay", "Scenes · dialogue · timestamps"),
        ("Understand story", "Characters · conflict · emotion"),
        ("Generate promo cuts", "Hooks · endings · continuity"),
        ("Evaluate & rank", "Quality gate · diversity · score"),
    ], 1):
        st.markdown(f"<div class='pp-step'><div class='pp-step-n'>{number}</div><div><div class='pp-step-title'>{_safe(title)}</div><div class='pp-step-sub'>{_safe(subtitle)}</div></div></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### Cost & privacy")
    st.caption("No paid APIs are called by the app. Groq is used for Story Intelligence and Promo Evaluation. Provider free-tier quotas can change.")
    st.caption("Uploaded scripts and derived artifacts are stored in this workspace. Only submit content you are authorized to process.")


tab_generate, tab_dashboard, tab_outputs, tab_about = st.tabs(
    ["✨ Create promos", "🏆 Best promo", "📦 Run library", "ⓘ How it works"]
)

with tab_generate:
    st.markdown("<div class='pp-section-kicker'>Start a new run</div><h2 style='margin-top:0'>Upload your screenplay</h2>", unsafe_allow_html=True)
    st.write("Choose one or more timestamped TXT episodes. PromoPilot will handle every processing stage and return the best-ranked promo for each episode.")
    uploads = st.file_uploader(
        "Drop screenplay TXT files here",
        type=["txt"],
        accept_multiple_files=True,
        help="Use the timestamped scene/event format supported by the PromoPilot parser.",
        key="screenplay_uploads_full_pipeline",
    )
    show_id = st.text_input("Show / series ID", value="kuku_show_001", help="Keep this ID consistent for episodes belonging to the same show.")
    if uploads:
        st.markdown("<div class='pp-section-kicker'>Ready to process</div>", unsafe_allow_html=True)
        upload_cols = st.columns(min(3, len(uploads)))
        for index, item in enumerate(uploads):
            with upload_cols[index % len(upload_cols)]:
                st.markdown(f"<div class='pp-mini-card'><div class='pp-rank'>EPISODE FILE</div><div style='font-weight:800;margin-top:7px'>{_safe(item.name)}</div><div class='pp-muted' style='font-size:.76rem;margin-top:5px'>{item.size/1024:.1f} KB</div></div>", unsafe_allow_html=True)

    with st.expander("⚙ Advanced processing settings", expanded=False):
        st.caption("Recommended settings tuned for Groq free-tier rate limits and the previously validated Episode 1 workflow.")
        max_candidates = st.slider("Maximum promo candidates per episode", 3, 20, 10, 1)
        api_delay = st.slider("Delay between Groq calls (seconds)", 10, 60, 18, 2)
        min_score = st.slider("Minimum score for the recommended shortlist", 0, 100, 40, 5)
        st.caption("Story Intelligence: 6-event chunks, 1-event overlap, 1,400-token completion budget. Promo Evaluator: 900-token completion budget.")
    groq_ready = bool(get_groq_api_key())
    can_run = bool(uploads) and bool(show_id.strip()) and groq_ready
    st.markdown("<div class='pp-footer-note'>The first run can take time depending on screenplay length and Groq free-tier capacity. Successful chunks and evaluations are checkpointed for retries.</div>", unsafe_allow_html=True)
    button_label = "Resume / retry processing" if st.session_state.active_run_id else "🚀 Generate my promos"
    if st.button(button_label, type="primary", disabled=not can_run, use_container_width=True):
        signature = hashlib.sha256("|".join(f"{item.name}:{hashlib.sha256(item.getvalue()).hexdigest()}" for item in uploads).encode("utf-8")).hexdigest()
        if not st.session_state.active_run_id or st.session_state.active_upload_signature != signature:
            st.session_state.active_run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_") + signature[:8]
            st.session_state.active_upload_signature = signature
        run_id = st.session_state.active_run_id
        config = PipelineConfig(
            story_max_completion_tokens=1400,
            story_chunk_size=6,
            story_chunk_overlap=1,
            story_delay_between_chunks=float(api_delay),
            max_candidates=max_candidates,
            evaluator_max_completion_tokens=900,
            evaluator_delay_between_calls=float(api_delay),
            min_shortlist_score=float(min_score),
            top_n=5,
            overlap_threshold=0.80,
        )
        log_lines: list[str] = []
        total_steps = 1 + (7 * len(uploads))
        completed_steps = [0]
        with st.status("Warming up PromoPilot…", expanded=True) as status:
            progress = st.progress(0.0, text="Preparing your screenplay")
            log_panel = st.empty()
            def on_event(kind: str, stage: str, message: str) -> None:
                if kind == "start":
                    status.update(label=stage, state="running", expanded=True)
                elif kind == "success":
                    completed_steps[0] += 1
                    progress.progress(min(completed_steps[0] / total_steps, 1.0), text=f"Completed {completed_steps[0]}/{total_steps} processing stages")
                    status.update(label=f"Complete · {stage}", state="running", expanded=True)
                if message:
                    log_lines.append(f"[{stage}] {message}")
                    log_panel.code("\n".join(log_lines[-35:]), language="text")
            try:
                results = run_full_pipeline(
                    [(file.name, file.getvalue()) for file in uploads], root=ROOT, show_id=show_id,
                    run_id=run_id, config=config, on_event=on_event,
                )
                progress.progress(1.0, text="All stages completed")
                status.update(label="PromoPilot finished successfully", state="complete", expanded=False)
                st.session_state.latest_results = results
                st.session_state.active_run_id = None
                st.session_state.active_upload_signature = None
            except (PipelineStageError, ValueError, OSError, json.JSONDecodeError) as exc:
                status.update(label="Processing stopped · see error details", state="error", expanded=True)
                st.error(str(exc))
                st.warning("The run workspace and valid checkpoints were retained. Keep the same files and retry to resume where possible.")
            except Exception as exc:
                status.update(label="Unexpected pipeline error", state="error", expanded=True)
                st.exception(exc)
                st.warning("The run workspace was retained. Review the error before retrying.")

    if not groq_ready:
        st.warning("Configure GROQ_API_KEY in your local .env file or Streamlit Cloud Secrets to enable the AI stages.")
    if st.session_state.latest_results:
        st.divider()
        render_run_results(st.session_state.latest_results, key_prefix="latest_run")

with tab_dashboard:
    ranking_paths = discover_rankings()
    st.markdown("<div class='pp-section-kicker'>Winner spotlight</div><h2 style='margin-top:0'>Your best promo</h2>", unsafe_allow_html=True)
    if not ranking_paths:
        st.info("No completed promo runs yet. Upload a screenplay in Create promos to generate your first recommendation.")
    else:
        labels = [str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path) for path in ranking_paths]
        selected = st.selectbox("Select an episode result", labels)
        selected_path = ranking_paths[labels.index(selected)]
        try:
            ranking_data = load_json(selected_path)
            candidate_payload = candidate_payload_for_ranking(selected_path, ranking_data)
            display_ranking(ranking_data, candidate_payload=candidate_payload, key_prefix="dashboard")
            if candidate_payload is None:
                st.info("Ranking loaded. The source candidate artifact is not beside this ranking file, so hook/ending transcript details may be unavailable here. The complete best-promo JSON remains downloadable.")
        except ValueError as exc:
            st.error(str(exc))

with tab_outputs:
    st.markdown("<div class='pp-section-kicker'>Your workspace</div><h2 style='margin-top:0'>Past runs & downloads</h2>", unsafe_allow_html=True)
    if st.session_state.latest_results:
        render_run_results(st.session_state.latest_results, key_prefix="outputs_run")
    run_files = sorted(RUNS_DIR.glob("**/*_ranking_final.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if run_files:
        st.markdown("### Previously generated rankings")
        for path in run_files[:30]:
            try:
                data = load_json(path)
                episode_id = data.get("episode_id", path.stem)
                st.download_button(f"Download {episode_id} final ranking", data=path.read_bytes(), file_name=path.name, mime="application/json", key=f"old_{hashlib.md5(str(path).encode()).hexdigest()}")
            except ValueError as exc:
                st.warning(f"Could not read {path.name}: {exc}")
    if not st.session_state.latest_results and not run_files:
        st.info("Your generated reports will appear here after your first run.")

with tab_about:
    st.markdown("<div class='pp-section-kicker'>Behind the recommendations</div><h2 style='margin-top:0'>How PromoPilot works</h2>", unsafe_allow_html=True)
    stages = [
        ("01", "Screenplay parser", "Extracts scenes, speakers, dialogue and timestamped source events."),
        ("02", "Story intelligence", "Uses Groq to identify characters, relationships, conflict, emotion and dramatic moments."),
        ("03", "Story normalizer", "Resolves speaker aliases and maintains a show-scoped character registry."),
        ("04", "Candidate generator", "Builds continuous promo windows from source-grounded hooks and endings."),
        ("05", "Promo evaluator", "Scores hook, cliffhanger, clarity, emotion, pacing and clip integrity."),
        ("06", "Evaluation quality gate", "Checks timing, incomplete dialogue and post-cut context deterministically."),
        ("07", "Ranking engine", "Ranks candidates, manages overlap, applies quality blocks and surfaces the highest score."),
    ]
    for number, title, description in stages:
        st.markdown(f"<div class='pp-step'><div class='pp-step-n'>{number}</div><div><div class='pp-step-title'>{_safe(title)}</div><div class='pp-step-sub'>{_safe(description)}</div></div></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### Cost, privacy & output")
    st.write("PromoPilot uses local Python processing and Groq for the two AI stages. It does not call paid APIs. Free-tier quotas, model availability and hosting limits can change.")
    st.warning("Relevant screenplay text and promo candidate context are sent to Groq. Upload only scripts you are authorized to process. The app creates timestamped promo cut recommendations; it does not render MP4 files.")
    st.code("pip install -r requirements.txt\n# Add GROQ_API_KEY=... to .env\nstreamlit run app.py", language="bash")
