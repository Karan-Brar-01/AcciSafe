"""
Impact-Net — Emergency Operations Center dashboard (Streamlit).

Command-and-control UI for edge crash triage. Vision logic lives in pipeline.py;
this module is presentation + orchestration only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ultralytics import YOLO

from pipeline import (
    load_yolo_model,
    process_video,
    severity_label,
)

# ---------------------------------------------------------------------------
# Page / theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Impact-Net | Crash Triage AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "_uploads"

MODEL_OPTIONS: dict[str, str] = {
    "YOLOv8n-Traffic": "yolov8n.pt",
    "YOLOv8s-Traffic": "yolov8s.pt",
    "YOLOv8m-Traffic": "yolov8m.pt",
}

# Sensitivity → (YOLO conf floor, sudden-decel Δv km/h threshold)
SENSITIVITY_PRESETS: dict[str, tuple[float, float]] = {
    "Low": (0.50, 18.0),
    "Medium": (0.35, 12.0),
    "High": (0.22, 9.0),
}

EOC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --eoc-bg: #0b0f14;
  --eoc-panel: #121820;
  --eoc-panel-2: #18212c;
  --eoc-border: #243041;
  --eoc-text: #e8eef6;
  --eoc-muted: #8b9bb0;
  --eoc-cyan: #2ee6d6;
  --eoc-amber: #f5a524;
  --eoc-red: #ff4d4f;
  --eoc-green: #3ddc97;
  --eoc-glow-cyan: 0 0 18px rgba(46, 230, 214, 0.35);
  --eoc-glow-red: 0 0 22px rgba(255, 77, 79, 0.45);
  --eoc-glow-amber: 0 0 18px rgba(245, 165, 36, 0.4);
  --eoc-glow-green: 0 0 18px rgba(61, 220, 151, 0.4);
}

html, body, [class*="css"] {
  font-family: "IBM Plex Sans", sans-serif;
}

.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(46, 230, 214, 0.07), transparent 55%),
    radial-gradient(900px 500px at 100% 0%, rgba(255, 77, 79, 0.05), transparent 50%),
    linear-gradient(180deg, #0b0f14 0%, #0e141c 100%);
  color: var(--eoc-text);
}

/* Hide Streamlit chrome clutter */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

.block-container {
  padding-top: 1.25rem !important;
  padding-bottom: 2rem !important;
  max-width: 1440px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d1219 0%, #101722 100%);
  border-right: 1px solid var(--eoc-border);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label {
  color: var(--eoc-text) !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] [data-baseweb="select"] {
  background-color: var(--eoc-panel-2);
}

/* Hero header */
.eoc-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.25rem;
  padding: 1rem 1.25rem;
  border: 1px solid var(--eoc-border);
  border-radius: 10px;
  background: linear-gradient(135deg, rgba(18, 24, 32, 0.95), rgba(24, 33, 44, 0.85));
  box-shadow: inset 0 1px 0 rgba(46, 230, 214, 0.08);
}
.eoc-brand {
  font-family: "IBM Plex Mono", monospace;
  font-size: 1.65rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: var(--eoc-text);
  line-height: 1.1;
}
.eoc-brand span {
  color: var(--eoc-cyan);
  text-shadow: var(--eoc-glow-cyan);
}
.eoc-sub {
  margin-top: 0.35rem;
  color: var(--eoc-muted);
  font-size: 0.92rem;
  letter-spacing: 0.02em;
}
.eoc-status-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.45rem 0.85rem;
  border-radius: 999px;
  border: 1px solid var(--eoc-border);
  background: rgba(14, 20, 28, 0.9);
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--eoc-muted);
  white-space:nowrap;
}
.eoc-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--eoc-cyan);
  box-shadow: var(--eoc-glow-cyan);
  animation: pulse-dot 1.8s ease-in-out infinite;
}
.eoc-dot.idle { background: #5a6a7d; box-shadow: none; animation: none; }
.eoc-dot.busy { background: var(--eoc-amber); box-shadow: var(--eoc-glow-amber); }
.eoc-dot.ok { background: var(--eoc-green); box-shadow: var(--eoc-glow-green); }
.eoc-dot.crit { background: var(--eoc-red); box-shadow: var(--eoc-glow-red); }

@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.55; transform: scale(0.85); }
}

/* Panels */
.eoc-panel {
  border: 1px solid var(--eoc-border);
  border-radius: 10px;
  background: var(--eoc-panel);
  padding: 1rem 1.1rem;
  margin-bottom: 0.85rem;
}
.eoc-panel-title {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--eoc-muted);
  margin-bottom: 0.75rem;
}

/* Severity card */
.severity-card {
  border-radius: 12px;
  padding: 1.15rem 1.25rem;
  border: 1px solid transparent;
  margin-bottom: 0.85rem;
  position: relative;
  overflow: hidden;
}
.severity-card::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(120deg, rgba(255,255,255,0.04), transparent 45%);
  pointer-events: none;
}
.severity-card.safe {
  background: linear-gradient(135deg, rgba(61, 220, 151, 0.12), rgba(18, 24, 32, 0.95));
  border-color: rgba(61, 220, 151, 0.45);
  box-shadow: var(--eoc-glow-green);
}
.severity-card.moderate {
  background: linear-gradient(135deg, rgba(245, 165, 36, 0.14), rgba(18, 24, 32, 0.95));
  border-color: rgba(245, 165, 36, 0.5);
  box-shadow: var(--eoc-glow-amber);
}
.severity-card.critical {
  background: linear-gradient(135deg, rgba(255, 77, 79, 0.16), rgba(18, 24, 32, 0.95));
  border-color: rgba(255, 77, 79, 0.55);
  box-shadow: var(--eoc-glow-red);
}
.severity-kicker {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.7rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  opacity: 0.85;
}
.severity-title {
  font-size: 1.45rem;
  font-weight: 700;
  margin: 0.25rem 0 0.35rem 0;
  letter-spacing: 0.01em;
}
.severity-detail {
  color: var(--eoc-muted);
  font-size: 0.92rem;
  line-height: 1.45;
}

/* Metrics neon */
div[data-testid="stMetric"] {
  background: var(--eoc-panel-2);
  border: 1px solid var(--eoc-border);
  border-radius: 10px;
  padding: 0.85rem 1rem;
}
div[data-testid="stMetric"] label {
  color: var(--eoc-muted) !important;
  font-family: "IBM Plex Mono", monospace !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-size: 0.7rem !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: var(--eoc-cyan) !important;
  text-shadow: var(--eoc-glow-cyan);
  font-family: "IBM Plex Mono", monospace;
  font-weight: 600;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.35rem;
  background: transparent;
  border-bottom: 1px solid var(--eoc-border);
}
.stTabs [data-baseweb="tab"] {
  background: var(--eoc-panel);
  border-radius: 8px 8px 0 0;
  color: var(--eoc-muted);
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.78rem;
  letter-spacing: 0.06em;
}
.stTabs [aria-selected="true"] {
  color: var(--eoc-cyan) !important;
  border-bottom: 2px solid var(--eoc-cyan) !important;
}

/* Primary button */
.stButton > button[kind="primary"],
.stButton > button {
  background: linear-gradient(90deg, #1a8f86, #2ee6d6) !important;
  color: #041016 !important;
  border: none !important;
  font-weight: 700 !important;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-family: "IBM Plex Mono", monospace !important;
  box-shadow: var(--eoc-glow-cyan);
}
.stButton > button:hover {
  filter: brightness(1.08);
}

/* Empty state */
.eoc-empty {
  border: 1px dashed var(--eoc-border);
  border-radius: 10px;
  padding: 2.5rem 1.5rem;
  text-align: center;
  color: var(--eoc-muted);
  background: rgba(18, 24, 32, 0.55);
}
.eoc-empty strong {
  display: block;
  color: var(--eoc-text);
  font-size: 1.05rem;
  margin-bottom: 0.4rem;
}

video {
  border-radius: 8px;
  border: 1px solid var(--eoc-border);
  width: 100%;
  background: #000;
}
</style>
"""


# ---------------------------------------------------------------------------
# Cached resources / helpers
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_cached_model(weights_path: str) -> YOLO:
    """Load YOLO once per weights path for the lifetime of the server process."""
    return load_yolo_model(weights_path)


def list_sample_videos() -> list[Path]:
    if not SAMPLES_DIR.is_dir():
        return []
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return sorted(
        [p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in exts and p.is_file()],
        key=lambda p: (0 if p.name.startswith("safe_") else 1, p.name.lower()),
    )


def sample_option_label(path: Path) -> str:
    """Sidebar label with SAFE vs CRASH tag for demo variety."""
    if path.name.startswith("safe_"):
        return f"[SAFE] {path.name}"
    return f"[CRASH] {path.name}"


def severity_band(level: int) -> tuple[str, str, str]:
    """Return (css_class, headline, tone_label)."""
    if level <= 2:
        return "safe", "SAFE / MINOR", "Green — monitoring posture"
    if level == 3:
        return "moderate", "MODERATE — LANE IMPACT", "Amber — elevated response"
    return "critical", "CRITICAL DISPATCH REQUIRED", "Red — trauma priority"


def build_telemetry_figure(
    telemetry: pd.DataFrame,
    collision_frame_idx: Optional[int],
) -> go.Figure:
    fig = go.Figure()
    if telemetry is None or telemetry.empty:
        fig.update_layout(
            title="No telemetry available",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18,24,32,0.9)",
            font=dict(color="#8b9bb0", family="IBM Plex Sans"),
            height=320,
        )
        return fig

    x = telemetry["timestamp"] if "timestamp" in telemetry.columns else telemetry["frame_idx"]
    x_title = "Time (s)" if "timestamp" in telemetry.columns else "Frame"

    if "max_acceleration" in telemetry.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=telemetry["max_acceleration"],
                name="Max acceleration (Δv)",
                mode="lines",
                line=dict(color="#2ee6d6", width=2),
                hovertemplate="%{x:.2f}<br>accel=%{y:.2f}<extra></extra>",
            )
        )
    if "anomaly_score" in telemetry.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=telemetry["anomaly_score"],
                name="Anomaly score (KSI)",
                mode="lines",
                line=dict(color="#f5a524", width=2, dash="dot"),
                hovertemplate="%{x:.2f}<br>anomaly=%{y:.2f}<extra></extra>",
            )
        )

    if collision_frame_idx is not None and "frame_idx" in telemetry.columns:
        hit = telemetry.loc[telemetry["frame_idx"] == collision_frame_idx]
        if not hit.empty:
            x_hit = float(hit.iloc[0]["timestamp"] if "timestamp" in hit.columns else collision_frame_idx)
            fig.add_vline(
                x=x_hit,
                line_width=2,
                line_dash="dash",
                line_color="#ff4d4f",
                annotation_text="IMPACT",
                annotation_position="top",
                annotation_font=dict(color="#ff4d4f", size=11, family="IBM Plex Mono"),
            )

    fig.update_layout(
        margin=dict(l=40, r=20, t=36, b=40),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12, 17, 24, 0.92)",
        font=dict(color="#c5d0de", family="IBM Plex Sans", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            title=x_title,
            gridcolor="rgba(36, 48, 65, 0.85)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Signal",
            gridcolor="rgba(36, 48, 65, 0.85)",
            zeroline=False,
        ),
        hovermode="x unified",
    )
    return fig


def resolve_input_video(
    sample_choice: str,
    uploaded_file: Any,
    sample_map: dict[str, Path],
) -> Optional[Path]:
    """Prefer uploaded custom clip; otherwise selected sample."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
        dest = UPLOAD_DIR / f"upload_{uploaded_file.name}"
        dest.write_bytes(uploaded_file.getbuffer())
        return dest
    if sample_choice and sample_choice in sample_map:
        return sample_map[sample_choice]
    return None


def render_header(system_state: str) -> None:
    dot_class = {
        "STANDBY": "idle",
        "ANALYZING": "busy",
        "CLEAR": "ok",
        "ALERT": "crit",
    }.get(system_state, "idle")
    st.markdown(
        f"""
        <div class="eoc-header">
          <div>
            <div class="eoc-brand">IMPACT<span>-NET</span></div>
            <div class="eoc-sub">Edge-AI Traffic Crash Triage &amp; Severity Assessment — Emergency Operations Center</div>
          </div>
          <div class="eoc-status-pill">
            <span class="eoc-dot {dot_class}"></span>
            SYS · {system_state}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def main() -> None:
    st.markdown(EOC_CSS, unsafe_allow_html=True)

    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_source" not in st.session_state:
        st.session_state.last_source = None
    if "system_state" not in st.session_state:
        st.session_state.system_state = "STANDBY"

    # ----- Sidebar -----
    with st.sidebar:
        st.markdown("### Mission Control")
        st.caption("Configure detection posture, then run analysis on a feed.")

        model_label = st.selectbox(
            "Detection model",
            options=list(MODEL_OPTIONS.keys()),
            index=0,
            help="YOLOv8n-Traffic is optimized for CPU edge triage.",
        )
        weights = MODEL_OPTIONS[model_label]

        sensitivity = st.select_slider(
            "Sensitivity threshold",
            options=list(SENSITIVITY_PRESETS.keys()),
            value="Medium",
            help="High = lower detection floor + more aggressive Δv flagging.",
        )
        conf_threshold, sudden_decel = SENSITIVITY_PRESETS[sensitivity]

        st.markdown("---")
        samples = list_sample_videos()
        sample_map = {sample_option_label(p): p for p in samples}
        sample_labels = ["— Select sample —"] + [sample_option_label(p) for p in samples]
        if not samples:
            st.warning("No clips in `./samples/`. Run `python3 download_samples.py`.")
        sample_choice = st.selectbox(
            "Pre-loaded video",
            options=sample_labels,
            index=min(1, len(sample_labels) - 1) if samples else 0,
            help="SAFE clips should triage ~1–2; CRASH clips should triage ~4–5.",
        )

        uploaded = st.file_uploader(
            "Upload custom feed",
            type=["mp4", "avi", "mov"],
            help="Overrides the sample selection when provided.",
        )

        st.markdown("---")
        run_clicked = st.button("▶  Run Analysis", type="primary", use_container_width=True)

        st.caption(
            f"Active posture · conf ≥ {conf_threshold:.2f} · Δv ≥ {sudden_decel:.0f} km/h"
        )

    render_header(st.session_state.system_state)

    # ----- Run pipeline -----
    if run_clicked:
        source = resolve_input_video(
            sample_choice if sample_choice != "— Select sample —" else "",
            uploaded,
            sample_map,
        )
        if source is None:
            st.error("Select a sample clip or upload a video before running analysis.")
        else:
            st.session_state.system_state = "ANALYZING"

            progress = st.progress(0.0, text="Initializing Impact-Net pipeline…")
            status = st.empty()

            def on_progress(frac: float, message: str) -> None:
                progress.progress(min(1.0, max(0.0, frac)), text=message)
                status.markdown(
                    f'<div class="eoc-status-pill"><span class="eoc-dot busy"></span>'
                    f'{message}</div>',
                    unsafe_allow_html=True,
                )

            try:
                with st.spinner("Warming detection model…"):
                    model = get_cached_model(weights)

                result = process_video(
                    str(source),
                    progress_callback=on_progress,
                    model=model,
                    model_path=weights,
                    output_dir=str(OUTPUT_DIR),
                    conf_threshold=conf_threshold,
                    sudden_decel_kmh=sudden_decel,
                )
                st.session_state.last_result = result
                st.session_state.last_source = str(source)
                level = int(result.get("max_severity_level") or 1)
                st.session_state.system_state = "ALERT" if level >= 3 else "CLEAR"
                progress.progress(1.0, text="Analysis complete")
                status.empty()
            except FileNotFoundError as exc:
                st.session_state.system_state = "STANDBY"
                st.error(f"Input video missing: {exc}")
            except Exception as exc:  # noqa: BLE001 — surface to operator console
                st.session_state.system_state = "STANDBY"
                st.error(f"Analysis failed: {exc}")
                st.exception(exc)

    result = st.session_state.last_result
    source_path = st.session_state.last_source

    left, right = st.columns([1.15, 1.0], gap="large")

    # ----- Left: visual feed -----
    with left:
        st.markdown(
            '<div class="eoc-panel"><div class="eoc-panel-title">Visual Feed</div>',
            unsafe_allow_html=True,
        )
        if not result:
            st.markdown(
                """
                <div class="eoc-empty">
                  <strong>No active incident feed</strong>
                  Configure the sidebar and press <em>Run Analysis</em> to triage a clip.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            tab_orig, tab_ai = st.tabs(["Original Feed", "AI Annotated Feed"])
            with tab_orig:
                if source_path and Path(source_path).is_file():
                    st.video(source_path)
                else:
                    st.info("Original source file is no longer available on disk.")
            with tab_ai:
                out_vid = result.get("output_video_path")
                if out_vid and Path(out_vid).is_file():
                    st.video(out_vid)
                else:
                    st.warning("Annotated H.264 output was not produced.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ----- Right: telemetry -----
    with right:
        st.markdown(
            '<div class="eoc-panel-title">Telemetry &amp; Triage</div>',
            unsafe_allow_html=True,
        )

        if not result:
            st.markdown(
                """
                <div class="eoc-empty">
                  <strong>Awaiting triage output</strong>
                  Severity, kinetics, and dispatch payload appear here after a run.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            level = int(result.get("max_severity_level") or 1)
            delta_v = float(result.get("max_delta_v") or 0.0)
            payload = result.get("dispatch_payload") or {}
            confidence = float(payload.get("incident_confidence") or 0.0)
            telemetry: pd.DataFrame = result.get("frame_telemetry")
            if not isinstance(telemetry, pd.DataFrame):
                telemetry = pd.DataFrame()

            # Edge-case messaging
            mean_vehicles = float(telemetry["vehicle_count"].mean()) if not telemetry.empty else 0.0
            if mean_vehicles < 0.05 and level <= 1 and delta_v < 1.0:
                st.info(
                    "No vehicles confidently detected across the feed. "
                    "Severity held at monitoring baseline — treat as a clean / non-incident clip."
                )
            elif level <= 1 and float(telemetry["anomaly_score"].max() if not telemetry.empty else 0) < 0.5:
                st.success(
                    "Feed processed with no actionable collision signature. "
                    "Posture remains routine monitoring."
                )

            css_class, headline, tone = severity_band(level)
            label = payload.get("severity_label") or severity_label(level)
            triage = payload.get("triage_recommendation") or ""
            st.markdown(
                f"""
                <div class="severity-card {css_class}">
                  <div class="severity-kicker">Real-time severity · {tone}</div>
                  <div class="severity-title">{headline}</div>
                  <div class="severity-detail">
                    Level {level} — {label}<br/>{triage}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Kinetic Severity", f"{level} / 5")
            m2.metric("Peak Δv", f"{delta_v:.1f} km/h")
            m3.metric("Detection Confidence", f"{confidence * 100:.0f}%")

            st.markdown(
                '<div class="eoc-panel-title" style="margin-top:0.75rem;">Kinetics over time</div>',
                unsafe_allow_html=True,
            )
            impact_idx = result.get("collision_frame_idx")
            # Only mark impact line when severity suggests a real event
            mark_idx = int(impact_idx) if level >= 2 and impact_idx is not None else (
                int(impact_idx) if (impact_idx is not None and delta_v >= sudden_decel * 0.8) else None
            )
            # Prefer marking whenever we have a meaningful anomaly
            if not telemetry.empty and "anomaly_score" in telemetry.columns:
                if float(telemetry["anomaly_score"].max()) >= 1.0 and impact_idx is not None:
                    mark_idx = int(impact_idx)

            st.plotly_chart(
                build_telemetry_figure(telemetry, mark_idx),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            with st.expander("Emergency Dispatch Payload (911 / EMS)", expanded=level >= 3):
                st.caption("Structured JSON ready for CAD / computer-aided dispatch handoff.")
                st.json(payload)

            if payload.get("collision_frame_idx") is not None:
                st.caption(
                    f"Impact frame `{payload.get('collision_frame_idx')}` · "
                    f"t = {payload.get('timestamp_sec', 0):.2f}s · "
                    f"priority `{payload.get('dispatch_priority', 'n/a')}`"
                )


if __name__ == "__main__":
    main()
