import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

alt.data_transformers.disable_max_rows()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Media Lens – US Political News Coverage",
    page_icon="📰",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Hero */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    /* Section headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 0.3rem;
    }
    .section-desc {
        color: #555;
        margin-bottom: 1rem;
        font-size: 1.05rem;
    }
    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1.2rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        flex: 1;
        background: #f8f9fb;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .metric-card .num {
        font-size: 2rem;
        font-weight: 800;
    }
    .metric-card .label {
        font-size: 0.9rem;
        color: #777;
    }
    /* Insight boxes */
    .insight-box {
        background: #eef3ff;
        border-left: 4px solid #4a7cff;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.98rem;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 1rem 0;
        border-top: 1px solid #eee;
    }
    div[data-testid="stSidebar"] {
        background: #f8f9fb;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Colour palette (consistent across charts)
# ---------------------------------------------------------------------------
OUTLET_COLORS = {
    "NYTimes": "#1f77b4",
    "FoxNews": "#d62728",
    "CNN": "#ff7f0e",
    "WashingtonPost": "#2ca02c",
    "NBCNews": "#9467bd",
    "Politico": "#8c564b",
    "WSJ": "#e377c2",
}

TOPIC_COLORS = {
    "Elections": "#4a7cff",
    "Government": "#ff6b6b",
    "Immigration": "#ffa94d",
    "ForeignPolicy": "#51cf66",
    "Economy": "#845ef7",
    "Political Figures": "#f06595",
}

# ---------------------------------------------------------------------------
# Data loading & cleaning (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    tone_vol = pd.read_csv("gdelt_us_politics_tone_and_topics_long.csv")
    topic_share = pd.read_csv("gdelt_us_politics_topic_share.csv")
    events_df = pd.read_csv("us_politics_event_annotations.csv")

    # 1. Parse dates
    tone_vol["date"] = pd.to_datetime(tone_vol["date"], format="%Y%m%dT%H%M%SZ")
    topic_share["date"] = pd.to_datetime(topic_share["date"], format="%Y%m%dT%H%M%SZ")
    events_df["event_date"] = pd.to_datetime(events_df["event_date"], errors="coerce")

    # 2. Remove 2026 data (only 1 incomplete day)
    tone_vol = tone_vol[tone_vol["year"] != 2026].copy()
    topic_share = topic_share[topic_share["year"] != 2026].copy()
    events_df = events_df[
        (events_df["event_date"].dt.year >= 2017)
        & (events_df["event_date"].dt.year <= 2025)
    ].copy()

    # 3. Null-out missing data: when volume == 0 the outlet had no articles,
    #    so the corresponding tone value is meaningless and must also be null.
    #    Build a set of (date, outlet, topic) keys where volume is zero.
    vol_mask = (tone_vol["metric"] == "volume") & (tone_vol["value"] == 0)
    missing_keys = tone_vol.loc[vol_mask, ["date", "outlet", "topic"]]

    #    Mark volume == 0 rows as NaN
    tone_vol.loc[vol_mask, "value"] = pd.NA

    #    Also mark the matching tone rows as NaN
    tone_idx = tone_vol[tone_vol["metric"] == "tone"].merge(
        missing_keys, on=["date", "outlet", "topic"], how="inner"
    ).index
    # Use merge indicator to find tone rows whose (date, outlet, topic) is in missing_keys
    tone_rows = tone_vol[tone_vol["metric"] == "tone"].copy()
    tone_rows["_drop"] = False
    merged = tone_rows[["date", "outlet", "topic"]].reset_index().merge(
        missing_keys, on=["date", "outlet", "topic"], how="inner"
    )
    tone_vol.loc[merged["index"], "value"] = pd.NA

    #    Drop all NaN value rows
    tone_vol = tone_vol.dropna(subset=["value"]).copy()

    topic_share.loc[topic_share["value"] == 0, "value"] = pd.NA
    topic_share = topic_share.dropna(subset=["value"]).copy()
    # Also drop rows where topic_share is NaN (caused by total_volume == 0)
    topic_share = topic_share.dropna(subset=["topic_share"]).copy()

    # 4. Cap extreme tone outliers at +/- 10 (artifacts from very low article counts)
    tone_vol.loc[tone_vol["metric"] == "tone", "value"] = tone_vol.loc[
        tone_vol["metric"] == "tone", "value"
    ].clip(lower=-10, upper=10)

    # 5. Remove total blackout date (2025-12-06 – GDELT ingestion failure)
    blackout = pd.Timestamp("2025-12-06")
    tone_vol = tone_vol[tone_vol["date"] != blackout]
    topic_share = topic_share[topic_share["date"] != blackout]

    # 6. Flag outlet reliability – mark outlets with >50% zero-days in a month
    #    as unreliable for that period. We handle this by simply keeping cleaned data;
    #    the sidebar lets users filter outlets in/out as needed.

    # 7. Normalize and scope event annotations
    events_df["topic"] = events_df["topic"].replace(
        {
            "PoliticalFigures": "Political Figures",
            "ForeignPolicy": "ForeignPolicy",
        }
    )
    events_df = events_df[events_df["topic"] != "OutletMeta"].copy()

    # Duplicate election-related Government events onto Elections (explicit rule)
    election_mask = (
        (events_df["topic"] == "Government")
        & (
            events_df["subtopic"].str.contains("election", case=False, na=False)
            | events_df["event_name"].str.contains("election", case=False, na=False)
        )
    )
    election_dupes = events_df[election_mask].copy()
    election_dupes["topic"] = "Elections"
    events_df = pd.concat([events_df, election_dupes], ignore_index=True)

    # Forward event window for overlap matching
    events_df["event_start"] = events_df["event_date"]
    events_df["event_end"] = events_df["event_date"] + pd.to_timedelta(
        events_df["window_days"], unit="D"
    )

    return tone_vol, topic_share, events_df


tone_vol, topic_share, events_df = load_data()

# Split tone and volume
tone_df = tone_vol[tone_vol["metric"] == "tone"].copy()
volume_df = tone_vol[tone_vol["metric"] == "volume"].copy()

OUTLETS = sorted(tone_df["outlet"].unique())
TOPICS = sorted(tone_df["topic"].unique())
YEARS = sorted(tone_df["year"].unique())

# ---------------------------------------------------------------------------
# Sidebar – Global filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Filters")
    year_range = st.slider(
        "Year range",
        min_value=int(min(YEARS)),
        max_value=int(max(YEARS)),
        value=(2017, 2025),
    )
    selected_outlets = st.multiselect(
        "Outlets",
        OUTLETS,
        default=OUTLETS,
    )
    selected_topics = st.multiselect(
        "Topics",
        TOPICS,
        default=TOPICS,
    )
    smoothing = st.select_slider(
        "Smoothing window (days)",
        options=[1, 7, 14, 30, 60, 90],
        value=30,
    )
    heatmap_granularity = st.radio(
        "Heatmap granularity",
        ["Yearly", "Semesterly", "Quarterly"],
        horizontal=True,
        index=0,
    )
    st.markdown("---")
    st.markdown("### Data Quality")
    st.markdown(
        "<small>"
        "**Cleaning applied:** removed 2026 (incomplete), replaced zero-value "
        "gaps with NaN, capped tone outliers at +/-10, dropped the 2025-12-06 "
        "blackout date.<br><br>"
        "**Note:** WSJ data degrades from Feb 2024; Politico & WashingtonPost "
        "drop out from Sep 2024. NYTimes is partially degraded from late 2024. "
        "Consider narrowing the year range or deselecting these outlets for "
        "the most recent period."
        "</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<small>Data from the <b>GDELT Project</b> (2017-2025). "
        "Tone values represent average sentiment; volume is the "
        "normalized share of total news output.</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
def apply_filters(df):
    mask = (
        (df["year"] >= year_range[0])
        & (df["year"] <= year_range[1])
        & (df["outlet"].isin(selected_outlets))
        & (df["topic"].isin(selected_topics))
    )
    return df[mask].copy()


tone_f = apply_filters(tone_df)
volume_f = apply_filters(volume_df)
topic_share_f = apply_filters(topic_share)
events_f = events_df[
    (events_df["event_date"].dt.year >= year_range[0])
    & (events_df["event_date"].dt.year <= year_range[1])
    & (events_df["topic"].isin(selected_topics))
].copy()

if not selected_outlets or not selected_topics:
    st.warning("Please select at least one outlet and one topic from the sidebar.")
    st.stop()

if tone_f.empty:
    st.warning("No tone data available for the current filters.")
    st.stop()

# Helper: rolling smooth
def smooth(df, value_col="value", window=30):
    if window <= 1:
        return df
    df = df.sort_values("date")
    df[value_col] = (
        df.groupby(["outlet", "topic"])[value_col]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )
    return df


tone_smooth = smooth(tone_f.copy(), "value", smoothing)
volume_smooth = smooth(volume_f.copy(), "value", smoothing)

# ═══════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<p class="hero-title">Media Lens</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    "Exploring how America's top news outlets cover politics — "
    "what they talk about, and the tone they use."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown(
    "Political news shapes public opinion, yet different outlets can paint "
    "vastly different pictures of the same events. This tool lets you explore "
    "**9 years** of coverage data from **7 major US news outlets** across "
    "**6 key political topics**, powered by the GDELT Project's global news "
    "monitoring database."
)

# Key metrics
n_articles_proxy = len(volume_f)
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="num">7</div>
        <div class="label">News Outlets</div>
    </div>
    <div class="metric-card">
        <div class="num">6</div>
        <div class="label">Political Topics</div>
    </div>
    <div class="metric-card">
        <div class="num">9</div>
        <div class="label">Years of Data</div>
    </div>
    <div class="metric-card">
        <div class="num">270K+</div>
        <div class="label">Daily Measurements</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Helpers for periodized heatmap + topic share smoothing
# ═══════════════════════════════════════════════════════════════════════════
def add_period_columns(df, granularity):
    out = df.copy()
    years = out["date"].dt.year.astype(int)

    if granularity == "Yearly":
        out["period_key"] = years
        out["period_label"] = years.astype(str)
        out["period_start"] = pd.to_datetime(years.astype(str) + "-01-01")
        out["period_end"] = pd.to_datetime(years.astype(str) + "-12-31")
    elif granularity == "Semesterly":
        semesters = ((out["date"].dt.month - 1) // 6 + 1).astype(int)
        out["period_key"] = years * 10 + semesters
        out["period_label"] = years.astype(str) + "-H" + semesters.astype(str)
        out["period_start"] = pd.to_datetime(
            years.astype(str) + semesters.map({1: "-01-01", 2: "-07-01"})
        )
        out["period_end"] = pd.to_datetime(
            years.astype(str) + semesters.map({1: "-06-30", 2: "-12-31"})
        )
    else:  # Quarterly
        quarters = ((out["date"].dt.month - 1) // 3 + 1).astype(int)
        out["period_key"] = years * 10 + quarters
        out["period_label"] = years.astype(str) + "-Q" + quarters.astype(str)
        quarter_period = out["date"].dt.to_period("Q")
        out["period_start"] = quarter_period.dt.start_time
        out["period_end"] = quarter_period.dt.end_time.dt.normalize()

    return out


def smooth_topic_share(df, window):
    if df.empty:
        return df

    out = df.sort_values("date").copy()
    if window <= 1:
        out["topic_share_smoothed"] = out["topic_share"]
    else:
        out["topic_share_smoothed"] = (
            out.groupby(["outlet", "topic"])["topic_share"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    denom = out.groupby(["date", "outlet"])["topic_share_smoothed"].transform("sum")
    denom = denom.replace(0, pd.NA)
    out["topic_share_smoothed"] = out["topic_share_smoothed"] / denom
    out = out.dropna(subset=["topic_share_smoothed"])

    return out


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Topic Evolution Heatmap  ★ MAIN VISUALIZATION ★
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">1 &middot; Topic Evolution Heatmap</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="section-desc">'
    "Click a cell to view outlet-level tone breakdown. "
    "Cells with an <b>i</b> marker include relevant US political events for that topic/period. "
    "Use the sidebar to switch yearly, semesterly, or quarterly granularity."
    "</p>",
    unsafe_allow_html=True,
)

TOPIC_ORDER = [
    "Elections",
    "Government",
    "Immigration",
    "ForeignPolicy",
    "Economy",
    "Political Figures",
]

tone_period = add_period_columns(tone_f, heatmap_granularity)

# Aggregate tone by period and topic
topic_period_stats = (
    tone_period.groupby(["topic", "period_key", "period_label", "period_start", "period_end"])
    .agg(
        avg_tone=("value", "mean"),
        std_tone=("value", "std"),
        n_days=("value", "count"),
    )
    .reset_index()
)
topic_period_stats["std_tone"] = topic_period_stats["std_tone"].fillna(0)

# Period-over-period tone change
topic_period_stats = topic_period_stats.sort_values(["topic", "period_key"])
topic_period_stats["prev_tone"] = topic_period_stats.groupby("topic")["avg_tone"].shift(1)
topic_period_stats["period_change"] = topic_period_stats["avg_tone"] - topic_period_stats["prev_tone"]
topic_period_stats["period_change_label"] = topic_period_stats["period_change"].apply(
    lambda x: "n/a" if pd.isna(x) else (f"+{x:.2f}" if x > 0 else f"{x:.2f}")
)

# Per-outlet breakdown in matching period buckets
topic_period_outlet = (
    tone_period.groupby(["topic", "period_key", "period_label", "outlet"])["value"]
    .mean()
    .reset_index()
    .rename(columns={"value": "avg_tone"})
)

# Build event overlaps: event window [event_date, event_date + window_days]
cell_periods = topic_period_stats[
    ["topic", "period_key", "period_label", "period_start", "period_end"]
].drop_duplicates()

event_matches = cell_periods.merge(events_f, on="topic", how="left")
event_matches = event_matches[
    event_matches["event_start"].notna()
    & (event_matches["event_start"] <= event_matches["period_end"])
    & (event_matches["event_end"] >= event_matches["period_start"])
].copy()

if event_matches.empty:
    event_summary = pd.DataFrame(
        columns=["topic", "period_key", "period_label", "event_count", "event_summary"]
    )
    cell_event_panel = cell_periods[["topic", "period_key", "period_label"]].copy()
    cell_event_panel["row_num"] = 1
    cell_event_panel["panel_line"] = "No annotated event for this period/topic."
    cell_event_panel["annotation_short"] = ""
    cell_event_panel["source_url"] = ""
    cell_event_panel["is_fallback"] = True
else:
    event_matches = event_matches.sort_values(["topic", "period_key", "event_date", "event_name"])
    event_matches["event_date_label"] = event_matches["event_date"].dt.strftime("%Y-%m-%d")
    event_matches["event_line"] = event_matches["event_date_label"] + " - " + event_matches["event_name"]

    event_summary = (
        event_matches.groupby(["topic", "period_key", "period_label"])
        .agg(
            event_count=("event_name", "count"),
            event_summary=("event_line", lambda s: " | ".join(s.tolist()[:3])),
        )
        .reset_index()
    )

    event_rows = event_matches[
        [
            "topic",
            "period_key",
            "period_label",
            "event_date",
            "event_name",
            "annotation_short",
            "source_url",
        ]
    ].copy()
    event_rows["row_num"] = event_rows.groupby(["topic", "period_key"]).cumcount() + 1
    event_rows["event_date_label"] = event_rows["event_date"].dt.strftime("%Y-%m-%d")
    event_rows["panel_line"] = event_rows["event_date_label"] + " - " + event_rows["event_name"]
    event_rows["is_fallback"] = False

    event_cells = event_rows[["topic", "period_key"]].drop_duplicates()
    fallback_rows = cell_periods.merge(
        event_cells.assign(_has_event=1), on=["topic", "period_key"], how="left"
    )
    fallback_rows = fallback_rows[fallback_rows["_has_event"].isna()].copy()
    fallback_rows["row_num"] = 1
    fallback_rows["panel_line"] = "No annotated event for this period/topic."
    fallback_rows["annotation_short"] = ""
    fallback_rows["source_url"] = ""
    fallback_rows["is_fallback"] = True

    keep_cols = [
        "topic",
        "period_key",
        "period_label",
        "row_num",
        "panel_line",
        "annotation_short",
        "source_url",
        "is_fallback",
    ]
    cell_event_panel = pd.concat(
        [event_rows[keep_cols], fallback_rows[keep_cols]],
        ignore_index=True,
    )

# Merge event summary into heatmap cells
topic_period_rich = topic_period_stats.merge(
    event_summary, on=["topic", "period_key", "period_label"], how="left"
)
topic_period_rich["event_count"] = topic_period_rich["event_count"].fillna(0).astype(int)
topic_period_rich["event_summary"] = topic_period_rich["event_summary"].fillna(
    "No annotated event for this period/topic."
)

period_order = (
    topic_period_rich[["period_key", "period_label"]]
    .drop_duplicates()
    .sort_values("period_key")["period_label"]
    .tolist()
)
available_topics = topic_period_rich["topic"].dropna().unique().tolist()
y_sorted = [t for t in TOPIC_ORDER if t in available_topics]

if not available_topics:
    st.warning("No heatmap data for the current filters.")
    st.stop()

default_topic = next((t for t in TOPIC_ORDER if t in available_topics), available_topics[0])
default_period_key = int(
    topic_period_rich.loc[
        topic_period_rich["topic"] == default_topic,
        "period_key",
    ].max()
)

cell_sel = alt.selection_point(
    fields=["topic", "period_key"],
    on="click",
    empty=False,
    value=[{"topic": default_topic, "period_key": default_period_key}],
)

tone_min = float(topic_period_rich["avg_tone"].min())
tone_max = float(topic_period_rich["avg_tone"].max())
if tone_min == tone_max:
    tone_min -= 0.1
    tone_max += 0.1

time_axis_title = {
    "Yearly": "Year",
    "Semesterly": "Semester",
    "Quarterly": "Quarter",
}[heatmap_granularity]

heatmap_base = alt.Chart(topic_period_rich).encode(
    x=alt.X(
        "period_label:O",
        sort=period_order,
        title=time_axis_title,
        axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=13),
    ),
    y=alt.Y(
        "topic:N",
        sort=y_sorted,
        title=None,
        axis=alt.Axis(labelFontSize=14),
    ),
)

heatmap_rects = heatmap_base.mark_rect(cornerRadius=6).encode(
    color=alt.Color(
        "avg_tone:Q",
        title="Average Tone",
        scale=alt.Scale(scheme="cividis", domain=[tone_min, tone_max]),
        legend=alt.Legend(
            title="More negative  ->  less negative / positive",
            orient="bottom",
            direction="horizontal",
            gradientLength=340,
            titleFontSize=11,
        ),
    ),
    stroke=alt.condition(cell_sel, alt.value("#0f172a"), alt.value("#ffffff")),
    strokeWidth=alt.condition(cell_sel, alt.value(3), alt.value(1.5)),
    tooltip=[
        alt.Tooltip("topic:N", title="Topic"),
        alt.Tooltip("period_label:N", title=time_axis_title),
        alt.Tooltip("avg_tone:Q", format=".2f", title="Avg Tone"),
        alt.Tooltip("std_tone:Q", format=".2f", title="Std Dev"),
        alt.Tooltip("n_days:Q", title="Data Points"),
        alt.Tooltip("period_change_label:N", title="Period-over-Period"),
        alt.Tooltip("event_count:Q", title="Relevant Events"),
        alt.Tooltip("event_summary:N", title="Event Summary"),
    ],
)

heatmap_text = heatmap_base.mark_text(fontSize=14, fontWeight="bold").encode(
    text=alt.Text("avg_tone:Q", format=".1f"),
    color=alt.condition(alt.datum.avg_tone > -1.0, alt.value("#111827"), alt.value("white")),
)

event_markers = topic_period_rich[topic_period_rich["event_count"] > 0].copy()
marker_i = alt.Chart(event_markers).mark_text(
    text="i",
    dx=22,
    dy=-13,
    color="#111827",
    fontSize=11,
    fontWeight="bold",
).encode(
    x=alt.X("period_label:O", sort=period_order),
    y=alt.Y("topic:N", sort=y_sorted),
)

heatmap_chart = (heatmap_rects + heatmap_text + marker_i).add_params(cell_sel).properties(
    height=420,
    title=alt.Title(
        text="How tone for each topic changed over time",
        subtitle="Click for outlet details. Cells with i include event annotations.",
        fontSize=16,
        subtitleFontSize=12,
        subtitleColor="#6b7280",
        anchor="middle",
    ),
)

outlet_bars = (
    alt.Chart(topic_period_outlet)
    .transform_filter(cell_sel)
    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    .encode(
        x=alt.X(
            "outlet:N",
            title="Outlet",
            sort=alt.EncodingSortField(field="avg_tone", order="ascending"),
            axis=alt.Axis(labelAngle=-25, labelFontSize=12),
        ),
        y=alt.Y("avg_tone:Q", title="Avg Tone", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "outlet:N",
            scale=alt.Scale(
                domain=list(OUTLET_COLORS.keys()),
                range=list(OUTLET_COLORS.values()),
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("topic:N", title="Topic"),
            alt.Tooltip("period_label:N", title=time_axis_title),
            alt.Tooltip("outlet:N", title="Outlet"),
            alt.Tooltip("avg_tone:Q", format=".2f", title="Avg Tone"),
        ],
    )
)
outlet_zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
outlet_chart = (outlet_bars + outlet_zero).properties(
    height=270,
    title="Outlet tone breakdown (click a heatmap cell)",
)

event_panel_base = alt.Chart(cell_event_panel).transform_filter(cell_sel).encode(
    y=alt.Y("row_num:O", title=None, axis=None, sort="ascending")
)
event_panel_marks = event_panel_base.mark_circle(size=60).encode(
    x=alt.value(6),
    color=alt.condition("datum.is_fallback", alt.value("#9ca3af"), alt.value("#1d4ed8")),
    tooltip=[
        alt.Tooltip("period_label:N", title=time_axis_title),
        alt.Tooltip("topic:N", title="Topic"),
        alt.Tooltip("panel_line:N", title="Event"),
        alt.Tooltip("annotation_short:N", title="Annotation"),
        alt.Tooltip("source_url:N", title="Source URL"),
    ],
)
event_panel_text = event_panel_base.mark_text(align="left", dx=16, fontSize=12).encode(
    x=alt.value(18),
    text=alt.Text("panel_line:N"),
    color=alt.condition("datum.is_fallback", alt.value("#6b7280"), alt.value("#1f2937")),
)
event_panel_chart = (event_panel_marks + event_panel_text).properties(
    height=150,
    title="Event notes for selected cell",
)

st.altair_chart(
    alt.vconcat(
        heatmap_chart,
        outlet_chart,
        event_panel_chart,
        spacing=10,
    ).resolve_scale(color="independent"),
    use_container_width=True,
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Topic share over time (with event annotations)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">2 &middot; What Topics Dominate the News?</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="section-desc">'
    "See how topic share changes over time, with event markers and multi-outlet selection."
    "</p>",
    unsafe_allow_html=True,
)

outlet_options = ["All outlets"] + selected_outlets
selected_topic_outlets = st.multiselect(
    "Select outlet(s) for topic share:",
    options=outlet_options,
    default=["All outlets"],
    key="topic_share_outlets",
)

if not selected_topic_outlets or "All outlets" in selected_topic_outlets:
    topic_outlet_scope = selected_outlets
else:
    topic_outlet_scope = [o for o in selected_topic_outlets if o in selected_outlets]

if not topic_outlet_scope:
    st.warning("Select at least one outlet for the topic-share chart.")
    st.stop()

ts_selected = topic_share_f[topic_share_f["outlet"].isin(topic_outlet_scope)].copy()
ts_smoothed = smooth_topic_share(ts_selected, smoothing)
ts_smoothed["month"] = ts_smoothed["date"].dt.to_period("M").dt.to_timestamp()
ts_monthly = (
    ts_smoothed.groupby(["month", "topic"])["topic_share_smoothed"]
    .mean()
    .reset_index()
    .rename(columns={"topic_share_smoothed": "topic_share"})
)

topic_selection = alt.selection_point(fields=["topic"], bind="legend")

stacked_area = (
    alt.Chart(ts_monthly)
    .mark_area()
    .encode(
        x=alt.X("month:T", title="Date"),
        y=alt.Y(
            "topic_share:Q",
            title="Share of Coverage",
            stack="normalize",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%"),
        ),
        color=alt.Color(
            "topic:N",
            title="Topic",
            scale=alt.Scale(
                domain=list(TOPIC_COLORS.keys()),
                range=list(TOPIC_COLORS.values()),
            ),
        ),
        opacity=alt.condition(topic_selection, alt.value(1), alt.value(0.25)),
        tooltip=[
            "month:T",
            "topic:N",
            alt.Tooltip("topic_share:Q", format=".1%", title="Share"),
        ],
    )
    .add_params(topic_selection)
    .properties(height=400)
    .interactive(bind_x=True, bind_y=False)
)

events_topic_overlay = events_f[events_f["topic"].isin(ts_monthly["topic"].unique())].copy()

if events_topic_overlay.empty:
    topic_share_chart = stacked_area
else:
    event_rules = alt.Chart(events_topic_overlay).mark_rule(opacity=0.2).encode(
        x=alt.X("event_date:T"),
        color=alt.Color(
            "topic:N",
            scale=alt.Scale(
                domain=list(TOPIC_COLORS.keys()),
                range=list(TOPIC_COLORS.values()),
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("event_date:T", title="Event Date"),
            alt.Tooltip("topic:N", title="Topic"),
            alt.Tooltip("event_name:N", title="Event"),
            alt.Tooltip("annotation_short:N", title="Annotation"),
            alt.Tooltip("source_url:N", title="Source URL"),
        ],
    )

    event_points = (
        alt.Chart(events_topic_overlay)
        .transform_calculate(y_pos="1")
        .mark_point(filled=True, shape="diamond", size=90, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("event_date:T"),
            y=alt.Y("y_pos:Q"),
            color=alt.Color(
                "topic:N",
                scale=alt.Scale(
                    domain=list(TOPIC_COLORS.keys()),
                    range=list(TOPIC_COLORS.values()),
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("event_date:T", title="Event Date"),
                alt.Tooltip("topic:N", title="Topic"),
                alt.Tooltip("event_name:N", title="Event"),
                alt.Tooltip("annotation_short:N", title="Annotation"),
                alt.Tooltip("source_url:N", title="Source URL"),
            ],
        )
    )

    event_info = (
        alt.Chart(events_topic_overlay)
        .transform_calculate(y_pos="1")
        .mark_text(text="i", dy=-8, fontSize=9, fontWeight="bold", color="#111827")
        .encode(x=alt.X("event_date:T"), y=alt.Y("y_pos:Q"))
    )

    topic_share_chart = stacked_area + event_rules + event_points + event_info

st.altair_chart(topic_share_chart, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<b>Key insight:</b> Topic attention shifts over time, and event markers help students connect "
    "specific political moments with changes in media focus."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Tone over time
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">3 &middot; The Tone of Political Coverage Over Time</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="section-desc">'
    "Drag to select a time range on the top chart — the bar chart below instantly "
    "updates to show average outlet tone for that period. Click the legend to isolate outlets."
    "</p>",
    unsafe_allow_html=True,
)

# Monthly aggregation for the line chart (keeps data manageable)
tone_monthly = tone_f.copy()
tone_monthly["month"] = tone_monthly["date"].dt.to_period("M").dt.to_timestamp()
tone_monthly_agg = (
    tone_monthly.groupby(["month", "outlet"])["value"]
    .mean()
    .reset_index()
)

# Shared brush selection for cross-filter chart
brush = alt.selection_interval(encodings=["x"])
legend_sel = alt.selection_point(fields=["outlet"], bind="legend")

# Top chart: line with brush
brush_line = (
    alt.Chart(tone_monthly_agg)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("month:T", title="Date"),
        y=alt.Y("value:Q", title="Avg Tone", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "outlet:N",
            title="Outlet",
            scale=alt.Scale(
                domain=list(OUTLET_COLORS.keys()),
                range=list(OUTLET_COLORS.values()),
            ),
        ),
        opacity=alt.condition(legend_sel, alt.value(1), alt.value(0.1)),
        tooltip=["month:T", "outlet:N", alt.Tooltip("value:Q", format=".2f", title="Tone")],
    )
    .properties(height=300)
    .add_params(brush, legend_sel)
)

brush_zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")

# Bottom chart: bar filtered by brush
brush_bars = (
    alt.Chart(tone_monthly_agg)
    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    .encode(
        x=alt.X(
            "outlet:N",
            title="Outlet",
            sort=alt.EncodingSortField(field="value", order="ascending"),
            axis=alt.Axis(labelAngle=-30),
        ),
        y=alt.Y("mean(value):Q", title="Avg Tone (selected period)", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "outlet:N",
            scale=alt.Scale(
                domain=list(OUTLET_COLORS.keys()),
                range=list(OUTLET_COLORS.values()),
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("outlet:N", title="Outlet"),
            alt.Tooltip("mean(value):Q", format=".2f", title="Avg Tone"),
        ],
    )
    .transform_filter(brush)
    .properties(height=250, title="Outlet tone for brushed period")
)

brush_bar_zero = (
    alt.Chart(pd.DataFrame({"y": [0]}))
    .mark_rule(strokeDash=[4, 4], color="gray")
    .encode(y="y:Q")
)

cross_filter = alt.vconcat(
    brush_line + brush_zero,
    brush_bars + brush_bar_zero,
).resolve_legend(color="independent")

st.altair_chart(cross_filter, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<b>Key insight:</b> Most outlets tend toward negative tone in political coverage "
    "(below the zero line). Notice how sentiment dips around major political events "
    "and then partially recovers."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Tone Distribution Box Plots
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">4 &middot; Tone Distribution by Outlet</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="section-desc">'
    "Averages can hide a lot. These box plots show the full spread of monthly "
    "tone values — revealing which outlets are volatile and which are consistent."
    "</p>",
    unsafe_allow_html=True,
)

# Monthly tone per outlet for distribution
tone_box = tone_f.copy()
tone_box["month"] = tone_box["date"].dt.to_period("M").dt.to_timestamp()
tone_box_monthly = (
    tone_box.groupby(["month", "outlet"])["value"]
    .mean()
    .reset_index()
)

box_plot = (
    alt.Chart(tone_box_monthly)
    .mark_boxplot(extent="min-max", size=40)
    .encode(
        x=alt.X(
            "outlet:N",
            title="Outlet",
            sort=alt.EncodingSortField(field="value", op="median", order="ascending"),
            axis=alt.Axis(labelAngle=-30, labelFontSize=12),
        ),
        y=alt.Y("value:Q", title="Monthly Avg Tone", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "outlet:N",
            title="Outlet",
            scale=alt.Scale(
                domain=list(OUTLET_COLORS.keys()),
                range=list(OUTLET_COLORS.values()),
            ),
            legend=None,
        ),
    )
    .properties(height=380)
)

# Overlay strip/jitter for individual months
strip = (
    alt.Chart(tone_box_monthly)
    .mark_circle(size=20, opacity=0.3)
    .encode(
        x=alt.X("outlet:N", sort=alt.EncodingSortField(field="value", op="median", order="ascending")),
        y=alt.Y("value:Q"),
        color=alt.Color(
            "outlet:N",
            scale=alt.Scale(
                domain=list(OUTLET_COLORS.keys()),
                range=list(OUTLET_COLORS.values()),
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("outlet:N", title="Outlet"),
            alt.Tooltip("month:T", title="Month"),
            alt.Tooltip("value:Q", format=".2f", title="Tone"),
        ],
    )
)

st.altair_chart(box_plot + strip, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<b>Key insight:</b> Some outlets (like WSJ) show a much wider spread in tone, meaning their "
    "coverage varies greatly month to month. Others are more consistently negative. "
    "The individual dots let you spot the extreme months."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Diverging Bars – Deviation from Average
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">5 &middot; How Each Outlet Deviates from the Average</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="section-desc">'
    "For each topic, how does each outlet compare to the overall average? "
    "Bars extending left are more negative than average; bars extending right are less negative."
    "</p>",
    unsafe_allow_html=True,
)

# Compute per-outlet, per-topic avg tone and deviation from topic mean
outlet_topic_tone = (
    tone_f.groupby(["outlet", "topic"])["value"]
    .mean()
    .reset_index()
    .rename(columns={"value": "outlet_tone"})
)
topic_avg = (
    tone_f.groupby("topic")["value"]
    .mean()
    .reset_index()
    .rename(columns={"value": "topic_avg"})
)
deviation_df = outlet_topic_tone.merge(topic_avg, on="topic")
deviation_df["deviation"] = deviation_df["outlet_tone"] - deviation_df["topic_avg"]
deviation_df["direction"] = np.where(
    deviation_df["deviation"] >= 0, "Less negative", "More negative"
)

div_base = alt.Chart(deviation_df).encode(
    y=alt.Y(
        "outlet:N",
        title=None,
        sort=alt.EncodingSortField(field="deviation", order="ascending"),
        axis=alt.Axis(labelFontSize=11),
    ),
    x=alt.X(
        "deviation:Q",
        title="Deviation from topic average",
        axis=alt.Axis(format=".2f"),
    ),
    color=alt.Color(
        "direction:N",
        title="Relative tone",
        scale=alt.Scale(
            domain=["More negative", "Less negative"],
            range=["#d62728", "#2ca02c"],
        ),
        legend=alt.Legend(orient="bottom"),
    ),
    tooltip=[
        alt.Tooltip("outlet:N", title="Outlet"),
        alt.Tooltip("topic:N", title="Topic"),
        alt.Tooltip("outlet_tone:Q", format=".2f", title="Outlet Tone"),
        alt.Tooltip("topic_avg:Q", format=".2f", title="Topic Avg"),
        alt.Tooltip("deviation:Q", format=".2f", title="Deviation"),
    ],
)

div_bars = div_base.mark_bar(cornerRadius=3).properties(height=80, width=500)

# Altair cannot layer after faceting, so layer first then facet.
# Use the same source table to keep facet data at the top level.
div_zero = (
    alt.Chart(deviation_df)
    .mark_rule(strokeDash=[4, 4], color="#666")
    .encode(x=alt.datum(0))
)

diverging = alt.layer(div_bars, div_zero).facet(
    row=alt.Row(
        "topic:N",
        title=None,
        sort=TOPIC_ORDER,
        header=alt.Header(labelFontSize=13, labelFontWeight="bold"),
    )
)

st.altair_chart(diverging, use_container_width=True)

st.markdown(
    '<div class="insight-box">'
    "<b>Key insight:</b> Fox News tends to be more negative than average on Immigration "
    "and Foreign Policy, while Economy coverage shows the smallest outlet-to-outlet divergence. "
    "This reveals which topics drive the most editorial disagreement."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Topic deep-dive – pick a topic, compare outlets
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">6 &middot; Deep Dive: Compare Outlets on a Topic</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="section-desc">'
    "Select a topic to compare outlet tone side-by-side over time."
    "</p>",
    unsafe_allow_html=True,
)

deep_topic = st.selectbox("Choose a topic:", selected_topics, index=0, key="deep_topic")

deep_tone = tone_smooth[tone_smooth["topic"] == deep_topic].copy()
deep_tone_agg = deep_tone.groupby(["date", "outlet"])["value"].mean().reset_index()

outlet_sel2 = alt.selection_point(fields=["outlet"], bind="legend")

deep_tone_chart = (
    alt.Chart(deep_tone_agg)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Tone", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "outlet:N",
            title="Outlet",
            scale=alt.Scale(
                domain=list(OUTLET_COLORS.keys()),
                range=list(OUTLET_COLORS.values()),
            ),
        ),
        opacity=alt.condition(outlet_sel2, alt.value(1), alt.value(0.1)),
        tooltip=["date:T", "outlet:N", alt.Tooltip("value:Q", format=".2f")],
    )
    .add_params(outlet_sel2)
    .properties(height=350, title=f"Tone over Time – {deep_topic}")
    .interactive()
)

zero_line2 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")

st.altair_chart(deep_tone_chart + zero_line2, use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: Outlet Sentiment Ranking – Bump Chart
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">7 &middot; Outlet Sentiment Rankings Over Time</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="section-desc">'
    "Which outlet is the most negative each year? This bump chart ranks outlets from "
    "most negative (rank 1, top) to least negative. Watch the lines cross as rankings shift."
    "</p>",
    unsafe_allow_html=True,
)

# Compute yearly avg tone per outlet, then rank
yearly_tone = (
    tone_f.groupby(["year", "outlet"])["value"]
    .mean()
    .reset_index()
)
yearly_tone["rank"] = yearly_tone.groupby("year")["value"].rank(method="min").astype(int)

outlet_sel3 = alt.selection_point(fields=["outlet"], bind="legend")

bump_lines = (
    alt.Chart(yearly_tone)
    .mark_line(strokeWidth=3)
    .encode(
        x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0, labelFontSize=13)),
        y=alt.Y(
            "rank:O",
            title="Rank (1 = most negative)",
            sort="ascending",
            axis=alt.Axis(labelFontSize=13),
        ),
        color=alt.Color(
            "outlet:N",
            title="Outlet",
            scale=alt.Scale(
                domain=list(OUTLET_COLORS.keys()),
                range=list(OUTLET_COLORS.values()),
            ),
        ),
        opacity=alt.condition(outlet_sel3, alt.value(1), alt.value(0.15)),
        tooltip=[
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip("outlet:N", title="Outlet"),
            alt.Tooltip("rank:Q", title="Rank"),
            alt.Tooltip("value:Q", format=".2f", title="Avg Tone"),
        ],
    )
    .add_params(outlet_sel3)
)

bump_points = (
    alt.Chart(yearly_tone)
    .mark_circle(size=100)
    .encode(
        x=alt.X("year:O"),
        y=alt.Y("rank:O", sort="ascending"),
        color=alt.Color(
            "outlet:N",
            scale=alt.Scale(
                domain=list(OUTLET_COLORS.keys()),
                range=list(OUTLET_COLORS.values()),
            ),
        ),
        opacity=alt.condition(outlet_sel3, alt.value(1), alt.value(0.15)),
        tooltip=[
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip("outlet:N", title="Outlet"),
            alt.Tooltip("rank:Q", title="Rank"),
            alt.Tooltip("value:Q", format=".2f", title="Avg Tone"),
        ],
    )
)

# Labels on the right side (last year)
max_year = yearly_tone["year"].max()
bump_labels = (
    alt.Chart(yearly_tone[yearly_tone["year"] == max_year])
    .mark_text(align="left", dx=8, fontSize=12, fontWeight="bold")
    .encode(
        x=alt.X("year:O"),
        y=alt.Y("rank:O", sort="ascending"),
        text=alt.Text("outlet:N"),
        color=alt.Color(
            "outlet:N",
            scale=alt.Scale(
                domain=list(OUTLET_COLORS.keys()),
                range=list(OUTLET_COLORS.values()),
            ),
        ),
        opacity=alt.condition(outlet_sel3, alt.value(1), alt.value(0.15)),
    )
)

st.altair_chart(
    (bump_lines + bump_points + bump_labels).properties(height=400),
    use_container_width=True,
)

st.markdown(
    '<div class="insight-box">'
    "<b>Key insight:</b> Rankings are not static — outlets swap positions frequently. "
    "An outlet that was the most negative one year can become moderate the next, "
    "suggesting tone is driven by editorial choices around specific events rather than "
    "a fixed institutional bias."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# KEY TAKEAWAYS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">Key Takeaways</p>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Negativity dominates political coverage.** Across all 7 outlets and 6 topics,
    the average tone of political reporting is consistently negative. This reflects
    a well-documented pattern in media studies, but seeing it quantified across
    outlets makes the scale clear.

    **Coverage priorities differ.** While Government and Elections dominate everywhere,
    the relative attention given to Immigration, Economy, and Foreign Policy varies
    meaningfully between outlets — reflecting different editorial priorities and
    audience expectations.
    """)

with col2:
    st.markdown("""
    **Election years amplify negativity.** Sentiment dips are most pronounced around
    election cycles (2018, 2020, 2024), suggesting that political campaign coverage
    drives more negative framing.

    **No outlet is truly "neutral".** Every outlet in this dataset shows a measurable
    negative bias in political tone, though the degree varies. This matters for media
    literacy — understanding that all sources carry some sentiment framing helps
    students consume news more critically.
    """)

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="footer">'
    "Media Lens &middot; DSBA Data Visualization Project 2026 &middot; "
    "Data from the GDELT Project &middot; Built with Streamlit & Altair"
    "</div>",
    unsafe_allow_html=True,
)
