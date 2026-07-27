"""
A Decade of Distress — Occupational Suicide Trends in India, 2015-2024
Streamlit dashboard companion to the analysis notebook.

Run locally with:  streamlit run app.py
Expects NCRB_Suicide_Data_2015_2024.xlsx either uploaded via the sidebar,
or placed in the same folder as this script.
"""

import os
import io
import requests

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_rel, binomtest, pearsonr
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="A Decade of Distress — India Suicide Trends", layout="wide")

YEARS = list(range(2015, 2025))
YEAR_PCT_COLS = [f"{y} (%)" for y in YEARS]

GEOJSON_URL = "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/india.geojson"
GEOJSON_LOCAL_CACHE = "india.geojson"

NAME_MAP = {
    "Andaman & Nicobar Islands": "Andaman and Nicobar Islands",
    "Dadra & Nagar Haveli / Daman & Diu": "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi (UT)": "Delhi",
    "Jammu & Kashmir": "Jammu and Kashmir",
}

FLAGGED_STATES_NOTE = (
    "Meghalaya, Mizoram, Andaman & Nicobar Islands, Goa, and Bihar show volatile "
    "year-to-year rate swings for specific years — verified against source NCRB PDFs "
    "and confirmed as accurate, not transcription errors. This volatility is a real "
    "artifact of small population denominators combined with NCRB's periodic revisions "
    "to population-projection sources across report years. Chandigarh's 2018 count "
    "(360) remains an unexplained outlier present in the source data itself."
)


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
@st.cache_data
def load_data(file_bytes: bytes):
    xl = pd.ExcelFile(io.BytesIO(file_bytes))

    national = xl.parse("National_Trend", skiprows=2, nrows=11)
    prof = xl.parse("Profession_wise", skiprows=2, nrows=9)
    causes = xl.parse("Causes_of_Suicide", skiprows=2, nrows=17)
    counts = xl.parse("State_wise_Counts", skiprows=2, nrows=36)
    rates = xl.parse("State_wise_Rates", skiprows=2, nrows=36)

    counts.columns = ["State/UT"] + YEARS
    rates.columns = ["State/UT"] + YEARS
    for df in (counts, rates):
        for y in YEARS:
            df[y] = pd.to_numeric(df[y], errors="coerce")

    return national, prof, causes, counts, rates


@st.cache_resource
def load_geo():
    import geopandas as gpd

    if not os.path.exists(GEOJSON_LOCAL_CACHE):
        r = requests.get(GEOJSON_URL, timeout=30)
        with open(GEOJSON_LOCAL_CACHE, "wb") as f:
            f.write(r.content)

    india_districts = gpd.read_file(GEOJSON_LOCAL_CACHE)
    states_geo = india_districts.dissolve(by="st_nm", as_index=False)[["st_nm", "geometry"]]
    return states_geo


# ----------------------------------------------------------------------
# Sidebar — data source
# ----------------------------------------------------------------------
st.sidebar.title("Data source")
uploaded = st.sidebar.file_uploader("Upload NCRB_Suicide_Data_2015_2024.xlsx", type=["xlsx"])

DEFAULT_PATH = "NCRB_Suicide_Data_2015_2024.xlsx"
file_bytes = None
if uploaded is not None:
    file_bytes = uploaded.read()
elif os.path.exists(DEFAULT_PATH):
    with open(DEFAULT_PATH, "rb") as f:
        file_bytes = f.read()
    st.sidebar.success(f"Loaded {DEFAULT_PATH} from the app folder.")
else:
    st.sidebar.warning("Upload the Excel file to begin, or place it next to app.py.")
    st.stop()

national, prof, causes, counts, rates = load_data(file_bytes)

st.title("A Decade of Distress")
st.caption("Occupational Distress and Suicide Trajectories in India, 2015–2024")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "National Trends",
    "Occupation × Cause Link",
    "State Explorer",
    "Statistical Robustness",
    "Forecast",
])

# ----------------------------------------------------------------------
# TAB 1 — National Trends
# ----------------------------------------------------------------------
with tab1:
    st.header("National Trends, 2015–2024")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Profession-wise % share of suicides")
        fig, ax = plt.subplots(figsize=(7, 5))
        for _, row in prof.iterrows():
            lw = 3 if row["Profession Category"] == "Daily Wage Earner" else 1
            alpha = 1.0 if row["Profession Category"] == "Daily Wage Earner" else 0.6
            ax.plot(YEARS, row[YEAR_PCT_COLS].astype(float), marker="o",
                    linewidth=lw, alpha=alpha, label=row["Profession Category"])
        ax.set_xlabel("Year")
        ax.set_ylabel("% of total suicides")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        dwe = prof[prof["Profession Category"] == "Daily Wage Earner"].iloc[0]
        change = dwe["2024 (%)"] - dwe["2015 (%)"]
        st.metric("Daily Wage Earner share, 2015 → 2024",
                   f"{dwe['2024 (%)']:.1f}%",
                   f"+{change:.1f} pts since 2015")

    with col2:
        st.subheader("Top causes of suicide")
        n_causes = st.slider("Number of top causes to show", 3, 10, 6)
        causes_top = causes.set_index("Cause")[YEAR_PCT_COLS].astype(float)
        causes_top = causes_top.loc[causes_top[f"{YEARS[-1]} (%)"].sort_values(ascending=False).head(n_causes).index]
        fig, ax = plt.subplots(figsize=(7, 5))
        for cause, row in causes_top.iterrows():
            ax.plot(YEARS, row.values, marker="o", label=cause)
        ax.set_xlabel("Year")
        ax.set_ylabel("% of total suicides")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    st.subheader("National suicide count & rate, 2015–2024")
    st.dataframe(national, width='stretch')

# ----------------------------------------------------------------------
# TAB 2 — Occupation x Cause Link
# ----------------------------------------------------------------------
with tab2:
    st.header("Does the character of suicide causation shift alongside its occupational composition?")
    st.markdown(
        "Correlating two series that both trend upward over time produces a "
        "**misleadingly high correlation purely from the shared trend** — a well-known "
        "statistical pitfall. Toggle below to see the difference between the raw "
        "(misleading) version and the properly detrended (honest) version."
    )

    view = st.radio("View", ["Detrended (honest)", "Raw levels (misleading — for comparison)"], horizontal=True)

    prof_t = prof.set_index("Profession Category")[YEAR_PCT_COLS].T
    prof_t.index = YEARS
    causes_t = causes.set_index("Cause")[YEAR_PCT_COLS].T
    causes_t.index = YEARS

    if view.startswith("Detrended"):
        prof_use = prof_t.diff()
        causes_use = causes_t.diff()
        st.caption("Correlation of year-over-year changes (first differences).")
    else:
        prof_use = prof_t
        causes_use = causes_t
        st.caption("Correlation of raw yearly % levels — inflated by shared trend, shown for contrast only.")

    corr_matrix = pd.DataFrame(index=prof_use.columns, columns=causes_use.columns, dtype=float)
    for p_ in prof_use.columns:
        for c in causes_use.columns:
            pair = pd.concat([prof_use[p_], causes_use[c]], axis=1).dropna()
            if len(pair) > 2:
                corr_matrix.loc[p_, c] = pair.iloc[:, 0].corr(pair.iloc[:, 1])

    fig, ax = plt.subplots(figsize=(14, 4.5))
    im = ax.imshow(corr_matrix.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_yticklabels(corr_matrix.index, fontsize=8)
    plt.colorbar(im, label="Pearson r")
    st.pyplot(fig)

    st.subheader("Daily Wage Earner — strongest links (detrended)")
    dwe_diff_corr = causes_t.diff().corrwith(prof_t.diff()["Daily Wage Earner"]).sort_values(ascending=False)
    top_cause = dwe_diff_corr.index[0]
    pair = pd.concat([prof_t.diff()["Daily Wage Earner"], causes_t.diff()[top_cause]], axis=1).dropna()
    r, p = pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
    st.write(f"Strongest detrended relationship: **{top_cause}** (r = {r:.3f}, p = {p:.3f})")
    if p >= 0.05:
        st.warning(f"Not statistically significant at n={len(pair)} year-pairs — report as suggestive, not proven.")
    st.dataframe(dwe_diff_corr.round(3).rename("Pearson r").to_frame(), width='stretch')

# ----------------------------------------------------------------------
# TAB 3 — State Explorer
# ----------------------------------------------------------------------
with tab3:
    st.header("State-level suicide rate, 2015–2024")
    st.info(FLAGGED_STATES_NOTE)

    try:
        states_geo = load_geo()
        rates_geo_ready = rates.copy()
        rates_geo_ready["geo_name"] = rates_geo_ready["State/UT"].map(lambda s: NAME_MAP.get(s, s))
        merged = states_geo.merge(rates_geo_ready, left_on="st_nm", right_on="geo_name", how="left")

        map_year = st.select_slider("Select year", options=YEARS, value=2024)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Suicide rate — {map_year}")
            fig, ax = plt.subplots(figsize=(6, 6))
            merged.plot(column=map_year, cmap="OrRd", legend=True, ax=ax,
                        edgecolor="black", linewidth=0.3, missing_kwds={"color": "lightgrey"})
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            st.subheader("Change, 2015 → 2024")
            merged["change"] = merged[2024] - merged[2015]
            vmax = merged["change"].abs().max()
            fig, ax = plt.subplots(figsize=(6, 6))
            merged.plot(column="change", cmap="RdBu_r", legend=True, ax=ax,
                        edgecolor="black", linewidth=0.3, missing_kwds={"color": "lightgrey"},
                        vmin=-vmax, vmax=vmax)
            ax.axis("off")
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Map rendering unavailable in this environment ({e}). Ranked list below still works.")

    st.subheader("Ranked spotlight: biggest movers, 2015 → 2024")
    st.caption(
        "Presented as a ranked list rather than a forced cluster typology — k-means "
        "trajectory clustering was tested and did not produce a meaningful multi-state "
        "grouping (it only isolated single outlier states)."
    )
    traj = rates.set_index("State/UT")[YEARS].dropna()
    ranked = pd.DataFrame({
        "2015 rate": traj[2015],
        "2024 rate": traj[2024],
        "Total change": traj[2024] - traj[2015],
    }).sort_values("Total change", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Biggest risers**")
        st.dataframe(ranked.head(5).round(1), width='stretch')
    with col2:
        st.markdown("**Biggest decliners**")
        st.dataframe(ranked.tail(5).sort_values("Total change").round(1), width='stretch')

# ----------------------------------------------------------------------
# TAB 4 — Statistical Robustness
# ----------------------------------------------------------------------
with tab4:
    st.header("Is the state-level rate rise statistically robust?")

    window = st.radio("Test window", ["Full decade (2015 vs 2024)", "Recent window (2019 vs 2024)"], horizontal=True)
    y0, y1 = (2015, 2024) if window.startswith("Full") else (2019, 2024)

    pivot = rates.set_index("State/UT")[[y0, y1]].dropna()
    diffs = pivot[y1] - pivot[y0]
    n_total = len(pivot)

    stat, p = wilcoxon(pivot[y0], pivot[y1])
    t_stat, t_p = ttest_rel(pivot[y0], pivot[y1])
    n_pos, n_neg = int((diffs > 0).sum()), int((diffs < 0).sum())
    bt = binomtest(n_pos, n_pos + n_neg, p=0.5)

    col1, col2, col3 = st.columns(3)
    col1.metric("Wilcoxon p-value", f"{p:.4f}", "Significant" if p < 0.05 else "Not significant")
    col2.metric("Paired t-test p-value", f"{t_p:.4f}", "Significant" if t_p < 0.05 else "Not significant")
    col3.metric("Sign test p-value", f"{bt.pvalue:.4f}", "Significant" if bt.pvalue < 0.05 else "Not significant")

    st.write(f"n = {n_total} states | {n_pos} increased, {n_neg} decreased ({n_pos/n_total*100:.0f}% increased)")

    st.subheader("Leave-one-out sensitivity")
    loo_results = []
    for state in pivot.index:
        sub = pivot.drop(state)
        s, pv = wilcoxon(sub[y0], sub[y1])
        loo_results.append((state, pv))
    loo_df = pd.DataFrame(loo_results, columns=["Dropped state", "p-value"]).sort_values("p-value")
    n_flips = (loo_df["p-value"] > 0.05).sum()
    st.write(f"Removing a single state flips the result to non-significant for **{n_flips} of {n_total}** states.")
    if n_flips > n_total * 0.3:
        st.warning("This result is fragile — a large share of single-state removals change the conclusion. "
                   "Do not report this p-value without this caveat.")
    st.dataframe(loo_df.round(5), width='stretch', height=250)

# ----------------------------------------------------------------------
# TAB 5 — Forecast
# ----------------------------------------------------------------------
with tab5:
    st.header("Short-horizon projection")
    st.warning(
        "With only 10 historical data points, treat this as an illustration of "
        "'where the current trend leads,' not a precise prediction. This is a "
        "forecast of an aggregate national statistic — not a model of individual "
        "suicide risk, which this data cannot and should not be used for."
    )

    category = st.selectbox("Profession category to forecast",
                             prof["Profession Category"].tolist(),
                             index=prof["Profession Category"].tolist().index("Daily Wage Earner"))
    horizon = st.slider("Forecast horizon (years)", 1, 5, 3)

    series = prof[prof["Profession Category"] == category][YEAR_PCT_COLS].values.flatten().astype(float)
    model = ExponentialSmoothing(series, trend="add", initialization_method="estimated").fit()
    forecast = model.forecast(horizon)
    forecast_years = list(range(YEARS[-1] + 1, YEARS[-1] + 1 + horizon))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(YEARS, series, marker="o", label="Observed (2015–2024)")
    ax.plot(forecast_years, forecast, marker="o", linestyle="--", color="red", label=f"Forecast ({forecast_years[0]}–{forecast_years[-1]})")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of total suicides")
    ax.set_title(f"{category} — Observed + Forecast")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    fc_table = pd.DataFrame({"Year": forecast_years, "Projected %": forecast.round(1)})
    st.dataframe(fc_table, width='stretch')

st.sidebar.markdown("---")
st.sidebar.caption("Built on NCRB ADSI data, 2015–2024. All figures verified against source PDFs.")
