"""
A Decade of Distress — Occupational Suicide Trends in India, 2015-2024
Streamlit dashboard companion to the analysis notebook.

Run locally with:  streamlit run app.py
Expects NCRB_Suicide_Data_2015_2024.xlsx in the same folder as this script.
"""

import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

DATA_FILE = "NCRB_Suicide_Data_2015_2024.xlsx"


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    xl = pd.ExcelFile(DATA_FILE)

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
    import requests

    if not os.path.exists(GEOJSON_LOCAL_CACHE):
        r = requests.get(GEOJSON_URL, timeout=30)
        with open(GEOJSON_LOCAL_CACHE, "wb") as f:
            f.write(r.content)

    india_districts = gpd.read_file(GEOJSON_LOCAL_CACHE)
    states_geo = india_districts.dissolve(by="st_nm", as_index=False)[["st_nm", "geometry"]]
    return states_geo


if not os.path.exists(DATA_FILE):
    st.error(f"Could not find {DATA_FILE}. Make sure it's in the same folder as app.py.")
    st.stop()

national, prof, causes, counts, rates = load_data()

# ----------------------------------------------------------------------
# Left-pane navigation
# ----------------------------------------------------------------------
st.sidebar.title("A Decade of Distress")
st.sidebar.caption("Suicide trends in India, 2015–2024")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Who is affected: Occupations", "Why it happens: Causes", "Where: State Map", "What's next: Forecast"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Source: National Crime Records Bureau (NCRB), Government of India — annual reports, 2015–2024.")

dwe = prof[prof["Profession Category"] == "Daily Wage Earner"].iloc[0]
dwe_change = dwe["2024 (%)"] - dwe["2015 (%)"]

# ----------------------------------------------------------------------
# PAGE — Overview
# ----------------------------------------------------------------------
if page == "Overview":
    st.title("A Decade of Distress")
    st.subheader("How suicide in India has changed, 2015–2024")
    st.write(
        "This dashboard looks at official government data on suicide in India over the last "
        "10 years, and focuses on one striking pattern: **who** is affected has been changing."
    )

    nat_2015 = national[national["Year"] == 2015].iloc[0]
    nat_2024 = national[national["Year"] == 2024].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total suicides in 2024", f"{nat_2024['Total Suicides']:,.0f}",
               f"+{nat_2024['Total Suicides'] - nat_2015['Total Suicides']:,.0f} vs. 2015")
    c2.metric("National suicide rate (per 1 lakh people)", f"{nat_2024['Suicide Rate (per lakh)']:.1f}",
               f"+{nat_2024['Suicide Rate (per lakh)'] - nat_2015['Suicide Rate (per lakh)']:.1f} vs. 2015")
    c3.metric("Share who were daily-wage workers", f"{dwe['2024 (%)']:.0f}%",
               f"+{dwe_change:.0f} points vs. 2015")

    st.markdown("### The key finding, in one sentence")
    st.info(
        f"In 2015, about **1 in 6** suicide victims in India was a daily-wage worker. "
        f"By 2024, it was nearly **1 in 3** — the single biggest shift in who is affected "
        f"by suicide in India over the last decade."
    )

    st.markdown("### National trend over time")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(national["Year"], national["Suicide Rate (per lakh)"], marker="o", color="firebrick")
    ax.set_xlabel("Year")
    ax.set_ylabel("Suicides per 1 lakh (100,000) people")
    ax.set_title("India's overall suicide rate, 2015–2024")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    st.caption(
        "The overall rate has risen over the decade, but — as the next pages show — "
        "this rise is not spread evenly. It's concentrated among specific groups and specific states."
    )

# ----------------------------------------------------------------------
# PAGE — Occupations
# ----------------------------------------------------------------------
elif page == "Who is affected: Occupations":
    st.title("Who is affected: Occupation")
    st.write(
        "Every suicide recorded by police in India is tagged with the victim's occupation. "
        "Here's how the mix of occupations has shifted over the decade."
    )

    prof_display = prof.set_index("Profession Category")[YEAR_PCT_COLS].copy()
    prof_display.columns = YEARS
    change_2015_2024 = (prof_display[2024] - prof_display[2015]).sort_values(ascending=False)

    st.markdown("### Which groups grew, and which shrank?")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["firebrick" if v > 0 else "steelblue" for v in change_2015_2024.values]
    ax.barh(change_2015_2024.index, change_2015_2024.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Change in share of all suicides, 2015 → 2024 (percentage points)")
    ax.set_title("Change by occupation group, 2015–2024")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(
        "Red bars = this group makes up a bigger share of suicides now than in 2015. "
        "Blue bars = a smaller share. Daily Wage Earner shows the largest increase by far."
    )

    st.markdown("### The full decade, year by year")
    highlight = st.selectbox("Highlight a group", prof["Profession Category"].tolist(),
                              index=prof["Profession Category"].tolist().index("Daily Wage Earner"))
    fig, ax = plt.subplots(figsize=(10, 5))
    for _, row in prof.iterrows():
        cat = row["Profession Category"]
        is_highlight = cat == highlight
        ax.plot(YEARS, row[YEAR_PCT_COLS].astype(float),
                marker="o",
                linewidth=3.5 if is_highlight else 1,
                alpha=1.0 if is_highlight else 0.25,
                color="firebrick" if is_highlight else "grey",
                label=cat if is_highlight else None)
    ax.set_xlabel("Year")
    ax.set_ylabel("% of all suicides")
    ax.set_title(f"{highlight}: share of all suicides, 2015–2024")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# ----------------------------------------------------------------------
# PAGE — Causes
# ----------------------------------------------------------------------
elif page == "Why it happens: Causes":
    st.title("Why it happens: Stated causes")
    st.write(
        "Alongside occupation, police records also note the stated reason for each suicide "
        "(family problems, financial trouble, illness, and so on)."
    )

    causes_display = causes.set_index("Cause")[YEAR_PCT_COLS].copy()
    causes_display.columns = YEARS
    top6 = causes_display.loc[causes_display[2024].sort_values(ascending=False).head(6).index]

    st.markdown("### Most common stated causes")
    fig, ax = plt.subplots(figsize=(9, 5))
    for cause, row in top6.iterrows():
        ax.plot(YEARS, row.values, marker="o", label=cause)
    ax.set_xlabel("Year")
    ax.set_ylabel("% of all suicides")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    st.caption("'Family Problems' and 'Illness' remain the two most commonly recorded reasons throughout the decade.")

    st.markdown("---")
    st.markdown("### Does the *reason* change alongside the *occupation* shift?")
    st.write(
        "If more people affected by suicide are daily-wage workers, does the *reason* given "
        "also shift toward money-related causes (debt, addiction) rather than family-related ones? "
        "Here's what a decade of data actually shows."
    )

    st.caption(
        "Note: both the occupation trend and the cause trends have been steadily rising for 10 years, "
        "so a simple side-by-side comparison can be misleading — two unrelated things that are both "
        "just generally increasing will look 'related' even if they aren't. The comparison below removes "
        "that effect to show the real year-to-year relationship."
    )

    prof_t = prof.set_index("Profession Category")[YEAR_PCT_COLS].T
    prof_t.index = YEARS
    causes_t = causes.set_index("Cause")[YEAR_PCT_COLS].T
    causes_t.index = YEARS
    prof_diff = prof_t.diff()
    causes_diff = causes_t.diff()

    occ_choice = st.selectbox("Choose an occupation group to check", prof["Profession Category"].tolist(),
                               index=prof["Profession Category"].tolist().index("Daily Wage Earner"))
    link = causes_diff.corrwith(prof_diff[occ_choice]).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["firebrick" if v > 0 else "steelblue" for v in link.values]
    ax.barh(link.index, link.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Strength of year-to-year link (-1 = opposite, +1 = moves together)")
    ax.set_title(f"Which causes move together with '{occ_choice}'?")
    plt.tight_layout()
    st.pyplot(fig)

    top_cause = link.index[0]
    top_val = link.iloc[0]
    st.info(
        f"For **{occ_choice}**, the strongest year-to-year link is with **{top_cause}** "
        f"(strength: {top_val:.2f} out of 1.0). This is a real pattern in the data, but with "
        f"only 10 years to compare, it should be treated as a hint worth investigating further, "
        f"not a proven fact."
    )

# ----------------------------------------------------------------------
# PAGE — State Map
# ----------------------------------------------------------------------
elif page == "Where: State Map":
    st.title("Where: State-by-state view")
    st.write("Suicide rates vary widely across India's states. Explore how each state has changed.")

    try:
        states_geo = load_geo()
        rates_geo_ready = rates.copy()
        rates_geo_ready["geo_name"] = rates_geo_ready["State/UT"].map(lambda s: NAME_MAP.get(s, s))
        merged = states_geo.merge(rates_geo_ready, left_on="st_nm", right_on="geo_name", how="left")

        map_year = st.select_slider("Select a year", options=YEARS, value=2024)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Suicide rate by state — {map_year}**")
            fig, ax = plt.subplots(figsize=(6, 6))
            merged.plot(column=map_year, cmap="OrRd", legend=True, ax=ax,
                        edgecolor="black", linewidth=0.3, missing_kwds={"color": "lightgrey"})
            ax.axis("off")
            st.pyplot(fig)
            st.caption("Darker = higher suicide rate. Grey = no data for that year.")

        with col2:
            st.markdown("**Where has it gotten better or worse since 2015?**")
            merged["change"] = merged[2024] - merged[2015]
            vmax = merged["change"].abs().max()
            fig, ax = plt.subplots(figsize=(6, 6))
            merged.plot(column="change", cmap="RdBu_r", legend=True, ax=ax,
                        edgecolor="black", linewidth=0.3, missing_kwds={"color": "lightgrey"},
                        vmin=-vmax, vmax=vmax)
            ax.axis("off")
            st.pyplot(fig)
            st.caption("Red = got worse since 2015. Blue = improved since 2015.")
    except Exception as e:
        st.warning("The map couldn't load in this environment, but the rankings below still work.")

    st.markdown("---")
    st.markdown("### States with the biggest changes")
    traj = rates.set_index("State/UT")[YEARS].dropna()
    ranked = pd.DataFrame({
        "2015": traj[2015],
        "2024": traj[2024],
        "Change": traj[2024] - traj[2015],
    }).sort_values("Change", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Got worse the most**")
        st.dataframe(ranked.head(5).round(1), width='stretch')
    with col2:
        st.markdown("**Improved the most**")
        st.dataframe(ranked.tail(5).sort_values("Change").round(1), width='stretch')

    st.caption(
        "A few small states/UTs (e.g. Meghalaya, Mizoram, Andaman & Nicobar Islands) show unusually "
        "large swings in some years — this has been checked against the original government reports "
        "and is a real feature of the official data for those small-population areas, not an error."
    )

# ----------------------------------------------------------------------
# PAGE — Forecast
# ----------------------------------------------------------------------
elif page == "What's next: Forecast":
    st.title("If the trend continues")
    st.write(
        "Based only on the pattern of the last 10 years, here's a simple projection of where "
        "a given trend might go next — **not** a guarantee, just an illustration of momentum."
    )

    category = st.selectbox("Choose an occupation group", prof["Profession Category"].tolist(),
                             index=prof["Profession Category"].tolist().index("Daily Wage Earner"))
    horizon = st.slider("How many years ahead?", 1, 5, 3)

    series = prof[prof["Profession Category"] == category][YEAR_PCT_COLS].values.flatten().astype(float)
    model = ExponentialSmoothing(series, trend="add", initialization_method="estimated").fit()
    forecast = model.forecast(horizon)
    forecast_years = list(range(YEARS[-1] + 1, YEARS[-1] + 1 + horizon))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(YEARS, series, marker="o", label="Actual (2015–2024)")
    ax.plot(forecast_years, forecast, marker="o", linestyle="--", color="red",
             label=f"Projected ({forecast_years[0]}–{forecast_years[-1]})")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of all suicides")
    ax.set_title(f"{category}: actual trend + simple projection")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.caption(
        "This projection is based on only 10 years of data, so treat it as a rough sketch of "
        "'where things are headed if nothing changes' — not a precise prediction."
    )
