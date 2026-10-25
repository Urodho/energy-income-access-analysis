from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Renewables, Income and Electricity Access", layout="wide")

DATA = Path(__file__).parent / "data" / "processed"
ORDER = ["Low", "Middle", "High"]


@st.cache_data
def load_data():
    joined = pd.read_csv(DATA / "energy_income_joined.csv")
    latest = pd.read_csv(DATA / "country_latest.csv")
    results = pd.read_csv(DATA / "hypothesis_results.csv")
    return joined, latest, results


joined, latest, results = load_data()

# ---------- Title and introduction ----------
st.title("Renewable Electricity, Income and Access to Electricity")
st.caption("Evidence from countries worldwide, with Kenya as a case study")

st.header("The problem")
st.write(
    "Countries must expand electricity that is affordable, reliable and clean. "
    "It is not clear whether a higher renewable share goes together with higher "
    "income or better access. **Research question:** across countries and years, "
    "is a higher share of electricity from renewables associated with higher GDP "
    "per capita and higher access to electricity?"
)

st.header("Data sources")
st.write(
    "- **Our World in Data (energy dataset):** renewables share of electricity.\n"
    "- **World Bank World Development Indicators:** GDP per capita and access to electricity.\n"
    "- The two were joined on ISO country code and year."
)

# ---------- Key figures ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Joined rows", f"{len(joined):,}")
c2.metric("Countries", joined["iso_code"].nunique())
c3.metric("Years covered", f"{joined['year'].min()} to {joined['year'].max()}")
c4.metric("Countries used in tests", len(latest))

# ---------- Interactive exploration ----------
st.header("Explore the data")
st.sidebar.header("Filters")
ymin, ymax = int(joined["year"].min()), int(joined["year"].max())
years = st.sidebar.slider("Years", ymin, ymax, (1990, ymax))
groups = st.sidebar.multiselect("Income group", ORDER, default=ORDER)

view = joined[joined["year"].between(years[0], years[1]) & joined["income_group"].isin(groups)]
st.write(f"Showing {len(view):,} country-year rows.")

left, right = st.columns(2)

with left:
    d = view.dropna(subset=["log_gdp_per_capita", "renewables_share_elec"])
    if d.empty:
        st.info("No data for these filters.")
    else:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.scatterplot(data=d, x="log_gdp_per_capita", y="renewables_share_elec",
                        hue="income_group", hue_order=ORDER, alpha=0.4, s=15, ax=ax)
        ax.set_title("GDP per Capita vs Renewables Share of Electricity")
        ax.set_xlabel("Log of GDP per capita (higher = richer)")
        ax.set_ylabel("Renewables share of electricity (%)")
        st.pyplot(fig)

with right:
    d2 = view.dropna(subset=["high_renewables"])
    share = (d2.groupby("income_group")["high_renewables"].mean() * 100).reindex(ORDER).dropna()
    if share.empty:
        st.info("No data for these filters.")
    else:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(x=share.index, y=share.values, order=list(share.index), ax=ax)
        ax.set_title("Country-Years with High Renewables (50%+), by Income Group")
        ax.set_xlabel("Income group")
        ax.set_ylabel("Country-years with high renewables (%)")
        st.pyplot(fig)

# ---------- Statistical findings ----------
st.header("Main statistical findings")
cols = ["log_gdp_per_capita", "electricity_access", "renewables_share_elec"]
corr = joined[cols].corr(method="spearman").round(2)
fig, ax = plt.subplots(figsize=(6, 3.8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
ax.set_title("Spearman Correlations Between Key Variables")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
st.pyplot(fig)
st.write(
    "GDP per capita and electricity access are strongly linked. Renewables share has a "
    "weak negative link with both. Correlation does not prove that one thing causes another."
)

st.header("Three hypothesis tests (significance level 0.05)")
st.write(
    "These tests use one row per country: the latest year with all three measures. "
    "This avoids counting the same country many times."
)
show = results.copy()
show["statistic"] = show["statistic"].round(3)
show["p_value"] = show["p_value"].round(3)
st.dataframe(show, hide_index=True)

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(data=latest, x="income_group", y="renewables_share_elec", order=ORDER, ax=ax)
ken = latest[latest["iso_code"] == "KEN"]
if not ken.empty:
    k = ken.iloc[0]
    ax.scatter(ORDER.index(k["income_group"]), k["renewables_share_elec"],
               color="red", s=100, zorder=5, label="Kenya")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
ax.set_title("Renewables Share of Electricity by Income Group")
ax.set_xlabel("Income group")
ax.set_ylabel("Renewables share of electricity (%)")
st.pyplot(fig)

# ---------- Kenya ----------
st.header("Kenya as a case study")
if not ken.empty:
    k = ken.iloc[0]
    same = latest[latest["income_group"] == k["income_group"]]
    a, b, c = st.columns(3)
    a.metric("GDP per capita (constant 2015 US$)", f"{k['gdp_per_capita']:,.0f}")
    b.metric("Renewables share of electricity (%)", f"{k['renewables_share_elec']:.1f}",
             f"{k['renewables_share_elec'] - same['renewables_share_elec'].mean():+.1f} vs group average")
    c.metric("Access to electricity (%)", f"{k['electricity_access']:.1f}",
             f"{k['electricity_access'] - same['electricity_access'].mean():+.1f} vs group average")
    st.caption(f"Latest year with data: {int(k['year'])}. Income group: {k['income_group']}.")

# ---------- Conclusion, recommendations, limitations ----------
st.header("Plain-language conclusion")
st.write(
    "Richer countries tend to have much higher access to electricity. Across all "
    "country-years, richer countries tend to have a somewhat lower renewables share. "
    "But when each country is counted once, none of the three tests found a difference "
    "that is clearly more than chance. Income level alone is not a reliable guide to a "
    "country's renewable share."
)

st.header("Practical recommendations")
st.write(
    "- Do not plan energy investments from income level alone.\n"
    "- Treat access to electricity and renewable share as separate goals. Kenya shows "
    "that a high renewables share does not mean everyone has access.\n"
    "- Improve data coverage, because large parts of the data are missing."
)

st.header("Limitations")
st.write(
    "- Renewables share is missing in about 44% of joined rows, and electricity access in "
    "about 42%, so each analysis uses fewer rows than the 10,926 in the joined table.\n"
    "- The three income groups are my own thirds of GDP per capita, not official classes.\n"
    "- The 50% and 99% thresholds were my choice.\n"
    "- The groups in the A/B comparison were not randomly assigned, so no cause is proven.\n"
    "- Small places such as Monaco and Kosovo are not in the joined data.\n"
    "- With about 66 countries per group, the tests can only detect fairly large differences."
)