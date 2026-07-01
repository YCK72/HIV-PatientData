import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load & standardize ──────────────────────────────────────────────────
df = pd.read_csv('../Data/Cleaned/cleaned_data.csv')

group_mapping = {
    "HIV - Pain": "HIV - Pain",
    "HIV-PAIN": "HIV - Pain",
    "HIV - No Pain": "HIV - No Pain",
    "HIV- No Pain": "HIV - No Pain",
    "No HIV - Pain": "No HIV - Pain",
    "No HIV - No Pain": "No HIV - No Pain"
}
df["Group"] = df["Group"].map(group_mapping)

output_dir = '../docs/graphs'
os.makedirs(output_dir, exist_ok=True)

GROUP_COLORS = {
    "HIV - Pain": "#e74c3c",
    "HIV - No Pain": "#e67e22",
    "No HIV - Pain": "#3498db",
    "No HIV - No Pain": "#2ecc71"
}

# ── 2. Helper: save figure ─────────────────────────────────────────────────
def save_fig(fig, filename):
    path = os.path.join(output_dir, filename)
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"Saved: {filename}")

# ── 3. CORRELATION 1: Pain Severity vs Mental Health ──────────────────────
mental_health_cols = [
    "Major Depression Now", "Anxiety Disorder Now",
    "Bipolar Disorder Now", "Other Mental Health Conditions Now"
]

rows = []
for col in mental_health_cols:
    for group in df["Group"].unique():
        g = df[df["Group"] == group].dropna(subset=[col, "Bodily Pain Rating in Past Week"])
        if len(g) < 5:
            continue
        r, p = stats.pointbiserialr(g[col], g["Bodily Pain Rating in Past Week"])
        rows.append({
            "Condition": col.replace(" Now", ""),
            "Group": group,
            "Correlation (r)": round(r, 3),
            "P-Value": round(p, 4),
            "Significant": "Yes" if p < 0.05 else "No"
        })

corr1_df = pd.DataFrame(rows)

fig1 = px.bar(
    corr1_df, x="Condition", y="Correlation (r)",
    color="Group", barmode="group",
    color_discrete_map=GROUP_COLORS,
    pattern_shape="Significant", pattern_shape_map={"Yes": "", "No": "/"},
    title="Correlation 1: Pain Severity vs Mental Health Conditions by Group",
    labels={"Correlation (r)": "Point-Biserial r"},
    hover_data=["P-Value", "Significant"]
)
fig1.add_hline(y=0, line_dash="dash", line_color="black")
fig1.update_layout(xaxis_tickangle=-20)
save_fig(fig1, "01_pain_vs_mental_health.html")

# ── 4. CORRELATION 2: Chronic Pain vs Mental Health ───────────────────────
chronic_cols = ["Pain > 3 Months", "Pain >12hrs or More 6 months"]
rows = []
for pain_col in chronic_cols:
    for mh_col in mental_health_cols:
        for group in df["Group"].unique():
            g = df[df["Group"] == group].dropna(subset=[pain_col, mh_col])
            if len(g) < 5:
                continue
            r, p = stats.pointbiserialr(g[mh_col], g[pain_col])
            rows.append({
                "Pain Duration": pain_col,
                "MH Condition": mh_col.replace(" Now", ""),
                "Group": group,
                "Correlation (r)": round(r, 3),
                "P-Value": round(p, 4),
                "Significant": "Yes" if p < 0.05 else "No"
            })

corr2_df = pd.DataFrame(rows)

fig2 = px.bar(
    corr2_df, x="MH Condition", y="Correlation (r)",
    color="Group", barmode="group", facet_col="Pain Duration",
    color_discrete_map=GROUP_COLORS,
    title="Correlation 2: Chronic Pain Duration vs Mental Health Conditions",
    hover_data=["P-Value", "Significant"]
)
fig2.add_hline(y=0, line_dash="dash", line_color="black")
save_fig(fig2, "02_chronic_pain_vs_mental_health.html")

# ── 5. CORRELATION 3: ART Adherence vs Pain ───────────────────────────────
rows = []
hiv_df = df[df["Group"].isin(["HIV - Pain", "HIV - No Pain"])].dropna(
    subset=["Doses ART missed?", "Bodily Pain Rating in Past Week"]
)
r, p = stats.spearmanr(hiv_df["Doses ART missed?"], hiv_df["Bodily Pain Rating in Past Week"])
rows.append({"Comparison": "ART Missed vs Pain Rating", "r": round(r, 3), "p": round(p, 4)})

for col in ["Pain > 3 Months", "Pain >12hrs or More 6 months"]:
    g = hiv_df.dropna(subset=[col])
    r, p = stats.pointbiserialr(g[col], g["Doses ART missed?"])
    rows.append({"Comparison": f"ART Missed vs {col}", "r": round(r, 3), "p": round(p, 4)})

corr3_df = pd.DataFrame(rows)

fig3 = px.bar(
    corr3_df, x="Comparison", y="r",
    title="Correlation 3: ART Adherence vs Pain (HIV Patients Only)",
    labels={"r": "Spearman/Point-Biserial r"},
    color="Comparison", text="p"
)
fig3.update_traces(texttemplate="p=%{text}", textposition="outside")
fig3.add_hline(y=0, line_dash="dash", line_color="black")
save_fig(fig3, "03_art_adherence_vs_pain.html")

# ── 6. CORRELATION 4: ART Drug Class vs Peripheral Neuropathy ─────────────
art_cols = [
    "ART?_Nucleoside reverse transcriptase inhibitors (NRTIs)",
    "ART?_Integrase Strand Transfer Inhibitor (INSTI)",
    "ART?_Non-nucleoside reverse transcriptase inhibitor (NNRTI)",
    "ART?_Protease Inhibitor (PI)"
]
art_labels = ["NRTIs", "INSTI", "NNRTI", "PI"]

rows = []
hiv_df2 = df[df["Group"].isin(["HIV - Pain", "HIV - No Pain"])]
for col, label in zip(art_cols, art_labels):
    g = hiv_df2.dropna(subset=[col, "Peripheral Neuropathy Now"])
    if len(g) < 5:
        continue
    chi2, p, _, _ = stats.chi2_contingency(
        pd.crosstab(g[col], g["Peripheral Neuropathy Now"])
    )
    rows.append({
        "ART Class": label,
        "Chi2": round(chi2, 3),
        "P-Value": round(p, 4),
        "Significant": "Yes" if p < 0.05 else "No"
    })

corr4_df = pd.DataFrame(rows)

fig4 = px.bar(
    corr4_df, x="ART Class", y="Chi2",
    color="Significant",
    color_discrete_map={"Yes": "#e74c3c", "No": "#95a5a6"},
    title="Correlation 4: ART Drug Class vs Peripheral Neuropathy (Chi-Square)",
    hover_data=["P-Value"]
)
save_fig(fig4, "04_art_class_vs_neuropathy.html")

# ── 7. CORRELATION 5: Comorbidity Clustering (Metabolic Syndrome) ─────────
metabolic_pairs = [
    ("Weight", "Diabetes Now"),
    ("Weight", "HBP Now"),
    ("HBP Now", "Heart Disease Now"),
    ("Diabetes Now", "Kidney Disease Now")
]

rows = []
for col_a, col_b in metabolic_pairs:
    for group in df["Group"].unique():
        g = df[df["Group"] == group].dropna(subset=[col_a, col_b])
        if len(g) < 5:
            continue
        if df[col_a].nunique() > 2:
            r, p = stats.pointbiserialr(g[col_b], g[col_a])
        else:
            chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(g[col_a], g[col_b]))
            r = chi2
        rows.append({
            "Pair": f"{col_a} vs {col_b}",
            "Group": group,
            "Statistic": round(r, 3),
            "P-Value": round(p, 4),
            "Significant": "Yes" if p < 0.05 else "No"
        })

corr5_df = pd.DataFrame(rows)

fig5 = px.bar(
    corr5_df, x="Pair", y="Statistic",
    color="Group", barmode="group",
    color_discrete_map=GROUP_COLORS,
    title="Correlation 5: Metabolic Comorbidity Clustering by Group",
    hover_data=["P-Value", "Significant"]
)
fig5.update_layout(xaxis_tickangle=-20)
save_fig(fig5, "05_metabolic_clustering.html")

# ── 8. CORRELATION 6: Age vs Comorbidity Burden ───────────────────────────
comorbidity_now_cols = [
    c for c in df.columns
    if c.endswith("Now") and c not in ["Asthma/Breathing Problems  Now"]
]

df["Comorbidity_Count"] = df[comorbidity_now_cols].sum(axis=1)

fig6 = px.scatter(
    df, x="Age", y="Comorbidity_Count",
    color="Group", trendline="ols",
    color_discrete_map=GROUP_COLORS,
    title="Correlation 6: Age vs Total Comorbidity Count by Group",
    labels={"Comorbidity_Count": "Number of Current Conditions"}
)
save_fig(fig6, "06_age_vs_comorbidity.html")

rows = []
for group in df["Group"].unique():
    g = df[df["Group"] == group].dropna(subset=["Age", "Comorbidity_Count"])
    r, p = stats.spearmanr(g["Age"], g["Comorbidity_Count"])
    rows.append({"Group": group, "Spearman r": round(r, 3), "P-Value": round(p, 4)})

print("\nCorrelation 6 - Age vs Comorbidity:")
print(pd.DataFrame(rows))

# ── 9. CORRELATION 7: Age vs Pain Severity ────────────────────────────────
fig7 = px.scatter(
    df, x="Age", y="Bodily Pain Rating in Past Week",
    color="Group", trendline="ols",
    color_discrete_map=GROUP_COLORS,
    title="Correlation 7: Age vs Pain Severity by Group"
)
save_fig(fig7, "07_age_vs_pain.html")

# ── 10. CORRELATION 8: Smoking vs Pain Severity ───────────────────────────
rows = []
for group in df["Group"].unique():
    g = df[df["Group"] == group].dropna(
        subset=["Smoke_Cigarettes", "Bodily Pain Rating in Past Week"]
    )
    if len(g) < 5:
        continue
    r, p = stats.pointbiserialr(g["Smoke_Cigarettes"], g["Bodily Pain Rating in Past Week"])
    rows.append({
        "Group": group, "Smoke Type": "Cigarettes",
        "r": round(r, 3), "P-Value": round(p, 4)
    })
    g2 = df[df["Group"] == group].dropna(
        subset=["Smoke_E-Cigarettes", "Bodily Pain Rating in Past Week"]
    )
    if len(g2) < 5:
        continue
    r2, p2 = stats.pointbiserialr(g2["Smoke_E-Cigarettes"], g2["Bodily Pain Rating in Past Week"])
    rows.append({
        "Group": group, "Smoke Type": "E-Cigarettes",
        "r": round(r2, 3), "P-Value": round(p2, 4)
    })

corr8_df = pd.DataFrame(rows)

fig8 = px.bar(
    corr8_df, x="Group", y="r",
    color="Smoke Type", barmode="group",
    title="Correlation 8: Smoking vs Pain Severity by Group",
    hover_data=["P-Value"]
)
fig8.add_hline(y=0, line_dash="dash", line_color="black")
save_fig(fig8, "08_smoking_vs_pain.html")

# ── 11. CORRELATION 9: Sleep Condition vs Pain & Mental Health ────────────
sleep_targets = ["Bodily Pain Rating in Past Week"] + mental_health_cols

rows = []
for target in sleep_targets:
    for group in df["Group"].unique():
        g = df[df["Group"] == group].dropna(subset=["Sleep Condition Now", target])
        if len(g) < 5:
            continue
        r, p = stats.pointbiserialr(g["Sleep Condition Now"], g[target])
        rows.append({
            "Target": target.replace(" Now", ""),
            "Group": group,
            "r": round(r, 3),
            "P-Value": round(p, 4),
            "Significant": "Yes" if p < 0.05 else "No"
        })

corr9_df = pd.DataFrame(rows)

fig9 = px.bar(
    corr9_df, x="Target", y="r",
    color="Group", barmode="group",
    color_discrete_map=GROUP_COLORS,
    title="Correlation 9: Sleep Condition vs Pain & Mental Health",
    hover_data=["P-Value", "Significant"]
)
fig9.add_hline(y=0, line_dash="dash", line_color="black")
fig9.update_layout(xaxis_tickangle=-20)
save_fig(fig9, "09_sleep_vs_pain_mentalhealth.html")

# ── 12. CORRELATION 10: Polypharmacy vs Pain & Hospitalization ────────────
med_class_cols = [c for c in df.columns if c.startswith("Medication Class_")]
df["Med_Count"] = df[med_class_cols].sum(axis=1)

fig10a = px.scatter(
    df, x="Med_Count", y="Bodily Pain Rating in Past Week",
    color="Group", trendline="ols",
    color_discrete_map=GROUP_COLORS,
    title="Correlation 10a: Number of Medication Classes vs Pain Severity"
)
save_fig(fig10a, "10a_polypharmacy_vs_pain.html")

fig10b = px.box(
    df, x="Hospitalized (Past Year)", y="Med_Count",
    color="Group",
    color_discrete_map=GROUP_COLORS,
    title="Correlation 10b: Hospitalization vs Number of Medication Classes"
)
save_fig(fig10b, "10b_polypharmacy_vs_hospitalization.html")

# ── 13. CORRELATION 11: Pain Region vs Peripheral Neuropathy ──────────────
pain_region_cols = [
    "Parts That Tend to Be Most Pain (0: No | 1: Yes | NA: Not Disclosed)_Feet/Ankle",
    "Parts That Tend to Be Most Pain (0: No | 1: Yes | NA: Not Disclosed)_Thigh/Knee/Leg Pain"
]

rows = []
for col in pain_region_cols:
    for group in df["Group"].unique():
        g = df[df["Group"] == group].dropna(subset=[col, "Peripheral Neuropathy Now"])
        if len(g) < 5:
            continue
        chi2, p, _, _ = stats.chi2_contingency(
            pd.crosstab(g[col], g["Peripheral Neuropathy Now"])
        )
        rows.append({
            "Pain Region": col.split("_")[-1],
            "Group": group,
            "Chi2": round(chi2, 3),
            "P-Value": round(p, 4),
            "Significant": "Yes" if p < 0.05 else "No"
        })

corr11_df = pd.DataFrame(rows)

fig11 = px.bar(
    corr11_df, x="Pain Region", y="Chi2",
    color="Group", barmode="group",
    color_discrete_map=GROUP_COLORS,
    title="Correlation 11: Lower Limb Pain Region vs Peripheral Neuropathy",
    hover_data=["P-Value", "Significant"]
)
save_fig(fig11, "11_pain_region_vs_neuropathy.html")

# ── 14. CORRELATION 12: Group-level comorbidity prevalence overview ────────
comorbidity_cols = [
    "HBP Now", "Heart Disease Now", "Diabetes Now", "Cancer Now",
    "Kidney Disease Now", "Sleep Condition Now", "Major Depression Now",
    "Anxiety Disorder Now", "Peripheral Neuropathy Now", "Stroke Now"
]

prev_rows = []
for col in comorbidity_cols:
    for group in df["Group"].unique():
        g = df[df["Group"] == group]
        prevalence = g[col].mean() * 100
        prev_rows.append({
            "Condition": col.replace(" Now", ""),
            "Group": group,
            "Prevalence (%)": round(prevalence, 1)
        })

prev_df = pd.DataFrame(prev_rows)

fig12 = px.bar(
    prev_df, x="Condition", y="Prevalence (%)",
    color="Group", barmode="group",
    color_discrete_map=GROUP_COLORS,
    title="Correlation 12: Comorbidity Prevalence Across All Four Groups (%)",
)
fig12.update_layout(xaxis_tickangle=-30)
save_fig(fig12, "12_comorbidity_prevalence_by_group.html")

print("\nAll graphs saved to ../web/graphs/")
print("Run the website next.")