import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

# ── 1. Paths (work regardless of which directory you run from) ─────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(script_dir, '..', 'Data', 'Cleaned', 'cleaned_data.csv')
output_dir = os.path.join(script_dir, '..', 'docs', 'graphs')
os.makedirs(output_dir, exist_ok=True)

# ── 2. Load & standardize ──────────────────────────────────────────────────
df = pd.read_csv(data_path)

group_mapping = {
    "HIV - Pain":        "HIV - Pain",
    "HIV-PAIN":          "HIV - Pain",
    "HIV - No Pain":     "HIV - No Pain",
    "HIV- No Pain":      "HIV - No Pain",
    "No HIV - Pain":     "No HIV - Pain",
    "No HIV - No Pain":  "No HIV - No Pain"
}
df["Group"] = df["Group"].map(group_mapping)

GROUP_COLORS = {
    "HIV - Pain":        "#e74c3c",
    "HIV - No Pain":     "#e67e22",
    "No HIV - Pain":     "#3498db",
    "No HIV - No Pain":  "#2ecc71"
}

print(f"Loaded {len(df)} patients · groups: {df['Group'].value_counts().to_dict()}")

# ── 3. Helpers ─────────────────────────────────────────────────────────────
def save_fig(fig, filename):
    path = os.path.join(output_dir, filename)
    fig.write_html(path, include_plotlyjs='cdn')
    print(f"  Saved: {filename}")

def mobile_layout(fig, height=500):
    fig.update_layout(
        autosize=True,
        height=height,
        margin=dict(l=10, r=10, t=60, b=130),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.38,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            itemwidth=90
        ),
        font=dict(size=11),
        title_font=dict(size=13),
        xaxis=dict(tickfont=dict(size=9), title_font=dict(size=10), tickangle=-30),
        yaxis=dict(tickfont=dict(size=9), title_font=dict(size=10))
    )
    return fig

# ── 4. CORRELATION 1: Pain Severity vs Mental Health ──────────────────────
mental_health_cols = [
    "Major Depression Now", "Anxiety Disorder Now",
    "Bipolar Disorder Now", "Other Mental Health Conditions Now"
]

rows = []
for col in mental_health_cols:
    for group in df["Group"].unique():
        g = df[df["Group"] == group].dropna(
            subset=[col, "Bodily Pain Rating in Past Week"])
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

fig1 = px.bar(
    pd.DataFrame(rows), x="Condition", y="Correlation (r)",
    color="Group", barmode="group",
    color_discrete_map=GROUP_COLORS,
    title="Pain Severity vs Mental Health Conditions",
    labels={"Correlation (r)": "Point-Biserial r"},
    hover_data=["P-Value", "Significant"]
)
fig1.add_hline(y=0, line_dash="dash", line_color="black")
fig1 = mobile_layout(fig1, height=500)
save_fig(fig1, "01_pain_vs_mental_health.html")

# ── 5. CORRELATION 2: Chronic Pain Duration vs Mental Health ──────────────
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
                "Pain Duration": "Pain > 3 Months" if "3 Months" in pain_col
                                  else "Pain > 12hrs/day",
                "MH Condition": mh_col.replace(" Now", ""),
                "Group": group,
                "Correlation (r)": round(r, 3),
                "P-Value": round(p, 4),
                "Significant": "Yes" if p < 0.05 else "No"
            })

fig2 = px.bar(
    pd.DataFrame(rows), x="MH Condition", y="Correlation (r)",
    color="Group", barmode="group",
    facet_row="Pain Duration",
    color_discrete_map=GROUP_COLORS,
    title="Chronic Pain Duration vs Mental Health Conditions",
    hover_data=["P-Value", "Significant"],
    height=750
)
fig2.add_hline(y=0, line_dash="dash", line_color="black")
fig2.update_layout(
    autosize=True,
    height=750,
    margin=dict(l=10, r=10, t=60, b=150),
    legend=dict(
        orientation="h", yanchor="bottom", y=-0.22,
        xanchor="center", x=0.5, font=dict(size=10)
    ),
    font=dict(size=11),
    title_font=dict(size=13)
)
fig2.for_each_xaxis(lambda ax: ax.update(tickangle=-30, tickfont=dict(size=9)))
fig2.for_each_yaxis(lambda ax: ax.update(tickfont=dict(size=9)))
save_fig(fig2, "02_chronic_pain_vs_mental_health.html")

# ── 6. CORRELATION 3: ART Adherence vs Pain ───────────────────────────────
rows = []
hiv_df = df[df["Group"].isin(["HIV - Pain", "HIV - No Pain"])].dropna(
    subset=["Doses ART missed?", "Bodily Pain Rating in Past Week"])

r, p = stats.spearmanr(hiv_df["Doses ART missed?"],
                        hiv_df["Bodily Pain Rating in Past Week"])
rows.append({"Comparison": "ART Missed\nvs Pain Rating",
             "r": round(r, 3), "p": round(p, 4)})

for col in ["Pain > 3 Months", "Pain >12hrs or More 6 months"]:
    g = hiv_df.dropna(subset=[col])
    r, p = stats.pointbiserialr(g[col], g["Doses ART missed?"])
    label = "ART Missed\nvs Pain >3mo" if "3 Months" in col \
        else "ART Missed\nvs Pain >12hrs"
    rows.append({"Comparison": label, "r": round(r, 3), "p": round(p, 4)})

fig3 = px.bar(
    pd.DataFrame(rows), x="Comparison", y="r",
    title="ART Adherence vs Pain (HIV Patients Only)",
    labels={"r": "Spearman / Point-Biserial r"},
    color="Comparison", text="p"
)
fig3.update_traces(texttemplate="p=%{text}", textposition="outside")
fig3.add_hline(y=0, line_dash="dash", line_color="black")
fig3 = mobile_layout(fig3, height=480)
save_fig(fig3, "03_art_adherence_vs_pain.html")

# ── 7. CORRELATION 4: ART Drug Class vs Peripheral Neuropathy ─────────────
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
        pd.crosstab(g[col], g["Peripheral Neuropathy Now"]))
    rows.append({
        "ART Class": label,
        "Chi2": round(chi2, 3),
        "P-Value": round(p, 4),
        "Significant": "Yes" if p < 0.05 else "No"
    })

fig4 = px.bar(
    pd.DataFrame(rows), x="ART Class", y="Chi2",
    color="Significant",
    color_discrete_map={"Yes": "#e74c3c", "No": "#95a5a6"},
    title="ART Drug Class vs Peripheral Neuropathy (Chi-Square)",
    hover_data=["P-Value"]
)
fig4 = mobile_layout(fig4, height=450)
save_fig(fig4, "04_art_class_vs_neuropathy.html")

# ── 8. CORRELATION 5: Metabolic Comorbidity Clustering ────────────────────
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
            chi2, p, _, _ = stats.chi2_contingency(
                pd.crosstab(g[col_a], g[col_b]))
            r = chi2
        rows.append({
            "Pair": f"{col_a}\nvs {col_b}",
            "Group": group,
            "Statistic": round(r, 3),
            "P-Value": round(p, 4),
            "Significant": "Yes" if p < 0.05 else "No"
        })

fig5 = px.bar(
    pd.DataFrame(rows), x="Pair", y="Statistic",
    color="Group", barmode="group",
    color_discrete_map=GROUP_COLORS,
    title="Metabolic Comorbidity Clustering by Group",
    hover_data=["P-Value", "Significant"]
)
fig5 = mobile_layout(fig5, height=500)
save_fig(fig5, "05_metabolic_clustering.html")

# ── 9. CORRELATION 6: Age vs Comorbidity Burden ───────────────────────────
comorbidity_now_cols = [
    c for c in df.columns
    if c.endswith("Now") and c != "Asthma/Breathing Problems  Now"
]
df["Comorbidity_Count"] = df[comorbidity_now_cols].sum(axis=1)

fig6 = px.scatter(
    df, x="Age", y="Comorbidity_Count",
    color="Group", trendline="ols",
    color_discrete_map=GROUP_COLORS,
    title="Age vs Total Comorbidity Count by Group",
    labels={"Comorbidity_Count": "No. of Current Conditions"}
)
fig6 = mobile_layout(fig6, height=500)
save_fig(fig6, "06_age_vs_comorbidity.html")

rows = []
for group in df["Group"].unique():
    g = df[df["Group"] == group].dropna(subset=["Age", "Comorbidity_Count"])
    r, p = stats.spearmanr(g["Age"], g["Comorbidity_Count"])
    rows.append({"Group": group, "Spearman r": round(r, 3), "P-Value": round(p, 4)})
print("\nCorrelation 6 - Age vs Comorbidity:")
print(pd.DataFrame(rows).to_string(index=False))

# ── 10. CORRELATION 7: Age vs Pain Severity ───────────────────────────────
fig7 = px.scatter(
    df, x="Age", y="Bodily Pain Rating in Past Week",
    color="Group", trendline="ols",
    color_discrete_map=GROUP_COLORS,
    title="Age vs Pain Severity by Group"
)
fig7 = mobile_layout(fig7, height=500)
save_fig(fig7, "07_age_vs_pain.html")

# ── 11. CORRELATION 8: Smoking vs Pain ───────────────────────────────────
rows = []
for group in df["Group"].unique():
    for smoke_col, label in [
        ("Smoke_Cigarettes", "Cigarettes"),
        ("Smoke_E-Cigarettes", "E-Cigarettes")
    ]:
        g = df[df["Group"] == group].dropna(
            subset=[smoke_col, "Bodily Pain Rating in Past Week"])
        if len(g) < 5:
            continue
        r, p = stats.pointbiserialr(
            g[smoke_col], g["Bodily Pain Rating in Past Week"])
        rows.append({
            "Group": group, "Smoke Type": label,
            "r": round(r, 3), "P-Value": round(p, 4)
        })

fig8 = px.bar(
    pd.DataFrame(rows), x="Group", y="r",
    color="Smoke Type", barmode="group",
    title="Smoking vs Pain Severity by Group",
    hover_data=["P-Value"]
)
fig8.add_hline(y=0, line_dash="dash", line_color="black")
fig8 = mobile_layout(fig8, height=500)
save_fig(fig8, "08_smoking_vs_pain.html")

# ── 12. CORRELATION 9: Sleep vs Pain & Mental Health ─────────────────────
sleep_targets = ["Bodily Pain Rating in Past Week"] + mental_health_cols

rows = []
for target in sleep_targets:
    for group in df["Group"].unique():
        g = df[df["Group"] == group].dropna(
            subset=["Sleep Condition Now", target])
        if len(g) < 5:
            continue
        r, p = stats.pointbiserialr(g["Sleep Condition Now"], g[target])
        rows.append({
            "Target": target.replace(" Now", "").replace(
                "Bodily Pain Rating in Past Week", "Pain Rating"),
            "Group": group,
            "r": round(r, 3),
            "P-Value": round(p, 4),
            "Significant": "Yes" if p < 0.05 else "No"
        })

fig9 = px.bar(
    pd.DataFrame(rows), x="Target", y="r",
    color="Group", barmode="group",
    color_discrete_map=GROUP_COLORS,
    title="Sleep Condition vs Pain & Mental Health",
    hover_data=["P-Value", "Significant"]
)
fig9.add_hline(y=0, line_dash="dash", line_color="black")
fig9 = mobile_layout(fig9, height=500)
save_fig(fig9, "09_sleep_vs_pain_mentalhealth.html")

# ── 13. CORRELATION 10: Polypharmacy ──────────────────────────────────────
med_class_cols = [c for c in df.columns if c.startswith("Medication Class_")]
df["Med_Count"] = df[med_class_cols].sum(axis=1)

fig10a = px.scatter(
    df, x="Med_Count", y="Bodily Pain Rating in Past Week",
    color="Group", trendline="ols",
    color_discrete_map=GROUP_COLORS,
    title="No. of Medication Classes vs Pain Severity",
    labels={"Med_Count": "Medication Classes"}
)
fig10a = mobile_layout(fig10a, height=500)
save_fig(fig10a, "10a_polypharmacy_vs_pain.html")

fig10b = px.box(
    df, x="Hospitalized (Past Year)", y="Med_Count",
    color="Group",
    color_discrete_map=GROUP_COLORS,
    title="Hospitalization vs No. of Medication Classes",
    labels={
        "Med_Count": "Medication Classes",
        "Hospitalized (Past Year)": "Hospitalized"
    }
)
fig10b = mobile_layout(fig10b, height=500)
save_fig(fig10b, "10b_polypharmacy_vs_hospitalization.html")

# ── 14. CORRELATION 11: Pain Region vs Neuropathy ─────────────────────────
pain_region_cols = [
    "Parts That Tend to Be Most Pain (0: No | 1: Yes | NA: Not Disclosed)_Feet/Ankle",
    "Parts That Tend to Be Most Pain (0: No | 1: Yes | NA: Not Disclosed)_Thigh/Knee/Leg Pain"
]

rows = []
for col in pain_region_cols:
    for group in df["Group"].unique():
        g = df[df["Group"] == group].dropna(
            subset=[col, "Peripheral Neuropathy Now"])
        if len(g) < 5:
            continue
        chi2, p, _, _ = stats.chi2_contingency(
            pd.crosstab(g[col], g["Peripheral Neuropathy Now"]))
        rows.append({
            "Pain Region": col.split("_")[-1],
            "Group": group,
            "Chi2": round(chi2, 3),
            "P-Value": round(p, 4),
            "Significant": "Yes" if p < 0.05 else "No"
        })

fig11 = px.bar(
    pd.DataFrame(rows), x="Pain Region", y="Chi2",
    color="Group", barmode="group",
    color_discrete_map=GROUP_COLORS,
    title="Lower Limb Pain Region vs Peripheral Neuropathy",
    hover_data=["P-Value", "Significant"]
)
fig11 = mobile_layout(fig11, height=480)
save_fig(fig11, "11_pain_region_vs_neuropathy.html")

# ── 15. CORRELATION 12: Comorbidity Prevalence ────────────────────────────
comorbidity_cols = [
    "HBP Now", "Heart Disease Now", "Diabetes Now", "Cancer Now",
    "Kidney Disease Now", "Sleep Condition Now", "Major Depression Now",
    "Anxiety Disorder Now", "Peripheral Neuropathy Now", "Stroke Now"
]

prev_rows = []
for col in comorbidity_cols:
    for group in df["Group"].unique():
        g = df[df["Group"] == group]
        prev_rows.append({
            "Condition": col.replace(" Now", ""),
            "Group": group,
            "Prevalence (%)": round(g[col].mean() * 100, 1)
        })

fig12 = px.bar(
    pd.DataFrame(prev_rows), x="Condition", y="Prevalence (%)",
    color="Group", barmode="group",
    color_discrete_map=GROUP_COLORS,
    title="Comorbidity Prevalence Across All Four Groups (%)"
)
fig12 = mobile_layout(fig12, height=520)
save_fig(fig12, "12_comorbidity_prevalence_by_group.html")

print("\nAll graphs saved to docs/graphs/")