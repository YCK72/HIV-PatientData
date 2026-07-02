
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import kaleido
except ImportError:
    print("ERROR: kaleido not installed.  Run: pip install kaleido")
    exit(1)

# ── 1. Paths ───────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(script_dir, '..', 'Data', 'Cleaned', 'cleaned_data.csv')
output_dir = os.path.join(script_dir, '..', 'Data', 'Exported_Images')
os.makedirs(output_dir, exist_ok=True)

# ── 2. Load & standardize ──────────────────────────────────────────────────
df = pd.read_csv(data_path)
group_mapping = {
    "HIV - Pain": "HIV - Pain", "HIV-PAIN": "HIV - Pain",
    "HIV - No Pain": "HIV - No Pain", "HIV- No Pain": "HIV - No Pain",
    "No HIV - Pain": "No HIV - Pain", "No HIV - No Pain": "No HIV - No Pain"
}
df["Group"] = df["Group"].map(group_mapping)

GROUP_ORDER  = ["HIV - Pain", "HIV - No Pain", "No HIV - Pain", "No HIV - No Pain"]
GROUP_COLORS = {
    "HIV - Pain": "#e74c3c", "HIV - No Pain": "#e67e22",
    "No HIV - Pain": "#3498db", "No HIV - No Pain": "#2ecc71"
}

# Derived columns
comorbidity_now_cols = [c for c in df.columns
                        if c.endswith("Now") and "Asthma" not in c]
df["Comorbidity_Count"] = df[comorbidity_now_cols].sum(axis=1)
med_class_cols = [c for c in df.columns if c.startswith("Medication Class_")]
df["Med_Count"] = df[med_class_cols].sum(axis=1)
pain_region_prefix = "Parts That Tend to Be Most Pain (0: No | 1: Yes | NA: Not Disclosed)_"
pain_region_cols   = [c for c in df.columns if c.startswith(pain_region_prefix)]
df["Pain_Region_Count"] = df[pain_region_cols].sum(axis=1)
df["HIV_Status"]  = df["Group"].apply(lambda g: "HIV+" if "No Pain" not in str(g) and "HIV" in str(g) else "HIV-")
df["Pain_Status"] = df["Group"].apply(lambda g: "Pain" if "Pain" in str(g) and "No Pain" not in str(g) else "No Pain")

pain_col = "Bodily Pain Rating in Past Week"
print(f"Loaded {len(df)} patients\nSaving to: {output_dir}\n")

# ── 3. Helpers ─────────────────────────────────────────────────────────────
W, SCALE = 1400, 2   # width px, retina scale

def save_fig(fig, filename, height=None):
    h = height or fig.layout.height or 600
    base = os.path.join(output_dir, filename)
    fig.write_image(base + ".png", width=W, height=h, scale=SCALE)
    fig.write_image(base + ".pdf", width=W, height=h)
    print(f"  {filename}.png + .pdf")

def base_layout(fig, height=600):
    fig.update_layout(
        autosize=False, width=W, height=height,
        margin=dict(l=80, r=40, t=80, b=180),
        legend=dict(orientation="h", yanchor="bottom", y=-0.32,
                    xanchor="center", x=0.5, font=dict(size=13), itemwidth=120),
        font=dict(size=13), title_font=dict(size=17),
        xaxis=dict(tickfont=dict(size=12), title_font=dict(size=13), tickangle=-25),
        yaxis=dict(tickfont=dict(size=12), title_font=dict(size=13)),
        plot_bgcolor="white", paper_bgcolor="white"
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# PAPER 1
# ═══════════════════════════════════════════════════════════════════════════

# P1-Fig02c: Age violin
fig = px.violin(df, x="Group", y="Age", color="Group", box=True, points="all",
                color_discrete_map=GROUP_COLORS,
                category_orders={"Group": GROUP_ORDER},
                title="Figure 2C: Age Distribution by Group")
fig = base_layout(fig, 600)
save_fig(fig, "p1_02c_age_violin")

# P1-Fig02d: Weight violin
fig = px.violin(df, x="Group", y="Weight", color="Group", box=True, points="all",
                color_discrete_map=GROUP_COLORS,
                category_orders={"Group": GROUP_ORDER},
                title="Figure 2D: Weight Distribution by Group")
fig = base_layout(fig, 600)
save_fig(fig, "p1_02d_weight_violin")

# P1-Fig02e: Age histograms
fig = px.histogram(df, x="Age", color="Group", facet_col="Group",
                   facet_col_wrap=2, nbins=15,
                   color_discrete_map=GROUP_COLORS,
                   category_orders={"Group": GROUP_ORDER},
                   title="Figure 2E: Age Histogram by Group")
fig.update_layout(height=620, showlegend=False, width=W,
                  plot_bgcolor="white", paper_bgcolor="white",
                  margin=dict(l=60, r=40, t=80, b=60))
save_fig(fig, "p1_02e_age_histogram", 620)

# P1-Fig03a: Pain severity violin
fig = px.violin(df, x="Group", y=pain_col, color="Group", box=True, points="all",
                color_discrete_map=GROUP_COLORS,
                category_orders={"Group": GROUP_ORDER},
                title="Figure 3A: Pain Severity Distribution by Group")
fig = base_layout(fig, 600)
save_fig(fig, "p1_03a_pain_severity_violin")

# P1-Fig03b: Pain histogram
fig = px.histogram(df, x=pain_col, color="Group", barmode="overlay",
                   opacity=0.6, nbins=10,
                   color_discrete_map=GROUP_COLORS,
                   category_orders={"Group": GROUP_ORDER},
                   title="Figure 3B: Pain Severity Histogram by Group",
                   labels={pain_col: "Pain Rating (0–10)"})
fig = base_layout(fig, 560)
save_fig(fig, "p1_03b_pain_severity_histogram")

# P1-Fig03c: Pain stacked proportions
pain_prop_rows = []
for group in GROUP_ORDER:
    g = df[df["Group"] == group][pain_col].dropna()
    for rating in range(0, 11):
        pain_prop_rows.append({"Group": group, "Pain Rating": str(rating),
                                "Proportion (%)": round((g == rating).sum() / len(g) * 100, 1)})
fig = px.bar(pd.DataFrame(pain_prop_rows), x="Group",
             y="Proportion (%)", color="Pain Rating", barmode="stack",
             title="Figure 3C: Pain Rating Proportion by Group (%)")
fig = base_layout(fig, 580)
save_fig(fig, "p1_03c_pain_severity_stacked")

# P1-Fig04: Chronic pain prevalence
chronic_rows = []
for col, label in [("Pain > 3 Months","Pain > 3 Months"),
                   ("Pain >12hrs or More 6 months","Pain >12 hrs/day")]:
    for group in GROUP_ORDER:
        g = df[df["Group"] == group][col].dropna()
        chronic_rows.append({"Group": group, "Criterion": label,
                              "Prevalence (%)": round(g.mean() * 100, 1)})
fig = px.bar(pd.DataFrame(chronic_rows), x="Group", y="Prevalence (%)",
             color="Criterion", barmode="group",
             title="Figure 4: Chronic Pain Prevalence by Group (%)",
             category_orders={"Group": GROUP_ORDER})
fig = base_layout(fig, 560)
save_fig(fig, "p1_04_chronic_pain_prevalence")

# P1-Fig05: Pain location heatmap
pain_region_labels = {c: c.replace(pain_region_prefix, "") for c in pain_region_cols}
heat_rows = []
for col, label in pain_region_labels.items():
    for group in GROUP_ORDER:
        g = df[df["Group"] == group][col].dropna()
        heat_rows.append({"Region": label, "Group": group,
                           "Prevalence (%)": round(g.mean() * 100, 1)})
heat_df = pd.DataFrame(heat_rows).pivot(index="Region", columns="Group",
                                         values="Prevalence (%)")
fig = px.imshow(heat_df, color_continuous_scale="RdYlBu_r",
                text_auto=True, aspect="auto",
                title="Figure 5: Pain Location Prevalence Heatmap (%)")
fig.update_layout(height=max(420, len(pain_region_cols)*40+120),
                  width=W, margin=dict(l=220, r=40, t=80, b=80),
                  plot_bgcolor="white", paper_bgcolor="white")
save_fig(fig, "p1_05_pain_location_heatmap")

# P1-Fig06: Pain location frequencies
freq_rows = []
for col, label in pain_region_labels.items():
    freq_rows.append({"Region": label,
                       "Prevalence (%)": round(df[col].mean() * 100, 1)})
freq_df = pd.DataFrame(freq_rows).sort_values("Prevalence (%)", ascending=True)
fig = px.bar(freq_df, x="Prevalence (%)", y="Region", orientation="h",
             title="Figure 6: Pain Location Frequencies (Overall %)",
             color="Prevalence (%)", color_continuous_scale="Reds")
fig.update_layout(height=max(420, len(pain_region_cols)*32+120),
                  width=W, margin=dict(l=220, r=40, t=80, b=60),
                  plot_bgcolor="white", paper_bgcolor="white")
save_fig(fig, "p1_06_pain_location_frequencies")

# P1-Fig07: Region count histogram
fig = px.histogram(df, x="Pain_Region_Count", color="Group",
                   barmode="overlay", opacity=0.65, nbins=12,
                   color_discrete_map=GROUP_COLORS,
                   category_orders={"Group": GROUP_ORDER},
                   title="Figure 7: Number of Painful Regions by Group",
                   labels={"Pain_Region_Count": "Number of Painful Regions"})
fig = base_layout(fig, 560)
save_fig(fig, "p1_07_pain_region_count_histogram")

# P1-Fig08: Severity vs regions
fig = px.scatter(df, x="Pain_Region_Count", y=pain_col, color="Group",
                 trendline="ols", color_discrete_map=GROUP_COLORS,
                 category_orders={"Group": GROUP_ORDER},
                 title="Figure 8: Pain Severity vs Number of Pain Regions",
                 labels={"Pain_Region_Count": "Number of Painful Regions",
                          pain_col: "Pain Rating"})
fig = base_layout(fig, 580)
save_fig(fig, "p1_08_pain_severity_vs_regions")

# P1-Fig09: Forest plot (logistic regression)
try:
    import statsmodels.formula.api as smf
    df["Pain_Binary"] = (df["Pain_Status"] == "Pain").astype(int)
    df["HIV_Binary"]  = (df["HIV_Status"]  == "HIV+").astype(int)
    predictor_map = {
        "Age": "Age", "Weight": "Weight", "HIV": "HIV_Binary",
        "Smoking": "Smoke_Cigarettes", "Depression": "Major Depression Now",
        "Sleep Disorder": "Sleep Condition Now", "Neuropathy": "Peripheral Neuropathy Now",
        "Med Count": "Med_Count", "Comorbidity Count": "Comorbidity_Count",
    }
    valid_preds = {k: v for k, v in predictor_map.items() if v in df.columns}
    model_df = df[["Pain_Binary"] + list(valid_preds.values())].dropna()
    model_df.columns = ["Pain_Binary"] + list(valid_preds.keys())
    formula = "Pain_Binary ~ " + " + ".join([f"Q('{k}')" for k in valid_preds])
    model = smf.logit(formula, data=model_df).fit(disp=False)
    or_rows = []
    for term, coef in model.params.items():
        if "Intercept" in term: continue
        ci_lo = model.conf_int().loc[term, 0]
        ci_hi = model.conf_int().loc[term, 1]
        p_val = model.pvalues[term]
        label = term.replace("Q('","").replace("')","")
        or_rows.append({"Predictor": label,
                         "OR": round(np.exp(coef), 3),
                         "CI_Lo": round(np.exp(ci_lo), 3),
                         "CI_Hi": round(np.exp(ci_hi), 3),
                         "P": round(p_val, 4),
                         "Significant": p_val < 0.05})
    or_df = pd.DataFrame(or_rows).sort_values("OR")
    fig = go.Figure()
    for _, row in or_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["OR"]], y=[row["Predictor"]], mode="markers",
            marker=dict(size=12, color="#e74c3c" if row["Significant"] else "#95a5a6"),
            error_x=dict(type="data", symmetric=False,
                         array=[row["CI_Hi"]-row["OR"]],
                         arrayminus=[row["OR"]-row["CI_Lo"]]),
            showlegend=False))
    fig.add_vline(x=1, line_dash="dash", line_color="black")
    fig.update_layout(
        title="Figure 9/10: Adjusted Logistic Regression — Forest Plot",
        xaxis_title="Odds Ratio (95% CI)", yaxis_title="Predictor",
        height=max(450, len(or_df)*50+120), width=W,
        margin=dict(l=180, r=40, t=80, b=60),
        plot_bgcolor="white", paper_bgcolor="white")
    save_fig(fig, "p1_09_logistic_regression_forest")
except ImportError:
    print("  statsmodels not installed — skipping forest plot")

# ═══════════════════════════════════════════════════════════════════════════
# PAPER 2
# ═══════════════════════════════════════════════════════════════════════════

# P2-Fig02: Comorbidity burden
burden_df = df.groupby("Group")["Comorbidity_Count"].agg(
    Mean="mean", SEM=lambda x: stats.sem(x, nan_policy="omit")).reset_index()
burden_df["Group"] = pd.Categorical(burden_df["Group"], categories=GROUP_ORDER)
burden_df = burden_df.sort_values("Group")
fig = go.Figure()
for _, row in burden_df.iterrows():
    fig.add_trace(go.Bar(name=row["Group"], x=[row["Group"]], y=[row["Mean"]],
                          error_y=dict(type="data", array=[row["SEM"]]),
                          marker_color=GROUP_COLORS.get(row["Group"], "#aaa")))
fig.update_layout(title="Figure 2 (P2): Mean Comorbidity Count by Group (± SEM)",
                   yaxis_title="Mean Number of Conditions", showlegend=False,
                   height=540, width=W, margin=dict(l=80, r=40, t=80, b=80),
                   plot_bgcolor="white", paper_bgcolor="white")
save_fig(fig, "p2_02_comorbidity_burden")

# P2-Fig03: Comorbidity heatmap
comorbidity_cols_full = [c for c in ["Diabetes Now","HBP Now","Stroke Now","Cancer Now",
    "Kidney Disease Now","Peripheral Neuropathy Now","Major Depression Now",
    "Anxiety Disorder Now","Sleep Condition Now","Heart Disease Now"] if c in df.columns]
heat2_rows = {col.replace(" Now",""):
              {g: round(df[df["Group"]==g][col].mean()*100, 1) for g in GROUP_ORDER}
              for col in comorbidity_cols_full}
heat2_df = pd.DataFrame(heat2_rows).T[GROUP_ORDER]
fig = px.imshow(heat2_df, color_continuous_scale="RdYlBu_r",
                text_auto=True, aspect="auto",
                title="Figure 3 (P2): Comorbidity Prevalence Heatmap by Group (%)")
fig.update_layout(height=480, width=W, margin=dict(l=200, r=40, t=80, b=80),
                   plot_bgcolor="white", paper_bgcolor="white")
save_fig(fig, "p2_03_comorbidity_heatmap")

# P2-Fig04–06: Disorder bars (reuse logic from analysis.py)
for label_map, title, fname, barmode in [
    ({"Stroke":"Stroke Now","Seizure Disorder":"Seizure Disorder Now",
      "Neurological Disease":"Neurological Disease Now",
      "Peripheral Neuropathy":"Peripheral Neuropathy Now"},
     "Figure 4 (P2): Neurological Disorder Prevalence (%)", "p2_04_neurological_disorders", "stack"),
    ({"Major Depression":"Major Depression Now","Anxiety Disorder":"Anxiety Disorder Now",
      "Bipolar Disorder":"Bipolar Disorder Now","Other MH":"Other Mental Health Conditions Now"},
     "Figure 5 (P2): Psychiatric Disorder Prevalence (%)", "p2_05_psychiatric_disorders", "group"),
    ({"Diabetes":"Diabetes Now","Hypertension":"HBP Now",
      "Heart Disease":"Heart Disease Now","Kidney Disease":"Kidney Disease Now"},
     "Figure 6 (P2): Metabolic Disorder Prevalence (%)", "p2_06_metabolic_disorders", "group"),
]:
    cols = {k: v for k, v in label_map.items() if v in df.columns}
    rows = []
    for lbl, col in cols.items():
        for group in GROUP_ORDER:
            g = df[df["Group"]==group][col].dropna()
            rows.append({"Group": group, "Condition": lbl,
                          "Prevalence (%)": round(g.mean()*100, 1)})
    fig = px.bar(pd.DataFrame(rows),
                  x="Group" if barmode=="stack" else "Condition",
                  y="Prevalence (%)",
                  color="Condition" if barmode=="stack" else "Group",
                  barmode=barmode,
                  color_discrete_map=GROUP_COLORS if barmode=="group" else None,
                  category_orders={"Group": GROUP_ORDER},
                  title=title)
    fig = base_layout(fig, 560)
    save_fig(fig, fname)

# P2-Fig07: ART drug classes
art_map = {"NRTIs":"ART?_Nucleoside reverse transcriptase inhibitors (NRTIs)",
           "INSTI":"ART?_Integrase Strand Transfer Inhibitor (INSTI)",
           "NNRTI":"ART?_Non-nucleoside reverse transcriptase inhibitor (NNRTI)",
           "PI":"ART?_Protease Inhibitor (PI)"}
art_map = {k: v for k, v in art_map.items() if v in df.columns}
hiv_only = df[df["Group"].isin(["HIV - Pain","HIV - No Pain"])]
art_rows = [{"Drug Class": k, "Count": int(hiv_only[v].sum())}
             for k, v in art_map.items()]
fig = px.pie(pd.DataFrame(art_rows), names="Drug Class", values="Count",
              title="Figure 7A (P2): ART Drug Class Distribution",
              color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_layout(height=500, width=W, margin=dict(l=40, r=40, t=80, b=80),
                   paper_bgcolor="white")
save_fig(fig, "p2_07a_art_drug_classes_pie")

# P2-Fig09: ART adherence histogram
if "Doses ART missed?" in df.columns:
    fig = px.histogram(hiv_only, x="Doses ART missed?", color="Group",
                        barmode="overlay", opacity=0.65, nbins=15,
                        color_discrete_map=GROUP_COLORS,
                        title="Figure 9 (P2): ART Missed Doses Distribution (HIV Only)")
    fig = base_layout(fig, 540)
    save_fig(fig, "p2_09_art_adherence_histogram")

# P2-Fig11: Smoking by group
smoke_rows = []
for smoke_col, label in [("Smoke_Cigarettes","Cigarettes"),("Smoke_E-Cigarettes","E-Cigarettes")]:
    if smoke_col in df.columns:
        for group in GROUP_ORDER:
            g = df[df["Group"]==group][smoke_col].dropna()
            smoke_rows.append({"Group":group,"Type":label,
                                "Prevalence (%)":round(g.mean()*100,1)})
fig = px.bar(pd.DataFrame(smoke_rows), x="Group", y="Prevalence (%)",
              color="Type", barmode="group",
              category_orders={"Group": GROUP_ORDER},
              title="Figure 11 (P2): Smoking Prevalence by Group (%)")
fig = base_layout(fig, 540)
save_fig(fig, "p2_11_smoking_by_group")

# P2-Fig12: Smoking vs pain boxplot
if "Smoke_Cigarettes" in df.columns:
    smoke_pain = df.dropna(subset=["Smoke_Cigarettes", pain_col]).copy()
    smoke_pain["Smoker"] = smoke_pain["Smoke_Cigarettes"].map({1:"Smoker",0:"Non-Smoker"})
    fig = px.box(smoke_pain, x="Smoker", y=pain_col, color="Group",
                  color_discrete_map=GROUP_COLORS,
                  category_orders={"Group": GROUP_ORDER},
                  title="Figure 12 (P2): Pain Rating by Smoking Status and Group")
    fig = base_layout(fig, 560)
    save_fig(fig, "p2_12_smoking_vs_pain_boxplot")

# P2-Fig14: Sleep vs pain violin
if "Sleep Condition Now" in df.columns:
    sleep_viol = df.dropna(subset=["Sleep Condition Now", pain_col]).copy()
    sleep_viol["Sleep Disorder"] = sleep_viol["Sleep Condition Now"].map(
        {1:"Sleep Disorder",0:"No Sleep Disorder"})
    fig = px.violin(sleep_viol, x="Sleep Disorder", y=pain_col,
                     color="Group", box=True,
                     color_discrete_map=GROUP_COLORS,
                     category_orders={"Group": GROUP_ORDER},
                     title="Figure 14 (P2): Pain Rating by Sleep Disorder Status")
    fig = base_layout(fig, 580)
    save_fig(fig, "p2_14_sleep_vs_pain_violin")

# P2-Fig15: Polypharmacy histogram
fig = px.histogram(df, x="Med_Count", color="Group",
                    barmode="overlay", opacity=0.65, nbins=10,
                    color_discrete_map=GROUP_COLORS,
                    category_orders={"Group": GROUP_ORDER},
                    title="Figure 15 (P2): Medication Class Count Distribution by Group",
                    labels={"Med_Count":"Number of Medication Classes"})
fig = base_layout(fig, 540)
save_fig(fig, "p2_15_polypharmacy_histogram")

# P2-Fig17: Polypharmacy vs hospitalization
if "Hospitalized (Past Year)" in df.columns:
    fig = px.box(df, x="Hospitalized (Past Year)", y="Med_Count",
                  color="Group", color_discrete_map=GROUP_COLORS,
                  category_orders={"Group": GROUP_ORDER},
                  title="Figure 17 (P2): Medication Count by Hospitalization Status",
                  labels={"Med_Count":"Number of Medication Classes",
                           "Hospitalized (Past Year)":"Hospitalized (0=No, 1=Yes)"})
    fig = base_layout(fig, 560)
    save_fig(fig, "p2_17_polypharmacy_vs_hospitalization")

# P2-Fig18: Polypharmacy vs pain
fig = px.scatter(df, x="Med_Count", y=pain_col, color="Group", trendline="ols",
                  color_discrete_map=GROUP_COLORS,
                  category_orders={"Group": GROUP_ORDER},
                  title="Figure 18 (P2): Medication Count vs Pain Severity",
                  labels={"Med_Count":"Number of Medication Classes",pain_col:"Pain Rating"})
fig = base_layout(fig, 580)
save_fig(fig, "p2_18_polypharmacy_vs_pain")

# P2-Fig21: Age vs medication count
fig = px.scatter(df, x="Age", y="Med_Count", color="Group", trendline="ols",
                  color_discrete_map=GROUP_COLORS,
                  category_orders={"Group": GROUP_ORDER},
                  title="Figure 21 (P2): Age vs Medication Count by Group",
                  labels={"Med_Count":"Number of Medication Classes"})
fig = base_layout(fig, 580)
save_fig(fig, "p2_21_age_vs_medication_count")

# P2-Fig22: Weight vs pain
fig = px.scatter(df, x="Weight", y=pain_col, color="Group", trendline="ols",
                  color_discrete_map=GROUP_COLORS,
                  category_orders={"Group": GROUP_ORDER},
                  title="Figure 22 (P2): Weight vs Pain Severity by Group",
                  labels={pain_col:"Pain Rating"})
fig = base_layout(fig, 580)
save_fig(fig, "p2_22_weight_vs_pain")

# P2-Fig23: Weight vs diabetes
if "Diabetes Now" in df.columns:
    d = df.dropna(subset=["Diabetes Now","Weight"]).copy()
    d["Diabetes"] = d["Diabetes Now"].map({1:"Diabetes",0:"No Diabetes"})
    fig = px.box(d, x="Diabetes", y="Weight", color="Group",
                  color_discrete_map=GROUP_COLORS,
                  category_orders={"Group": GROUP_ORDER},
                  title="Figure 23 (P2): Weight by Diabetes Status and Group")
    fig = base_layout(fig, 560)
    save_fig(fig, "p2_23_weight_vs_diabetes")

# P2-Fig24: Weight vs hypertension
if "HBP Now" in df.columns:
    d = df.dropna(subset=["HBP Now","Weight"]).copy()
    d["Hypertension"] = d["HBP Now"].map({1:"HBP",0:"No HBP"})
    fig = px.box(d, x="Hypertension", y="Weight", color="Group",
                  color_discrete_map=GROUP_COLORS,
                  category_orders={"Group": GROUP_ORDER},
                  title="Figure 24 (P2): Weight by Hypertension Status and Group")
    fig = base_layout(fig, 560)
    save_fig(fig, "p2_24_weight_vs_hypertension")

# P2-Fig25: Hospitalization
if "Hospitalized (Past Year)" in df.columns:
    hosp_rows = [{"Group": g, "Hospitalized (%)":
                   round(df[df["Group"]==g]["Hospitalized (Past Year)"].mean()*100,1)}
                  for g in GROUP_ORDER]
    fig = px.bar(pd.DataFrame(hosp_rows), x="Group", y="Hospitalized (%)",
                  color="Group", color_discrete_map=GROUP_COLORS,
                  category_orders={"Group": GROUP_ORDER},
                  title="Figure 25 (P2): Hospitalization Rate by Group (%)")
    fig.update_traces(showlegend=False)
    fig = base_layout(fig, 520)
    save_fig(fig, "p2_25_hospitalization_by_group")

# P2-Fig27: Correlation matrix
corr_vars = {"Age":"Age","Weight":"Weight","Pain Rating":pain_col,
              "Med Count":"Med_Count","Comorbidity Count":"Comorbidity_Count",
              "ART Missed":"Doses ART missed?"}
corr_vars = {k: v for k, v in corr_vars.items() if v in df.columns}
corr_df = df[list(corr_vars.values())].rename(columns={v:k for k,v in corr_vars.items()})
fig = px.imshow(corr_df.corr(method="spearman"),
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                text_auto=".2f", aspect="auto",
                title="Figure 27 (P2): Spearman Correlation Matrix")
fig.update_layout(height=540, width=W, margin=dict(l=160, r=40, t=80, b=80),
                   plot_bgcolor="white", paper_bgcolor="white")
save_fig(fig, "p2_27_correlation_matrix")

# ── Age category figures ───────────────────────────────────────────────────
age_bins   = [18, 30, 40, 50, 60, 100]
age_labels = ["18–29","30–39","40–49","50–59","60+"]
df["Age_Category"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)

age_cat_df = df.groupby(["Group","Age_Category"], observed=True).size().reset_index(name="Count")
age_cat_df["Pct"] = age_cat_df.groupby("Group")["Count"].transform(
    lambda x: round(x/x.sum()*100, 1))
fig = px.bar(age_cat_df, x="Group", y="Pct", color="Age_Category", barmode="stack",
              category_orders={"Group": GROUP_ORDER, "Age_Category": age_labels},
              title="Age Category Distribution by Group (%)", labels={"Pct":"Percentage (%)"})
fig = base_layout(fig, 560)
save_fig(fig, "age_01_age_categories_stacked")

pain_by_age = [{"Age Category": cat, "Pain Prevalence (%)":
                 round((df[df["Age_Category"]==cat]["Pain_Status"]=="Pain").mean()*100,1)}
                for cat in age_labels]
fig = px.line(pd.DataFrame(pain_by_age), x="Age Category",
               y="Pain Prevalence (%)", markers=True,
               title="Pain Prevalence by Age Category")
fig = base_layout(fig, 500)
save_fig(fig, "age_02_pain_prevalence_by_age")

# ── Original 12 correlations ───────────────────────────────────────────────
mental_health_cols = ["Major Depression Now","Anxiety Disorder Now",
                       "Bipolar Disorder Now","Other Mental Health Conditions Now"]

# Corr 1
rows = []
for col in mental_health_cols:
    for group in df["Group"].unique():
        g = df[df["Group"]==group].dropna(subset=[col, pain_col])
        if len(g) < 5: continue
        r, p = stats.pointbiserialr(g[col], g[pain_col])
        rows.append({"Condition":col.replace(" Now",""),"Group":group,
                     "Correlation (r)":round(r,3),"P-Value":round(p,4),
                     "Significant":"Yes" if p<0.05 else "No"})
fig = px.bar(pd.DataFrame(rows), x="Condition", y="Correlation (r)",
              color="Group", barmode="group", color_discrete_map=GROUP_COLORS,
              title="Correlation 1: Pain Severity vs Mental Health Conditions",
              labels={"Correlation (r)":"Point-Biserial r"},
              hover_data=["P-Value","Significant"])
fig.add_hline(y=0, line_dash="dash", line_color="black")
fig = base_layout(fig, 600)
save_fig(fig, "01_pain_vs_mental_health")

# Corr 6
fig = px.scatter(df, x="Age", y="Comorbidity_Count", color="Group", trendline="ols",
                  color_discrete_map=GROUP_COLORS,
                  title="Correlation 6: Age vs Total Comorbidity Count by Group",
                  labels={"Comorbidity_Count":"Number of Current Conditions"})
fig = base_layout(fig, 600)
save_fig(fig, "06_age_vs_comorbidity")

# Corr 7
fig = px.scatter(df, x="Age", y=pain_col, color="Group", trendline="ols",
                  color_discrete_map=GROUP_COLORS,
                  title="Correlation 7: Age vs Pain Severity by Group")
fig = base_layout(fig, 600)
save_fig(fig, "07_age_vs_pain")

# Corr 10a & 10b
fig = px.scatter(df, x="Med_Count", y=pain_col, color="Group", trendline="ols",
                  color_discrete_map=GROUP_COLORS,
                  title="Correlation 10a: Medication Classes vs Pain Severity",
                  labels={"Med_Count":"Number of Medication Classes"})
fig = base_layout(fig, 600)
save_fig(fig, "10a_polypharmacy_vs_pain")

if "Hospitalized (Past Year)" in df.columns:
    fig = px.box(df, x="Hospitalized (Past Year)", y="Med_Count",
                  color="Group", color_discrete_map=GROUP_COLORS,
                  title="Correlation 10b: Hospitalization vs Medication Classes",
                  labels={"Med_Count":"Number of Medication Classes",
                           "Hospitalized (Past Year)":"Hospitalized (0=No, 1=Yes)"})
    fig = base_layout(fig, 600)
    save_fig(fig, "10b_polypharmacy_vs_hospitalization")

# Corr 12
comorbidity_cols_12 = [c for c in ["HBP Now","Heart Disease Now","Diabetes Now","Cancer Now",
    "Kidney Disease Now","Sleep Condition Now","Major Depression Now",
    "Anxiety Disorder Now","Peripheral Neuropathy Now","Stroke Now"] if c in df.columns]
prev_rows = []
for col in comorbidity_cols_12:
    for group in df["Group"].unique():
        g = df[df["Group"]==group]
        prev_rows.append({"Condition":col.replace(" Now",""),"Group":group,
                           "Prevalence (%)":round(g[col].mean()*100,1)})
fig = px.bar(pd.DataFrame(prev_rows), x="Condition", y="Prevalence (%)",
              color="Group", barmode="group", color_discrete_map=GROUP_COLORS,
              title="Correlation 12: Comorbidity Prevalence Across All Four Groups (%)")
fig = base_layout(fig, 620)
save_fig(fig, "12_comorbidity_prevalence_by_group")

print(f"\nDone! All figures exported as PNG + PDF\nLocation: {output_dir}")