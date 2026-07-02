import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# ── 1. Paths ───────────────────────────────────────────────────────────────
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

GROUP_ORDER  = ["HIV - Pain", "HIV - No Pain", "No HIV - Pain", "No HIV - No Pain"]
GROUP_COLORS = {
    "HIV - Pain":        "#e74c3c",
    "HIV - No Pain":     "#e67e22",
    "No HIV - Pain":     "#3498db",
    "No HIV - No Pain":  "#2ecc71"
}

# ── Derived columns ────────────────────────────────────────────────────────
# Comorbidity count
comorbidity_now_cols = [c for c in df.columns
                        if c.endswith("Now") and "Asthma" not in c]
df["Comorbidity_Count"] = df[comorbidity_now_cols].sum(axis=1)

# Medication count
med_class_cols = [c for c in df.columns if c.startswith("Medication Class_")]
df["Med_Count"] = df[med_class_cols].sum(axis=1)

# Pain region count — adjust prefix to match your actual column names
pain_region_prefix = "Parts That Tend to Be Most Pain (0: No | 1: Yes | NA: Not Disclosed)_"
pain_region_cols   = [c for c in df.columns if c.startswith(pain_region_prefix)]
df["Pain_Region_Count"] = df[pain_region_cols].sum(axis=1)

# HIV flag
df["HIV_Status"] = df["Group"].apply(
    lambda g: "HIV+" if "HIV - Pain" in str(g) or "HIV - No Pain" in str(g) else "HIV-")
df["Pain_Status"] = df["Group"].apply(
    lambda g: "Pain" if "Pain" in str(g) and "No Pain" not in str(g) else "No Pain")

print(f"Loaded {len(df)} patients")

# ── 3. Helpers ─────────────────────────────────────────────────────────────
PLOTLY_CONFIG = {
    'scrollZoom': False,
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d',
        'autoScale2d', 'resetScale2d', 'lasso2d', 'select2d'
    ]
}

def save_fig(fig, filename):
    path = os.path.join(output_dir, filename)
    fig.write_html(path, include_plotlyjs='cdn', config=PLOTLY_CONFIG)
    print(f"  Saved: {filename}")

def base_layout(fig, height=580):
    fig.update_layout(
        autosize=True, height=height,
        margin=dict(l=50, r=30, t=70, b=160),
        legend=dict(orientation="h", yanchor="bottom", y=-0.38,
                    xanchor="center", x=0.5, font=dict(size=12), itemwidth=110),
        font=dict(size=12), title_font=dict(size=15),
        xaxis=dict(tickfont=dict(size=11), title_font=dict(size=12), tickangle=-25),
        yaxis=dict(tickfont=dict(size=11), title_font=dict(size=12))
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# PAPER 1 FIGURES
# ═══════════════════════════════════════════════════════════════════════════

# ── P1-Fig01: CONSORT Flow Diagram ────────────────────────────────────────
group_counts = df["Group"].value_counts()
n_hiv_pain     = int(group_counts.get("HIV - Pain",    0))
n_hiv_nopain   = int(group_counts.get("HIV - No Pain", 0))
n_nohiv_pain   = int(group_counts.get("No HIV - Pain", 0))
n_nohiv_nopain = int(group_counts.get("No HIV - No Pain", 0))
n_included     = len(df)
n_total        = n_included + 30   # adjust if you know the exact exclusion count
n_excluded     = n_total - n_included

consort = go.Figure()
boxes = [
    # (x_center, y_center, width, height, label, color)
    (0.5,  0.93, 0.38, 0.10, f"Recruited<br><b>{n_total}</b> patients",                             "#2c3e50"),
    (0.5,  0.73, 0.38, 0.10, f"Excluded  <b>{n_excluded}</b><br>(Missing >50% or empty records)",   "#e74c3c"),
    (0.5,  0.53, 0.38, 0.10, f"Included<br><b>{n_included}</b> patients",                           "#27ae60"),
    (0.13, 0.25, 0.20, 0.13, f"HIV + Pain<br><b>{n_hiv_pain}</b>",                                  "#e74c3c"),
    (0.38, 0.25, 0.20, 0.13, f"HIV + No Pain<br><b>{n_hiv_nopain}</b>",                             "#e67e22"),
    (0.63, 0.25, 0.20, 0.13, f"No HIV + Pain<br><b>{n_nohiv_pain}</b>",                             "#3498db"),
    (0.88, 0.25, 0.20, 0.13, f"No HIV + No Pain<br><b>{n_nohiv_nopain}</b>",                        "#2ecc71"),
]
for (x, y, w, h, label, color) in boxes:
    consort.add_shape(type="rect",
        x0=x-w/2, x1=x+w/2, y0=y-h/2, y1=y+h/2,
        xref="paper", yref="paper",
        line=dict(color=color, width=2),
        fillcolor=color, opacity=0.15)
    consort.add_annotation(x=x, y=y, text=label,
        xref="paper", yref="paper",
        showarrow=False, font=dict(size=12, color="#2c3e50"), align="center")

# Arrows drawn as lines + arrowhead triangles using shapes
# (avoids axref="paper" which is not supported in older Plotly versions)
arrow_lines = [
    (0.5, 0.88, 0.5, 0.78),    # recruited → excluded
    (0.5, 0.68, 0.5, 0.58),    # excluded → included
    (0.5, 0.48, 0.13, 0.32),   # included → HIV Pain
    (0.5, 0.48, 0.38, 0.32),   # included → HIV No Pain
    (0.5, 0.48, 0.63, 0.32),   # included → No HIV Pain
    (0.5, 0.48, 0.88, 0.32),   # included → No HIV No Pain
]
for (x0, y0, x1, y1) in arrow_lines:
    # Line segment
    consort.add_shape(type="line",
        x0=x0, y0=y0, x1=x1, y1=y1,
        xref="paper", yref="paper",
        line=dict(color="#555", width=1.5))
    # Arrowhead: small filled triangle at the endpoint
    # Use a tiny scatter marker instead (triangle-down)
    consort.add_trace(go.Scatter(
        x=[x1], y=[y1], mode="markers",
        marker=dict(symbol="triangle-down", size=10, color="#555"),
        showlegend=False, hoverinfo="skip"
    ))
consort.update_layout(
    title="Figure 1: Study Flow Diagram (CONSORT)",
    xaxis=dict(visible=False, range=[0, 1]),
    yaxis=dict(visible=False, range=[0.05, 1.05]),
    height=620, plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=20, r=20, t=60, b=20)
)
save_fig(consort, "p1_01_consort_flow.html")

# ── P1-Fig02a: Sex Distribution by Group ──────────────────────────────────
# Auto-detect sex column (exact match first, then partial)
sex_col = next((c for c in df.columns if c.lower() in [
    "sex", "gender", "sex at birth", "biological sex",
    "patient sex", "sex (m/f)"]), None)
if sex_col is None:
    sex_col = next((c for c in df.columns if any(
        x in c.lower() for x in ["sex", "gender", "male", "female"])), None)
if sex_col:
    sex_df = df.groupby(["Group", sex_col]).size().reset_index(name="Count")
    sex_df["Pct"] = sex_df.groupby("Group")["Count"].transform(
        lambda x: round(x / x.sum() * 100, 1))
    fig_sex = px.bar(sex_df, x="Group", y="Pct", color=sex_col,
                     barmode="stack", category_orders={"Group": GROUP_ORDER},
                     title="Figure 2A: Sex Distribution by Group (%)",
                     labels={"Pct": "Percentage (%)"})
    fig_sex = base_layout(fig_sex, height=560)
    save_fig(fig_sex, "p1_02a_sex_by_group.html")

# ── P1-Fig02b: Ethnicity/Race Distribution ────────────────────────────────
# NOTE: adjust to your actual column name (e.g. "Race", "Race/Ethnicity", "Ethnicity")
race_col = "Race/Ethnicity"
if race_col not in df.columns:
    race_col = next((c for c in df.columns if "race" in c.lower()
                     or "ethnic" in c.lower()), None)
if race_col:
    race_df = df.groupby(["Group", race_col]).size().reset_index(name="Count")
    race_df["Pct"] = race_df.groupby("Group")["Count"].transform(
        lambda x: round(x / x.sum() * 100, 1))
    fig_race = px.bar(race_df, x=race_col, y="Pct", color="Group",
                      barmode="group", color_discrete_map=GROUP_COLORS,
                      category_orders={"Group": GROUP_ORDER},
                      title="Figure 2B: Ethnicity/Race Distribution by Group (%)",
                      labels={"Pct": "Percentage (%)"})
    fig_race = base_layout(fig_race, height=580)
    save_fig(fig_race, "p1_02b_ethnicity_by_group.html")

# ── P1-Fig02c: Age Distribution — Violin ──────────────────────────────────
fig_age_viol = px.violin(df, x="Group", y="Age", color="Group",
                          box=True, points="all",
                          color_discrete_map=GROUP_COLORS,
                          category_orders={"Group": GROUP_ORDER},
                          title="Figure 2C: Age Distribution by Group")
fig_age_viol = base_layout(fig_age_viol, height=580)
save_fig(fig_age_viol, "p1_02c_age_violin.html")

# ── P1-Fig02d: Weight Distribution — Violin ───────────────────────────────
fig_wt_viol = px.violin(df, x="Group", y="Weight", color="Group",
                         box=True, points="all",
                         color_discrete_map=GROUP_COLORS,
                         category_orders={"Group": GROUP_ORDER},
                         title="Figure 2D: Weight Distribution by Group")
fig_wt_viol = base_layout(fig_wt_viol, height=580)
save_fig(fig_wt_viol, "p1_02d_weight_violin.html")

# ── P1-Fig02e: Age Histogram by Group ─────────────────────────────────────
fig_age_hist = px.histogram(df, x="Age", color="Group", facet_col="Group",
                             facet_col_wrap=2, nbins=15,
                             color_discrete_map=GROUP_COLORS,
                             category_orders={"Group": GROUP_ORDER},
                             title="Figure 2E: Age Histogram by Group")
fig_age_hist.update_layout(height=620, showlegend=False,
                            margin=dict(l=50, r=30, t=70, b=40))
save_fig(fig_age_hist, "p1_02e_age_histogram.html")

# ── P1-Fig03: Pain Severity Distribution ──────────────────────────────────
pain_col = "Bodily Pain Rating in Past Week"
fig_pain_viol = px.violin(df, x="Group", y=pain_col, color="Group",
                           box=True, points="all",
                           color_discrete_map=GROUP_COLORS,
                           category_orders={"Group": GROUP_ORDER},
                           title="Figure 3A: Pain Severity Distribution by Group")
fig_pain_viol = base_layout(fig_pain_viol, height=580)
save_fig(fig_pain_viol, "p1_03a_pain_severity_violin.html")

fig_pain_hist = px.histogram(df, x=pain_col, color="Group",
                              barmode="overlay", opacity=0.6, nbins=10,
                              color_discrete_map=GROUP_COLORS,
                              category_orders={"Group": GROUP_ORDER},
                              title="Figure 3B: Pain Severity Histogram by Group",
                              labels={pain_col: "Pain Rating (0–10)"})
fig_pain_hist = base_layout(fig_pain_hist, height=560)
save_fig(fig_pain_hist, "p1_03b_pain_severity_histogram.html")

# Stacked proportion bar: rating 0–10 by group
pain_prop_rows = []
for group in GROUP_ORDER:
    g = df[df["Group"] == group][pain_col].dropna()
    for rating in range(0, 11):
        pain_prop_rows.append({
            "Group": group,
            "Pain Rating": str(rating),
            "Proportion (%)": round((g == rating).sum() / len(g) * 100, 1)
        })
fig_pain_prop = px.bar(pd.DataFrame(pain_prop_rows),
                        x="Group", y="Proportion (%)", color="Pain Rating",
                        barmode="stack",
                        title="Figure 3C: Pain Rating Proportion by Group (%)")
fig_pain_prop = base_layout(fig_pain_prop, height=580)
save_fig(fig_pain_prop, "p1_03c_pain_severity_stacked.html")

# ── P1-Fig04: Chronic Pain Prevalence ─────────────────────────────────────
chronic_rows = []
for col, label in [("Pain > 3 Months", "Pain > 3 Months"),
                   ("Pain >12hrs or More 6 months", "Pain >12 hrs/day")]:
    for group in GROUP_ORDER:
        g = df[df["Group"] == group][col].dropna()
        chronic_rows.append({"Group": group, "Criterion": label,
                              "Prevalence (%)": round(g.mean() * 100, 1)})
fig_chronic = px.bar(pd.DataFrame(chronic_rows),
                      x="Group", y="Prevalence (%)", color="Criterion",
                      barmode="group",
                      title="Figure 4: Chronic Pain Prevalence by Group (%)",
                      category_orders={"Group": GROUP_ORDER})
fig_chronic = base_layout(fig_chronic, height=560)
save_fig(fig_chronic, "p1_04_chronic_pain_prevalence.html")

# ── P1-Fig05: Pain Location Heatmap ───────────────────────────────────────
pain_region_labels = {c: c.replace(pain_region_prefix, "") for c in pain_region_cols}
heat_rows = []
for col, label in pain_region_labels.items():
    for group in GROUP_ORDER:
        g = df[df["Group"] == group][col].dropna()
        heat_rows.append({"Region": label, "Group": group,
                           "Prevalence (%)": round(g.mean() * 100, 1)})
heat_df = pd.DataFrame(heat_rows).pivot(
    index="Region", columns="Group", values="Prevalence (%)")
fig_heat = px.imshow(heat_df, color_continuous_scale="RdYlBu_r",
                     text_auto=True, aspect="auto",
                     title="Figure 5: Pain Location Prevalence Heatmap (%)")
fig_heat.update_layout(height=max(400, len(pain_region_cols) * 40 + 120),
                        margin=dict(l=200, r=30, t=70, b=60))
save_fig(fig_heat, "p1_05_pain_location_heatmap.html")

# ── P1-Fig06: Pain Location Frequencies — Horizontal Bar ──────────────────
freq_rows = []
for col, label in pain_region_labels.items():
    pct = round(df[col].mean() * 100, 1)
    freq_rows.append({"Region": label, "Prevalence (%)": pct})
freq_df = pd.DataFrame(freq_rows).sort_values("Prevalence (%)", ascending=True)
fig_freq = px.bar(freq_df, x="Prevalence (%)", y="Region", orientation="h",
                   title="Figure 6: Pain Location Frequencies (Overall %)",
                   color="Prevalence (%)", color_continuous_scale="Reds")
fig_freq.update_layout(height=max(400, len(pain_region_cols) * 32 + 120),
                        margin=dict(l=200, r=30, t=70, b=60))
save_fig(fig_freq, "p1_06_pain_location_frequencies.html")

# ── P1-Fig07: Number of Painful Regions — Histogram ───────────────────────
fig_nregions = px.histogram(df, x="Pain_Region_Count", color="Group",
                             barmode="overlay", opacity=0.65, nbins=12,
                             color_discrete_map=GROUP_COLORS,
                             category_orders={"Group": GROUP_ORDER},
                             title="Figure 7: Number of Painful Regions by Group",
                             labels={"Pain_Region_Count": "Number of Painful Regions"})
fig_nregions = base_layout(fig_nregions, height=560)
save_fig(fig_nregions, "p1_07_pain_region_count_histogram.html")

# ── P1-Fig08: Pain Severity vs Number of Pain Regions ─────────────────────
fig_sev_reg = px.scatter(df, x="Pain_Region_Count", y=pain_col,
                          color="Group", trendline="ols",
                          color_discrete_map=GROUP_COLORS,
                          category_orders={"Group": GROUP_ORDER},
                          title="Figure 8: Pain Severity vs Number of Pain Regions",
                          labels={"Pain_Region_Count": "Number of Painful Regions",
                                  pain_col: "Pain Rating"})
fig_sev_reg = base_layout(fig_sev_reg, height=580)
save_fig(fig_sev_reg, "p1_08_pain_severity_vs_regions.html")

# ── P1-Fig09 & Fig10: Logistic Regression Forest Plot ─────────────────────
# Requires statsmodels; falls back gracefully if not available
try:
    import statsmodels.formula.api as smf

    # Binary outcome: pain (Pain vs No Pain)
    df["Pain_Binary"] = (df["Pain_Status"] == "Pain").astype(int)
    df["HIV_Binary"]  = (df["HIV_Status"]  == "HIV+").astype(int)

    # Candidate predictors — adjust column names to match your dataset
    predictor_map = {
        "Age":               "Age",
        "Weight":            "Weight",
        "HIV":               "HIV_Binary",
        "Smoking":           "Smoke_Cigarettes",
        "Depression":        "Major Depression Now",
        "Sleep Disorder":    "Sleep Condition Now",
        "Neuropathy":        "Peripheral Neuropathy Now",
        "Med Count":         "Med_Count",
        "Comorbidity Count": "Comorbidity_Count",
    }
    # Keep only columns that exist in the dataset
    valid_preds = {k: v for k, v in predictor_map.items() if v in df.columns}

    model_df = df[["Pain_Binary"] + list(valid_preds.values())].dropna()
    model_df.columns = ["Pain_Binary"] + list(valid_preds.keys())

    # Drop zero-variance predictors (causes singular Hessian / LinAlgError)
    zero_var = [c for c in model_df.columns[1:] if model_df[c].nunique() < 2]
    for c in zero_var:
        print(f"  Dropping zero-variance predictor: {c}")
    model_df = model_df.drop(columns=zero_var)

    predictors = [c for c in model_df.columns if c != "Pain_Binary"]
    formula = "Pain_Binary ~ " + " + ".join([f"Q('{k}')" for k in predictors])

    # Try Newton-Raphson first; fall back to BFGS if Hessian is singular
    try:
        model = smf.logit(formula, data=model_df).fit(disp=False)
    except Exception:
        print("  Newton failed, retrying with bfgs...")
        model = smf.logit(formula, data=model_df).fit(method="bfgs", disp=False)

    or_rows = []
    for term, coef in model.params.items():
        if "Intercept" in term:
            continue
        ci_lo = model.conf_int().loc[term, 0]
        ci_hi = model.conf_int().loc[term, 1]
        p_val = model.pvalues[term]
        label = term.replace("Q('", "").replace("')", "")
        or_rows.append({
            "Predictor":   label,
            "OR":          round(np.exp(coef),  3),
            "CI_Lo":       round(np.exp(ci_lo), 3),
            "CI_Hi":       round(np.exp(ci_hi), 3),
            "P":           round(p_val,          4),
            "Significant": p_val < 0.05
        })

    or_df = pd.DataFrame(or_rows).sort_values("OR")

    fig_forest = go.Figure()
    for _, row in or_df.iterrows():
        fig_forest.add_trace(go.Scatter(
            x=[row["OR"]], y=[row["Predictor"]],
            mode="markers",
            marker=dict(size=10, color="#e74c3c" if row["Significant"] else "#95a5a6"),
            error_x=dict(type="data", symmetric=False,
                         array=[row["CI_Hi"] - row["OR"]],
                         arrayminus=[row["OR"] - row["CI_Lo"]]),
            hovertemplate=(f"OR={row['OR']:.3f} [{row['CI_Lo']:.3f}–{row['CI_Hi']:.3f}]"
                           f"<br>p={row['P']}<extra>{row['Predictor']}</extra>"),
            showlegend=False
        ))
    fig_forest.add_vline(x=1, line_dash="dash", line_color="black")
    fig_forest.update_layout(
        title="Figure 9/10: Adjusted Logistic Regression — Predictors of Pain (Forest Plot)",
        xaxis_title="Odds Ratio (95% CI)",
        yaxis_title="Predictor",
        height=max(450, len(or_df) * 45 + 120),
        margin=dict(l=160, r=40, t=70, b=60),
        plot_bgcolor="white"
    )
    save_fig(fig_forest, "p1_09_logistic_regression_forest.html")

except ImportError:
    print("  statsmodels not installed — skipping forest plot (pip install statsmodels)")
except Exception as e:
    print(f"  Logistic regression failed: {e} — skipping forest plot")


# ═══════════════════════════════════════════════════════════════════════════
# PAPER 2 FIGURES
# ═══════════════════════════════════════════════════════════════════════════

# ── P2-Fig02: Comorbidity Burden — Average Count by Group ─────────────────
burden_df = df.groupby("Group")["Comorbidity_Count"].agg(
    Mean="mean", SEM=lambda x: stats.sem(x, nan_policy="omit")).reset_index()
burden_df["Group"] = pd.Categorical(burden_df["Group"], categories=GROUP_ORDER)
burden_df = burden_df.sort_values("Group")
fig_burden = go.Figure()
for _, row in burden_df.iterrows():
    fig_burden.add_trace(go.Bar(
        name=row["Group"], x=[row["Group"]], y=[row["Mean"]],
        error_y=dict(type="data", array=[row["SEM"]]),
        marker_color=GROUP_COLORS.get(row["Group"], "#aaa")
    ))
fig_burden.update_layout(
    title="Figure 2 (P2): Mean Comorbidity Count by Group (± SEM)",
    yaxis_title="Mean Number of Conditions",
    showlegend=False, height=540,
    margin=dict(l=60, r=30, t=70, b=80), plot_bgcolor="white"
)
save_fig(fig_burden, "p2_02_comorbidity_burden.html")

# ── P2-Fig03: Comorbidity Prevalence Heatmap ──────────────────────────────
comorbidity_cols_full = [
    "Diabetes Now", "HBP Now", "Stroke Now", "Cancer Now",
    "Kidney Disease Now", "Peripheral Neuropathy Now",
    "Major Depression Now", "Anxiety Disorder Now",
    "Sleep Condition Now", "Heart Disease Now"
]
comorbidity_cols_full = [c for c in comorbidity_cols_full if c in df.columns]
heat2_rows = {}
for col in comorbidity_cols_full:
    heat2_rows[col.replace(" Now", "")] = {
        g: round(df[df["Group"] == g][col].mean() * 100, 1)
        for g in GROUP_ORDER
    }
heat2_df = pd.DataFrame(heat2_rows).T[GROUP_ORDER]
fig_heat2 = px.imshow(heat2_df, color_continuous_scale="RdYlBu_r",
                       text_auto=True, aspect="auto",
                       title="Figure 3 (P2): Comorbidity Prevalence Heatmap by Group (%)")
fig_heat2.update_layout(height=480, margin=dict(l=200, r=30, t=70, b=100))
save_fig(fig_heat2, "p2_03_comorbidity_heatmap.html")

# ── P2-Fig04: Neurological Disorders ──────────────────────────────────────
neuro_cols = {
    "Stroke":              "Stroke Now",
    "Seizure Disorder":    "Seizure Disorder Now",
    "Neurological Disease":"Neurological Disease Now",
    "Peripheral Neuropathy":"Peripheral Neuropathy Now"
}
neuro_cols = {k: v for k, v in neuro_cols.items() if v in df.columns}
neuro_rows = []
for label, col in neuro_cols.items():
    for group in GROUP_ORDER:
        g = df[df["Group"] == group][col].dropna()
        neuro_rows.append({"Group": group, "Disorder": label,
                            "Prevalence (%)": round(g.mean() * 100, 1)})
fig_neuro = px.bar(pd.DataFrame(neuro_rows), x="Group", y="Prevalence (%)",
                    color="Disorder", barmode="stack",
                    category_orders={"Group": GROUP_ORDER},
                    title="Figure 4 (P2): Neurological Disorder Prevalence by Group (%)")
fig_neuro = base_layout(fig_neuro, height=560)
save_fig(fig_neuro, "p2_04_neurological_disorders.html")

# ── P2-Fig05: Psychiatric Disorders ───────────────────────────────────────
psych_cols = {
    "Major Depression":   "Major Depression Now",
    "Anxiety Disorder":   "Anxiety Disorder Now",
    "Bipolar Disorder":   "Bipolar Disorder Now",
    "Other MH":           "Other Mental Health Conditions Now"
}
psych_cols = {k: v for k, v in psych_cols.items() if v in df.columns}
psych_rows = []
for label, col in psych_cols.items():
    for group in GROUP_ORDER:
        g = df[df["Group"] == group][col].dropna()
        psych_rows.append({"Group": group, "Disorder": label,
                            "Prevalence (%)": round(g.mean() * 100, 1)})
fig_psych = px.bar(pd.DataFrame(psych_rows), x="Disorder", y="Prevalence (%)",
                    color="Group", barmode="group",
                    color_discrete_map=GROUP_COLORS,
                    category_orders={"Group": GROUP_ORDER},
                    title="Figure 5 (P2): Psychiatric Disorder Prevalence by Group (%)")
fig_psych = base_layout(fig_psych, height=560)
save_fig(fig_psych, "p2_05_psychiatric_disorders.html")

# ── P2-Fig06: Metabolic Disorders ─────────────────────────────────────────
metabolic_cols = {
    "Diabetes":      "Diabetes Now",
    "Hypertension":  "HBP Now",
    "Heart Disease": "Heart Disease Now",
    "Kidney Disease":"Kidney Disease Now"
}
metabolic_cols = {k: v for k, v in metabolic_cols.items() if v in df.columns}
met_rows = []
for label, col in metabolic_cols.items():
    for group in GROUP_ORDER:
        g = df[df["Group"] == group][col].dropna()
        met_rows.append({"Group": group, "Condition": label,
                          "Prevalence (%)": round(g.mean() * 100, 1)})
fig_met = px.bar(pd.DataFrame(met_rows), x="Condition", y="Prevalence (%)",
                  color="Group", barmode="group",
                  color_discrete_map=GROUP_COLORS,
                  category_orders={"Group": GROUP_ORDER},
                  title="Figure 6 (P2): Metabolic Disorder Prevalence by Group (%)")
fig_met = base_layout(fig_met, height=560)
save_fig(fig_met, "p2_06_metabolic_disorders.html")

# ── P2-Fig07: ART Drug Classes ────────────────────────────────────────────
art_map = {
    "NRTIs": "ART?_Nucleoside reverse transcriptase inhibitors (NRTIs)",
    "INSTI":  "ART?_Integrase Strand Transfer Inhibitor (INSTI)",
    "NNRTI": "ART?_Non-nucleoside reverse transcriptase inhibitor (NNRTI)",
    "PI":    "ART?_Protease Inhibitor (PI)"
}
art_map = {k: v for k, v in art_map.items() if v in df.columns}
hiv_only = df[df["Group"].isin(["HIV - Pain", "HIV - No Pain"])]
art_rows = []
for label, col in art_map.items():
    art_rows.append({"Drug Class": label,
                     "Count": int(hiv_only[col].sum()),
                     "Prevalence (%)": round(hiv_only[col].mean() * 100, 1)})
art_pie = px.pie(pd.DataFrame(art_rows), names="Drug Class", values="Count",
                  title="Figure 7A (P2): ART Drug Class Distribution (HIV Patients)",
                  color_discrete_sequence=px.colors.qualitative.Set2)
art_pie.update_layout(height=500, margin=dict(l=30, r=30, t=70, b=80))
save_fig(art_pie, "p2_07a_art_drug_classes_pie.html")

art_bar_rows = []
for label, col in art_map.items():
    for group in ["HIV - Pain", "HIV - No Pain"]:
        g = df[df["Group"] == group][col].dropna()
        art_bar_rows.append({"Group": group, "Drug Class": label,
                              "Prevalence (%)": round(g.mean() * 100, 1)})
fig_art_bar = px.bar(pd.DataFrame(art_bar_rows),
                      x="Drug Class", y="Prevalence (%)", color="Group",
                      barmode="group",
                      color_discrete_map=GROUP_COLORS,
                      title="Figure 7B (P2): ART Drug Class Prevalence — HIV Subgroups (%)")
fig_art_bar = base_layout(fig_art_bar, height=540)
save_fig(fig_art_bar, "p2_07b_art_drug_classes_bar.html")

# ── P2-Fig09: ART Adherence Histogram ─────────────────────────────────────
art_adhere_col = "Doses ART missed?"
if art_adhere_col in df.columns:
    fig_adhere_hist = px.histogram(hiv_only, x=art_adhere_col, color="Group",
                                    barmode="overlay", opacity=0.65, nbins=15,
                                    color_discrete_map=GROUP_COLORS,
                                    title="Figure 9 (P2): ART Missed Doses Distribution (HIV Only)",
                                    labels={art_adhere_col: "Doses of ART Missed"})
    fig_adhere_hist = base_layout(fig_adhere_hist, height=540)
    save_fig(fig_adhere_hist, "p2_09_art_adherence_histogram.html")

# ── P2-Fig11: Smoking by Group ────────────────────────────────────────────
smoke_rows = []
for smoke_col, label in [("Smoke_Cigarettes", "Cigarettes"),
                          ("Smoke_E-Cigarettes", "E-Cigarettes")]:
    if smoke_col in df.columns:
        for group in GROUP_ORDER:
            g = df[df["Group"] == group][smoke_col].dropna()
            smoke_rows.append({"Group": group, "Type": label,
                                "Prevalence (%)": round(g.mean() * 100, 1)})
fig_smoke = px.bar(pd.DataFrame(smoke_rows), x="Group", y="Prevalence (%)",
                    color="Type", barmode="group",
                    category_orders={"Group": GROUP_ORDER},
                    title="Figure 11 (P2): Smoking Prevalence by Group (%)")
fig_smoke = base_layout(fig_smoke, height=540)
save_fig(fig_smoke, "p2_11_smoking_by_group.html")

# ── P2-Fig12: Smoking vs Pain Severity — Boxplot ──────────────────────────
if "Smoke_Cigarettes" in df.columns:
    smoke_pain = df.dropna(subset=["Smoke_Cigarettes", pain_col]).copy()
    smoke_pain["Smoker"] = smoke_pain["Smoke_Cigarettes"].map({1: "Smoker", 0: "Non-Smoker"})
    fig_smoke_pain = px.box(smoke_pain, x="Smoker", y=pain_col, color="Group",
                             color_discrete_map=GROUP_COLORS,
                             category_orders={"Group": GROUP_ORDER},
                             title="Figure 12 (P2): Pain Rating by Smoking Status and Group",
                             labels={pain_col: "Pain Rating"})
    fig_smoke_pain = base_layout(fig_smoke_pain, height=560)
    save_fig(fig_smoke_pain, "p2_12_smoking_vs_pain_boxplot.html")

# ── P2-Fig13: Sleep Disorders Prevalence by Group ─────────────────────────
if "Sleep Condition Now" in df.columns:
    sleep_rows = []
    for group in GROUP_ORDER:
        g = df[df["Group"] == group]["Sleep Condition Now"].dropna()
        sleep_rows.append({"Group": group,
                            "Prevalence (%)": round(g.mean() * 100, 1)})
    fig_sleep = px.bar(pd.DataFrame(sleep_rows), x="Group", y="Prevalence (%)",
                        color="Group", color_discrete_map=GROUP_COLORS,
                        category_orders={"Group": GROUP_ORDER},
                        title="Figure 13 (P2): Sleep Disorder Prevalence by Group (%)")
    fig_sleep.update_traces(showlegend=False)
    fig_sleep = base_layout(fig_sleep, height=520)
    save_fig(fig_sleep, "p2_13_sleep_prevalence.html")

# ── P2-Fig14: Sleep Disorder vs Pain — Violin ─────────────────────────────
if "Sleep Condition Now" in df.columns:
    sleep_viol = df.dropna(subset=["Sleep Condition Now", pain_col]).copy()
    sleep_viol["Sleep Disorder"] = sleep_viol["Sleep Condition Now"].map(
        {1: "Sleep Disorder", 0: "No Sleep Disorder"})
    fig_sleep_pain = px.violin(sleep_viol, x="Sleep Disorder", y=pain_col,
                                color="Group", box=True,
                                color_discrete_map=GROUP_COLORS,
                                category_orders={"Group": GROUP_ORDER},
                                title="Figure 14 (P2): Pain Rating by Sleep Disorder Status",
                                labels={pain_col: "Pain Rating"})
    fig_sleep_pain = base_layout(fig_sleep_pain, height=580)
    save_fig(fig_sleep_pain, "p2_14_sleep_vs_pain_violin.html")

# ── P2-Fig15: Polypharmacy Distribution ───────────────────────────────────
fig_poly_hist = px.histogram(df, x="Med_Count", color="Group",
                              barmode="overlay", opacity=0.65, nbins=10,
                              color_discrete_map=GROUP_COLORS,
                              category_orders={"Group": GROUP_ORDER},
                              title="Figure 15 (P2): Medication Class Count Distribution by Group",
                              labels={"Med_Count": "Number of Medication Classes"})
fig_poly_hist = base_layout(fig_poly_hist, height=540)
save_fig(fig_poly_hist, "p2_15_polypharmacy_histogram.html")

# ── P2-Fig16: Medication Classes Stacked Bar ───────────────────────────────
# Rename med class columns for display
med_display = {}
for col in med_class_cols:
    label = col.replace("Medication Class_", "").strip()
    med_display[col] = label

med_stack_rows = []
for col, label in med_display.items():
    for group in GROUP_ORDER:
        g = df[df["Group"] == group][col].dropna()
        med_stack_rows.append({"Group": group, "Medication Class": label,
                                "Prevalence (%)": round(g.mean() * 100, 1)})
fig_med = px.bar(pd.DataFrame(med_stack_rows), x="Group",
                  y="Prevalence (%)", color="Medication Class",
                  barmode="stack",
                  category_orders={"Group": GROUP_ORDER},
                  title="Figure 16 (P2): Medication Class Prevalence by Group (% stacked)")
fig_med.update_layout(height=620, margin=dict(l=50, r=30, t=70, b=200),
                       legend=dict(orientation="h", yanchor="bottom",
                                   y=-0.55, xanchor="center", x=0.5))
save_fig(fig_med, "p2_16_medication_classes_stacked.html")

# ── P2-Fig17: Polypharmacy vs Hospitalization — Boxplot ───────────────────
if "Hospitalized (Past Year)" in df.columns:
    fig_poly_hosp = px.box(df, x="Hospitalized (Past Year)", y="Med_Count",
                            color="Group", color_discrete_map=GROUP_COLORS,
                            category_orders={"Group": GROUP_ORDER},
                            title="Figure 17 (P2): Medication Count by Hospitalization Status",
                            labels={"Med_Count": "Number of Medication Classes",
                                    "Hospitalized (Past Year)": "Hospitalized (0=No, 1=Yes)"})
    fig_poly_hosp = base_layout(fig_poly_hosp, height=560)
    save_fig(fig_poly_hosp, "p2_17_polypharmacy_vs_hospitalization.html")

# ── P2-Fig18: Polypharmacy vs Pain — Scatter ──────────────────────────────
fig_poly_pain = px.scatter(df, x="Med_Count", y=pain_col,
                            color="Group", trendline="ols",
                            color_discrete_map=GROUP_COLORS,
                            category_orders={"Group": GROUP_ORDER},
                            title="Figure 18 (P2): Medication Count vs Pain Severity",
                            labels={"Med_Count": "Number of Medication Classes",
                                    pain_col: "Pain Rating"})
fig_poly_pain = base_layout(fig_poly_pain, height=580)
save_fig(fig_poly_pain, "p2_18_polypharmacy_vs_pain.html")

# ── P2-Fig21: Age vs Medication Count ─────────────────────────────────────
fig_age_med = px.scatter(df, x="Age", y="Med_Count",
                          color="Group", trendline="ols",
                          color_discrete_map=GROUP_COLORS,
                          category_orders={"Group": GROUP_ORDER},
                          title="Figure 21 (P2): Age vs Medication Count by Group",
                          labels={"Med_Count": "Number of Medication Classes"})
fig_age_med = base_layout(fig_age_med, height=580)
save_fig(fig_age_med, "p2_21_age_vs_medication_count.html")

# ── P2-Fig22: Weight vs Pain — Scatter ────────────────────────────────────
fig_wt_pain = px.scatter(df, x="Weight", y=pain_col,
                          color="Group", trendline="ols",
                          color_discrete_map=GROUP_COLORS,
                          category_orders={"Group": GROUP_ORDER},
                          title="Figure 22 (P2): Weight vs Pain Severity by Group",
                          labels={pain_col: "Pain Rating"})
fig_wt_pain = base_layout(fig_wt_pain, height=580)
save_fig(fig_wt_pain, "p2_22_weight_vs_pain.html")

# ── P2-Fig23: Weight vs Diabetes — Boxplot ────────────────────────────────
if "Diabetes Now" in df.columns:
    diab_box = df.dropna(subset=["Diabetes Now", "Weight"]).copy()
    diab_box["Diabetes"] = diab_box["Diabetes Now"].map({1: "Diabetes", 0: "No Diabetes"})
    fig_wt_diab = px.box(diab_box, x="Diabetes", y="Weight",
                          color="Group", color_discrete_map=GROUP_COLORS,
                          category_orders={"Group": GROUP_ORDER},
                          title="Figure 23 (P2): Weight by Diabetes Status and Group")
    fig_wt_diab = base_layout(fig_wt_diab, height=560)
    save_fig(fig_wt_diab, "p2_23_weight_vs_diabetes.html")

# ── P2-Fig24: Weight vs Hypertension — Boxplot ────────────────────────────
if "HBP Now" in df.columns:
    hbp_box = df.dropna(subset=["HBP Now", "Weight"]).copy()
    hbp_box["Hypertension"] = hbp_box["HBP Now"].map({1: "HBP", 0: "No HBP"})
    fig_wt_hbp = px.box(hbp_box, x="Hypertension", y="Weight",
                         color="Group", color_discrete_map=GROUP_COLORS,
                         category_orders={"Group": GROUP_ORDER},
                         title="Figure 24 (P2): Weight by Hypertension Status and Group")
    fig_wt_hbp = base_layout(fig_wt_hbp, height=560)
    save_fig(fig_wt_hbp, "p2_24_weight_vs_hypertension.html")

# ── P2-Fig25: Hospitalization by Group ────────────────────────────────────
if "Hospitalized (Past Year)" in df.columns:
    hosp_rows = []
    for group in GROUP_ORDER:
        g = df[df["Group"] == group]["Hospitalized (Past Year)"].dropna()
        hosp_rows.append({"Group": group,
                           "Hospitalized (%)": round(g.mean() * 100, 1)})
    fig_hosp = px.bar(pd.DataFrame(hosp_rows), x="Group", y="Hospitalized (%)",
                       color="Group", color_discrete_map=GROUP_COLORS,
                       category_orders={"Group": GROUP_ORDER},
                       title="Figure 25 (P2): Hospitalization Rate by Group (%)")
    fig_hosp.update_traces(showlegend=False)
    fig_hosp = base_layout(fig_hosp, height=520)
    save_fig(fig_hosp, "p2_25_hospitalization_by_group.html")

# ── P2-Fig27: Correlation Matrix ──────────────────────────────────────────
corr_vars = {
    "Age":              "Age",
    "Weight":           "Weight",
    "Pain Rating":      pain_col,
    "Med Count":        "Med_Count",
    "Comorbidity Count":"Comorbidity_Count",
    "ART Missed":       "Doses ART missed?"
}
corr_vars = {k: v for k, v in corr_vars.items() if v in df.columns}
corr_df = df[list(corr_vars.values())].rename(columns={v: k for k, v in corr_vars.items()})
corr_matrix = corr_df.corr(method="spearman")
fig_corr = px.imshow(corr_matrix, color_continuous_scale="RdBu_r",
                      zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
                      title="Figure 27 (P2): Spearman Correlation Matrix — Continuous Variables")
fig_corr.update_layout(height=540, margin=dict(l=140, r=30, t=70, b=120))
save_fig(fig_corr, "p2_27_correlation_matrix.html")

# ── Age Category Figures ───────────────────────────────────────────────────
age_bins   = [18, 30, 40, 50, 60, 100]
age_labels = ["18–29", "30–39", "40–49", "50–59", "60+"]
df["Age_Category"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)

# Age categories stacked bar
age_cat_df = df.groupby(["Group", "Age_Category"], observed=True).size().reset_index(name="Count")
age_cat_df["Pct"] = age_cat_df.groupby("Group")["Count"].transform(
    lambda x: round(x / x.sum() * 100, 1))
fig_age_cat = px.bar(age_cat_df, x="Group", y="Pct", color="Age_Category",
                      barmode="stack",
                      category_orders={"Group": GROUP_ORDER,
                                       "Age_Category": age_labels},
                      title="Age Category Distribution by Group (%)",
                      labels={"Pct": "Percentage (%)"})
fig_age_cat = base_layout(fig_age_cat, height=560)
save_fig(fig_age_cat, "age_01_age_categories_stacked.html")

# Pain prevalence by age category
pain_by_age = []
for cat in age_labels:
    g = df[df["Age_Category"] == cat]
    pain_by_age.append({"Age Category": cat,
                         "Pain Prevalence (%)": round(
                             (g["Pain_Status"] == "Pain").mean() * 100, 1)})
fig_pain_age = px.line(pd.DataFrame(pain_by_age),
                        x="Age Category", y="Pain Prevalence (%)", markers=True,
                        title="Pain Prevalence by Age Category")
fig_pain_age = base_layout(fig_pain_age, height=500)
save_fig(fig_pain_age, "age_02_pain_prevalence_by_age.html")

# Comorbidity count by age category
comor_by_age = df.groupby("Age_Category", observed=True)["Comorbidity_Count"].mean().reset_index()
fig_comor_age = px.bar(comor_by_age, x="Age_Category", y="Comorbidity_Count",
                        title="Mean Comorbidity Count by Age Category",
                        labels={"Age_Category": "Age Category",
                                 "Comorbidity_Count": "Mean Comorbidities"},
                        color="Comorbidity_Count",
                        color_continuous_scale="Reds")
fig_comor_age = base_layout(fig_comor_age, height=500)
save_fig(fig_comor_age, "age_03_comorbidity_by_age.html")

# ── Sex-Specific Figures ───────────────────────────────────────────────────
# Auto-detect sex/gender column — tries exact names first, then partial match
sex_col = next((c for c in df.columns if c.lower() in [
    "sex", "gender", "sex at birth", "biological sex",
    "patient sex", "sex (m/f)"]), None)
if sex_col is None:
    sex_col = next((c for c in df.columns if any(
        x in c.lower() for x in ["sex", "gender", "male", "female"])), None)
if sex_col is None:
    print("  NOTE: No sex/gender column found — skipping sex figures.")
    print("  Tip: manually set sex_col = 'YourColumnName' here.")
if sex_col:
    # Pain prevalence by sex
    sex_pain = df.groupby(sex_col).apply(
        lambda x: (x["Pain_Status"] == "Pain").mean() * 100).reset_index()
    sex_pain.columns = [sex_col, "Pain Prevalence (%)"]
    fig_sex_pain = px.bar(sex_pain, x=sex_col, y="Pain Prevalence (%)",
                           color=sex_col,
                           title="Pain Prevalence by Sex (%)")
    fig_sex_pain = base_layout(fig_sex_pain, height=500)
    save_fig(fig_sex_pain, "sex_01_pain_prevalence_by_sex.html")

    # Pain severity by sex
    fig_sex_sev = px.violin(df.dropna(subset=[sex_col, pain_col]),
                             x=sex_col, y=pain_col, box=True, color=sex_col,
                             title="Pain Severity by Sex",
                             labels={pain_col: "Pain Rating"})
    fig_sex_sev = base_layout(fig_sex_sev, height=540)
    save_fig(fig_sex_sev, "sex_02_pain_severity_by_sex.html")

    # Comorbidity count by sex
    fig_sex_comor = px.box(df.dropna(subset=[sex_col]),
                            x=sex_col, y="Comorbidity_Count", color=sex_col,
                            title="Comorbidity Count by Sex",
                            labels={"Comorbidity_Count": "Number of Conditions"})
    fig_sex_comor = base_layout(fig_sex_comor, height=520)
    save_fig(fig_sex_comor, "sex_03_comorbidity_by_sex.html")

# ═══════════════════════════════════════════════════════════════════════════
# ORIGINAL 12 CORRELATIONS (unchanged — kept for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════

mental_health_cols = [
    "Major Depression Now", "Anxiety Disorder Now",
    "Bipolar Disorder Now", "Other Mental Health Conditions Now"
]

# ── CORRELATION 1 ─────────────────────────────────────────────────────────
rows = []
for col in mental_health_cols:
    for group in df["Group"].unique():
        g = df[df["Group"] == group].dropna(subset=[col, pain_col])
        if len(g) < 5: continue
        r, p = stats.pointbiserialr(g[col], g[pain_col])
        rows.append({"Condition": col.replace(" Now",""), "Group": group,
                     "Correlation (r)": round(r,3), "P-Value": round(p,4),
                     "Significant": "Yes" if p<0.05 else "No"})
fig1 = px.bar(pd.DataFrame(rows), x="Condition", y="Correlation (r)",
              color="Group", barmode="group", color_discrete_map=GROUP_COLORS,
              title="Correlation 1: Pain Severity vs Mental Health Conditions",
              labels={"Correlation (r)": "Point-Biserial r"},
              hover_data=["P-Value","Significant"])
fig1.add_hline(y=0, line_dash="dash", line_color="black")
fig1 = base_layout(fig1, height=600)
save_fig(fig1, "01_pain_vs_mental_health.html")

# ── CORRELATION 2 ─────────────────────────────────────────────────────────
chronic_cols = ["Pain > 3 Months", "Pain >12hrs or More 6 months"]
rows = []
for pain_c in chronic_cols:
    for mh_col in mental_health_cols:
        for group in df["Group"].unique():
            g = df[df["Group"] == group].dropna(subset=[pain_c, mh_col])
            if len(g) < 5: continue
            r, p = stats.pointbiserialr(g[mh_col], g[pain_c])
            rows.append({"Pain Duration": "Pain > 3 Months" if "3 Months" in pain_c
                          else "Pain > 12hrs/day",
                          "MH Condition": mh_col.replace(" Now",""), "Group": group,
                          "Correlation (r)": round(r,3), "P-Value": round(p,4),
                          "Significant": "Yes" if p<0.05 else "No"})
fig2 = px.bar(pd.DataFrame(rows), x="MH Condition", y="Correlation (r)",
              color="Group", barmode="group", facet_row="Pain Duration",
              color_discrete_map=GROUP_COLORS,
              title="Correlation 2: Chronic Pain Duration vs Mental Health",
              hover_data=["P-Value","Significant"])
fig2.add_hline(y=0, line_dash="dash", line_color="black")
fig2.update_layout(autosize=True, height=800, margin=dict(l=50,r=30,t=70,b=180),
    legend=dict(orientation="h",yanchor="bottom",y=-0.25,xanchor="center",x=0.5,
                font=dict(size=12),itemwidth=110), font=dict(size=12), title_font=dict(size=15))
fig2.for_each_xaxis(lambda ax: ax.update(tickangle=-25, tickfont=dict(size=11)))
fig2.for_each_yaxis(lambda ax: ax.update(tickfont=dict(size=11)))
save_fig(fig2, "02_chronic_pain_vs_mental_health.html")

# ── CORRELATION 3 ─────────────────────────────────────────────────────────
rows = []
hiv_df = df[df["Group"].isin(["HIV - Pain","HIV - No Pain"])].dropna(
    subset=["Doses ART missed?", pain_col])
r, p = stats.spearmanr(hiv_df["Doses ART missed?"], hiv_df[pain_col])
rows.append({"Comparison":"ART Missed vs Pain Rating","r":round(r,3),"p":round(p,4)})
for col in ["Pain > 3 Months","Pain >12hrs or More 6 months"]:
    g = hiv_df.dropna(subset=[col])
    r, p = stats.pointbiserialr(g[col], g["Doses ART missed?"])
    rows.append({"Comparison": "ART Missed vs Pain > 3 Months" if "3 Months" in col
                  else "ART Missed vs Pain > 12hrs/day", "r":round(r,3),"p":round(p,4)})
fig3 = px.bar(pd.DataFrame(rows), x="Comparison", y="r",
              title="Correlation 3: ART Adherence vs Pain (HIV Patients Only)",
              labels={"r":"Spearman / Point-Biserial r"}, color="Comparison", text="p")
fig3.update_traces(texttemplate="p=%{text}", textposition="outside")
fig3.add_hline(y=0, line_dash="dash", line_color="black")
fig3 = base_layout(fig3, height=580)
save_fig(fig3, "03_art_adherence_vs_pain.html")

# ── CORRELATION 4 ─────────────────────────────────────────────────────────
art_cols = ["ART?_Nucleoside reverse transcriptase inhibitors (NRTIs)",
            "ART?_Integrase Strand Transfer Inhibitor (INSTI)",
            "ART?_Non-nucleoside reverse transcriptase inhibitor (NNRTI)",
            "ART?_Protease Inhibitor (PI)"]
art_labels = ["NRTIs","INSTI","NNRTI","PI"]
rows = []
hiv_df2 = df[df["Group"].isin(["HIV - Pain","HIV - No Pain"])]
for col, label in zip(art_cols, art_labels):
    g = hiv_df2.dropna(subset=[col,"Peripheral Neuropathy Now"])
    if len(g) < 5: continue
    chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(g[col],g["Peripheral Neuropathy Now"]))
    rows.append({"ART Class":label,"Chi2":round(chi2,3),"P-Value":round(p,4),
                 "Significant":"Yes" if p<0.05 else "No"})
fig4 = px.bar(pd.DataFrame(rows), x="ART Class", y="Chi2",
              color="Significant", color_discrete_map={"Yes":"#e74c3c","No":"#95a5a6"},
              title="Correlation 4: ART Drug Class vs Peripheral Neuropathy (Chi-Square)",
              hover_data=["P-Value"])
fig4 = base_layout(fig4, height=560)
save_fig(fig4, "04_art_class_vs_neuropathy.html")

# ── CORRELATION 5 ─────────────────────────────────────────────────────────
metabolic_pairs = [("Weight","Diabetes Now"),("Weight","HBP Now"),
                   ("HBP Now","Heart Disease Now"),("Diabetes Now","Kidney Disease Now")]
rows = []
for col_a, col_b in metabolic_pairs:
    for group in df["Group"].unique():
        g = df[df["Group"] == group].dropna(subset=[col_a,col_b])
        if len(g) < 5: continue
        if df[col_a].nunique() > 2:
            r, p = stats.pointbiserialr(g[col_b], g[col_a])
        else:
            chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(g[col_a],g[col_b]))
            r = chi2
        rows.append({"Pair":f"{col_a} vs {col_b}","Group":group,
                     "Statistic":round(r,3),"P-Value":round(p,4),
                     "Significant":"Yes" if p<0.05 else "No"})
fig5 = px.bar(pd.DataFrame(rows), x="Pair", y="Statistic",
              color="Group", barmode="group", color_discrete_map=GROUP_COLORS,
              title="Correlation 5: Metabolic Comorbidity Clustering by Group",
              hover_data=["P-Value","Significant"])
fig5 = base_layout(fig5, height=600)
save_fig(fig5, "05_metabolic_clustering.html")

# ── CORRELATION 6 ─────────────────────────────────────────────────────────
fig6 = px.scatter(df, x="Age", y="Comorbidity_Count", color="Group", trendline="ols",
                   color_discrete_map=GROUP_COLORS,
                   title="Correlation 6: Age vs Total Comorbidity Count by Group",
                   labels={"Comorbidity_Count":"Number of Current Conditions"})
fig6 = base_layout(fig6, height=600)
save_fig(fig6, "06_age_vs_comorbidity.html")

# ── CORRELATION 7 ─────────────────────────────────────────────────────────
fig7 = px.scatter(df, x="Age", y=pain_col, color="Group", trendline="ols",
                   color_discrete_map=GROUP_COLORS,
                   title="Correlation 7: Age vs Pain Severity by Group")
fig7 = base_layout(fig7, height=600)
save_fig(fig7, "07_age_vs_pain.html")

# ── CORRELATION 8 ─────────────────────────────────────────────────────────
rows = []
for group in df["Group"].unique():
    for smoke_col, label in [("Smoke_Cigarettes","Cigarettes"),("Smoke_E-Cigarettes","E-Cigarettes")]:
        g = df[df["Group"]==group].dropna(subset=[smoke_col, pain_col])
        if len(g) < 5: continue
        r, p = stats.pointbiserialr(g[smoke_col], g[pain_col])
        rows.append({"Group":group,"Smoke Type":label,"r":round(r,3),"P-Value":round(p,4)})
fig8 = px.bar(pd.DataFrame(rows), x="Group", y="r", color="Smoke Type", barmode="group",
              title="Correlation 8: Smoking vs Pain Severity by Group", hover_data=["P-Value"])
fig8.add_hline(y=0, line_dash="dash", line_color="black")
fig8 = base_layout(fig8, height=600)
save_fig(fig8, "08_smoking_vs_pain.html")

# ── CORRELATION 9 ─────────────────────────────────────────────────────────
sleep_targets = [pain_col] + mental_health_cols
rows = []
for target in sleep_targets:
    for group in df["Group"].unique():
        g = df[df["Group"]==group].dropna(subset=["Sleep Condition Now", target])
        if len(g) < 5: continue
        r, p = stats.pointbiserialr(g["Sleep Condition Now"], g[target])
        rows.append({"Target": target.replace(" Now","").replace(pain_col,"Pain Rating"),
                     "Group":group,"r":round(r,3),"P-Value":round(p,4),
                     "Significant":"Yes" if p<0.05 else "No"})
fig9 = px.bar(pd.DataFrame(rows), x="Target", y="r", color="Group", barmode="group",
              color_discrete_map=GROUP_COLORS,
              title="Correlation 9: Sleep Condition vs Pain & Mental Health",
              hover_data=["P-Value","Significant"])
fig9.add_hline(y=0, line_dash="dash", line_color="black")
fig9 = base_layout(fig9, height=600)
save_fig(fig9, "09_sleep_vs_pain_mentalhealth.html")

# ── CORRELATION 10 ────────────────────────────────────────────────────────
fig10a = px.scatter(df, x="Med_Count", y=pain_col, color="Group", trendline="ols",
                    color_discrete_map=GROUP_COLORS,
                    title="Correlation 10a: Medication Classes vs Pain Severity",
                    labels={"Med_Count":"Number of Medication Classes"})
fig10a = base_layout(fig10a, height=600)
save_fig(fig10a, "10a_polypharmacy_vs_pain.html")

fig10b = px.box(df, x="Hospitalized (Past Year)", y="Med_Count",
                color="Group", color_discrete_map=GROUP_COLORS,
                title="Correlation 10b: Hospitalization vs Medication Classes",
                labels={"Med_Count":"Number of Medication Classes",
                        "Hospitalized (Past Year)":"Hospitalized (0=No, 1=Yes)"})
fig10b = base_layout(fig10b, height=600)
save_fig(fig10b, "10b_polypharmacy_vs_hospitalization.html")

# ── CORRELATION 11 ────────────────────────────────────────────────────────
rows = []
for col in pain_region_cols:
    for group in df["Group"].unique():
        g = df[df["Group"]==group].dropna(subset=[col,"Peripheral Neuropathy Now"])
        if len(g) < 5: continue
        chi2, p, _, _ = stats.chi2_contingency(
            pd.crosstab(g[col], g["Peripheral Neuropathy Now"]))
        rows.append({"Pain Region":col.split("_")[-1],"Group":group,
                     "Chi2":round(chi2,3),"P-Value":round(p,4),
                     "Significant":"Yes" if p<0.05 else "No"})
fig11 = px.bar(pd.DataFrame(rows), x="Pain Region", y="Chi2",
               color="Group", barmode="group", color_discrete_map=GROUP_COLORS,
               title="Correlation 11: Lower Limb Pain Region vs Peripheral Neuropathy",
               hover_data=["P-Value","Significant"])
fig11 = base_layout(fig11, height=580)
save_fig(fig11, "11_pain_region_vs_neuropathy.html")

# ── CORRELATION 12 ────────────────────────────────────────────────────────
comorbidity_cols_12 = ["HBP Now","Heart Disease Now","Diabetes Now","Cancer Now",
    "Kidney Disease Now","Sleep Condition Now","Major Depression Now",
    "Anxiety Disorder Now","Peripheral Neuropathy Now","Stroke Now"]
prev_rows = []
for col in comorbidity_cols_12:
    for group in df["Group"].unique():
        g = df[df["Group"]==group]
        prev_rows.append({"Condition":col.replace(" Now",""),"Group":group,
                           "Prevalence (%)":round(g[col].mean()*100,1)})
fig12 = px.bar(pd.DataFrame(prev_rows), x="Condition", y="Prevalence (%)",
               color="Group", barmode="group", color_discrete_map=GROUP_COLORS,
               title="Correlation 12: Comorbidity Prevalence Across All Four Groups (%)")
fig12 = base_layout(fig12, height=620)
save_fig(fig12, "12_comorbidity_prevalence_by_group.html")

print("\nAll graphs saved to docs/graphs/")