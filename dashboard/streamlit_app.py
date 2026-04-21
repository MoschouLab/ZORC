"""
ZORC — P11b Streamlit Gene Explorer
Lab tool for querying P-body mRNA enrichment predictions.
Run from project root: streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.impute import SimpleImputer

# ── Paths (relative to project root) ─────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DB_PATH       = ROOT / "data" / "zorc_database.db"
FM_PATH       = ROOT / "data" / "processed" / "08_zorc_feature_matrix.csv"
BASE_MODEL    = ROOT / "results" / "09f_rf_base_model.pkl"
FINAL_MODEL   = ROOT / "results" / "09f_rf_final_model.pkl"
PREDS_PATH    = ROOT / "results" / "09f_predictions_final.csv"
SHAP_IMP_PATH = ROOT / "results" / "09f_shap_final.csv"
PROBES_PATH   = ROOT / "results" / "11a_xenium_probe_candidates.csv"
HC_PATH       = ROOT / "data" / "processed" / "colleague_high_confidence_set.csv"

META_COLS = {
    "geneID", "gene_id", "transcript_id", "class", "condition", "qc_fail",
    "isoform_source", "event_type", "bioemu_tier", "bioemu_status",
    "feature_source", "cluster_id", "split", "aiupred_status",
}
DROP_FEATURES = {"packing_x_idr", "rrach_per_cds_kb"}
EPS_RMSF = 0.1

CONFIDENCE_THRESHOLDS = {"High": 0.75, "Medium": 0.55}
PALETTE = {"positive": "#2E86AB", "negative": "#E84855", "neutral": "#A8A8A8"}

st.set_page_config(
    page_title="ZORC Gene Explorer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Data loading helpers (cached) ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_feature_matrix() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(FM_PATH)
    if "rmsf_nterm50" in df.columns and "rmsf_cterm50" in df.columns:
        df["rmsf_nterm_cterm_ratio"] = df["rmsf_nterm50"] / (df["rmsf_cterm50"] + EPS_RMSF)
    if "utr3_au_content" in df.columns and "utr3_length" in df.columns:
        df["utr3_au_x_length"] = df["utr3_au_content"] * df["utr3_length"]
    feature_cols = [c for c in df.columns if c not in META_COLS and c not in DROP_FEATURES]
    return df, feature_cols


@st.cache_resource(show_spinner=False)
def load_models():
    with open(BASE_MODEL, "rb") as f:
        rf_base = pickle.load(f)
    with open(FINAL_MODEL, "rb") as f:
        rf_cal = pickle.load(f)
    return rf_base, rf_cal


@st.cache_resource(show_spinner=False)
def build_imputer_and_explainer(feature_cols: tuple[str, ...]):
    df, _ = load_feature_matrix()
    X = df[list(feature_cols)].values.astype(np.float32)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    rf_base, _ = load_models()
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_imp), min(200, len(X_imp)), replace=False)
    explainer = shap.TreeExplainer(
        rf_base, X_imp[bg_idx], feature_perturbation="interventional"
    )
    return imputer, explainer, X_imp


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDS_PATH)


@st.cache_data(show_spinner=False)
def load_shap_importance() -> pd.DataFrame:
    return pd.read_csv(SHAP_IMP_PATH)


@st.cache_data(show_spinner=False)
def load_probes() -> pd.DataFrame:
    return pd.read_csv(PROBES_PATH)


@st.cache_data(show_spinner=False)
def load_hc_set() -> pd.DataFrame:
    return pd.read_csv(HC_PATH)


def db_query(sql: str) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(f"ATTACH '{DB_PATH}' AS z (TYPE SQLITE)")
    result = con.execute(sql).df()
    con.close()
    return result


# ── Shared data ───────────────────────────────────────────────────────────────

def get_shared_data():
    fm, feature_cols = load_feature_matrix()
    fc_tuple = tuple(feature_cols)
    imputer, explainer, X_imp = build_imputer_and_explainer(fc_tuple)
    preds = load_predictions()
    return fm, feature_cols, imputer, explainer, X_imp, preds


# ── Confidence label ──────────────────────────────────────────────────────────

def confidence_label(prob: float) -> str:
    if prob >= CONFIDENCE_THRESHOLDS["High"] or prob <= (1 - CONFIDENCE_THRESHOLDS["High"]):
        return "High"
    if prob >= CONFIDENCE_THRESHOLDS["Medium"] or prob <= (1 - CONFIDENCE_THRESHOLDS["Medium"]):
        return "Medium"
    return "Low"


# ── Page 1: Gene Search ───────────────────────────────────────────────────────

def page_gene_search():
    st.title("🔍 Gene Search")
    st.markdown("Enter an *Arabidopsis thaliana* AGI code to retrieve the ZORC P-body enrichment prediction.")

    fm, feature_cols, imputer, explainer, X_imp, preds = get_shared_data()

    gene_input = st.text_input(
        "AGI code (e.g. AT5G47010)",
        value="AT5G47010",
        max_chars=12,
    ).strip().upper()

    if not gene_input:
        return

    # ── Look up gene in SQLite ────────────────────────────────────────────────
    row_db = db_query(f"""
        SELECT g.gene_id, g.gene_name, g.condition, g.class,
               g.bioemu_tier, g.split,
               ROUND(p.prob_pos, 4) AS prob_pos, p.pred,
               f.mrna_length, f.cds_length, f.utr3_length, f.utr5_length,
               ROUND(f.utr3_au_content, 4)       AS utr3_au_content,
               ROUND(f.rrach_count, 0)            AS rrach_count,
               ROUND(f.rrach_per_kb, 2)           AS rrach_per_kb,
               ROUND(f.idr_percent, 2)            AS idr_percent,
               ROUND(f.rmsf_mean, 2)              AS rmsf_mean,
               ROUND(f.rmsf_nterm50, 2)           AS rmsf_nterm50,
               ROUND(f.rmsf_cterm50, 2)           AS rmsf_cterm50,
               ROUND(f.rmsf_nterm_cterm_ratio, 3) AS rmsf_ratio,
               ROUND(f.n_residues, 0)             AS n_residues,
               ROUND(f.mfe_per_nt, 4)             AS mfe_per_nt
        FROM z.genes g
        JOIN z.predictions p USING (gene_id)
        JOIN z.features f USING (gene_id)
        WHERE g.gene_id = '{gene_input}'
    """)

    if row_db.empty:
        st.warning(f"Gene **{gene_input}** not found in the ZORC database. "
                   "Check the AGI code or verify the gene is in the coregulon.")
        return

    row = row_db.iloc[0]
    prob = float(row["prob_pos"])
    label = "P-body enriched" if row["pred"] == 1 else "Not enriched"
    conf  = confidence_label(prob)
    color = PALETTE["positive"] if row["pred"] == 1 else PALETTE["negative"]

    # ── Header row ────────────────────────────────────────────────────────────
    gname = row["gene_name"] if pd.notna(row["gene_name"]) else ""
    st.subheader(f"{gene_input}  {'  |  ' + gname if gname else ''}")

    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    col_meta1.metric("Prediction", label)
    col_meta2.metric("Confidence", conf)
    col_meta3.metric("Condition", row["condition"])
    col_meta4.metric("BioEmu tier", int(row["bioemu_tier"]) if pd.notna(row["bioemu_tier"]) else "N/A")

    # ── Probability gauge ─────────────────────────────────────────────────────
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob, 3),
        number={"font": {"size": 32}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 0.25], "color": "#FADADD"},
                {"range": [0.25, 0.55], "color": "#FFF3CD"},
                {"range": [0.55, 0.75], "color": "#D4EDDA"},
                {"range": [0.75, 1.0], "color": "#C3E6CB"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 0.5,
            },
        },
        title={"text": "P(P-body enriched)", "font": {"size": 16}},
    ))
    fig_gauge.update_layout(height=260, margin=dict(t=40, b=10, l=40, r=40))

    # ── SHAP waterfall (per gene) ─────────────────────────────────────────────
    gene_row_fm = fm[fm["geneID"] == gene_input]
    shap_fig = None
    if not gene_row_fm.empty:
        idx = gene_row_fm.index[0]
        X_single_raw = gene_row_fm[feature_cols].values.astype(np.float32)
        X_single_imp = imputer.transform(X_single_raw)
        sv_raw = np.array(explainer.shap_values(X_single_imp))
        # sv_raw shape: (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        if sv_raw.ndim == 3 and sv_raw.shape[0] == 1:
            sv = sv_raw[0, :, 1]   # (n_features, n_classes)[class=1]
        elif sv_raw.ndim == 3:
            sv = sv_raw[:, 0, 1]   # (n_classes, n_samples, n_features) → class=1, sample=0
        else:
            sv = sv_raw.ravel()
        shap_df = (
            pd.DataFrame({"feature": feature_cols, "shap": sv})
            .reindex(pd.Series(sv).abs().sort_values(ascending=False).index)
            .head(10)
            .sort_values("shap")
        )
        colors = [PALETTE["positive"] if v > 0 else PALETTE["negative"] for v in shap_df["shap"]]
        fig_shap = go.Figure(go.Bar(
            x=shap_df["shap"],
            y=shap_df["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in shap_df["shap"]],
            textposition="outside",
        ))
        fig_shap.add_vline(x=0, line_color="black", line_width=1)
        fig_shap.update_layout(
            title="Top 10 SHAP contributions (this gene)",
            xaxis_title="SHAP value → pushes toward P-body enriched",
            yaxis_title="",
            height=360,
            margin=dict(t=40, b=30, l=10, r=80),
        )
        shap_fig = fig_shap

    col_gauge, col_shap = st.columns([1, 2])
    with col_gauge:
        st.plotly_chart(fig_gauge, use_container_width=True)
    with col_shap:
        if shap_fig:
            st.plotly_chart(shap_fig, use_container_width=True)
        else:
            st.info("SHAP unavailable (gene not in feature matrix).")

    # ── Feature values vs population violin plots ─────────────────────────────
    st.markdown("### Feature context — gene vs population")
    VIOLIN_FEATURES = [
        "utr3_au_content", "rrach_count", "idr_percent",
        "rmsf_nterm_cterm_ratio", "mfe_per_nt",
    ]
    fm_plot = fm.copy()
    fm_plot["label"] = fm_plot["class"].map({1: "Positive", 0: "Negative"})

    gene_vals = {
        "utr3_au_content":       row["utr3_au_content"],
        "rrach_count":           row["rrach_count"],
        "idr_percent":           row["idr_percent"],
        "rmsf_nterm_cterm_ratio": row["rmsf_ratio"],
        "mfe_per_nt":            row["mfe_per_nt"],
    }

    vcols = st.columns(len(VIOLIN_FEATURES))
    for i, feat in enumerate(VIOLIN_FEATURES):
        if feat not in fm.columns:
            continue
        fig_v = go.Figure()
        for cls, col_name in [(1, PALETTE["positive"]), (0, PALETTE["negative"])]:
            vals = fm_plot.loc[fm_plot["class"] == cls, feat].dropna()
            fig_v.add_trace(go.Violin(
                y=vals,
                name="Pos" if cls == 1 else "Neg",
                box_visible=False,
                meanline_visible=True,
                line_color=col_name,
                fillcolor=col_name,
                opacity=0.5,
                showlegend=False,
            ))
        gene_val = gene_vals.get(feat)
        if pd.notna(gene_val):
            fig_v.add_hline(
                y=float(gene_val),
                line_color="black",
                line_dash="dash",
                line_width=2,
                annotation_text=f"{float(gene_val):.3g}",
                annotation_position="top right",
            )
        fig_v.update_layout(
            title=feat.replace("_", " "),
            height=220,
            margin=dict(t=35, b=5, l=5, r=5),
            yaxis_title="",
        )
        with vcols[i]:
            st.plotly_chart(fig_v, use_container_width=True)

    # ── Key feature table ─────────────────────────────────────────────────────
    with st.expander("All feature values for this gene"):
        display_cols = [
            "mrna_length", "cds_length", "utr3_length", "utr5_length",
            "utr3_au_content", "rrach_count", "rrach_per_kb",
            "idr_percent", "rmsf_mean", "rmsf_nterm50", "rmsf_cterm50",
            "rmsf_ratio", "n_residues", "mfe_per_nt",
        ]
        feat_table = {c: row.get(c, np.nan) for c in display_cols}
        st.dataframe(
            pd.DataFrame(feat_table, index=["value"]).T.reset_index().rename(
                columns={"index": "feature", "value": "value"}
            ),
            use_container_width=True,
        )

    # ── External links ────────────────────────────────────────────────────────
    st.markdown("### External resources")
    tair_url     = f"https://www.arabidopsis.org/servlets/TairObject?type=locus&name={gene_input}"
    uniprot_url  = f"https://www.uniprot.org/uniprotkb?query={gene_input}&organism_id=3702"
    af2_url      = f"https://alphafold.ebi.ac.uk/search/text/{gene_input}"
    st.markdown(
        f"[TAIR — {gene_input}]({tair_url})  |  "
        f"[UniProt search]({uniprot_url})  |  "
        f"[AlphaFold2 entry]({af2_url})"
    )


# ── Page 2: Xenium Probe Candidates ──────────────────────────────────────────

def page_xenium():
    st.title("🔬 Xenium Probe Candidates")
    st.markdown(
        "Candidate genes for **Xenium In Situ** (10X Genomics) spatial transcriptomics validation.  \n"
        "**150 positive probes** (class=1, P(pos) > 0.75) and **150 negative controls** "
        "(class=0, P(pos) < 0.25) selected by the ZORC final model (P9f)."
    )

    probes = load_probes()

    tab_pos, tab_neg, tab_all = st.tabs(["Positive probes (150)", "Negative controls (150)", "All (300)"])

    def probe_table_and_scatter(df: pd.DataFrame, probe_label: str):
        is_pos = probe_label == "positive_probe"
        color  = PALETTE["positive"] if is_pos else PALETTE["negative"]

        # Scatter: utr3_au_content vs rrach_count
        fig = go.Figure(go.Scatter(
            x=df["utr3_au_content"],
            y=df["rrach_count"],
            mode="markers",
            marker=dict(
                color=df["prob_pos"],
                colorscale="Blues" if is_pos else "Reds",
                colorbar=dict(title="P(pos)"),
                size=8,
                line=dict(color="white", width=0.5),
            ),
            text=df["gene_id"] + "<br>" + df["gene_name"].fillna(""),
            hovertemplate="%{text}<br>P(pos): %{marker.color:.3f}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="UTR3 AU content",
            yaxis_title="RRACH count",
            title=f"{'Positive probes' if is_pos else 'Negative controls'} — feature space",
            height=380,
            margin=dict(t=40, b=40, l=40, r=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        display = df[["gene_id", "gene_name", "condition", "prob_pos",
                       "rrach_count", "utr3_au_content", "idr_percent"]].copy()
        display.columns = ["Gene ID", "Name", "Condition", "P(pos)",
                           "RRACH count", "UTR3 AU content", "IDR%"]
        st.dataframe(display, use_container_width=True, height=380)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇ Download {'positive' if is_pos else 'negative'} probes CSV",
            data=csv,
            file_name=f"zorc_{'positive' if is_pos else 'negative'}_probes.csv",
            mime="text/csv",
        )

    with tab_pos:
        pos_df = probes[probes["probe_type"] == "positive_probe"].reset_index(drop=True)
        probe_table_and_scatter(pos_df, "positive_probe")

    with tab_neg:
        neg_df = probes[probes["probe_type"] == "negative_probe"].reset_index(drop=True)
        probe_table_and_scatter(neg_df, "negative_probe")

    with tab_all:
        fig_all = go.Figure()
        for ptype, color, name in [
            ("positive_probe", PALETTE["positive"], "Positive probes"),
            ("negative_probe", PALETTE["negative"], "Negative controls"),
        ]:
            sub = probes[probes["probe_type"] == ptype]
            fig_all.add_trace(go.Scatter(
                x=sub["utr3_au_content"],
                y=sub["rrach_count"],
                mode="markers",
                name=name,
                marker=dict(color=color, size=7, opacity=0.7,
                            line=dict(color="white", width=0.3)),
                text=sub["gene_id"] + "<br>" + sub["gene_name"].fillna(""),
                hovertemplate="%{text}<br>P(pos): %{customdata:.3f}<extra></extra>",
                customdata=sub["prob_pos"],
            ))
        fig_all.update_layout(
            xaxis_title="UTR3 AU content",
            yaxis_title="RRACH count",
            title="All 300 probe candidates",
            height=420,
            margin=dict(t=40, b=40, l=40, r=40),
        )
        st.plotly_chart(fig_all, use_container_width=True)
        st.dataframe(probes, use_container_width=True, height=400)
        st.download_button(
            "⬇ Download all 300 probes CSV",
            probes.to_csv(index=False).encode("utf-8"),
            "11a_xenium_probe_candidates.csv",
            "text/csv",
        )


# ── Page 3: Model Card ────────────────────────────────────────────────────────

def page_model_card():
    st.title("📋 Model Card")
    st.markdown(
        "**ZORC P9f — RandomForest + Platt calibration** "
        "(*Arabidopsis thaliana* P-body mRNA enrichment prediction)"
    )

    # ── Performance metrics ───────────────────────────────────────────────────
    st.subheader("Performance metrics (test set, held-out)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Test AUROC",   "0.7963")
    m2.metric("Test AUPRC",   "0.8431")
    m3.metric("F1-macro",     "0.7229")
    m4.metric("F1-positive",  "0.7904")
    m5.metric("F1-negative",  "0.6554")

    st.markdown("*Note: F1-neg ceiling ~0.66 reflects ~22% biologically ambiguous labels in the negative class, not a model limitation.*")

    # ── HC validation ─────────────────────────────────────────────────────────
    st.subheader("High-confidence validation set (25 genes)")
    hc = load_hc_set()
    preds = load_predictions()
    hc_merged = hc.merge(
        preds[["gene_id", "prob_pos", "pred"]].rename(columns={"gene_id": "geneID"}),
        on="geneID",
        how="left",
    )
    hc_merged["correct"] = (hc_merged["pred"] == 1).astype(int)
    n_correct = hc_merged["correct"].sum()
    st.metric("HC accuracy", f"{n_correct}/{len(hc_merged)}  ({100*n_correct/len(hc_merged):.0f}%)")
    only_miss = hc_merged[hc_merged["correct"] == 0][["geneID", "description", "prob_pos"]]
    if not only_miss.empty:
        st.markdown("**Only miss:**")
        st.dataframe(only_miss, use_container_width=True)
    st.dataframe(
        hc_merged[["geneID", "description", "condition", "prob_pos", "correct"]].rename(
            columns={"geneID": "gene_id", "prob_pos": "P(pos)", "correct": "correct?"}
        ),
        use_container_width=True,
        height=320,
    )

    # ── Confusion matrix (test set via predictions) ───────────────────────────
    st.subheader("Confusion matrix — test set (threshold = 0.50)")
    test_preds = preds[preds["split"] == "test"] if "split" in preds.columns else pd.DataFrame()
    if test_preds.empty:
        # fall back to all predictions with split info from feature matrix
        fm, _ = load_feature_matrix()
        split_map = fm.set_index("geneID")["split"].to_dict()
        test_preds = preds.copy()
        test_preds["split"] = test_preds["gene_id"].map(split_map)
        test_preds = test_preds[test_preds["split"] == "test"]

    if not test_preds.empty:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_preds["class"], test_preds["pred"])
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Pred: Neg", "Pred: Pos"],
            y=["True: Neg", "True: Pos"],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        ))
        fig_cm.update_layout(
            title=f"Confusion matrix  (n={len(test_preds)} test genes)",
            height=320,
            margin=dict(t=40, b=40, l=80, r=40),
        )
        st.plotly_chart(fig_cm, use_container_width=False)

    # ── Feature importance (SHAP) ─────────────────────────────────────────────
    st.subheader("Feature importance — mean |SHAP| (P9f final model)")
    shap_imp = load_shap_importance().head(20)
    fig_shap = go.Figure(go.Bar(
        x=shap_imp["mean_abs_shap"][::-1].tolist(),
        y=shap_imp["feature"][::-1].tolist(),
        orientation="h",
        marker_color=PALETTE["positive"],
    ))
    fig_shap.update_layout(
        xaxis_title="Mean |SHAP|",
        height=480,
        margin=dict(t=20, b=30, l=10, r=20),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # ── Dataset description ───────────────────────────────────────────────────
    st.subheader("Dataset")
    st.markdown("""
| Property | Value |
|---|---|
| Organism | *Arabidopsis thaliana* |
| Total genes | 1,510 |
| Positives (class=1) | 884 — T-RIP enriched + protein log2FC ≥ 0 |
| Negatives (class=0) | 688 — T-RIP depleted |
| Features | 61 (41 RNA + 18 protein/IDR + 2 engineered) |
| Split | 70/15/15 train/val/test, CD-HIT 40% anti-leakage |
| RNA source | Liu et al. 2024 *Plant Cell* (T-RIP) |
| Protein source | Liu et al. 2023 *EMBO J* (APEAL proteomics) |
| Model | RandomForestClassifier 500 trees + Platt calibration |
| Phase | P9f (final) — April 2026 |
    """)

    st.subheader("Known limitations")
    st.markdown("""
- **49.5% imputed BioEmu features** for Tier 1 proteins (IDR < 10%) — dilutes SHAP contributions
  of conformational dynamics features.
- **Length confound:** `mrna_length` and `cds_length` are top predictors; length-only baseline
  not yet computed.
- **Label noise:** ~19% of negatives have P(pos) 0.35–0.65 (possible T-RIP / stress-granule
  co-purification artefacts).
- **AUROC ceiling ~0.80** confirmed across RF + XGBoost + feature engineering; likely reflects
  biological label noise rather than model limitation.
    """)


# ── Sidebar & routing ─────────────────────────────────────────────────────────

def main():
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/39/Arabidopsis_thaliana_2.jpg/320px-Arabidopsis_thaliana_2.jpg", width=200)
        st.title("ZORC")
        st.caption("Zip-code Of RNAs that Condense  \nMoschouLab / IMBB-FORTH  \nERC PLANTEX")
        st.divider()
        page = st.radio(
            "Navigate",
            ["Gene Search", "Xenium Probe Candidates", "Model Card"],
            index=0,
        )
        st.divider()
        st.caption("Model: RF P9f · AUROC 0.7963 · April 2026")

    if page == "Gene Search":
        page_gene_search()
    elif page == "Xenium Probe Candidates":
        page_xenium()
    else:
        page_model_card()


if __name__ == "__main__":
    main()
