"""
ZORC — P11c Plotly Dash Analytical Dashboard
==============================================
ML explorer for the ZORC P-body prediction pipeline.

Four linked panels:
  1. Feature Space UMAP — 2D embedding colored by class/tier/probability/error
  2. Model Performance Explorer — ROC+PR curves with threshold slider
  3. Feature Importance Deep-dive — SHAP bars filtered by category + scatter
  4. Experiment Comparison Table — all runs, connects to Panel 3 SHAP view

Run from project root:
    conda activate zorc_pipeline
    python dashboard/dash_app.py
Then open http://localhost:8050

Dependencies (zorc_pipeline env):
    pip install dash dash-bootstrap-components umap-learn
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import (
    Dash, Input, Output, State, callback,
    dash_table, dcc, html, ctx, no_update,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve, roc_auc_score, roc_curve,
)
import umap

warnings.filterwarnings("ignore")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
ENG_FM_PATH    = ROOT / "data" / "processed" / "08_zorc_feature_matrix_eng.csv"
BASE_FM_PATH   = ROOT / "data" / "processed" / "08_zorc_feature_matrix.csv"
UMAP_CACHE     = ROOT / "data" / "processed" / "umap_cache.npz"
RESULTS        = ROOT / "results"

# Prediction file registry — normalised to: gene_id, class, split, prob_pos, pred
PRED_FILES = {
    "P9":  RESULTS / "09_zorc_predictions.csv",
    "P9b": RESULTS / "09b_zorc_xgb_predictions.csv",
    "P9d": RESULTS / "09f_predictions_final.csv",   # P9d uses P9f preds (best disc.)
    "P9f": RESULTS / "09f_predictions_final.csv",
}

SHAP_FILES = {
    "P9":  RESULTS / "09_zorc_shap_importance.csv",
    "P9b": RESULTS / "09b_zorc_xgb_shap_importance.csv",
    "P9d": RESULTS / "09d_shap_rf_eng.csv",
    "P9f": RESULTS / "09f_shap_final.csv",
}

META_COLS = {
    "geneID", "gene_id", "transcript_id", "class", "condition", "qc_fail",
    "isoform_source", "event_type", "bioemu_tier", "bioemu_status",
    "feature_source", "cluster_id", "split", "aiupred_status",
    "rmsf_nterm_cterm_ratio", "packing_x_idr", "rrach_per_cds_kb",
    "utr3_au_x_length",
}
DROP_FEATURES = {"packing_x_idr", "rrach_per_cds_kb"}
EPS_RMSF = 0.1

PALETTE = {
    "pos": "#2E86AB", "neg": "#E84855",
    "grid": "#EBEBEB", "bg": "#FAFAFA",
}

# ── Feature categories ────────────────────────────────────────────────────────

RNA_FEATURES = [
    "mrna_length", "fA", "fU", "fG", "fC", "au_content", "gc_content",
    "di_AA", "di_AU", "di_AG", "di_AC", "di_UA", "di_UU", "di_UG", "di_UC",
    "di_GA", "di_GU", "di_GG", "di_GC", "di_CA", "di_CU", "di_CG", "di_CC",
    "utr5_length", "utr3_length", "cds_length",
    "utr5_fraction", "utr3_fraction", "cds_fraction",
    "utr3_au_content", "rrach_count", "rrach_per_kb",
    "aaach_count", "aaach_per_kb",
    "mfe", "mfe_per_nt", "frac_paired", "n_stemloops", "n_stemloops_per_kb",
    "long_3utr", "ptc_proxy",
]

PROTEIN_STATIC = [
    "n_residues", "idr_percent", "mean_disorder",
    "max_disorder_window", "n_idr_regions", "longest_idr_region",
]

PROTEIN_DYNAMIC = [
    "rmsf_mean", "rmsf_std", "rmsf_max", "rmsf_nterm50", "rmsf_cterm50",
    "rg_mean", "rg_std", "rg_cv", "contact_density", "packing_density",
    "sasa_per_residue", "pass_rate",
]

ENGINEERED_FEATURES = ["rmsf_nterm_cterm_ratio", "utr3_au_x_length"]

CATEGORY_MAP: dict[str, list[str]] = {
    "All":              RNA_FEATURES + PROTEIN_STATIC + PROTEIN_DYNAMIC + ENGINEERED_FEATURES,
    "RNA":              RNA_FEATURES,
    "Protein-static":  PROTEIN_STATIC,
    "Protein-dynamic": PROTEIN_DYNAMIC,
    "Engineered":      ENGINEERED_FEATURES,
}

def feature_category(feat: str) -> str:
    for cat, feats in CATEGORY_MAP.items():
        if cat != "All" and feat in feats:
            return cat
    return "RNA"

# ── Experiment comparison data ────────────────────────────────────────────────

EXPERIMENT_TABLE = [
    {"Run": "P9",  "Name": "RF baseline",       "Features": 56,
     "Val AUROC": 0.7990, "Test AUROC": 0.7974, "Test F1-mac": 0.7382,
     "HC": "24/25", "Notes": "Global median imputation, pre-AF2"},
    {"Run": "P9b", "Name": "XGBoost (post-AF2)", "Features": 60,
     "Val AUROC": 0.8001, "Test AUROC": 0.7879, "Test F1-mac": "—",
     "HC": "—",     "Notes": "NaN-native; best val AUROC"},
    {"Run": "P9d", "Name": "RF feat. eng. ★",    "Features": 63,
     "Val AUROC": 0.7969, "Test AUROC": 0.7963, "Test F1-mac": 0.7229,
     "HC": "24/25", "Notes": "Best test AUROC+F1"},
    {"Run": "P9f", "Name": "RF Platt-calibrated","Features": 61,
     "Val AUROC": 0.7979, "Test AUROC": 0.7862, "Test F1-mac": 0.7229,
     "HC": "24/25", "Notes": "Brier 0.1776; better calibration"},
]

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_preds(run: str) -> pd.DataFrame:
    """Load and normalise a predictions CSV to: gene_id, class, split, prob_pos, pred."""
    path = PRED_FILES[run]
    df = pd.read_csv(path)
    if "geneID" in df.columns:
        df = df.rename(columns={"geneID": "gene_id", "pred_class": "pred"})
    df["class"]    = df["class"].astype(int)
    df["pred"]     = df["pred"].astype(int)
    df["prob_pos"] = df["prob_pos"].astype(float)
    return df[["gene_id", "class", "split", "prob_pos", "pred"]]


def _load_shap(run: str) -> pd.DataFrame:
    df = pd.read_csv(SHAP_FILES[run])
    df = df[["feature", "mean_abs_shap"]].copy()
    df["category"] = df["feature"].apply(feature_category)
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def _load_fm() -> tuple[pd.DataFrame, list[str]]:
    """Load engineered feature matrix; fall back to base matrix if absent."""
    path = ENG_FM_PATH if ENG_FM_PATH.exists() else BASE_FM_PATH
    df = pd.read_csv(path)
    if "rmsf_nterm50" in df.columns and "rmsf_cterm50" in df.columns:
        if "rmsf_nterm_cterm_ratio" not in df.columns:
            df["rmsf_nterm_cterm_ratio"] = df["rmsf_nterm50"] / (df["rmsf_cterm50"] + EPS_RMSF)
    if "utr3_au_content" in df.columns and "utr3_length" in df.columns:
        if "utr3_au_x_length" not in df.columns:
            df["utr3_au_x_length"] = df["utr3_au_content"] * df["utr3_length"]
    id_col = "geneID" if "geneID" in df.columns else "gene_id"
    df = df.rename(columns={id_col: "gene_id"})
    feat_cols = [
        c for c in df.columns
        if c not in META_COLS and c not in DROP_FEATURES and c != "gene_id"
    ]
    return df, feat_cols


# ── Startup: load all data once ───────────────────────────────────────────────

print("Loading feature matrix…")
FM, FEAT_COLS = _load_fm()

print("Loading predictions…")
ALL_PREDS = {run: _load_preds(run) for run in PRED_FILES}
ALL_SHAP  = {run: _load_shap(run)  for run in SHAP_FILES}

# Build a joined frame for Panel 2/3 scatters (use P9d/P9f predictions as default)
_preds_p9f = ALL_PREDS["P9f"].set_index("gene_id")
FM_PRED = FM.join(_preds_p9f[["prob_pos", "pred"]], on="gene_id")


# ── UMAP precomputation ───────────────────────────────────────────────────────

def _compute_umap() -> np.ndarray:
    X = FM[FEAT_COLS].values.astype(np.float32)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
    return reducer.fit_transform(X_imp)


def _get_umap_embedding() -> np.ndarray:
    if UMAP_CACHE.exists():
        cached = np.load(UMAP_CACHE)
        if len(cached["xy"]) == len(FM):
            print("UMAP: loaded from cache.")
            return cached["xy"]
    print("UMAP: computing (first run, ~30–60 s)…")
    xy = _compute_umap()
    np.savez(UMAP_CACHE, xy=xy)
    print("UMAP: cached to", UMAP_CACHE)
    return xy


print("Computing / loading UMAP…")
UMAP_XY = _get_umap_embedding()

UMAP_DF = FM[["gene_id", "class", "split"]].copy()
UMAP_DF["umap_x"] = UMAP_XY[:, 0]
UMAP_DF["umap_y"] = UMAP_XY[:, 1]
if "bioemu_tier" in FM.columns:
    UMAP_DF["bioemu_tier"] = FM["bioemu_tier"].fillna(0).astype(int).astype(str)
UMAP_DF = UMAP_DF.join(
    ALL_PREDS["P9f"].set_index("gene_id")[["prob_pos", "pred"]], on="gene_id"
)
UMAP_DF["pred_error"] = (UMAP_DF["pred"] != UMAP_DF["class"]).astype(int)

print("All data loaded. Starting app…")


# ── App & layout ──────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="ZORC Analytical Dashboard",
)

_CARD = {"borderRadius": "8px", "boxShadow": "0 1px 4px rgba(0,0,0,.08)"}
_PLOT_H = 420

# Shared stores
STORES = html.Div([
    dcc.Store(id="selected-gene", data=None),
    dcc.Store(id="selected-run",  data="P9d"),
])

# ── Panel 1: UMAP ────────────────────────────────────────────────────────────

panel_umap = dbc.Card(dbc.CardBody([
    html.H5("Feature Space UMAP", className="card-title"),
    html.P(
        "2D UMAP of all features. Click a point to highlight the gene in Panel 3.",
        className="text-muted small",
    ),
    dbc.Row([
        dbc.Col([
            dbc.Label("Color by", html_for="umap-color"),
            dcc.Dropdown(
                id="umap-color",
                options=[
                    {"label": "Class (P-body vs negative)", "value": "class"},
                    {"label": "BioEmu tier",                "value": "bioemu_tier"},
                    {"label": "P(P-body enriched)",         "value": "prob_pos"},
                    {"label": "Prediction error",           "value": "pred_error"},
                    {"label": "Split (train/val/test)",     "value": "split"},
                ],
                value="class",
                clearable=False,
            ),
        ], md=4),
    ], className="mb-2"),
    dcc.Graph(id="umap-scatter", style={"height": f"{_PLOT_H}px"}),
]), style=_CARD)

# ── Panel 2: Model Performance Explorer ──────────────────────────────────────

panel_perf = dbc.Card(dbc.CardBody([
    html.H5("Model Performance Explorer", className="card-title"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Models to overlay"),
            dcc.Checklist(
                id="model-toggle",
                options=[
                    {"label": " RF baseline (P9)",  "value": "P9"},
                    {"label": " XGBoost (P9b)",      "value": "P9b"},
                    {"label": " RF P9f (final)",     "value": "P9f"},
                ],
                value=["P9", "P9f"],
                inline=True,
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "14px"},
            ),
        ], md=7),
        dbc.Col([
            dbc.Label("Threshold"),
            dcc.Slider(
                id="threshold-slider",
                min=0.30, max=0.70, step=0.01,
                value=0.50,
                marks={v: f"{v:.2f}" for v in [0.30, 0.40, 0.50, 0.60, 0.70]},
                tooltip={"always_visible": True, "placement": "bottom"},
            ),
        ], md=5),
    ], className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="roc-curve",  style={"height": f"{_PLOT_H}px"}), md=6),
        dbc.Col(dcc.Graph(id="pr-curve",   style={"height": f"{_PLOT_H}px"}), md=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="conf-matrix", style={"height": "300px"}), md=5),
        dbc.Col(html.Div(id="perf-metrics"), md=7, className="pt-2"),
    ], className="mt-2"),
]), style=_CARD)

# ── Panel 3: Feature Importance Deep-dive ────────────────────────────────────

panel_shap = dbc.Card(dbc.CardBody([
    html.H5("Feature Importance Deep-dive", className="card-title"),
    html.P(
        "SHAP importance for the selected experiment run. "
        "Choose a feature to explore its distribution vs P(pos).",
        className="text-muted small",
    ),
    dbc.Row([
        dbc.Col([
            dbc.Label("Experiment run"),
            dcc.Dropdown(
                id="shap-run",
                options=[{"label": f"{r['Run']} — {r['Name']}", "value": r["Run"]}
                         for r in EXPERIMENT_TABLE],
                value="P9d",
                clearable=False,
            ),
        ], md=4),
        dbc.Col([
            dbc.Label("Feature category"),
            dcc.Dropdown(
                id="shap-category",
                options=[{"label": c, "value": c} for c in CATEGORY_MAP],
                value="All",
                clearable=False,
            ),
        ], md=4),
        dbc.Col([
            dbc.Label("Scatter feature"),
            dcc.Dropdown(id="scatter-feature", value=None, clearable=True,
                         placeholder="click a SHAP bar or select…"),
        ], md=4),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="shap-bar",    style={"height": f"{_PLOT_H}px"}), md=6),
        dbc.Col(dcc.Graph(id="feat-scatter", style={"height": f"{_PLOT_H}px"}), md=6),
    ]),
]), style=_CARD)

# ── Panel 4: Experiment Comparison ───────────────────────────────────────────

_mlflow_note = (
    "MLflow available — run `mlflow ui --port 5000` for full experiment tracking."
    if MLFLOW_AVAILABLE else
    "MLflow not installed. Install via `pip install mlflow` for live tracking (P12a)."
)

panel_experiments = dbc.Card(dbc.CardBody([
    html.H5("Experiment Comparison", className="card-title"),
    html.P(_mlflow_note, className="text-muted small"),
    html.P(
        "Select a row to load its SHAP rankings in Panel 3.",
        className="text-muted small",
    ),
    dash_table.DataTable(
        id="exp-table",
        columns=[{"name": k, "id": k} for k in EXPERIMENT_TABLE[0]],
        data=EXPERIMENT_TABLE,
        row_selectable="single",
        selected_rows=[2],                    # P9d selected by default
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#2E86AB", "color": "white",
            "fontWeight": "bold", "fontSize": "12px",
        },
        style_cell={
            "fontSize": "12px", "padding": "6px 10px",
            "textAlign": "left", "whiteSpace": "normal",
        },
        style_data_conditional=[
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#f7f9fc",
            },
            {
                "if": {"filter_query": '{Run} = "P9d"'},
                "fontWeight": "bold", "backgroundColor": "#e8f4fd",
            },
        ],
    ),
]), style=_CARD)

# ── Main layout ───────────────────────────────────────────────────────────────

app.layout = dbc.Container([
    STORES,
    dbc.Row(dbc.Col(html.Div([
        html.H3("ZORC Analytical Dashboard", className="mb-0"),
        html.P(
            "P11c — ML explorer · MoschouLab · IMBB-FORTH · ERC PLANTEX",
            className="text-muted mb-0",
        ),
    ]), className="py-3")),
    dbc.Row([
        dbc.Col(panel_umap, md=6, className="mb-3"),
        dbc.Col(panel_perf, md=6, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(panel_shap,        md=8, className="mb-3"),
        dbc.Col(panel_experiments, md=4, className="mb-3"),
    ]),
], fluid=True, className="px-4")


# ── Callbacks ─────────────────────────────────────────────────────────────────

# ── 1a. UMAP scatter ─────────────────────────────────────────────────────────

COLOR_SCALES = {
    "class":       {"colorscale": None,       "is_cat": True},
    "bioemu_tier": {"colorscale": None,       "is_cat": True},
    "prob_pos":    {"colorscale": "RdYlGn",   "is_cat": False},
    "pred_error":  {"colorscale": None,       "is_cat": True},
    "split":       {"colorscale": None,       "is_cat": True},
}

CAT_COLORS = {
    "class":       {0: PALETTE["neg"], 1: PALETTE["pos"]},
    "bioemu_tier": {"0": "#A8A8A8", "1": "#FFB703", "2": "#219EBC", "3": "#8338EC"},
    "pred_error":  {0: PALETTE["pos"], 1: PALETTE["neg"]},
    "split":       {"train": "#6C757D", "val": "#FFC107", "test": "#20C997"},
}


@callback(Output("umap-scatter", "figure"), Input("umap-color", "value"))
def update_umap(color_by: str) -> go.Figure:
    df = UMAP_DF.copy()
    fig = go.Figure()
    info = COLOR_SCALES.get(color_by, {"is_cat": True})

    if info["is_cat"]:
        cat_colors = CAT_COLORS.get(color_by, {})
        for val, sub in df.groupby(color_by, sort=False):
            color = cat_colors.get(val, "#888")
            label = (
                ("P-body enriched" if val == 1 else "Not enriched")
                if color_by == "class"
                else f"Tier {val}" if color_by == "bioemu_tier"
                else str(val)
            )
            fig.add_trace(go.Scattergl(
                x=sub["umap_x"], y=sub["umap_y"],
                mode="markers",
                name=label,
                marker=dict(color=color, size=4, opacity=0.7),
                customdata=sub["gene_id"],
                hovertemplate="<b>%{customdata}</b><br>"
                              f"{color_by}: {val}<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scattergl(
            x=df["umap_x"], y=df["umap_y"],
            mode="markers",
            marker=dict(
                color=df[color_by],
                colorscale=info["colorscale"],
                colorbar=dict(title=color_by, thickness=12),
                size=4, opacity=0.7,
            ),
            customdata=df["gene_id"],
            hovertemplate="<b>%{customdata}</b><br>"
                          f"{color_by}: %{{marker.color:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        uirevision="umap",
        legend=dict(itemsizing="constant", font=dict(size=11)),
        margin=dict(t=20, b=30, l=30, r=10),
        plot_bgcolor=PALETTE["bg"],
        paper_bgcolor=PALETTE["bg"],
        xaxis=dict(showgrid=True, gridcolor=PALETTE["grid"], title="UMAP 1"),
        yaxis=dict(showgrid=True, gridcolor=PALETTE["grid"], title="UMAP 2"),
    )
    return fig


@callback(Output("selected-gene", "data"), Input("umap-scatter", "clickData"))
def store_selected_gene(click_data) -> str | None:
    if click_data is None:
        return no_update
    try:
        return click_data["points"][0]["customdata"]
    except (KeyError, IndexError):
        return no_update


# ── 2. ROC + PR curves + confusion matrix + metrics ──────────────────────────

MODEL_LINE = {
    "P9":  {"color": "#6C757D", "dash": "solid",  "name": "RF baseline P9"},
    "P9b": {"color": "#FFC107", "dash": "dash",   "name": "XGBoost P9b"},
    "P9f": {"color": "#2E86AB", "dash": "dot",    "name": "RF final P9f"},
}


def _test_preds(run: str) -> pd.DataFrame:
    return ALL_PREDS[run][ALL_PREDS[run]["split"] == "test"]


@callback(
    Output("roc-curve",   "figure"),
    Output("pr-curve",    "figure"),
    Output("conf-matrix", "figure"),
    Output("perf-metrics","children"),
    Input("model-toggle",    "value"),
    Input("threshold-slider","value"),
)
def update_perf(selected_runs: list[str], threshold: float):
    fig_roc = go.Figure()
    fig_pr  = go.Figure()

    # Diagonal reference
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#CCC", dash="dot"), showlegend=False,
    ))
    fig_pr.add_trace(go.Scatter(
        x=[0, 1], y=[0.5, 0.5], mode="lines",
        line=dict(color="#CCC", dash="dot"), showlegend=False,
    ))

    runs = selected_runs or ["P9f"]
    metric_rows = []

    for run in runs:
        tp = _test_preds(run)
        y_true, y_prob = tp["class"].values, tp["prob_pos"].values
        y_pred = (y_prob >= threshold).astype(int)
        style = MODEL_LINE.get(run, {"color": "#888", "dash": "solid", "name": run})

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{style['name']} (AUC={auroc:.3f})",
            line=dict(color=style["color"], dash=style["dash"], width=2),
        ))

        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        fig_pr.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines", name=f"{style['name']} (AP={auprc:.3f})",
            line=dict(color=style["color"], dash=style["dash"], width=2),
        ))

        f1_mac = f1_score(y_true, y_pred, average="macro",   zero_division=0)
        f1_pos = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        f1_neg = f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)
        metric_rows.append(
            dbc.Card(dbc.CardBody([
                html.P(style["name"], className="fw-bold mb-1 small"),
                dbc.Row([
                    dbc.Col([html.Span("AUROC", className="text-muted small"), html.Br(),
                             html.Span(f"{auroc:.4f}", className="fw-bold")], width=3),
                    dbc.Col([html.Span("AUPRC", className="text-muted small"), html.Br(),
                             html.Span(f"{auprc:.4f}", className="fw-bold")], width=3),
                    dbc.Col([html.Span("F1-pos", className="text-muted small"), html.Br(),
                             html.Span(f"{f1_pos:.4f}", className="fw-bold")], width=3),
                    dbc.Col([html.Span("F1-neg", className="text-muted small"), html.Br(),
                             html.Span(f"{f1_neg:.4f}", className="fw-bold")], width=3),
                ]),
            ]), className="mb-2", style={"border": f"2px solid {style['color']}"}),
        )

    for fig, title, xlabel, ylabel in [
        (fig_roc, "ROC curves — test set", "False Positive Rate", "True Positive Rate"),
        (fig_pr,  "PR curves — test set",  "Recall",              "Precision"),
    ]:
        fig.update_layout(
            title=dict(text=title, font=dict(size=13)),
            xaxis=dict(title=xlabel, showgrid=True, gridcolor=PALETTE["grid"]),
            yaxis=dict(title=ylabel, showgrid=True, gridcolor=PALETTE["grid"]),
            legend=dict(font=dict(size=10), x=0.01, y=0.01 if "ROC" in title else 0.99,
                        xanchor="left", yanchor="bottom" if "ROC" in title else "top"),
            plot_bgcolor=PALETTE["bg"], paper_bgcolor=PALETTE["bg"],
            margin=dict(t=35, b=40, l=50, r=10),
        )

    # Confusion matrix for the first selected run
    tp0 = _test_preds(runs[0])
    y0_true = tp0["class"].values
    y0_pred = (tp0["prob_pos"].values >= threshold).astype(int)
    cm = confusion_matrix(y0_true, y0_pred)
    style0 = MODEL_LINE.get(runs[0], {"color": "#2E86AB", "name": runs[0]})
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=["Pred: Neg", "Pred: Pos"], y=["True: Neg", "True: Pos"],
        text=cm, texttemplate="%{text}",
        colorscale=[[0, "#F9F9F9"], [1, style0["color"]]],
        showscale=False,
    ))
    fig_cm.update_layout(
        title=dict(text=f"CM — {style0['name']}  (thr={threshold:.2f})", font=dict(size=12)),
        margin=dict(t=35, b=30, l=70, r=10),
        height=300,
        plot_bgcolor=PALETTE["bg"], paper_bgcolor=PALETTE["bg"],
    )

    return fig_roc, fig_pr, fig_cm, metric_rows


# ── 3a. Sync selected-run from experiment table ───────────────────────────────

@callback(
    Output("selected-run", "data"),
    Output("shap-run",     "value"),
    Input("exp-table",     "selected_rows"),
    State("exp-table",     "data"),
)
def sync_run_from_table(selected_rows, table_data):
    if not selected_rows or not table_data:
        return no_update, no_update
    run = table_data[selected_rows[0]]["Run"]
    return run, run


# ── 3b. SHAP bar chart ───────────────────────────────────────────────────────

@callback(Output("shap-bar", "figure"),
          Input("shap-run",      "value"),
          Input("shap-category", "value"))
def update_shap_bar(run: str, category: str) -> go.Figure:
    df = ALL_SHAP.get(run or "P9d", ALL_SHAP["P9d"]).copy()
    allowed = set(CATEGORY_MAP.get(category, []))
    if allowed:
        df = df[df["feature"].isin(allowed)]
    df = df.head(20)
    fig = go.Figure(go.Bar(
        x=df["mean_abs_shap"][::-1],
        y=df["feature"][::-1],
        orientation="h",
        marker_color=df["category"].map({
            "RNA":             PALETTE["pos"],
            "Protein-static":  "#FB8500",
            "Protein-dynamic": "#8338EC",
            "Engineered":      "#06D6A0",
        }).fillna(PALETTE["pos"])[::-1],
        text=[f"{v:.4f}" for v in df["mean_abs_shap"][::-1]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>|SHAP|=%{x:.4f}<extra></extra>",
    ))
    # Legend patches via invisible scatter
    for cat, color in [
        ("RNA", PALETTE["pos"]), ("Protein-static", "#FB8500"),
        ("Protein-dynamic", "#8338EC"), ("Engineered", "#06D6A0"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=color, size=10, symbol="square"),
            name=cat, showlegend=True,
        ))
    fig.update_layout(
        title=dict(text=f"SHAP importance — {run} ({category})", font=dict(size=12)),
        xaxis_title="Mean |SHAP|",
        margin=dict(t=35, b=30, l=10, r=60),
        legend=dict(font=dict(size=10), x=0.55, y=0.05),
        plot_bgcolor=PALETTE["bg"], paper_bgcolor=PALETTE["bg"],
    )
    return fig


# ── 3c. Populate scatter-feature dropdown from SHAP bar click ────────────────

@callback(
    Output("scatter-feature", "options"),
    Output("scatter-feature", "value"),
    Input("shap-bar",    "clickData"),
    Input("shap-run",    "value"),
    Input("shap-category","value"),
    State("scatter-feature","value"),
)
def update_scatter_selector(click_data, run, category, current_feat):
    shap_df = ALL_SHAP.get(run or "P9d", ALL_SHAP["P9d"])
    allowed = set(CATEGORY_MAP.get(category, []))
    if allowed:
        shap_df = shap_df[shap_df["feature"].isin(allowed)]
    options = [{"label": r["feature"], "value": r["feature"]}
               for _, r in shap_df.head(30).iterrows()]

    # If user clicked a bar, extract the feature name from yaxis
    new_feat = current_feat
    if ctx.triggered_id == "shap-bar" and click_data:
        try:
            new_feat = click_data["points"][0]["y"]
        except (KeyError, IndexError):
            pass

    # Default to top feature if current selection is no longer in options
    valid = {o["value"] for o in options}
    if new_feat not in valid:
        new_feat = options[0]["value"] if options else None

    return options, new_feat


# ── 3d. Feature scatter (value vs P(pos)) ────────────────────────────────────

@callback(
    Output("feat-scatter", "figure"),
    Input("scatter-feature", "value"),
    Input("shap-run",        "value"),
    Input("selected-gene",   "data"),
)
def update_feat_scatter(feature: str | None, run: str, selected_gene: str | None) -> go.Figure:
    if not feature or feature not in FM_PRED.columns:
        fig = go.Figure()
        fig.add_annotation(text="Select a feature from the SHAP chart",
                           showarrow=False, font=dict(size=14, color="#888"))
        fig.update_layout(plot_bgcolor=PALETTE["bg"], paper_bgcolor=PALETTE["bg"],
                          margin=dict(t=20, b=20, l=20, r=20))
        return fig

    preds_run = ALL_PREDS.get(run or "P9d", ALL_PREDS["P9d"])
    df = FM[["gene_id", "class", feature]].merge(
        preds_run[["gene_id", "prob_pos"]], on="gene_id", how="left"
    ).dropna(subset=["prob_pos"])

    fig = go.Figure()
    for cls, color, name in [(1, PALETTE["pos"], "P-body enriched"),
                              (0, PALETTE["neg"], "Not enriched")]:
        sub = df[df["class"] == cls]
        fig.add_trace(go.Scattergl(
            x=sub[feature], y=sub["prob_pos"],
            mode="markers",
            name=name,
            marker=dict(color=color, size=4, opacity=0.55),
            customdata=sub["gene_id"],
            hovertemplate="<b>%{customdata}</b><br>"
                          f"{feature}: %{{x:.4g}}<br>"
                          "P(pos): %{y:.3f}<extra></extra>",
        ))

    # Highlight selected gene
    if selected_gene:
        gene_row = df[df["gene_id"] == selected_gene]
        if not gene_row.empty:
            fig.add_trace(go.Scatter(
                x=gene_row[feature], y=gene_row["prob_pos"],
                mode="markers",
                marker=dict(color="black", size=12, symbol="star",
                            line=dict(color="white", width=1.5)),
                name=selected_gene,
                hovertemplate=f"<b>{selected_gene}</b><br>"
                              f"{feature}: %{{x:.4g}}<br>"
                              "P(pos): %{y:.3f}<extra></extra>",
            ))

    fig.update_layout(
        title=dict(text=f"{feature} vs P(pos) — {run}", font=dict(size=12)),
        xaxis=dict(title=feature, showgrid=True, gridcolor=PALETTE["grid"]),
        yaxis=dict(title="P(P-body enriched)", showgrid=True, gridcolor=PALETTE["grid"],
                   range=[-0.05, 1.05]),
        plot_bgcolor=PALETTE["bg"], paper_bgcolor=PALETTE["bg"],
        margin=dict(t=35, b=40, l=50, r=10),
        legend=dict(font=dict(size=10)),
    )
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
