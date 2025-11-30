# app/eda_agent.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Any
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from io import BytesIO
import joblib

from .llm_agent import generate_report  # uses Gemini

# ensure matplotlib backend works in headless servers
plt.switch_backend("Agg")

OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# -------------------------
# EDA helpers
# -------------------------
def detect_column_types(df: pd.DataFrame) -> Dict[str, Any]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime = df.select_dtypes(include=["datetime64"]).columns.tolist()
    others = [c for c in df.columns if c not in numeric + categorical + datetime]
    return {"numeric": numeric, "categorical": categorical, "datetime": datetime, "others": others}

def generate_basic_summary(df: pd.DataFrame) -> dict:
    types = detect_column_types(df)
    summary = {
        "shape": df.shape,
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "missing_counts": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(3).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "types_detected": types,
    }
    if types["numeric"]:
        num_df = df[types["numeric"]]
        summary["numeric_describe"] = num_df.describe().to_dict()
        summary["skewness"] = num_df.skew().to_dict()
        summary["kurtosis"] = num_df.kurtosis().to_dict()
    cat_summary = {}
    for c in types["categorical"]:
        cat_summary[c] = {
            "unique": int(df[c].nunique(dropna=False)),
            "top5": df[c].value_counts(dropna=False).nlargest(5).to_dict(),
        }
    summary["categorical_summary"] = cat_summary
    return summary

# -------------------------
# Visualizations
# -------------------------
def save_numeric_plots(df: pd.DataFrame, cols: list, out_dir: Path = PLOTS_DIR) -> list:
    saved = []
    for col in cols:
        try:
            s = df[col].dropna()
            if s.empty:
                continue
            # histogram + KDE
            plt.figure(figsize=(6, 4))
            sns.histplot(s, kde=True)
            plt.title(f"Histogram: {col}")
            p1 = out_dir / f"hist_{col}.png"
            plt.tight_layout()
            plt.savefig(p1)
            plt.close()

            # boxplot
            plt.figure(figsize=(6, 3))
            sns.boxplot(x=s)
            plt.title(f"Boxplot: {col}")
            p2 = out_dir / f"box_{col}.png"
            plt.tight_layout()
            plt.savefig(p2)
            plt.close()

            saved.extend([str(p1), str(p2)])
        except Exception:
            continue
    return saved

def save_correlation_heatmap(df: pd.DataFrame, numeric_cols: list, out_dir: Path = PLOTS_DIR) -> Optional[str]:
    if len(numeric_cols) < 2:
        return None
    try:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        path = out_dir / "correlation_matrix.png"
        plt.title("Correlation matrix")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return str(path)
    except Exception:
        return None

def save_class_distribution(df: pd.DataFrame, target: str, out_dir: Path = PLOTS_DIR) -> Optional[str]:
    if not target or target not in df.columns:
        return None
    try:
        plt.figure(figsize=(6,4))
        df[target].value_counts(dropna=False).plot(kind="bar")
        plt.title(f"Class distribution: {target}")
        plt.tight_layout()
        path = out_dir / f"class_dist_{target}.png"
        plt.savefig(path)
        plt.close()
        return str(path)
    except Exception:
        return None

def detect_imbalance(df: pd.DataFrame, target: str, threshold: float = 0.7) -> dict:
    counts = df[target].value_counts(dropna=False)
    pct = (counts / counts.sum()).round(4).to_dict()
    majority_frac = float(counts.max() / counts.sum())
    is_imbalanced = majority_frac >= threshold
    recs = []
    if is_imbalanced:
        recs = [
            "Oversampling minority (SMOTE/ADASYN)",
            "Undersampling majority (RandomUnderSampler)",
            "Use class_weight in supported models",
            "Stratified splits for train/test"
        ]
    else:
        recs = ["Class balance looks acceptable."]
    return {
        "counts": counts.to_dict(),
        "pct": pct,
        "majority_fraction": majority_frac,
        "is_imbalanced": is_imbalanced,
        "recommendations": recs,
    }

# -------------------------
# Preprocessing pipeline (sklearn)
# -------------------------
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object","category","bool"]).columns.tolist()

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    transformers = []
    if numeric:
        transformers.append(("num", num_pipeline, numeric))
    if categorical:
        transformers.append(("cat", cat_pipeline, categorical))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor

# -------------------------
# PDF generation
# -------------------------
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def save_report_pdf(text: str, path: Path):
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin
    lines = text.splitlines()
    for line in lines:
        # simple wrapping
        if len(line) <= 95:
            c.drawString(margin, y, line)
            y -= 14
        else:
            # wrap long lines
            while len(line) > 0:
                chunk = line[:95]
                c.drawString(margin, y, chunk)
                line = line[95:]
                y -= 14
        if y < 80:
            c.showPage()
            y = height - margin
    c.save()

# -------------------------
# Feature importance (quick)
# -------------------------
def quick_feature_importance(df: pd.DataFrame, target: str, n_features: int = 10) -> Optional[dict]:
    if target not in df.columns:
        return None
    try:
        # simple approach: drop NA rows for training
        tmp = df.dropna(subset=[target])
        X = tmp.select_dtypes(include=[np.number]).fillna(0)
        y = tmp[target]
        if X.shape[1] == 0 or X.shape[0] < 10:
            return None
        # If target is categorical, try converting to numeric codes
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            y = y.astype("category").cat.codes
        clf = RandomForestClassifier(n_estimators=50, random_state=0)
        clf.fit(X, y)
        importances = dict(zip(X.columns, clf.feature_importances_))
        topk = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:n_features])
        return topk
    except Exception:
        return None

# -------------------------
# Top-level runner
# -------------------------
def run_auto_eda(df: pd.DataFrame, target: Optional[str] = None) -> dict:
    """
    Runs full EDA pipeline and returns a dict with:
    - summary (dict)
    - paths to saved plots
    - report_text (generated by Gemini)
    - cleaned_csv_path
    - pipeline_path
    - pdf_path
    """
    # basic summary
    summary = generate_basic_summary(df)

    # visualize numeric
    numeric_cols = summary["types_detected"]["numeric"]
    plot_paths = save_numeric_plots(df, numeric_cols)

    # correlation
    corr_path = save_correlation_heatmap(df, numeric_cols)
    if corr_path:
        plot_paths.append(corr_path)

    # imbalance
    imbalance = None
    class_plot = None
    if target and target in df.columns:
        imbalance = detect_imbalance(df, target)
        class_plot = save_class_distribution(df, target)
        if class_plot:
            plot_paths.append(class_plot)

    # feature importance
    feat_imp = None
    if target:
        feat_imp = quick_feature_importance(df, target)

    # preprocessor
    preproc = build_preprocessor(df)
    pipeline_path = OUTPUT_DIR / "preprocessor.pkl"
    joblib.dump(preproc, pipeline_path)

    # save cleaned CSV (simple normalization: drop duplicates and fillna)
    cleaned = df.copy()
    cleaned.columns = [str(c).strip().lower().replace(" ", "_") for c in cleaned.columns]
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.fillna("")  # keep it simple for download
    cleaned_path = OUTPUT_DIR / "cleaned_dataset.csv"
    cleaned.to_csv(cleaned_path, index=False)

    # LLM report
    report_text = generate_report(summary, imbalance)

    # PDF
    pdf_path = OUTPUT_DIR / "eda_report.pdf"
    save_report_pdf(report_text, pdf_path)

    return {
        "summary": summary,
        "plot_paths": plot_paths,
        "report_text": report_text,
        "cleaned_csv": str(cleaned_path),
        "pipeline_path": str(pipeline_path),
        "pdf_path": str(pdf_path),
        "feature_importance": feat_imp,
    }
