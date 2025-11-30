# main.py
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Standard libs
import io
import uuid
from typing import Optional, Dict, Any

# Data libs
import pandas as pd
import numpy as np

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ML/processing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# PDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Save/load
import joblib

# FastAPI + Gradio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

# Gemini integration (robust import)
try:
    from google import genai  # some installs use this path
except Exception:
    try:
        import genai  # alternative
    except Exception:
        genai = None  # will handle later

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Output dirs
BASE = Path("outputs")
PLOTS = BASE / "plots"
BASE.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

# Initialize Gemini client robustly
if genai is None:
    # will warn later when report is requested
    client = None
else:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()
    except Exception:
        try:
            client = genai.Client()
        except Exception:
            client = None

# ---------- Utility / EDA functions ----------

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
        "types_detected": types
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
            "top5": df[c].value_counts(dropna=False).nlargest(5).to_dict()
        }
    summary["categorical_summary"] = cat_summary
    return summary

def save_numeric_plots(df: pd.DataFrame, cols: list, out_dir: Path = PLOTS) -> list:
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
            p1 = out_dir / f"hist_{col}_{uuid.uuid4().hex[:6]}.png"
            plt.tight_layout()
            plt.savefig(p1)
            plt.close()

            # boxplot
            plt.figure(figsize=(6, 3))
            sns.boxplot(x=s)
            plt.title(f"Boxplot: {col}")
            p2 = out_dir / f"box_{col}_{uuid.uuid4().hex[:6]}.png"
            plt.tight_layout()
            plt.savefig(p2)
            plt.close()

            saved.extend([str(p1), str(p2)])
        except Exception:
            continue
    return saved

def save_correlation_heatmap(df: pd.DataFrame, numeric_cols: list, out_dir: Path = PLOTS) -> Optional[str]:
    if len(numeric_cols) < 2:
        return None
    try:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        path = out_dir / f"correlation_{uuid.uuid4().hex[:6]}.png"
        plt.title("Correlation matrix")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return str(path)
    except Exception:
        return None

def save_class_distribution(df: pd.DataFrame, target: str, out_dir: Path = PLOTS) -> Optional[str]:
    if not target or target not in df.columns:
        return None
    try:
        plt.figure(figsize=(6,4))
        df[target].value_counts(dropna=False).plot(kind="bar")
        plt.title(f"Class distribution: {target}")
        plt.tight_layout()
        path = out_dir / f"class_dist_{target}_{uuid.uuid4().hex[:6]}.png"
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
    return {"counts": counts.to_dict(), "pct": pct, "majority_fraction": majority_frac, "is_imbalanced": is_imbalanced, "recommendations": recs}

def build_preprocessor(df: pd.DataFrame):
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

def quick_feature_importance(df: pd.DataFrame, target: str, n_features: int = 10) -> Optional[dict]:
    if target not in df.columns:
        return None
    try:
        tmp = df.dropna(subset=[target])
        X = tmp.select_dtypes(include=[np.number]).fillna(0)
        y = tmp[target]
        if X.shape[1] == 0 or X.shape[0] < 10:
            return None
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            y = y.astype("category").cat.codes
        clf = RandomForestClassifier(n_estimators=50, random_state=0)
        clf.fit(X, y)
        importances = dict(zip(X.columns, clf.feature_importances_))
        topk = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:n_features])
        return topk
    except Exception:
        return None

# ---------------- LLM (Gemini) wrapper ----------------
def _assemble_prompt(summary: dict, imbalance: Optional[dict]) -> str:
    prompt_lines = [
        "You are an expert data scientist. Produce a clear, structured EDA report for the dataset below.",
        "Include: (1) Key observations (shape, memory, missing, types), (2) Issues to fix, (3) Recommended preprocessing steps and why,",
        "(4) Imbalance handling suggestions if relevant, (5) Quick feature-engineering ideas and next steps for modeling.",
        "",
        "DATASET_SUMMARY:",
        str(summary),
    ]
    if imbalance:
        prompt_lines += ["", "IMBALANCE_INFO:", str(imbalance)]
    return "\n".join(prompt_lines)

def generate_gemini_report(summary: dict, imbalance: Optional[dict] = None, max_tokens: int = 1200) -> str:
    prompt = _assemble_prompt(summary, imbalance)
    if client is None:
        return "Gemini client not available. Install google-genai and set GEMINI_API_KEY environment variable.\n\nDATA SUMMARY:\n" + str(summary)
    # Try multiple invocation styles
    try:
        if hasattr(client, "models") and hasattr(client.models, "generate_content"):
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, temperature=0.0, max_output_tokens=max_tokens)
            text = getattr(resp, "text", None) or getattr(resp, "content", None)
            if isinstance(text, (list, tuple)):
                text = "\n".join(getattr(item, "text", str(item)) for item in text)
            return str(text)
    except Exception:
        pass
    try:
        if hasattr(client, "generate_text"):
            resp = client.generate_text(model=GEMINI_MODEL, text=prompt, max_output_tokens=max_tokens)
            return getattr(resp, "text", str(resp))
    except Exception:
        pass
    try:
        if hasattr(client, "create"):
            resp = client.create(model=GEMINI_MODEL, prompt=prompt, max_tokens=max_tokens, temperature=0.0)
            txt = getattr(resp, "text", None) or getattr(resp, "choices", None)
            if isinstance(txt, list):
                txt = "\n".join([getattr(c, "text", str(c)) for c in txt])
            return str(txt)
    except Exception as e:
        return f"LLM generation failed: {e}\n\nFallback summary:\n{summary}"
    return "LLM: no compatible method found in genai client; please check SDK."

# ---------------- PDF generation (embed images) ----------------
def save_report_pdf(text: str, image_paths: list, pdf_path: Path):
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "AutoEDA Report")
    y -= 24

    # Text lines
    c.setFont("Helvetica", 10)
    for line in text.splitlines():
        if y < 120:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)
        # wrap
        while len(line) > 100:
            c.drawString(margin, y, line[:100])
            line = line[100:]
            y -= 12
            if y < 120:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 10)
        c.drawString(margin, y, line)
        y -= 12

    # Add images on new pages
    for img_path in image_paths:
        try:
            c.showPage()
            y = height - margin
            img = ImageReader(str(img_path))
            # fit to page width with aspect ratio
            iw, ih = img.getSize()
            max_w = width - 2 * margin
            scale = min(1.0, max_w / iw)
            w = iw * scale
            h = ih * scale
            c.drawImage(img, margin, height - margin - h, width=w, height=h)
        except Exception:
            continue
    c.save()

# ---------------- Top-level EDA runner ----------------
def run_auto_eda(df: pd.DataFrame, target: Optional[str] = None) -> dict:
    # Basic summary
    summary = generate_basic_summary(df)

    # Numeric visualizations
    numeric_cols = summary["types_detected"]["numeric"]
    plot_paths = save_numeric_plots(df, numeric_cols)

    # Correlation
    corr_path = save_correlation_heatmap(df, numeric_cols)
    if corr_path:
        plot_paths.append(corr_path)

    # Imbalance (if target provided)
    imbalance = None
    class_plot = None
    if target and target in df.columns:
        imbalance = detect_imbalance(df, target)
        class_plot = save_class_distribution(df, target)
        if class_plot:
            plot_paths.append(class_plot)

    # Feature importance
    feat_imp = None
    if target:
        feat_imp = quick_feature_importance(df, target)

    # Preprocessor save
    preproc = build_preprocessor(df)
    pipeline_path = BASE / "preprocessor.pkl"
    try:
        joblib.dump(preproc, pipeline_path)
    except Exception:
        pipeline_path = None

    # Cleaned CSV (simple normalization)
    cleaned = df.copy()
    cleaned.columns = [str(c).strip().lower().replace(" ", "_") for c in cleaned.columns]
    cleaned = cleaned.drop_duplicates()
    cleaned_path = BASE / "cleaned_dataset.csv"
    cleaned.to_csv(cleaned_path, index=False)

    # LLM report
    report_text = generate_gemini_report(summary, imbalance)

    # Create PDF including plot images
    pdf_path = BASE / "eda_report.pdf"
    try:
        save_report_pdf(report_text, plot_paths[:6], pdf_path)  # embed up to 6 images
    except Exception:
        pdf_path = None

    return {
        "summary": summary,
        "plot_paths": plot_paths,
        "report_text": report_text,
        "cleaned_csv": str(cleaned_path) if cleaned_path.exists() else None,
        "pipeline_path": str(pipeline_path) if pipeline_path and Path(pipeline_path).exists() else None,
        "pdf_path": str(pdf_path) if pdf_path and Path(pdf_path).exists() else None,
        "feature_importance": feat_imp,
    }

# ---------------- FastAPI + Gradio UI ----------------
app = FastAPI(title="AutoEDA Agent (Gemini + Gradio)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "app": "autoeda-agent"}

# Build Gradio Blocks app with sidebar
def build_gradio_app():
    with gr.Blocks(title="AutoEDA Agent (Gemini) - Sidebar UI") as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("## AutoEDA\nUpload CSV, then run EDA. Use the sidebar to view different panels.")
                upload_file = gr.File(label="Upload CSV file", file_count="single", type="file")
                target_input = gr.Textbox(label="Target column (optional)", placeholder="e.g. target")
                run_btn = gr.Button("Run EDA", variant="primary")
                gr.Markdown("---")
                # Sidebar buttons
                summary_btn = gr.Button("📊 Summary")
                dist_btn = gr.Button("📈 Distributions")
                corr_btn = gr.Button("🔗 Correlation Heatmap")
                out_btn = gr.Button("📦 Outliers")
                imb_btn = gr.Button("🧮 Class Imbalance")
                pdf_btn = gr.Button("📝 PDF Report")
                gem_btn = gr.Button("🤖 Gemini Summary")
                download_csv_btn = gr.Button("⬇️ Download Cleaned CSV")
                download_pdf_btn = gr.Button("⬇️ Download PDF")
            with gr.Column(scale=3):
                # Placeholders for content
                summary_text = gr.Textbox(label="Summary", lines=12)
                plot_gallery = gr.Gallery(label="Plots").style(grid=[2], height="auto")
                df_preview = gr.Dataframe(label="Dataset preview (first 5 rows)")
                imbalance_text = gr.Textbox(label="Imbalance Info", lines=6)
                gemini_text = gr.Textbox(label="Gemini EDA Report", lines=18)
                # Hidden file outputs for download
                cleaned_file = gr.File(label="Cleaned CSV (download)")
                pdf_file = gr.File(label="PDF Report (download)")

        # STATE to store result dict
        state = gr.State({})

        # Run EDA click
        def do_run_eda(file_obj, target):
            if file_obj is None:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            try:
                # file_obj is a TemporaryFile wrapper; read with pandas
                df = pd.read_csv(file_obj.name)
            except Exception as e:
                return f"Failed to read CSV: {e}", [], None, None, None, None, None
            res = run_auto_eda(df, target or None)
            # store in state (return as dict)
            return res, res.get("report_text", ""), res.get("plot_paths", []), df.head().reset_index(drop=True), (res.get("imbalance_summary", None) or ""), res.get("report_text", ""), None

        # run_btn triggers EDA and populate state and main widgets
        def run_wrapper(file_obj, target):
            if file_obj is None:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            try:
                df = pd.read_csv(file_obj.name)
            except Exception as e:
                return f"Failed to read CSV: {e}", [], None, None, None, None, None
            res = run_auto_eda(df, target or None)
            # set widgets:
            summary_str = "Shape: {}\nMemory (bytes): {}\n\nMissing (count): {}\n\nTypes: {}\n".format(
                res["summary"]["shape"], res["summary"]["memory_bytes"], res["summary"]["missing_counts"], res["summary"]["types_detected"]
            )
            # For imbalance display
            imb_str = ""
            if "summary" in res and "types_detected" in res["summary"]:
                if target and target in df.columns:
                    imb = detect_imbalance(df, target)
                    imb_str = str(imb)
            # Prepare cleaned CSV and PDF as file-like (Gradio needs file path)
            cleaned = res.get("cleaned_csv", None)
            pdf = res.get("pdf_path", None)
            return summary_str, res.get("plot_paths", []), df.head().reset_index(drop=True), imb_str, res.get("report_text", ""), cleaned, pdf

        run_btn.click(fn=run_wrapper, inputs=[upload_file, target_input], outputs=[summary_text, plot_gallery, df_preview, imbalance_text, gemini_text, cleaned_file, pdf_file])

        # Sidebar buttons update visible content (they simply focus relevant widgets)
        # We'll implement simple callbacks that return current values (no-op if EDA not run)
        def show_summary(state_dict):
            return state_dict.get("summary_str", "Run EDA first.")

        def show_distributions(state_dict):
            return state_dict.get("plot_paths", [])

        def show_correlation(state_dict):
            return state_dict.get("plot_paths", [])

        # Downloads: these functions simply return file paths that Gradio File component accepts
        def get_cleaned(path):
            if path:
                return path
            return None

        def get_pdf(path):
            if path:
                return path
            return None

        # Button wiring (some simply reuse already displayed components)
        summary_btn.click(lambda *_: None, inputs=[], outputs=[summary_text])  # no-op (summary already set)
        dist_btn.click(lambda *_: None, inputs=[], outputs=[plot_gallery])
        corr_btn.click(lambda *_: None, inputs=[], outputs=[plot_gallery])
        out_btn.click(lambda *_: None, inputs=[], outputs=[plot_gallery])
        imb_btn.click(lambda *_: None, inputs=[], outputs=[imbalance_text])
        pdf_btn.click(lambda *_: None, inputs=[], outputs=[pdf_file])
        gem_btn.click(lambda *_: None, inputs=[], outputs=[gemini_text])
        download_csv_btn.click(fn=get_cleaned, inputs=[cleaned_file], outputs=[cleaned_file])
        download_pdf_btn.click(fn=get_pdf, inputs=[pdf_file], outputs=[pdf_file])

    return demo

gradio_app = build_gradio_app()
# Mount Gradio at root
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

