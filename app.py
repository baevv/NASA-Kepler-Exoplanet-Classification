from pathlib import Path
import base64

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = BASE_DIR / "models/metadata.joblib"
LOGO_PATH = BASE_DIR / "assets/logo.png"
BACKGROUND_IMAGE_PATH = BASE_DIR / "assets/background.png"
ASSET_MASCOT_PATH = BASE_DIR / "assets/mascot.png"
ASSET_CONFIRMED_PATH = BASE_DIR / "assets/confirmed.png"
ASSET_CANDIDATE_PATH = BASE_DIR / "assets/candidate.png"
ASSET_FALSE_POSITIVE_PATH = BASE_DIR / "assets/false_positive.png"
ASSET_KEPLER_PATH = BASE_DIR / "assets/kepler.png"

CANDIDATE = "#D97706"
CONFIRMED = "#16A34A"
FALSE_POSITIVE = "#DC2626"

NASA_BLUE = "#1D4ED8"
BACKGROUND = "#F8FAFC"
CARD_BORDER = "#BFDBFE"
TEXT = "#1E293B"

PAGE_TITLES = [
    "Overview",
    "Prediction",
    "Model Comparison",
    "Class Overview",
    "Feature / Variable Analysis",
    "Dictionary",
]

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return X_df.drop(columns=self.columns, errors="ignore")


class MedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.num_cols_ = None

    def fit(self, X, y=None):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.num_cols_ = X_df.select_dtypes(include=[np.number]).columns.tolist()
        if self.num_cols_:
            self.imputer.fit(X_df[self.num_cols_])
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.num_cols_:
            X_df[self.num_cols_] = self.imputer.transform(X_df[self.num_cols_])
        return X_df


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, skewed_cols):
        self.skewed_cols = skewed_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in self.skewed_cols:
            if col in X_df.columns:
                X_df[col] = np.log1p(X_df[col].clip(lower=0))
        return X_df


class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.bounds_ = None
        self.num_cols_ = None

    def fit(self, X, y=None):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.num_cols_ = X_df.select_dtypes(include=[np.number]).columns.tolist()
        self.bounds_ = {}
        for col in self.num_cols_:
            self.bounds_[col] = (
                X_df[col].quantile(self.lower_q),
                X_df[col].quantile(self.upper_q),
            )
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if self.bounds_:
            for col, (lower, upper) in self.bounds_.items():
                if col in X_df.columns:
                    X_df[col] = X_df[col].clip(lower=lower, upper=upper)
        return X_df



@st.cache_data(show_spinner=False)
def load_metadata() -> dict | None:
    if not METADATA_PATH.exists():
        return None
    return joblib.load(METADATA_PATH)


@st.cache_data(show_spinner=False)
def load_feature_dataset() -> pd.DataFrame | None:
    dataset_path = BASE_DIR / "exoplanets_pull.csv"
    if not dataset_path.exists():
        return None
    return pd.read_csv(dataset_path)


@st.cache_resource(show_spinner=False)
def load_models(_metadata: dict | None) -> tuple[dict[str, object], list[tuple[str, Path]]]:
    if not _metadata:
        return {}, []

    loaded_models: dict[str, object] = {}
    missing_models: list[tuple[str, Path]] = []

    for model_name, model_path in _metadata.get("model_files", {}).items():
        path = BASE_DIR / Path(model_path)
        if not path.exists():
            missing_models.append((model_name, path))
            continue

        loaded_models[model_name] = joblib.load(path)

    return loaded_models, missing_models


def get_background_css() -> str:
    if BACKGROUND_IMAGE_PATH.exists():
        encoded_background = base64.b64encode(BACKGROUND_IMAGE_PATH.read_bytes()).decode("utf-8")
        return f"""
            .stApp {{
                background:
                    linear-gradient(
                        rgba(248, 250, 252, 0.72),
                        rgba(239, 246, 255, 0.80)
                    ),
                    url("data:image/png;base64,{encoded_background}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                color: var(--text);
                font-family: 'Inter', sans-serif;
            }}

            [data-testid="stHeader"] {{
                background: transparent;
            }}
        """

    return f"""
        .stApp {{
            background:
                radial-gradient(circle at 8% 8%, rgba(29, 78, 216, 0.10), transparent 28%),
                radial-gradient(circle at 95% 12%, rgba(191, 219, 254, 0.45), transparent 30%),
                linear-gradient(180deg, #F8FAFC 0%, #EFF6FF 100%);
            color: var(--text);
            font-family: 'Inter', sans-serif;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}
    """


def inject_css() -> None:
    background_css = get_background_css()
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Mono:wght@700&display=swap');

            :root {{
                --nasa-blue: {NASA_BLUE};
                --background: {BACKGROUND};
                --card-border: {CARD_BORDER};
                --text: {TEXT};
            }}

            {background_css}

            .block-container {{
                padding-top: 2rem;
            }}

            h1, h2, h3, h4, h5, h6 {{
                color: var(--text);
                font-family: 'Space Mono', monospace;
                letter-spacing: -0.02em;
            }}

            [data-testid="stTabs"] {{
                margin-top: 1.5rem;
            }}

            [data-testid="stTabs"] [role="tablist"] {{
                gap: 0.5rem;
            }}

            [data-testid="stTabs"] [role="tab"] {{
                background: #ffffff;
                border: 1px solid var(--card-border);
                border-radius: 999px;
                color: var(--text);
                padding: 0.5rem 1rem;
            }}

            [data-testid="stTabs"] [aria-selected="true"] {{
                border-color: var(--nasa-blue);
                color: var(--nasa-blue);
            }}

            .app-card {{
                background: #ffffff;
                border: 1px solid var(--card-border);
                border-radius: 18px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
                padding: 1.5rem;
                margin-top: 1rem;
            }}

            .hero-card {{
                background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            }}

            .placeholder-text {{
                color: var(--text);
                font-size: 1rem;
                line-height: 1.7;
                margin: 0;
            }}

            .model-count {{
                color: var(--nasa-blue);
                font-weight: 600;
                margin-top: 0.75rem;
            }}

            .overview-subtitle {{
                color: var(--text);
                font-size: 1.05rem;
                line-height: 1.7;
                margin: 0.35rem 0 0.9rem 0;
            }}

            .section-title {{
                color: var(--text);
                font-family: 'Space Mono', monospace;
                font-size: 1rem;
                margin: 2rem 0 0.5rem 0;
            }}

            .section-divider {{
                background: var(--nasa-blue);
                border-radius: 999px;
                height: 2px;
                margin: 0.25rem 0 1.25rem 0;
                opacity: 0.9;
                width: 100%;
            }}

            .metric-card {{
                background: #ffffff;
                border: 1px solid var(--card-border);
                border-radius: 18px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                min-height: 120px;
                padding: 1rem 1.1rem;
            }}

            .metric-label {{
                color: #475569;
                font-size: 0.84rem;
                font-weight: 600;
                letter-spacing: 0.04em;
                margin: 0 0 0.6rem 0;
                text-transform: uppercase;
            }}

            .metric-value {{
                color: var(--text);
                font-family: 'Space Mono', monospace;
                font-size: 1.25rem;
                line-height: 1.4;
                margin: 0;
                word-break: break-word;
            }}

            .chip-list {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.65rem;
                margin-top: 0.35rem;
            }}

            .model-chip {{
                background: #eff6ff;
                border: 1px solid var(--card-border);
                border-radius: 999px;
                color: var(--text);
                display: inline-block;
                font-size: 0.94rem;
                padding: 0.5rem 0.85rem;
            }}

            .ensemble-card {{
                background: #ffffff;
                border: 1px solid var(--card-border);
                border-radius: 18px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                height: 100%;
                min-height: 150px;
                padding: 1.1rem;
            }}

            .ensemble-title {{
                color: var(--text);
                font-family: 'Space Mono', monospace;
                font-size: 1rem;
                margin: 0 0 0.6rem 0;
            }}

            .ensemble-text {{
                color: var(--text);
                font-size: 0.97rem;
                line-height: 1.65;
                margin: 0;
            }}

            .prediction-result-card {{
                border-radius: 18px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
                margin-top: 1rem;
                padding: 1.25rem 1.4rem;
            }}

            .result-meta {{
                color: #475569;
                font-size: 0.85rem;
                font-weight: 600;
                letter-spacing: 0.04em;
                margin: 0 0 0.35rem 0;
                text-transform: uppercase;
            }}

            .result-value {{
                color: var(--text);
                font-family: 'Space Mono', monospace;
                font-size: 1.15rem;
                line-height: 1.5;
                margin: 0 0 0.8rem 0;
            }}

            .class-badge {{
                border-radius: 999px;
                color: #ffffff;
                display: inline-block;
                font-size: 0.83rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                padding: 0.28rem 0.7rem;
                text-transform: uppercase;
            }}

            .vote-card {{
                background: #ffffff;
                border: 1px solid var(--card-border);
                border-radius: 18px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                min-height: 115px;
                padding: 1rem 1.1rem;
            }}

            .vote-title {{
                color: #475569;
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                margin: 0 0 0.55rem 0;
                text-transform: uppercase;
            }}

            .vote-value {{
                color: var(--text);
                font-family: 'Space Mono', monospace;
                font-size: 1.15rem;
                margin: 0;
            }}

            .prediction-table table {{
                background: #ffffff;
                border: 1px solid var(--card-border);
                border-collapse: separate;
                border-radius: 18px;
                border-spacing: 0;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                overflow: hidden;
                width: 100%;
            }}

            .prediction-table thead th {{
                background: #eff6ff;
                color: var(--text);
                font-family: 'Space Mono', monospace;
                font-size: 0.86rem;
                padding: 0.85rem 1rem;
                text-align: left;
            }}

            .prediction-table tbody td {{
                border-top: 1px solid #dbeafe;
                color: var(--text);
                padding: 0.85rem 1rem;
            }}

            .prediction-mascot-img {{
                max-height: 110px;
                object-fit: contain;
                width: 100%;
            }}

            div[data-testid="stForm"] {{
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
                padding: 1.2rem 1.2rem 0.8rem 1.2rem;
            }}

            div[data-testid="stForm"] [data-baseweb="input"] > div {{
                background: #ffffff;
                border-color: #BFDBFE;
            }}

            div[data-testid="stForm"] [data-baseweb="input"] > div:hover {{
                border-color: #93C5FD;
            }}

            div[data-testid="stForm"] [data-baseweb="input"] > div:focus-within {{
                border-color: var(--nasa-blue);
                box-shadow: 0 0 0 1px rgba(29, 78, 216, 0.18);
            }}

            div[data-testid="stForm"] input {{
                background: transparent;
                color: var(--text);
                -webkit-text-fill-color: var(--text);
            }}

            .class-overview-card {{
                border-radius: 18px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
                min-height: 220px;
                padding: 1.25rem 1.35rem;
            }}

            .class-overview-title {{
                color: var(--text);
                font-family: 'Space Mono', monospace;
                font-size: 1rem;
                margin: 0 0 0.6rem 0;
            }}

            .class-overview-meta {{
                color: #475569;
                font-size: 0.92rem;
                line-height: 1.65;
                margin: 0 0 0.85rem 0;
            }}

            .class-overview-stat {{
                color: var(--text);
                font-family: 'Space Mono', monospace;
                font-size: 1rem;
                margin: 0.25rem 0 0 0;
            }}

            .insight-card {{
                background: #ffffff;
                border: 1px solid var(--card-border);
                border-radius: 18px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                min-height: 180px;
                padding: 1.15rem 1.25rem;
            }}

            .takeaway-card {{
                background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%);
                border-left: 5px solid #1D4ED8;
                border-radius: 18px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
                padding: 1.25rem 1.35rem;
            }}

            .dictionary-feature-card {{
                background: #EFF6FF;
                border: 1px solid #BFDBFE;
                border-radius: 18px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                min-height: 150px;
                padding: 1rem 1.1rem;
            }}

            .dictionary-formula {{
                color: #1D4ED8;
                font-family: 'Space Mono', monospace;
                font-size: 0.88rem;
                line-height: 1.55;
                margin: 0.45rem 0 0.55rem 0;
            }}

            .selector-badge {{
                background: #dbeafe;
                border: 1px solid #93c5fd;
                border-radius: 999px;
                color: #1d4ed8;
                display: inline-block;
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                margin-top: 1.85rem;
                padding: 0.35rem 0.8rem;
                text-transform: uppercase;
            }}

            .global-header {{
                align-items: center;
                background: #ffffff;
                border: 1px solid var(--card-border);
                border-radius: 18px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
                display: flex;
                flex-wrap: wrap;
                gap: 1.75rem;
                margin-top: 0.25rem;
                margin-bottom: 1.25rem;
                padding: 1.5rem 1.75rem;
            }}

            .logo-badge {{
                align-items: center;
                background: var(--nasa-blue);
                border-radius: 999px;
                display: flex;
                flex: 0 0 170px;
                height: 170px;
                justify-content: center;
                width: 170px;
            }}

            .header-logo-img {{
                width: 160px;
                height: 160px;
                object-fit: contain;
                border-radius: 20px;
            }}

            .logo-content {{
                flex: 1 1 280px;
                min-width: 0;
            }}

            .header-text {{
                flex: 1;
                display: flex;
                flex-direction: column;
                justify-content: center;
                min-width: 0;
            }}

            .logo-title {{
                color: var(--text);
                font-family: 'Space Mono', monospace;
                font-size: 3rem;
                font-weight: 900;
                line-height: 1.05;
                letter-spacing: -0.04em;
                margin: 0;
            }}

            .logo-subtitle {{
                color: var(--text);
                font-size: 1.55rem;
                line-height: 1.3;
                margin: 0.45rem 0 0 0;
                font-weight: 500;
            }}

            .header-divider {{
                background: #1D4ED8;
                border-radius: 999px;
                height: 4px;
                margin-top: 1rem;
                width: 100%;
            }}

            .overview-section-title {{
                color: var(--text);
                font-family: 'Space Mono', monospace;
                font-size: 1.08rem;
                margin: 2rem 0 0.65rem 0;
            }}

            .overview-hero {{
                background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
                border: 1px solid var(--card-border);
                border-radius: 22px;
                box-shadow: 0 14px 36px rgba(15, 23, 42, 0.08);
                display: flex;
                flex-wrap: wrap;
                gap: 1.5rem;
                margin-top: 1rem;
                padding: 1.65rem;
            }}

            .overview-hero-text {{
                flex: 1 1 420px;
                min-width: 0;
            }}

            .overview-hero-title {{
                color: var(--text);
                font-family: 'Space Mono', monospace;
                font-size: 1.7rem;
                line-height: 1.2;
                margin: 0 0 0.9rem 0;
            }}

            .overview-hero-text p {{
                color: var(--text);
                font-size: 1rem;
                line-height: 1.8;
                margin: 0 0 0.85rem 0;
            }}

            .overview-hero-visual {{
                align-items: center;
                display: flex;
                flex: 0 1 300px;
                flex-direction: column;
                gap: 1rem;
                justify-content: center;
                min-width: 220px;
            }}

            .overview-class-card {{
                background: #ffffff;
                border: 1px solid var(--card-border);
                border-radius: 18px;
                box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05);
                height: 100%;
                min-height: 220px;
                padding: 1.15rem;
            }}

            .overview-class-card h4 {{
                font-family: 'Space Mono', monospace;
                font-size: 1rem;
                margin: 0.8rem 0 0.55rem 0;
            }}

            .overview-class-card p {{
                color: var(--text);
                font-size: 0.96rem;
                line-height: 1.65;
                margin: 0;
            }}

            .overview-mini-card {{
                background: #ffffff;
                border: 1px solid var(--card-border);
                border-radius: 18px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                height: 100%;
                padding: 1.2rem;
            }}

            .overview-problem-card {{
                border-left: 4px solid #D97706;
                background: linear-gradient(180deg, #fffaf0 0%, #ffffff 100%);
            }}

            .overview-solution-card {{
                border-left: 4px solid #1D4ED8;
                background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%);
            }}

            .overview-card-title {{
                color: var(--text);
                font-family: 'Space Mono', monospace;
                font-size: 1rem;
                margin: 0 0 0.55rem 0;
            }}

            .overview-card-text {{
                color: var(--text);
                font-size: 0.97rem;
                line-height: 1.7;
                margin: 0;
            }}

            .asset-img {{
                display: block;
                height: auto;
                max-height: 200px;
                max-width: 100%;
                object-fit: contain;
            }}

            .overview-class-icon .asset-img {{
                margin: 0 auto;
                max-height: 100px;
            }}

            .hero-kepler-img {{
                max-height: 260px;
                object-fit: contain;
                width: 100%;
            }}

            .hero-mascot-img {{
                max-height: 220px;
                object-fit: contain;
                width: 100%;
            }}

            .kepler-strip-img {{
                max-height: 280px;
                object-fit: contain;
                width: 100%;
            }}

            .kepler-strip-card {{
                background: #ffffff;
                border: 1px solid var(--card-border);
                border-radius: 18px;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
                margin-top: 1rem;
                padding: 1.5rem;
            }}

            .overview-class-icon {{
                align-items: center;
                display: flex;
                justify-content: center;
                margin-bottom: 1rem;
                min-height: 96px;
                width: 100%;
            }}

            .overview-support-asset {{
                align-self: flex-end;
                max-width: 180px;
                width: 100%;
            }}

            .kepler-strip-visual {{
                align-items: center;
                display: flex;
                justify-content: center;
                min-width: 220px;
            }}

            .asset-fallback {{
                align-items: center;
                display: flex;
                font-size: 3rem;
                justify-content: center;
                line-height: 1;
            }}

            .overview-class-icon .asset-fallback {{
                font-size: 4rem;
                min-height: 96px;
            }}

            .overview-support-asset .asset-fallback {{
                font-size: 3rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_placeholder(title: str, description: str) -> None:
    st.title(title)
    st.markdown(
        f"""
        <section class="app-card">
            <p class="placeholder-text">{description}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def format_metric_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_metric_card(label: str, value: object) -> None:
    st.markdown(
        f"""
        <section class="metric-card">
            <p class="metric-label">{label}</p>
            <p class="metric-value">{value}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def get_class_color(class_name: str) -> str:
    color_map = {
        "CANDIDATE": CANDIDATE,
        "CONFIRMED": CONFIRMED,
        "FALSE POSITIVE": FALSE_POSITIVE,
    }
    return color_map.get(class_name, NASA_BLUE)


def get_logo_html() -> str:
    if LOGO_PATH.exists():
        encoded_logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
        return f'<img src="data:image/png;base64,{encoded_logo}" class="header-logo-img" alt="Dashboard logo" />'

    return """
    <div class="logo-badge" aria-hidden="true">
        <svg width="112" height="112" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="14" cy="14" r="2.1" fill="#FFFFFF"/>
            <ellipse cx="14" cy="14" rx="9.6" ry="4.6" stroke="#FFFFFF" stroke-width="1.6" fill="none" transform="rotate(-24 14 14)"/>
            <circle cx="19.8" cy="9.2" r="1.5" fill="#FFFFFF"/>
        </svg>
    </div>
    """


def get_image_html(path: Path, fallback: str, class_name: str = "asset-img", alt: str = "") -> str:
    if path.exists():
        suffix = path.suffix.lower()
        mime_type = "image/png" if suffix == ".png" else "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
        encoded_image = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f'<img src="data:{mime_type};base64,{encoded_image}" class="{class_name}" alt="{alt}" />'

    return f'<div class="{class_name} asset-fallback" aria-label="{alt}">{fallback}</div>'


def render_global_header() -> None:
    logo_html = get_logo_html()
    st.markdown(
        f"""
        <section class="global-header">
            {logo_html}
            <div class="logo-content">
                <div class="header-text">
                    <div class="logo-title" style="font-size: 48px; font-weight: 900; line-height: 1.05; letter-spacing: -1.5px; color: #1E293B;">
                        NASA Kepler Exoplanet Classification
                    </div>
                    <div class="logo-subtitle" style="font-size: 26px; font-weight: 500; line-height: 1.3; margin-top: 8px; color: #1E293B;">
                        Classifying Kepler objects of interest with machine learning
                    </div>
                </div>
                <div class="header-divider"></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_overview(metadata: dict, model_count: int) -> None:
    final_metrics = metadata.get("final_metrics", {})
    final_model_name = metadata.get("final_model_name", "N/A")
    class_names = metadata.get("class_names", [])
    model_names = list(metadata.get("model_files", {}).keys())
    feature_columns_train = metadata.get("feature_columns_train", [])
    raw_dataset_rows = metadata.get("raw_dataset_rows", metadata.get("dataset_rows"))
    if isinstance(raw_dataset_rows, (int, float)):
        total_observations = f"{int(raw_dataset_rows):,} KOI"
    else:
        total_observations = "9,564 KOI"

    mascot_html = get_image_html(ASSET_MASCOT_PATH, "🐕‍🚀", class_name="hero-mascot-img", alt="Mission mascot")
    kepler_html = get_image_html(ASSET_KEPLER_PATH, "🔭", class_name="kepler-strip-img", alt="Kepler telescope")
    confirmed_html = get_image_html(ASSET_CONFIRMED_PATH, "✅", class_name="asset-img", alt="Confirmed icon")
    candidate_html = get_image_html(ASSET_CANDIDATE_PATH, "🔍", class_name="asset-img", alt="Candidate icon")
    false_positive_html = get_image_html(ASSET_FALSE_POSITIVE_PATH, "⚠️", class_name="asset-img", alt="False positive icon")

    st.title("Overview")
    st.markdown(
        f"""
        <section class="overview-hero">
            <div class="overview-hero-text">
                <p class="overview-hero-title">Exoplanet Hunting Through Kepler’s Eyes</p>
                <p>The Kepler Space Telescope monitored thousands of stars over long periods and detected tiny brightness drops in their light curves. These dips may be caused by a planet passing in front of its host star, but not every signal is a real planet. Binary star systems, astrophysical noise, and measurement artifacts can also produce planet-like signals.</p>
                <p>This project uses machine learning to classify Kepler Objects of Interest into CANDIDATE, CONFIRMED, and FALSE POSITIVE classes, helping speed up the first-pass screening process for exoplanet candidates.</p>
            </div>
            <div class="overview-hero-visual">
                {mascot_html}
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="overview-section-title">Classification Dictionary</p>', unsafe_allow_html=True)
    class_cards = [
        ("CANDIDATE", candidate_html, "Planet-like signals that may represent exoplanets, but still require further validation.", CANDIDATE),
        ("CONFIRMED", confirmed_html, "Validated worlds beyond our Solar System; signals confirmed to be real exoplanets.", CONFIRMED),
        ("FALSE POSITIVE", false_positive_html, "Signals that look like planet transits but are actually caused by another phenomenon.", FALSE_POSITIVE),
    ]
    class_columns = st.columns(3)
    for column, (title, image_html, description, accent_color) in zip(class_columns, class_cards):
        with column:
            st.markdown(
                f"""
                <section class="overview-class-card" style="border-left: 4px solid {accent_color};">
                    <div class="overview-class-icon">{image_html}</div>
                    <h4>{title}</h4>
                    <p>{description}</p>
                </section>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <section class="kepler-strip-card">
            <div class="overview-hero" style="background: transparent; border: none; box-shadow: none; margin-top: 0; padding: 0; gap: 1.25rem;">
                <div class="overview-hero-text">
                    <p class="overview-hero-title" style="font-size: 1.3rem; margin-bottom: 0.7rem;">From Light Curves to Class Labels</p>
                    <p>Kepler detects repeated brightness dips in stellar light curves. These dips may indicate that an object is passing in front of a star, but the raw signal alone is not enough to decide whether it is a real exoplanet. The dashboard converts these observations into numerical features such as orbital period, transit depth, signal-to-noise ratio, planet radius, and stellar properties. After preprocessing and feature engineering, machine learning models use these patterns to classify each Kepler Object of Interest as CANDIDATE, CONFIRMED, or FALSE POSITIVE.</p>
                </div>
                <div class="kepler-strip-visual">
                    {kepler_html}
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="overview-section-title">Why Machine Learning?</p>', unsafe_allow_html=True)
    why_columns = st.columns(2)
    with why_columns[0]:
        st.markdown(
            """
            <section class="overview-mini-card overview-problem-card">
                <p class="overview-card-title">Problem</p>
                <p class="overview-card-text">Manually reviewing thousands of KOI signals is time-consuming. The CANDIDATE class is especially difficult because its patterns can overlap with both real exoplanets and false-positive signals.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
    with why_columns[1]:
        st.markdown(
            """
            <section class="overview-mini-card overview-solution-card">
                <p class="overview-card-title">Solution</p>
                <p class="overview-card-text">Machine learning models can evaluate transit properties, stellar parameters, and signal quality together to perform fast preliminary classification. In this project, XGBoost was selected as the final model.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<p class="overview-section-title">Dataset Quick Stats</p>', unsafe_allow_html=True)
    stats_columns = st.columns(4)
    stats_items = [
        ("Total KOI Records", total_observations),
        ("Classes", len(class_names) if class_names else 3),
        ("Features", len(feature_columns_train) if feature_columns_train else "40+"),
        ("Data Source", "NASA Exoplanet Archive"),
    ]
    for column, (label, value) in zip(stats_columns, stats_items):
        with column:
            render_metric_card(label, value)
    st.caption("KOI: Kepler Object of Interest")

    st.markdown('<p class="overview-section-title">Pipeline Flow</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <section class="app-card">
            <p class="placeholder-text">
                Raw Kepler data is cleaned, leakage columns are removed, ratio features are engineered, missing values are imputed,
                skewed variables are transformed, outliers are clipped, features are scaled, class imbalance is handled with SMOTE, and models are evaluated.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.graphviz_chart(
        """
        digraph Pipeline {
            graph [bgcolor="transparent", rankdir=LR, nodesep="0.35", ranksep="0.65"];
            node [
                shape=box,
                style="rounded,filled",
                fillcolor="#DBEAFE",
                color="#1D4ED8",
                fontcolor="#1E293B",
                fontname="Helvetica",
                margin="0.18,0.12",
                penwidth=1.4
            ];
            edge [
                color="#1D4ED8",
                penwidth=1.4,
                arrowsize=0.8
            ];

            raw [label="Raw Data"];
            drop [label="Drop Leakage Columns"];
            fe [label="Feature Engineering"];
            imp [label="Median Imputation"];
            log [label="Log Transform"];
            clip [label="Outlier Clipping"];
            scale [label="Standard Scaling"];
            smote [label="SMOTE"];
            train [label="Model Training"];
            eval [label="Evaluation"];

            raw -> drop -> fe -> imp -> log -> clip -> scale -> smote -> train -> eval;
        }
        """
    )

    st.markdown('<p class="overview-section-title">Final Results</p>', unsafe_allow_html=True)
    metric_columns = st.columns(5)
    metric_items = [
        ("Final Model", final_model_name),
        ("Macro F1", format_metric_value(final_metrics.get("Macro F1", final_metrics.get("macro_f1", "N/A")))),
        ("Accuracy", format_metric_value(final_metrics.get("Accuracy", final_metrics.get("accuracy", "N/A")))),
        ("Models", model_count),
        ("Classes", len(class_names)),
    ]
    for column, (label, value) in zip(metric_columns, metric_items):
        with column:
            render_metric_card(label, value)

    st.markdown('<p class="overview-section-title">Models Used</p>', unsafe_allow_html=True)
    st.markdown(
        '<section class="app-card"><div class="chip-list">'
        + "".join(f'<span class="model-chip">{name}</span>' for name in model_names)
        + "</div></section>",
        unsafe_allow_html=True,
    )

    st.markdown('<p class="overview-section-title">Ensemble Summary</p>', unsafe_allow_html=True)
    ensemble_columns = st.columns(3)
    ensemble_items = [
        ("Random Forest", "Bagging / Random Subspace"),
        ("XGBoost", "Boosting"),
        (
            "Stacking",
            "Meta-ensemble using XGBoost + Random Forest + SVM with Logistic Regression as final estimator",
        ),
    ]

    for column, (title, description) in zip(ensemble_columns, ensemble_items):
        with column:
            st.markdown(
                f"""
                <section class="ensemble-card">
                    <p class="ensemble-title">{title}</p>
                    <p class="ensemble-text">{description}</p>
                </section>
                """,
                unsafe_allow_html=True,
            )
    st.caption("Stacking was evaluated as an ensemble approach, while XGBoost was selected as the final model based on held-out Macro F1.")


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    ratio_specs = {
        "koi_prad_srad_ratio": ("koi_prad", "koi_srad"),
        "koi_snr_depth_ratio": ("koi_model_snr", "koi_depth"),
        "koi_insol_teq_ratio": ("koi_insol", "koi_teq"),
    }

    for ratio_name, (numerator_col, denominator_col) in ratio_specs.items():
        if numerator_col in df.columns and denominator_col in df.columns:
            df[ratio_name] = df[numerator_col] / (df[denominator_col] + 1e-6)

    return df


def validate_prediction_inputs(values: dict[str, float]) -> list[str]:
    validations = [
        ("koi_period", values.get("koi_period", 0.0) > 0, "koi_period must be greater than 0."),
        ("koi_duration", values.get("koi_duration", 0.0) > 0, "koi_duration must be greater than 0."),
        ("koi_depth", values.get("koi_depth", 0.0) >= 0, "koi_depth must be greater than or equal to 0."),
        ("koi_prad", values.get("koi_prad", 0.0) >= 0, "koi_prad must be greater than or equal to 0."),
        ("koi_teq", values.get("koi_teq", 0.0) >= 0, "koi_teq must be greater than or equal to 0."),
        ("koi_insol", values.get("koi_insol", 0.0) >= 0, "koi_insol must be greater than or equal to 0."),
        (
            "koi_model_snr",
            values.get("koi_model_snr", 0.0) >= 0,
            "koi_model_snr must be greater than or equal to 0.",
        ),
        ("koi_srad", values.get("koi_srad", 0.0) > 0, "koi_srad must be greater than 0."),
    ]

    return [message for _, is_valid, message in validations if not is_valid]


def build_prediction_dataframe(values: dict[str, float], metadata: dict) -> pd.DataFrame:
    feature_columns = metadata.get("feature_columns_train", [])
    feature_medians = metadata.get("feature_medians", {})
    base_row = {
        feature: float(feature_medians.get(feature, 0.0))
        for feature in feature_columns
    }
    base_row.update({feature: float(value) for feature, value in values.items()})

    input_df = pd.DataFrame([base_row])
    input_df = add_ratio_features(input_df)
    input_df = input_df[feature_columns]
    return input_df


def predict_all_models(
    input_df: pd.DataFrame,
    models: dict[str, object],
    label_encoder,
) -> tuple[list[dict[str, str | float | None]], list[str]]:
    results: list[dict[str, str | float | None]] = []
    errors: list[str] = []

    for model_name, model in models.items():
        try:
            encoded_prediction = model.predict(input_df)
            class_name = label_encoder.inverse_transform(encoded_prediction)[0]
            confidence = None

            if hasattr(model, "predict_proba"):
                confidence = float(model.predict_proba(input_df)[0].max())

            results.append(
                {
                    "Model": model_name,
                    "Prediction": class_name,
                    "Confidence": confidence,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "Model": model_name,
                    "Prediction": "ERROR",
                    "Confidence": None,
                }
            )
            errors.append(f"{model_name}: {exc}")

    return results, errors


def render_final_prediction_card(
    final_result: dict[str, str | float | None] | None,
    final_model_name: str,
) -> None:
    if not final_result:
        return

    prediction = str(final_result.get("Prediction", "N/A"))
    if prediction == "ERROR":
        border_color = NASA_BLUE
        background_color = "#EFF6FF"
        confidence_value = "N/A"
    else:
        border_color = get_class_color(prediction)
        background_color_map = {
            "CONFIRMED": "#F0FDF4",
            "FALSE POSITIVE": "#FEF2F2",
            "CANDIDATE": "#FFFBEB",
        }
        background_color = background_color_map.get(prediction, "#EFF6FF")
        confidence = final_result.get("Confidence")
        confidence_value = f"{confidence:.3f}" if isinstance(confidence, (int, float)) else "N/A"

    badge = (
        f'<span class="class-badge" style="background:{get_class_color(prediction)};">{prediction}</span>'
        if prediction != "ERROR"
        else '<span class="class-badge" style="background:#475569;">ERROR</span>'
    )

    st.markdown(
        f"""
        <section class="prediction-result-card" style="border-left: 5px solid {border_color}; background: {background_color};">
            <p class="result-meta">Final Selected Model</p>
            <p class="result-value">{final_model_name}</p>
            <p class="result-meta">Prediction</p>
            <p class="result-value">{badge}</p>
            <p class="result-meta">Confidence</p>
            <p class="result-value">{confidence_value}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_vote_summary(results: list[dict[str, str | float | None]], class_names: list[str]) -> None:
    valid_results = [result for result in results if result.get("Prediction") != "ERROR"]
    total_votes = len(valid_results)

    st.markdown('<p class="section-title">Model Agreement</p>', unsafe_allow_html=True)
    vote_columns = st.columns(len(class_names) if class_names else 1)

    for column, class_name in zip(vote_columns, class_names):
        vote_count = sum(1 for result in valid_results if result.get("Prediction") == class_name)
        with column:
            st.markdown(
                f"""
                <section class="vote-card" style="border-left: 4px solid {get_class_color(class_name)};">
                    <p class="vote-title">{class_name}</p>
                    <p class="vote-value">{vote_count} / {total_votes}</p>
                </section>
                """,
                unsafe_allow_html=True,
            )


def render_prediction(metadata: dict, models: dict[str, object]) -> None:
    st.title("Prediction")
    mascot_html = get_image_html(ASSET_MASCOT_PATH, "🐕‍🚀", class_name="prediction-mascot-img", alt="Prediction mascot")
    st.markdown(
        f"""
        <section class="app-card">
            <div style="display:flex; flex-wrap:wrap; align-items:center; gap:1rem;">
                <div style="flex:1 1 420px;">
                    <p class="overview-subtitle">Single KOI prediction using the trained machine learning models. Enter the main transit, planet, and stellar measurements below, or load one of the example profiles to test how the models classify a Kepler Object of Interest.</p>
                </div>
                <div style="flex:0 0 120px; max-width:120px;">
                    {mascot_html}
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    example_inputs = metadata.get("example_inputs", {})
    feature_medians = metadata.get("feature_medians", {})
    final_model_name = metadata.get("final_model_name", "N/A")
    class_names = metadata.get("class_names", [])
    label_encoder = metadata.get("label_encoder")

    minimal_fields = [
        ("koi_period", "KOI Period"),
        ("koi_duration", "KOI Duration"),
        ("koi_depth", "KOI Depth"),
        ("koi_prad", "KOI Planet Radius"),
        ("koi_teq", "KOI Equilibrium Temperature"),
        ("koi_insol", "KOI Insolation"),
        ("koi_model_snr", "KOI Model SNR"),
        ("koi_srad", "KOI Stellar Radius"),
    ]

    if "prediction_results" not in st.session_state:
        st.session_state.prediction_results = None
    if "prediction_validation_errors" not in st.session_state:
        st.session_state.prediction_validation_errors = []
    if "prediction_model_errors" not in st.session_state:
        st.session_state.prediction_model_errors = []

    st.markdown('<p class="section-title">Input Features</p>', unsafe_allow_html=True)
    st.caption("These inputs represent the most important measurements needed for a quick prediction. Missing advanced features are filled using training-set median defaults.")
    st.caption("Try a predefined KOI profile or enter your own values.")
    example_columns = st.columns(3)
    example_map = [
        ("Load Example: Candidate", "CANDIDATE"),
        ("Load Example: Confirmed Planet", "CONFIRMED"),
        ("Load Example: False Positive", "FALSE POSITIVE"),
    ]

    for column, (button_label, example_key) in zip(example_columns, example_map):
        with column:
            if st.button(button_label, use_container_width=True):
                selected_example = example_inputs.get(example_key, {})
                for field_name, _ in minimal_fields:
                    st.session_state[f"prediction_{field_name}"] = float(
                        selected_example.get(field_name, feature_medians.get(field_name, 0.0))
                    )
                st.session_state.prediction_results = None
                st.session_state.prediction_validation_errors = []
                st.session_state.prediction_model_errors = []

    st.markdown("#### Minimal KOI Input")
    st.caption(
        "Enter the 8 core KOI measurements used for quick prediction. Remaining model features are filled using training-set median defaults."
    )
    with st.form("prediction-form"):
        input_columns = st.columns(2)
        entered_values: dict[str, float] = {}

        for index, (field_name, field_label) in enumerate(minimal_fields):
            default_value = float(
                st.session_state.get(
                    f"prediction_{field_name}",
                    feature_medians.get(field_name, 0.0),
                )
            )
            with input_columns[index % 2]:
                entered_values[field_name] = float(
                    st.number_input(
                        field_label,
                        value=default_value,
                        step=0.1,
                        key=f"prediction_{field_name}",
                    )
                )

        submitted = st.form_submit_button("Predict KOI Class", use_container_width=True)

    if submitted:
        validation_errors = validate_prediction_inputs(entered_values)
        if validation_errors:
            st.session_state.prediction_results = None
            st.session_state.prediction_validation_errors = validation_errors
            st.session_state.prediction_model_errors = []
        else:
            input_df = build_prediction_dataframe(entered_values, metadata)
            results, prediction_errors = predict_all_models(input_df, models, label_encoder)
            st.session_state.prediction_results = results
            st.session_state.prediction_validation_errors = []
            st.session_state.prediction_model_errors = prediction_errors

    validation_errors = st.session_state.get("prediction_validation_errors", [])
    model_errors = st.session_state.get("prediction_model_errors", [])

    if validation_errors:
        st.error("\n".join(validation_errors))
        return

    results = st.session_state.get("prediction_results")
    if not results:
        st.markdown(
            """
            <section class="app-card">
                <p class="placeholder-text">Prediction results will appear here after you click “Predict KOI Class”.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    final_result = next(
        (result for result in results if result.get("Model") == final_model_name),
        None,
    )
    render_final_prediction_card(final_result, final_model_name)

    st.markdown('<p class="section-title">All Model Predictions</p>', unsafe_allow_html=True)
    table_rows = []
    for result in results:
        confidence = result.get("Confidence")
        prediction = str(result.get("Prediction", "N/A"))

        table_rows.append(
            {
                "Model": result.get("Model", "N/A"),
                "Prediction": prediction,
                "Confidence": f"{confidence:.3f}" if isinstance(confidence, (int, float)) else "N/A",
            }
        )

    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
    )

    render_vote_summary(results, class_names)

    if model_errors:
        st.warning("Some models returned errors:\n" + "\n".join(f"- {error}" for error in model_errors))

    st.info(
        "Some models may not provide confidence scores. For example, SVM RBF may be trained without probability=True, so confidence is shown as N/A."
    )


def get_selected_results_df(metadata: dict, selected_models: list[str]) -> pd.DataFrame:
    results_df = pd.DataFrame(metadata.get("tuned_results_table", []))
    if results_df.empty:
        return results_df
    return results_df[results_df["Model"].isin(selected_models)].reset_index(drop=True)


def build_class_f1_df(metadata: dict, selected_models: list[str]) -> pd.DataFrame:
    reports = metadata.get("classification_reports", {})
    rows = []
    for model_name in selected_models:
        report = reports.get(model_name, {})
        rows.append(
            {
                "Model": model_name,
                "CANDIDATE F1": float(report.get("CANDIDATE", {}).get("f1-score", 0.0)),
                "CONFIRMED F1": float(report.get("CONFIRMED", {}).get("f1-score", 0.0)),
                "FALSE POSITIVE F1": float(report.get("FALSE POSITIVE", {}).get("f1-score", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_selected_class_metrics_df(
    metadata: dict,
    selected_models: list[str],
    selected_class: str,
) -> pd.DataFrame:
    reports = metadata.get("classification_reports", {})
    rows = []
    for model_name in selected_models:
        class_metrics = reports.get(model_name, {}).get(selected_class, {})
        rows.append(
            {
                "Model": model_name,
                "Precision": float(class_metrics.get("precision", 0.0)),
                "Recall": float(class_metrics.get("recall", 0.0)),
                "F1-score": float(class_metrics.get("f1-score", 0.0)),
                "Support": float(class_metrics.get("support", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def render_metric_cards(metric_items: list[tuple[str, object]]) -> None:
    columns = st.columns(len(metric_items))
    for column, (label, value) in zip(columns, metric_items):
        with column:
            render_metric_card(label, value)


def short_model_name(model_name: str) -> str:
    short_names = {
        "Logistic Regression (Tuned)": "Logistic Regression",
        "Decision Tree (Tuned)": "Decision Tree",
        "Random Forest (Tuned)": "Random Forest",
        "XGBoost (Tuned)": "XGBoost",
        "SVM RBF (Tuned)": "SVM RBF",
        "Stacking (Tuned Ensemble)": "Stacking",
    }
    return short_names.get(model_name, model_name)


def get_plotly_layout(
    title: str,
    height: int = 360,
    margin: dict[str, int] | None = None,
    legend: dict[str, str | float] | None = None,
) -> dict:
    return {
        "title": title,
        "height": height,
        "paper_bgcolor": BACKGROUND,
        "plot_bgcolor": "#FFFFFF",
        "font": {"color": TEXT},
        "margin": margin or {"l": 35, "r": 20, "t": 45, "b": 35},
        "legend": legend
        or {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    }


def render_single_metric_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color: str = NASA_BLUE,
    y_range: list[float] | None = None,
    highlight_model_name: str | None = None,
    secondary_color: str = "#93C5FD",
    textposition: str = "auto",
) -> None:
    if df.empty:
        st.info("No data available for this view.")
        return

    chart_df = df.copy()
    chart_df["Chart Model"] = chart_df[x_col].map(short_model_name)

    if go is not None:
        bar_colors = []
        for model_name in chart_df[x_col]:
            if highlight_model_name and model_name == highlight_model_name:
                bar_colors.append(color)
            else:
                bar_colors.append(secondary_color)

        figure = go.Figure(
            data=[
                go.Bar(
                    x=chart_df["Chart Model"],
                    y=chart_df[y_col],
                    marker_color=bar_colors,
                    text=[f"{value:.3f}" for value in chart_df[y_col]],
                    textposition=textposition,
                    hovertemplate="<b>%{x}</b><br>"
                    + f"{y_col}: "
                    + "%{y:.3f}<extra></extra>",
                )
            ]
        )
        figure.update_layout(
            bargap=0.2,
            showlegend=False,
            **get_plotly_layout(title, height=360),
        )
        figure.update_xaxes(title_text="", tickangle=0)
        if y_range is None:
            figure.update_yaxes(title_text=y_col, zeroline=True, zerolinecolor="#94A3B8")
        else:
            figure.update_yaxes(
                title_text=y_col,
                range=y_range,
                zeroline=True,
                zerolinecolor="#94A3B8",
            )
        st.plotly_chart(figure, use_container_width=True)
        return

    fallback_df = chart_df.set_index("Chart Model")[[y_col]]
    st.bar_chart(fallback_df)


def render_grouped_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    color_map: dict[str, str],
    y_axis_title: str,
    height: int = 360,
    layout_margin: dict[str, int] | None = None,
    legend_layout: dict[str, str | float] | None = None,
) -> None:
    if df.empty:
        st.info("No data available for this view.")
        return

    chart_df = df.copy()
    chart_df["Chart Model"] = chart_df[x_col].map(short_model_name)

    if go is not None:
        figure = go.Figure()
        for y_col in y_cols:
            figure.add_trace(
                go.Bar(
                    x=chart_df["Chart Model"],
                    y=chart_df[y_col],
                    name=y_col,
                    marker_color=color_map.get(y_col, NASA_BLUE),
                    text=[f"{value:.3f}" for value in chart_df[y_col]],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>"
                    + f"{y_col}: "
                    + "%{y:.3f}<extra></extra>",
                )
            )

        figure.update_layout(
            barmode="group",
            bargap=0.18,
            bargroupgap=0.08,
            **get_plotly_layout(
                title,
                height=height,
                margin=layout_margin,
                legend=legend_layout,
            ),
        )
        figure.update_xaxes(title_text="", tickangle=0)
        figure.update_yaxes(title_text=y_axis_title, range=[0, 1])
        st.plotly_chart(figure, use_container_width=True)
        return

    fallback_df = chart_df[["Chart Model"] + y_cols].set_index("Chart Model")
    st.dataframe(fallback_df.style.format("{:.3f}"), use_container_width=True)


def render_per_class_f1_heatmap(class_f1_df: pd.DataFrame) -> None:
    if class_f1_df.empty:
        st.info("No class-wise F1 data available.")
        return

    heatmap_df = class_f1_df.copy()
    heatmap_df["Model"] = heatmap_df["Model"].map(short_model_name)
    heatmap_df = heatmap_df.rename(
        columns={
            "CANDIDATE F1": "CANDIDATE",
            "CONFIRMED F1": "CONFIRMED",
            "FALSE POSITIVE F1": "FALSE POSITIVE",
        }
    ).set_index("Model")

    if go is not None:
        figure = go.Figure(
            data=go.Heatmap(
                z=heatmap_df.values,
                x=list(heatmap_df.columns),
                y=list(heatmap_df.index),
                zmin=0,
                zmax=1,
                colorscale=[
                    [0.0, "#F8FAFC"],
                    [0.35, "#BFDBFE"],
                    [0.7, "#60A5FA"],
                    [1.0, "#1D4ED8"],
                ],
                colorbar={"title": "F1"},
                hovertemplate="Model: %{y}<br>Class: %{x}<br>F1: %{z:.3f}<extra></extra>",
            )
        )
        annotations = []
        for row_index, model_name in enumerate(heatmap_df.index):
            for col_index, class_name in enumerate(heatmap_df.columns):
                annotations.append(
                    {
                        "x": class_name,
                        "y": model_name,
                        "text": f"{heatmap_df.iloc[row_index, col_index]:.3f}",
                        "showarrow": False,
                        "font": {"color": TEXT},
                    }
                )
        figure.update_layout(
            **get_plotly_layout("Per-Class F1 Heatmap", height=360),
            annotations=annotations,
        )
        figure.update_xaxes(title_text="")
        figure.update_yaxes(title_text="")
        st.plotly_chart(figure, use_container_width=True)
        return

    st.dataframe(heatmap_df.style.format("{:.3f}"), use_container_width=True)


def render_precision_recall_scatter(
    selected_results_df: pd.DataFrame,
    final_model_name: str,
) -> None:
    if selected_results_df.empty:
        st.info("No performance data available for the selected models.")
        return

    chart_df = selected_results_df.copy()
    chart_df["Chart Model"] = chart_df["Model"].map(short_model_name)
    chart_df["Marker Size"] = 12 + (chart_df["Macro F1"].astype(float) * 28)

    if go is not None:
        figure = go.Figure()
        non_final_df = chart_df[chart_df["Model"] != final_model_name]
        final_df = chart_df[chart_df["Model"] == final_model_name]

        if not non_final_df.empty:
            figure.add_trace(
                go.Scatter(
                    x=non_final_df["Macro Precision"],
                    y=non_final_df["Macro Recall"],
                    mode="markers",
                    marker={
                        "size": non_final_df["Marker Size"],
                        "color": "#475569",
                        "opacity": 0.9,
                    },
                    customdata=non_final_df[["Model", "Accuracy", "Macro F1"]],
                    name="Selected Models",
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                    + "Macro Precision: %{x:.3f}<br>"
                    + "Macro Recall: %{y:.3f}<br>"
                    + "Accuracy: %{customdata[1]:.3f}<br>"
                    + "Macro F1: %{customdata[2]:.3f}<extra></extra>",
                )
            )

        if not final_df.empty:
            figure.add_trace(
                go.Scatter(
                    x=final_df["Macro Precision"],
                    y=final_df["Macro Recall"],
                    mode="markers+text",
                    text=final_df["Chart Model"],
                    textposition="top center",
                    marker={
                        "size": final_df["Marker Size"] + 6,
                        "color": "#93C5FD",
                        "line": {"color": NASA_BLUE, "width": 3},
                        "opacity": 1.0,
                    },
                    customdata=final_df[["Model", "Accuracy", "Macro F1"]],
                    name="Final Model",
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                    + "Macro Precision: %{x:.3f}<br>"
                    + "Macro Recall: %{y:.3f}<br>"
                    + "Accuracy: %{customdata[1]:.3f}<br>"
                    + "Macro F1: %{customdata[2]:.3f}<extra></extra>",
                )
            )

        figure.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line={"color": "#94A3B8", "dash": "dash"},
        )
        figure.update_layout(
            **get_plotly_layout(
                "Macro Precision vs Macro Recall",
                height=360,
                margin={"l": 35, "r": 20, "t": 60, "b": 80},
                legend={
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.20,
                    "xanchor": "center",
                    "x": 0.5,
                },
            )
        )
        figure.update_xaxes(title_text="Macro Precision", range=[0, 1])
        figure.update_yaxes(title_text="Macro Recall", range=[0, 1])
        st.plotly_chart(figure, use_container_width=True)
        return

    fallback_df = chart_df[["Model", "Macro Precision", "Macro Recall", "Accuracy", "Macro F1"]].copy()
    st.dataframe(fallback_df, use_container_width=True, hide_index=True)


def render_confusion_matrix(
    matrix: list[list[int]] | np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
) -> None:
    matrix_df = pd.DataFrame(matrix, index=class_names, columns=class_names)

    if go is not None:
        matrix_values = matrix_df.values
        max_value = int(matrix_values.max()) if matrix_values.size else 0
        annotations = []
        for row_index, true_class in enumerate(class_names):
            for col_index, predicted_class in enumerate(class_names):
                cell_value = int(matrix_values[row_index, col_index])
                text_color = "#FFFFFF" if max_value and cell_value >= (0.55 * max_value) else TEXT
                annotations.append(
                    {
                        "x": predicted_class,
                        "y": true_class,
                        "text": str(cell_value),
                        "showarrow": False,
                        "font": {"color": text_color, "size": 13},
                    }
                )

        figure = go.Figure(
            data=go.Heatmap(
                z=matrix_values,
                x=class_names,
                y=class_names,
                colorscale=[
                    [0.0, "#EFF6FF"],
                    [0.45, "#60A5FA"],
                    [1.0, NASA_BLUE],
                ],
                hoverongaps=False,
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            )
        )
        figure.update_layout(
            **get_plotly_layout(title, height=400),
            xaxis_title="Predicted Class",
            yaxis_title="True Class",
            annotations=annotations,
        )
        figure.update_xaxes(
            title_text="Predicted Class",
            categoryorder="array",
            categoryarray=class_names,
        )
        figure.update_yaxes(
            title_text="True Class",
            categoryorder="array",
            categoryarray=class_names,
            autorange="reversed",
        )
        st.plotly_chart(figure, use_container_width=True)
        return

    st.dataframe(matrix_df, use_container_width=True)


def normalize_base_model_name(model_name: str) -> str:
    base_name = model_name.replace(" (Tuned Ensemble)", "")
    base_name = base_name.replace(" (Tuned)", "")
    base_name = base_name.replace(" (Baseline)", "")
    return base_name


def get_class_distribution_df(metadata: dict) -> pd.DataFrame:
    class_names = metadata.get("class_names", [])
    class_distribution = metadata.get("class_distribution", {})
    class_percentages = metadata.get("class_percentages", {})

    rows = []
    for class_name in class_names:
        rows.append(
            {
                "Class": class_name,
                "Count": int(class_distribution.get(class_name, 0)),
                "Percentage": float(class_percentages.get(class_name, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def get_selected_model_class_metrics_df(metadata: dict, selected_model: str) -> pd.DataFrame:
    class_names = metadata.get("class_names", [])
    reports = metadata.get("classification_reports", {})
    selected_report = reports.get(selected_model, {})

    rows = []
    for class_name in class_names:
        class_metrics = selected_report.get(class_name, {})
        rows.append(
            {
                "Class": class_name,
                "Precision": float(class_metrics.get("precision", 0.0)),
                "Recall": float(class_metrics.get("recall", 0.0)),
                "F1-score": float(class_metrics.get("f1-score", 0.0)),
                "Support": int(class_metrics.get("support", 0)),
            }
        )
    return pd.DataFrame(rows)


def get_average_class_f1_df(metadata: dict) -> pd.DataFrame:
    class_names = metadata.get("class_names", [])
    reports = metadata.get("classification_reports", {})

    rows = []
    for class_name in class_names:
        class_scores = [
            float(model_report.get(class_name, {}).get("f1-score", 0.0))
            for model_report in reports.values()
            if isinstance(model_report, dict)
        ]
        rows.append(
            {
                "Class": class_name,
                "Average F1": float(np.mean(class_scores)) if class_scores else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("Average F1", ascending=True).reset_index(drop=True)


def get_confusion_insights(metadata: dict, selected_model: str) -> dict[str, object] | None:
    class_names = metadata.get("class_names", [])
    confusion_matrices = metadata.get("confusion_matrices", {})
    matrix = confusion_matrices.get(selected_model)
    if not class_names or matrix is None:
        return None

    matrix_array = np.array(matrix)
    off_diagonal = matrix_array.copy()
    np.fill_diagonal(off_diagonal, 0)

    if off_diagonal.size == 0 or int(off_diagonal.max()) == 0:
        largest_confusion = {
            "true_class": "None",
            "predicted_class": "None",
            "count": 0,
        }
    else:
        row_idx, col_idx = np.unravel_index(np.argmax(off_diagonal), off_diagonal.shape)
        largest_confusion = {
            "true_class": class_names[row_idx],
            "predicted_class": class_names[col_idx],
            "count": int(off_diagonal[row_idx, col_idx]),
        }

    per_class_confusions: dict[str, dict[str, object]] = {}
    for row_idx, class_name in enumerate(class_names):
        row_values = matrix_array[row_idx].copy()
        row_values[row_idx] = 0
        predicted_idx = int(np.argmax(row_values))
        count = int(row_values[predicted_idx])
        if count == 0:
            per_class_confusions[class_name] = {
                "predicted_class": "No major confusion",
                "count": 0,
            }
        else:
            per_class_confusions[class_name] = {
                "predicted_class": class_names[predicted_idx],
                "count": count,
            }

    return {
        "matrix": matrix_array,
        "largest_confusion": largest_confusion,
        "per_class_confusions": per_class_confusions,
    }


def render_class_meaning_card(
    class_name: str,
    meaning: str,
    importance: str,
    difficulty: str,
    count: int,
    percentage: float,
    border_color: str,
    background_color: str,
) -> None:
    st.markdown(
        f"""
        <section class="class-overview-card" style="border-left: 5px solid {border_color}; background: {background_color};">
            <p class="class-overview-title">{class_name}</p>
            <p class="class-overview-meta">{meaning}</p>
            <p class="class-overview-meta"><strong>Why important:</strong> {importance}</p>
            <p class="class-overview-meta"><strong>Expected difficulty:</strong> {difficulty}</p>
            <p class="class-overview-stat">Support: {count:,}</p>
            <p class="class-overview-stat">Percentage: {percentage:.1f}%</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_interpretation_card(
    class_name: str,
    precision: float,
    recall: float,
    f1_score: float,
) -> None:
    if precision >= 0.8 and recall >= 0.8:
        interpretation = "The model handles this class relatively well."
    elif precision < recall:
        interpretation = (
            "The model catches many true examples of this class, but it also assigns this label to some incorrect cases."
        )
    elif precision > recall:
        interpretation = (
            "When the model predicts this class, it is relatively reliable, but it misses some true examples."
        )
    else:
        interpretation = "Precision and recall are balanced, so the model is making a more even trade-off for this class."

    if class_name == "CANDIDATE":
        interpretation += " CANDIDATE is naturally ambiguous because it lies between confirmed planets and false positives."

    st.markdown(
        f"""
        <section class="insight-card" style="border-left: 4px solid {get_class_color(class_name)};">
            <p class="class-overview-title">{class_name}</p>
            <p class="class-overview-meta">Precision: {precision:.3f}</p>
            <p class="class-overview-meta">Recall: {recall:.3f}</p>
            <p class="class-overview-meta">F1-score: {f1_score:.3f}</p>
            <p class="placeholder-text">{interpretation}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_class_count_chart(class_distribution_df: pd.DataFrame) -> None:
    class_colors = [get_class_color(class_name) for class_name in class_distribution_df["Class"]]
    if go is not None:
        figure = go.Figure(
            data=[
                go.Bar(
                    x=class_distribution_df["Class"],
                    y=class_distribution_df["Count"],
                    marker_color=class_colors,
                    text=[f"{value:,}" for value in class_distribution_df["Count"]],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
                )
            ]
        )
        figure.update_layout(
            showlegend=False,
            **get_plotly_layout(
                "Class Count",
                height=360,
                margin={"l": 35, "r": 20, "t": 55, "b": 40},
            ),
        )
        figure.update_xaxes(title_text="")
        figure.update_yaxes(title_text="Count")
        st.plotly_chart(figure, use_container_width=True)
        return

    st.dataframe(class_distribution_df[["Class", "Count"]], use_container_width=True, hide_index=True)


def render_class_percentage_donut(class_distribution_df: pd.DataFrame) -> None:
    class_colors = [get_class_color(class_name) for class_name in class_distribution_df["Class"]]
    if go is not None:
        figure = go.Figure(
            data=[
                go.Pie(
                    labels=class_distribution_df["Class"],
                    values=class_distribution_df["Percentage"],
                    hole=0.52,
                    marker={"colors": class_colors},
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
                    sort=False,
                )
            ]
        )
        figure.update_layout(
            **get_plotly_layout(
                "Class Percentage",
                height=360,
                margin={"l": 35, "r": 20, "t": 55, "b": 40},
                legend={
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.08,
                    "xanchor": "center",
                    "x": 0.5,
                },
            ),
        )
        st.plotly_chart(figure, use_container_width=True)
        return

    fallback_df = class_distribution_df.copy()
    fallback_df["Percentage"] = fallback_df["Percentage"].map(lambda value: f"{value:.1f}%")
    st.dataframe(fallback_df[["Class", "Percentage"]], use_container_width=True, hide_index=True)


def render_support_vs_f1_scatter(selected_model_metrics_df: pd.DataFrame, selected_model: str) -> None:
    if go is not None:
        figure = go.Figure()
        for _, row in selected_model_metrics_df.iterrows():
            class_name = row["Class"]
            figure.add_trace(
                go.Scatter(
                    x=[row["Support"]],
                    y=[row["F1-score"]],
                    mode="markers+text",
                    text=[class_name],
                    textposition="top center",
                    marker={
                        "size": 16,
                        "color": get_class_color(class_name),
                        "line": {"color": "#1E293B", "width": 1},
                    },
                    name=class_name,
                    hovertemplate="<b>%{text}</b><br>Support: %{x}<br>F1-score: %{y:.3f}<extra></extra>",
                )
            )

        figure.update_layout(
            showlegend=False,
            **get_plotly_layout(
                f"Support vs F1-score — {selected_model}",
                height=380,
                margin={"l": 35, "r": 20, "t": 55, "b": 40},
            ),
        )
        figure.update_xaxes(title_text="Support")
        figure.update_yaxes(title_text="F1-score", range=[0, 1])
        st.plotly_chart(figure, use_container_width=True)
        return

    st.dataframe(
        selected_model_metrics_df[["Class", "Support", "F1-score"]],
        use_container_width=True,
        hide_index=True,
    )


def render_class_difficulty_chart(average_class_f1_df: pd.DataFrame) -> None:
    if go is not None:
        difficulty_colors = [get_class_color(class_name) for class_name in average_class_f1_df["Class"]]
        figure = go.Figure(
            data=[
                go.Bar(
                    x=average_class_f1_df["Class"],
                    y=average_class_f1_df["Average F1"],
                    marker_color=difficulty_colors,
                    text=[f"{value:.3f}" for value in average_class_f1_df["Average F1"]],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>Average F1: %{y:.3f}<extra></extra>",
                )
            ]
        )
        figure.update_layout(
            showlegend=False,
            **get_plotly_layout(
                "Class Difficulty Across All Models",
                height=340,
                margin={"l": 35, "r": 20, "t": 55, "b": 40},
            ),
        )
        figure.update_xaxes(title_text="")
        figure.update_yaxes(title_text="Average F1", range=[0, 1])
        st.plotly_chart(figure, use_container_width=True)
        return

    fallback_df = average_class_f1_df.copy()
    fallback_df["Average F1"] = fallback_df["Average F1"].map(lambda value: f"{value:.3f}")
    st.dataframe(fallback_df, use_container_width=True, hide_index=True)


def get_numeric_feature_options(metadata: dict, dataset_df: pd.DataFrame) -> list[str]:
    target_column = metadata.get("target_column")
    drop_columns = set(metadata.get("drop_columns", []))
    recommended_order = [
        "koi_model_snr",
        "koi_depth",
        "koi_prad",
        "koi_period",
        "koi_duration",
        "koi_teq",
        "koi_insol",
        "koi_srad",
    ]

    numeric_columns = dataset_df.select_dtypes(include=[np.number]).columns.tolist()
    filtered_numeric_columns = [
        column
        for column in numeric_columns
        if column != target_column and column not in drop_columns
    ]

    ordered_features = [feature for feature in recommended_order if feature in filtered_numeric_columns]
    remaining_features = [
        feature for feature in filtered_numeric_columns if feature not in ordered_features
    ]
    return ordered_features + remaining_features


def is_highly_skewed_feature(feature_name: str) -> bool:
    return feature_name in {
        "koi_model_snr",
        "koi_depth",
        "koi_prad",
        "koi_insol",
        "koi_srad",
        "koi_snr_depth_ratio",
        "koi_prad_srad_ratio",
        "koi_insol_teq_ratio",
    }


def apply_visual_log_transform(values: pd.Series) -> pd.Series:
    return np.log1p(np.clip(values.astype(float), a_min=0, a_max=None))


def trim_plot_outliers(plot_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    trimmed_df = plot_df.copy()
    for column in feature_columns:
        q01 = float(trimmed_df[column].quantile(0.01))
        q99 = float(trimmed_df[column].quantile(0.99))
        trimmed_df = trimmed_df[
            (trimmed_df[column] >= q01)
            & (trimmed_df[column] <= q99)
        ]
    return trimmed_df


def build_feature_summary_table(
    feature_df: pd.DataFrame,
    selected_feature: str,
    class_names: list[str],
) -> pd.DataFrame:
    summary_df = (
        feature_df.groupby("Class")[selected_feature]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "Count",
                "mean": "Mean",
                "median": "Median",
                "std": "Std",
                "min": "Min",
                "max": "Max",
            }
        )
    )

    summary_df["Class"] = pd.Categorical(summary_df["Class"], categories=class_names, ordered=True)
    summary_df = summary_df.sort_values("Class").reset_index(drop=True)
    return summary_df


def generate_feature_overlap_interpretation(
    feature_df: pd.DataFrame,
    selected_feature: str,
    class_names: list[str],
) -> str:
    stats: dict[str, dict[str, float]] = {}
    for class_name in class_names:
        class_values = feature_df[feature_df["Class"] == class_name][selected_feature].dropna()
        if class_values.empty:
            continue
        stats[class_name] = {
            "median": float(class_values.median()),
            "q1": float(class_values.quantile(0.25)),
            "q3": float(class_values.quantile(0.75)),
        }

    candidate_stats = stats.get("CANDIDATE")
    confirmed_stats = stats.get("CONFIRMED")
    false_positive_stats = stats.get("FALSE POSITIVE")
    if not candidate_stats or not confirmed_stats or not false_positive_stats:
        return "Not enough class-specific data is available to generate an overlap interpretation for this feature."

    messages: list[str] = []
    candidate_median = candidate_stats["median"]
    confirmed_median = confirmed_stats["median"]
    false_positive_median = false_positive_stats["median"]

    if min(confirmed_median, false_positive_median) <= candidate_median <= max(confirmed_median, false_positive_median):
        messages.append(
            "CANDIDATE lies between CONFIRMED and FALSE POSITIVE for this feature, which may contribute to ambiguity."
        )

    def iqr_overlaps(left: dict[str, float], right: dict[str, float]) -> bool:
        return max(left["q1"], right["q1"]) <= min(left["q3"], right["q3"])

    candidate_confirmed_overlap = iqr_overlaps(candidate_stats, confirmed_stats)
    candidate_false_positive_overlap = iqr_overlaps(candidate_stats, false_positive_stats)

    if candidate_confirmed_overlap:
        messages.append("CANDIDATE overlaps with CONFIRMED in the middle 50% range.")
    if candidate_false_positive_overlap:
        messages.append("CANDIDATE overlaps with FALSE POSITIVE in the middle 50% range.")
    if not candidate_confirmed_overlap and not candidate_false_positive_overlap:
        messages.append(
            "This feature shows clearer separation between classes, but one feature alone is not enough to explain the full model behavior."
        )

    return " ".join(messages)


@st.cache_data(show_spinner=False)
def build_pca_dataframe(
    _metadata: dict,
    selected_feature: str,
    sample_size: int,
) -> pd.DataFrame:
    dataset_df = load_feature_dataset()
    if dataset_df is None:
        return pd.DataFrame()

    target_column = _metadata.get("target_column")
    class_names = _metadata.get("class_names", [])
    random_state = int(_metadata.get("random_state", 42))
    drop_columns = set(_metadata.get("drop_columns", []))
    if not target_column or target_column not in dataset_df.columns:
        return pd.DataFrame()

    sampled_df = dataset_df[dataset_df[target_column].isin(class_names)].copy()
    if sampled_df.empty:
        return pd.DataFrame()

    if len(sampled_df) > sample_size:
        sampled_df = sampled_df.sample(n=sample_size, random_state=random_state)

    class_series = sampled_df[target_column].copy()
    hover_series = sampled_df[selected_feature].copy() if selected_feature in sampled_df.columns else None

    feature_df = sampled_df.drop(columns=[column for column in drop_columns if column in sampled_df.columns], errors="ignore")
    if target_column in feature_df.columns:
        feature_df = feature_df.drop(columns=[target_column], errors="ignore")
    numeric_feature_df = feature_df.select_dtypes(include=[np.number]).copy()
    if numeric_feature_df.empty:
        return pd.DataFrame()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed_array = imputer.fit_transform(numeric_feature_df)
    scaled_array = scaler.fit_transform(imputed_array)
    pca = PCA(n_components=2, random_state=random_state)
    components = pca.fit_transform(scaled_array)

    pca_df = pd.DataFrame(
        {
            "PCA1": components[:, 0],
            "PCA2": components[:, 1],
            "Class": class_series.values,
        }
    )
    if hover_series is not None:
        pca_df[selected_feature] = hover_series.values
    return pca_df


def render_bivariate_scatter(
    plot_df: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    class_names: list[str],
    use_log_x: bool,
    use_log_y: bool,
) -> None:
    if plot_df.empty:
        st.info("No data is available for the selected feature pair.")
        return

    x_axis_label = f"log1p({x_feature})" if use_log_x else x_feature
    y_axis_label = f"log1p({y_feature})" if use_log_y else y_feature

    if go is not None:
        figure = go.Figure()
        for class_name in class_names:
            class_df = plot_df[plot_df["Class"] == class_name]
            if class_df.empty:
                continue

            hover_columns = ["Class", x_feature, y_feature]
            for optional_column in ["kepid", "kepoi_name"]:
                if optional_column in class_df.columns:
                    hover_columns.append(optional_column)
            customdata = class_df[hover_columns].values

            hovertemplate = (
                "<b>%{customdata[0]}</b><br>"
                + f"{x_feature}: "
                + "%{customdata[1]:.3f}<br>"
                + f"{y_feature}: "
                + "%{customdata[2]:.3f}"
            )
            custom_index = 3
            if "kepid" in hover_columns:
                hovertemplate += "<br>kepid: %{customdata[" + str(custom_index) + "]}"
                custom_index += 1
            if "kepoi_name" in hover_columns:
                hovertemplate += "<br>kepoi_name: %{customdata[" + str(custom_index) + "]}"
            hovertemplate += "<extra></extra>"

            figure.add_trace(
                go.Scatter(
                    x=class_df["plot_x"],
                    y=class_df["plot_y"],
                    mode="markers",
                    name=class_name,
                    customdata=customdata,
                    marker={
                        "color": get_class_color(class_name),
                        "size": 4,
                        "opacity": 0.55,
                    },
                    hovertemplate=hovertemplate,
                )
            )

            centroid_x = float(class_df["plot_x"].median())
            centroid_y = float(class_df["plot_y"].median())
            figure.add_trace(
                go.Scatter(
                    x=[centroid_x],
                    y=[centroid_y],
                    mode="markers+text",
                    text=[class_name],
                    textposition="top center",
                    marker={
                        "color": get_class_color(class_name),
                        "size": 14,
                        "opacity": 1.0,
                        "line": {"color": "#000000", "width": 2},
                        "symbol": "diamond",
                    },
                    name=f"{class_name} centroid",
                    showlegend=False,
                    hovertemplate="<b>%{text}</b><br>X median: %{x:.3f}<br>Y median: %{y:.3f}<extra></extra>",
                )
            )

        figure.update_layout(
            **get_plotly_layout(
                f"{x_feature} vs {y_feature} by Class",
                height=460,
                margin={"l": 50, "r": 40, "t": 65, "b": 55},
                legend={
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.14,
                    "xanchor": "center",
                    "x": 0.5,
                },
            )
        )
        figure.update_xaxes(title_text=x_axis_label)
        figure.update_yaxes(title_text=y_axis_label)
        st.plotly_chart(figure, use_container_width=True)
        return

    st.dataframe(plot_df.head(500), use_container_width=True, hide_index=True)


def build_correlation_feature_set(
    metadata: dict,
    dataset_df: pd.DataFrame,
    mode: str,
    selected_feature: str,
) -> list[str]:
    numeric_features = get_numeric_feature_options(metadata, dataset_df)
    if mode == "Top important features":
        feature_importance_table = metadata.get("feature_importance_table", [])
        if feature_importance_table:
            importance_df = pd.DataFrame(feature_importance_table)
            if not importance_df.empty and {"Feature", "Importance"}.issubset(importance_df.columns):
                top_features = [
                    feature
                    for feature in importance_df.sort_values("Importance", ascending=False)["Feature"].tolist()
                    if feature in numeric_features
                ]
                return top_features[:12]

        fallback_order = [
            "koi_model_snr",
            "koi_prad",
            "koi_depth",
            "koi_period",
            "koi_duration",
            "koi_teq",
            "koi_insol",
            "koi_srad",
        ]
        ordered = [feature for feature in fallback_order if feature in numeric_features]
        remainder = [feature for feature in numeric_features if feature not in ordered]
        return (ordered + remainder)[:12]

    available_numeric = [feature for feature in numeric_features if feature in dataset_df.columns]
    if selected_feature not in available_numeric:
        return available_numeric[:12]

    corr_df = dataset_df[available_numeric].copy()
    imputer = SimpleImputer(strategy="median")
    imputed_df = pd.DataFrame(imputer.fit_transform(corr_df), columns=available_numeric)
    correlation_series = imputed_df.corr()[selected_feature].abs().sort_values(ascending=False)
    return correlation_series.index.tolist()[:12]


def render_correlation_heatmap(correlation_df: pd.DataFrame) -> None:
    if correlation_df.empty:
        st.info("Correlation heatmap could not be generated from the available features.")
        return

    if go is not None:
        annotations = []
        for row_label in correlation_df.index:
            for col_label in correlation_df.columns:
                annotations.append(
                    {
                        "x": col_label,
                        "y": row_label,
                        "text": f"{correlation_df.loc[row_label, col_label]:.2f}",
                        "showarrow": False,
                        "font": {"color": TEXT, "size": 11},
                    }
                )

        figure = go.Figure(
            data=go.Heatmap(
                z=correlation_df.values,
                x=list(correlation_df.columns),
                y=list(correlation_df.index),
                zmin=-1,
                zmax=1,
                zmid=0,
                colorscale="RdBu",
                colorbar={"title": "r"},
                hovertemplate="Feature A: %{y}<br>Feature B: %{x}<br>r: %{z:.2f}<extra></extra>",
            )
        )
        figure.update_layout(
            **get_plotly_layout(
                "Feature Correlation Heatmap",
                height=520,
                margin={"l": 50, "r": 40, "t": 65, "b": 55},
            ),
            annotations=annotations,
        )
        figure.update_xaxes(title_text="")
        figure.update_yaxes(title_text="", autorange="reversed")
        st.plotly_chart(figure, use_container_width=True)
        return

    st.dataframe(correlation_df.style.format("{:.2f}"), use_container_width=True)


def generate_correlation_interpretation(correlation_df: pd.DataFrame) -> str:
    if correlation_df.empty or len(correlation_df.columns) < 2:
        return "Not enough features are available to interpret pairwise correlations."

    best_pair: tuple[str, str] | None = None
    best_value = -1.0
    columns = list(correlation_df.columns)
    for i, left_feature in enumerate(columns):
        for right_feature in columns[i + 1 :]:
            corr_value = float(correlation_df.loc[left_feature, right_feature])
            if abs(corr_value) > best_value:
                best_value = abs(corr_value)
                best_pair = (left_feature, right_feature, corr_value)

    if best_pair is None:
        return "Not enough features are available to interpret pairwise correlations."

    left_feature, right_feature, corr_value = best_pair
    message = f"Strongest relationship: {left_feature} ↔ {right_feature} (r = {corr_value:.2f})."
    if abs(corr_value) > 0.85:
        message += " These features may carry redundant information."
    else:
        message += " No extremely high correlation was detected among the displayed features."
    return message


def render_feature_distribution_chart(
    feature_df: pd.DataFrame,
    selected_feature: str,
    plot_type: str,
    class_names: list[str],
    use_log_scale: bool,
    trim_outliers: bool,
) -> None:
    if feature_df.empty:
        st.info("No feature data is available for this chart.")
        return

    plot_df = feature_df.copy()
    trimmed_suffix = "full range"
    if trim_outliers:
        q01 = float(plot_df[selected_feature].quantile(0.01))
        q99 = float(plot_df[selected_feature].quantile(0.99))
        plot_df = plot_df[
            (plot_df[selected_feature] >= q01)
            & (plot_df[selected_feature] <= q99)
        ].copy()
        trimmed_suffix = "1–99% trimmed"

    y_axis_label = selected_feature
    scale_suffix = "raw scale"
    if use_log_scale:
        plot_df[selected_feature] = np.log1p(np.clip(plot_df[selected_feature], a_min=0, a_max=None))
        y_axis_label = f"log1p({selected_feature})"
        scale_suffix = "log1p"

    if go is not None:
        figure = go.Figure()
        for class_name in class_names:
            class_data = plot_df[plot_df["Class"] == class_name][selected_feature].dropna()
            if class_data.empty:
                continue

            if plot_type == "Violin Plot":
                figure.add_trace(
                    go.Violin(
                        x=[class_name] * len(class_data),
                        y=class_data,
                        name=class_name,
                        line={"color": get_class_color(class_name)},
                        fillcolor=get_class_color(class_name),
                        opacity=0.55,
                        box_visible=True,
                        meanline_visible=True,
                        points=False,
                    )
                )
            else:
                figure.add_trace(
                    go.Box(
                        x=[class_name] * len(class_data),
                        y=class_data,
                        name=class_name,
                        marker_color=get_class_color(class_name),
                        boxmean=True,
                    )
                )

        figure.update_layout(
            **get_plotly_layout(
                f"{selected_feature} Distribution by Class — {scale_suffix}, {trimmed_suffix}",
                height=420,
                margin={"l": 40, "r": 30, "t": 65, "b": 45},
            )
        )
        figure.update_layout(showlegend=False)
        figure.update_xaxes(
            title_text="Class",
            categoryorder="array",
            categoryarray=class_names,
        )
        figure.update_yaxes(title_text=y_axis_label)
        st.plotly_chart(figure, use_container_width=True)
        return

    summary_df = (
        plot_df.groupby("Class")[selected_feature]
        .describe()[["count", "mean", "std", "min", "50%", "max"]]
        .reset_index()
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def render_class_median_chart(
    summary_df: pd.DataFrame,
    selected_feature: str,
    class_names: list[str],
    use_log_scale: bool,
) -> None:
    if summary_df.empty:
        st.info("No summary data is available for the class median chart.")
        return

    chart_df = summary_df.copy()
    y_col = "Median"
    title = f"Class Median — {selected_feature}"
    y_axis_title = "Median"
    if use_log_scale:
        chart_df["Median Plot"] = np.log1p(np.clip(chart_df["Median"], a_min=0, a_max=None))
        y_col = "Median Plot"
        title = f"Class Median — log1p({selected_feature})"
        y_axis_title = f"log1p median of {selected_feature}"

    if go is not None:
        figure = go.Figure(
            data=[
                go.Bar(
                    x=chart_df["Class"],
                    y=chart_df[y_col],
                    marker_color=[get_class_color(class_name) for class_name in chart_df["Class"]],
                    text=[f"{value:.3f}" for value in chart_df[y_col]],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>Median: %{y:.3f}<extra></extra>",
                )
            ]
        )
        figure.update_layout(
            showlegend=False,
            **get_plotly_layout(
                title,
                height=420,
                margin={"l": 40, "r": 30, "t": 65, "b": 45},
            ),
        )
        figure.update_xaxes(title_text="", categoryorder="array", categoryarray=class_names)
        figure.update_yaxes(title_text=y_axis_title)
        st.plotly_chart(figure, use_container_width=True)
        return

    fallback_df = chart_df[["Class", y_col]].copy()
    st.dataframe(fallback_df, use_container_width=True, hide_index=True)


def render_pca_scatter(
    pca_df: pd.DataFrame,
    selected_feature: str,
    class_names: list[str],
    point_opacity: float,
    point_size: int,
    focus_central_view: bool,
) -> None:
    if pca_df.empty:
        st.info("PCA projection could not be generated from the available numeric features.")
        return

    if go is not None:
        figure = go.Figure()
        for class_name in class_names:
            class_df = pca_df[pca_df["Class"] == class_name]
            if class_df.empty:
                continue

            customdata = class_df[[selected_feature]].values if selected_feature in class_df.columns else None
            hovertemplate = "<b>%{text}</b><br>PCA1: %{x:.3f}<br>PCA2: %{y:.3f}"
            if customdata is not None:
                hovertemplate += f"<br>{selected_feature}: %{{customdata[0]:.3f}}"
            hovertemplate += "<extra></extra>"

            figure.add_trace(
                go.Scatter(
                    x=class_df["PCA1"],
                    y=class_df["PCA2"],
                    mode="markers",
                    name=class_name,
                    text=class_df["Class"],
                    customdata=customdata,
                    marker={
                        "color": get_class_color(class_name),
                        "size": point_size,
                        "opacity": point_opacity,
                    },
                    hovertemplate=hovertemplate,
                )
            )

            centroid_x = float(class_df["PCA1"].median())
            centroid_y = float(class_df["PCA2"].median())
            figure.add_trace(
                go.Scatter(
                    x=[centroid_x],
                    y=[centroid_y],
                    mode="markers+text",
                    text=[class_name],
                    textposition="top center",
                    marker={
                        "color": get_class_color(class_name),
                        "size": point_size + 8,
                        "opacity": 1.0,
                        "line": {"color": "#000000", "width": 2},
                        "symbol": "diamond",
                    },
                    name=f"{class_name} centroid",
                    hovertemplate="<b>%{text}</b><br>PCA1 median: %{x:.3f}<br>PCA2 median: %{y:.3f}<extra></extra>",
                    showlegend=False,
                )
            )

        figure.update_layout(
            **get_plotly_layout(
                "PCA Projection of Numeric Feature Space",
                height=420,
                margin={"l": 40, "r": 30, "t": 65, "b": 45},
                legend={
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.14,
                    "xanchor": "center",
                    "x": 0.5,
                },
            )
        )
        x_range = None
        y_range = None
        if focus_central_view:
            x_range = [
                float(pca_df["PCA1"].quantile(0.01)),
                float(pca_df["PCA1"].quantile(0.99)),
            ]
            y_range = [
                float(pca_df["PCA2"].quantile(0.01)),
                float(pca_df["PCA2"].quantile(0.99)),
            ]
        figure.update_xaxes(title_text="PCA1", range=x_range)
        figure.update_yaxes(title_text="PCA2", range=y_range)
        st.plotly_chart(figure, use_container_width=True)
        return

    display_df = pca_df.copy()
    st.dataframe(display_df.head(500), use_container_width=True, hide_index=True)


def render_feature_importance_section(metadata: dict) -> None:
    feature_importance_table = metadata.get("feature_importance_table", [])
    final_model_name = metadata.get("final_model_name", "")
    if not feature_importance_table:
        st.info("Feature importance is available only when the final model exposes feature_importances_.")
        return

    importance_df = pd.DataFrame(feature_importance_table)
    if importance_df.empty or not {"Feature", "Importance"}.issubset(importance_df.columns):
        st.info("Feature importance is available only when the final model exposes feature_importances_.")
        return

    top_importance_df = (
        importance_df.sort_values("Importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    chart_importance_df = (
        top_importance_df.sort_values("Importance", ascending=True)
        .sort_values("Importance", ascending=True)
        .reset_index(drop=True)
    )

    st.caption(
        "This plot shows feature importances from the final selected tree-based model. Not all algorithms expose directly comparable feature_importances_ values."
    )

    if go is not None:
        figure = go.Figure(
            data=[
                go.Bar(
                    x=chart_importance_df["Importance"],
                    y=chart_importance_df["Feature"],
                    orientation="h",
                    marker_color=NASA_BLUE,
                    text=[f"{value:.3f}" for value in chart_importance_df["Importance"]],
                    textposition="auto",
                    hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>",
                )
            ]
        )
        figure.update_layout(
            showlegend=False,
            **get_plotly_layout(
                f"Top Feature Importances — {final_model_name}" if final_model_name else "Top Feature Importances",
                height=420,
                margin={"l": 40, "r": 30, "t": 60, "b": 45},
            )
        )
        figure.update_xaxes(title_text="Importance")
        figure.update_yaxes(title_text="")
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.dataframe(top_importance_df, use_container_width=True, hide_index=True)

    display_df = top_importance_df.copy()
    display_df["Importance"] = display_df["Importance"].map(lambda value: f"{value:.3f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def build_feature_dictionary_df() -> pd.DataFrame:
    rows = [
        {
            "Display Feature": "koi_period",
            "Original Column": "koi_period",
            "Meaning": "Orbital period of the planet candidate, usually measured in days.",
            "Category": "Orbital property",
            "Why it matters": "Different planet and false-positive signals can show different periodic behavior.",
        },
        {
            "Display Feature": "koi_duration",
            "Original Column": "koi_duration",
            "Meaning": "Duration of the observed transit event.",
            "Category": "Transit property",
            "Why it matters": "Transit duration helps describe the shape and geometry of the signal.",
        },
        {
            "Display Feature": "koi_depth",
            "Original Column": "koi_depth",
            "Meaning": "Transit depth, representing how much the star brightness drops during transit.",
            "Category": "Transit property",
            "Why it matters": "Deeper transits often suggest larger objects or possible false positives.",
        },
        {
            "Display Feature": "koi_prad",
            "Original Column": "koi_prad",
            "Meaning": "Estimated planet radius.",
            "Category": "Planet property",
            "Why it matters": "Planet size is important for separating realistic planet candidates from non-planet signals.",
        },
        {
            "Display Feature": "koi_teq",
            "Original Column": "koi_teq",
            "Meaning": "Estimated equilibrium temperature of the planet candidate.",
            "Category": "Planet/environment property",
            "Why it matters": "Temperature provides context about the candidate’s physical environment.",
        },
        {
            "Display Feature": "koi_insol",
            "Original Column": "koi_insol",
            "Meaning": "Estimated stellar insolation received by the candidate relative to Earth.",
            "Category": "Planet/environment property",
            "Why it matters": "Insolation helps describe how much energy the object receives from its host star.",
        },
        {
            "Display Feature": "koi_model_snr",
            "Original Column": "koi_model_snr",
            "Meaning": "Signal-to-noise ratio of the transit model.",
            "Category": "Signal quality",
            "Why it matters": "Higher SNR generally means the transit signal is easier to detect and classify.",
        },
        {
            "Display Feature": "koi_srad",
            "Original Column": "koi_srad",
            "Meaning": "Estimated stellar radius of the host star.",
            "Category": "Stellar property",
            "Why it matters": "The host star size affects inferred planet size and transit interpretation.",
        },
        {
            "Display Feature": "koi_steff",
            "Original Column": "koi_steff",
            "Meaning": "Effective temperature of the host star.",
            "Category": "Stellar property",
            "Why it matters": "Stellar temperature helps characterize the host system and candidate environment.",
        },
        {
            "Display Feature": "koi_slogg",
            "Original Column": "koi_slogg",
            "Meaning": "Surface gravity of the host star.",
            "Category": "Stellar property",
            "Why it matters": "Surface gravity helps describe the type and evolutionary state of the host star.",
        },
        {
            "Display Feature": "koi_smet",
            "Original Column": "koi_smet",
            "Meaning": "Metallicity of the host star.",
            "Category": "Stellar property",
            "Why it matters": "Stellar composition can provide additional context about the planetary system.",
        },
        {
            "Display Feature": "koi_impact",
            "Original Column": "koi_impact",
            "Meaning": "Impact parameter of the transit, describing how centrally the object crosses the star.",
            "Category": "Transit geometry",
            "Why it matters": "Transit geometry affects the observed signal shape and classification.",
        },
        {
            "Display Feature": "koi_time0bk",
            "Original Column": "koi_time0bk",
            "Meaning": "Transit epoch or reference time of the first observed transit.",
            "Category": "Orbital/timing property",
            "Why it matters": "Timing information helps define the periodic transit pattern.",
        },
        {
            "Display Feature": "koi_tce_plnt_num",
            "Original Column": "koi_tce_plnt_num",
            "Meaning": "TCE planet number associated with the target star.",
            "Category": "Catalog/identifier feature",
            "Why it matters": "Indicates whether multiple threshold crossing events are associated with the same system.",
        },
        {
            "Display Feature": "koi_prad_srad_ratio (F.E)",
            "Original Column": "koi_prad_srad_ratio",
            "Meaning": "Engineered ratio between estimated planet radius and stellar radius. Formula: koi_prad / (koi_srad + 1e-6).",
            "Category": "Engineered ratio feature",
            "Why it matters": "Captures relative planet-to-star size information useful for transit interpretation.",
        },
        {
            "Display Feature": "koi_snr_depth_ratio (F.E)",
            "Original Column": "koi_snr_depth_ratio",
            "Meaning": "Engineered ratio between transit model signal-to-noise ratio and transit depth. Formula: koi_model_snr / (koi_depth + 1e-6).",
            "Category": "Engineered ratio feature",
            "Why it matters": "Combines signal strength and transit depth into one relative signal-quality feature.",
        },
        {
            "Display Feature": "koi_insol_teq_ratio (F.E)",
            "Original Column": "koi_insol_teq_ratio",
            "Meaning": "Engineered ratio between stellar insolation and equilibrium temperature. Formula: koi_insol / (koi_teq + 1e-6).",
            "Category": "Engineered ratio feature",
            "Why it matters": "Combines two environment-related measurements to provide additional context about the candidate’s physical environment.",
        },
    ]
    return pd.DataFrame(rows)


def render_model_comparison(metadata: dict) -> None:
    st.title("Model Comparison")
    st.markdown(
        """
        <section class="app-card">
            <p class="overview-subtitle">Dynamic model comparison dashboard</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    tuned_results_df = pd.DataFrame(metadata.get("tuned_results_table", []))
    if tuned_results_df.empty:
        st.warning("No tuned results are available in metadata.")
        return

    available_models = tuned_results_df["Model"].tolist()
    selected_models = st.multiselect(
        "Select models to compare",
        options=available_models,
        default=available_models,
    )

    if not selected_models:
        st.warning("Please select at least one model.")
        return

    selected_results_df = get_selected_results_df(metadata, selected_models)
    reports = metadata.get("classification_reports", {})
    class_names = metadata.get("class_names", [])

    best_overall_model = (
        selected_results_df.sort_values("Macro F1", ascending=False).iloc[0]["Model"]
        if not selected_results_df.empty
        else "N/A"
    )

    best_class_models = []
    for class_name in class_names:
        best_model = max(
            selected_models,
            key=lambda model: float(reports.get(model, {}).get(class_name, {}).get("f1-score", 0.0)),
        )
        best_class_models.append((f"Best {class_name} F1", best_model))

    average_f1_by_class = {}
    for class_name in class_names:
        class_scores = [
            float(reports.get(model, {}).get(class_name, {}).get("f1-score", 0.0))
            for model in selected_models
        ]
        average_f1_by_class[class_name] = float(np.mean(class_scores)) if class_scores else 0.0

    hardest_class = min(average_f1_by_class, key=average_f1_by_class.get) if average_f1_by_class else "N/A"

    render_metric_cards(
        [("Best Overall Model", best_overall_model)]
        + best_class_models
        + [("Hardest Class", hardest_class)]
    )

    with st.expander("Overall Performance", expanded=True):
        macro_f1_rank_df = selected_results_df.sort_values("Macro F1", ascending=False).reset_index(drop=True)
        overall_row_one = st.columns(2)
        with overall_row_one[0]:
            render_single_metric_bar_chart(
                macro_f1_rank_df,
                x_col="Model",
                y_col="Macro F1",
                title="Model Rank / Macro F1",
                color=NASA_BLUE,
                y_range=[0, 1],
                highlight_model_name=metadata.get("final_model_name", ""),
                secondary_color="#60A5FA",
            )
        with overall_row_one[1]:
            render_single_metric_bar_chart(
                selected_results_df,
                x_col="Model",
                y_col="Accuracy",
                title="Accuracy by Model",
                color=NASA_BLUE,
                y_range=[0, 1],
                secondary_color="#93C5FD",
            )

        overall_row_two = st.columns(2)
        with overall_row_two[0]:
            render_precision_recall_scatter(selected_results_df, metadata.get("final_model_name", ""))
        with overall_row_two[1]:
            render_grouped_bar_chart(
                selected_results_df,
                x_col="Model",
                y_cols=["Macro Precision", "Macro Recall", "Macro F1"],
                title="Macro Precision / Recall / F1",
                color_map={
                    "Macro Precision": "#1D4ED8",
                    "Macro Recall": "#0891B2",
                    "Macro F1": "#7C3AED",
                },
                y_axis_title="Score",
                height=360,
                layout_margin={"l": 35, "r": 20, "t": 85, "b": 40},
                legend_layout={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "center",
                    "x": 0.5,
                },
            )

        metrics_columns = [
            "Model",
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Weighted F1",
        ]
        display_df = selected_results_df[metrics_columns].copy()
        for column in metrics_columns[1:]:
            display_df[column] = display_df[column].map(lambda value: f"{float(value):.3f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with st.expander("Class-wise Performance Overview"):
        class_f1_df = build_class_f1_df(metadata, selected_models)
        classwise_columns = st.columns(2)
        with classwise_columns[0]:
            render_per_class_f1_heatmap(class_f1_df)
        with classwise_columns[1]:
            render_grouped_bar_chart(
                class_f1_df,
                x_col="Model",
                y_cols=["CANDIDATE F1", "CONFIRMED F1", "FALSE POSITIVE F1"],
                title="Class-wise F1 Across Selected Models",
                color_map={
                    "CANDIDATE F1": CANDIDATE,
                    "CONFIRMED F1": CONFIRMED,
                    "FALSE POSITIVE F1": FALSE_POSITIVE,
                },
                y_axis_title="F1 Score",
                height=360,
                layout_margin={"l": 35, "r": 20, "t": 85, "b": 40},
                legend_layout={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "center",
                    "x": 0.5,
                },
            )
        display_df = class_f1_df.copy()
        for column in ["CANDIDATE F1", "CONFIRMED F1", "FALSE POSITIVE F1"]:
            display_df[column] = display_df[column].map(lambda value: f"{float(value):.3f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(
            "CANDIDATE is usually the hardest class because it can overlap with both confirmed planets and false positives."
        )

    with st.expander("Selected Class Detail"):
        selected_class = st.selectbox("Select class", options=class_names)
        class_metrics_df = build_selected_class_metrics_df(metadata, selected_models, selected_class)
        selected_class_columns = st.columns(2)
        with selected_class_columns[0]:
            render_grouped_bar_chart(
                class_metrics_df,
                x_col="Model",
                y_cols=["Precision", "Recall", "F1-score"],
                title=f"{selected_class} Precision / Recall / F1-score",
                color_map={
                    "Precision": "#1D4ED8",
                    "Recall": "#0891B2",
                    "F1-score": "#7C3AED",
                },
                y_axis_title="Score",
                height=360,
                layout_margin={"l": 35, "r": 20, "t": 85, "b": 40},
                legend_layout={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "center",
                    "x": 0.5,
                },
            )
        display_df = class_metrics_df.copy()
        for column in ["Precision", "Recall", "F1-score"]:
            display_df[column] = display_df[column].map(lambda value: f"{float(value):.3f}")
        display_df["Support"] = display_df["Support"].map(lambda value: f"{int(value)}")
        with selected_class_columns[1]:
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        class_explanations = {
            "CANDIDATE": "This class represents planet candidates that are not yet confirmed and can overlap with both confirmed planets and false positives.",
            "CONFIRMED": "This class represents validated exoplanets.",
            "FALSE POSITIVE": "This class represents signals that are likely not real exoplanets.",
        }
        st.caption(class_explanations.get(selected_class, ""))

    with st.expander("Confusion Matrix Selector"):
        confusion_matrices = metadata.get("confusion_matrices", {})
        available_confusion_models = [
            model_name for model_name in selected_models if model_name in confusion_matrices
        ]
        if not available_confusion_models:
            st.info("No confusion matrices are available for the selected models.")
        else:
            selected_confusion_model = st.selectbox(
                "Select model for confusion matrix",
                options=available_confusion_models,
            )
            selected_matrix = confusion_matrices.get(selected_confusion_model, [])
            confusion_columns = st.columns([1.2, 1])
            with confusion_columns[0]:
                render_confusion_matrix(selected_matrix, class_names)

            matrix_array = np.array(selected_matrix)
            matrix_df = pd.DataFrame(selected_matrix, index=class_names, columns=class_names)
            off_diagonal = matrix_array.copy()
            np.fill_diagonal(off_diagonal, 0)

            with confusion_columns[1]:
                st.dataframe(matrix_df, use_container_width=True)
                if off_diagonal.size == 0 or int(off_diagonal.max()) == 0:
                    st.caption("No off-diagonal mistakes found.")
                else:
                    row_idx, col_idx = np.unravel_index(np.argmax(off_diagonal), off_diagonal.shape)
                    st.caption(
                        f"Most confused pair: {class_names[row_idx]} → {class_names[col_idx]} ({int(off_diagonal[row_idx, col_idx])})"
                    )

    with st.expander("Baseline vs Tuned"):
        baseline_results_df = pd.DataFrame(metadata.get("baseline_results_table", []))
        comparison_df = pd.DataFrame(metadata.get("comparison_table", []))

        # Normalize model-name column for compatibility with notebook outputs
        if "Model" not in comparison_df.columns and "Base Model" in comparison_df.columns:
            comparison_df = comparison_df.rename(columns={"Base Model": "Model"})

        column_aliases = {
            "Macro F1 Baseline": "Baseline F1",
            "Macro F1 Tuned": "Tuned F1",
            "Macro F1 Improvement": "Improvement",
            "Accuracy Baseline": "Baseline Accuracy",
            "Accuracy Tuned": "Tuned Accuracy",
            "Weighted F1 Baseline": "Baseline Weighted F1",
            "Weighted F1 Tuned": "Tuned Weighted F1",
        }

        comparison_df = comparison_df.rename(
            columns={old: new for old, new in column_aliases.items() if old in comparison_df.columns}
        )

        if "Model" not in comparison_df.columns:
            st.error("Model comparison data is missing the model name column.")
            st.write("Available columns:", list(comparison_df.columns))
            st.stop()

        available_baseline_names = (
            {normalize_base_model_name(model_name) for model_name in baseline_results_df["Model"].tolist()}
            if not baseline_results_df.empty
            else set()
        )
        selected_base_names = {
            normalize_base_model_name(model_name)
            for model_name in selected_models
            if normalize_base_model_name(model_name) in available_baseline_names
        }
        filtered_comparison_df = comparison_df[comparison_df["Model"].isin(selected_base_names)].reset_index(drop=True)

        if filtered_comparison_df.empty:
            st.info("No baseline comparison rows are available for the selected models.")
        else:
            display_df = filtered_comparison_df.copy()
            for column in ["Baseline F1", "Tuned F1", "Improvement"]:
                display_df[column] = display_df[column].map(lambda value: f"{float(value):.3f}")
            min_val = float(filtered_comparison_df["Improvement"].min())
            max_val = float(filtered_comparison_df["Improvement"].max())
            padding = max(0.005, (max_val - min_val) * 0.15)
            y_min = min(0.0, min_val - padding)
            y_max = max(0.0, max_val + padding)
            baseline_columns = st.columns(2)
            with baseline_columns[0]:
                render_single_metric_bar_chart(
                    filtered_comparison_df,
                    x_col="Model",
                    y_col="Improvement",
                    title="Macro F1 Improvement After Tuning",
                    color=NASA_BLUE,
                    y_range=[y_min, y_max],
                    secondary_color="#93C5FD",
                    textposition="auto",
                )
            with baseline_columns[1]:
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.caption(
                "Hyperparameter tuning optimizes cross-validation score, so test-set improvement is not guaranteed for every model."
            )

    with st.expander("Ensemble Summary"):
        ensemble_columns = st.columns(3)
        ensemble_items = [
            ("Random Forest", "Bagging / Random Subspace"),
            ("XGBoost", "Boosting"),
            (
                "Stacking",
                "Meta-ensemble using XGBoost + Random Forest + SVM with Logistic Regression as final estimator.",
            ),
        ]
        for column, (title, description) in zip(ensemble_columns, ensemble_items):
            with column:
                st.markdown(
                    f"""
                    <section class="ensemble-card">
                        <p class="ensemble-title">{title}</p>
                        <p class="ensemble-text">{description}</p>
                    </section>
                    """,
                    unsafe_allow_html=True,
                )

        st.caption(
            f"Stacking was tuned with GridSearchCV, but {metadata.get('final_model_name', 'the final model')} was selected as the final model because it achieved the best held-out test Macro F1."
        )


def render_class_overview(metadata: dict) -> None:
    st.title("Class Overview")
    st.markdown(
        """
        <section class="app-card">
            <p class="overview-subtitle">Understanding class distribution, class difficulty, and selected model behavior</p>
            <p class="placeholder-text">
                This page focuses on the target classes rather than the models. While Model Comparison answers
                “which model performs best?”, Class Overview answers “which class is hardest to predict and why?”.
                You can select a model to inspect how it behaves for each class.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    required_keys = [
        "class_names",
        "class_distribution",
        "class_percentages",
        "final_model_name",
        "classification_reports",
        "confusion_matrices",
    ]
    missing_keys = [key for key in required_keys if not metadata.get(key)]
    if missing_keys:
        st.warning(f"Missing class overview metadata: {', '.join(missing_keys)}")
        return

    class_names = metadata.get("class_names", [])
    class_distribution_df = get_class_distribution_df(metadata)
    final_model_name = metadata.get("final_model_name", "N/A")
    available_models = list(metadata.get("classification_reports", {}).keys())
    if not available_models:
        st.warning("No classification reports are available for class-level analysis.")
        return
    average_class_f1_df = get_average_class_f1_df(metadata)
    if class_distribution_df.empty or average_class_f1_df.empty:
        st.warning("Class overview metadata is incomplete. Unable to render this page.")
        return

    class_card_content = {
        "CANDIDATE": {
            "meaning": "Planet candidate, not yet confirmed.",
            "importance": "Represents uncertain cases that still need validation.",
            "difficulty": "Highest",
            "border_color": CANDIDATE,
            "background_color": "#FFFBEB",
        },
        "CONFIRMED": {
            "meaning": "Validated exoplanet.",
            "importance": "Represents successfully verified planet detections.",
            "difficulty": "Medium / easier than CANDIDATE",
            "border_color": CONFIRMED,
            "background_color": "#F0FDF4",
        },
        "FALSE POSITIVE": {
            "meaning": "Signal likely not caused by a real exoplanet.",
            "importance": "Helps filter non-planet detections.",
            "difficulty": "Usually easier due to stronger rejection patterns.",
            "border_color": FALSE_POSITIVE,
            "background_color": "#FEF2F2",
        },
    }

    st.markdown('<p class="section-title">Class Meaning Cards</p>', unsafe_allow_html=True)
    class_card_columns = st.columns(3)
    for column, class_name in zip(class_card_columns, class_names):
        card_content = class_card_content.get(class_name, {})
        class_row = class_distribution_df[class_distribution_df["Class"] == class_name]
        count = int(class_row["Count"].iloc[0]) if not class_row.empty else 0
        percentage = float(class_row["Percentage"].iloc[0]) if not class_row.empty else 0.0
        with column:
            render_class_meaning_card(
                class_name=class_name,
                meaning=card_content.get("meaning", ""),
                importance=card_content.get("importance", ""),
                difficulty=card_content.get("difficulty", ""),
                count=count,
                percentage=percentage,
                border_color=card_content.get("border_color", NASA_BLUE),
                background_color=card_content.get("background_color", "#FFFFFF"),
            )

    st.markdown('<p class="section-title">Dataset Distribution</p>', unsafe_allow_html=True)
    distribution_columns = st.columns(2)
    with distribution_columns[0]:
        render_class_count_chart(class_distribution_df)

    with distribution_columns[1]:
        render_class_percentage_donut(class_distribution_df)

    st.markdown('<p class="section-title">Selected Model Analysis</p>', unsafe_allow_html=True)
    default_model_index = available_models.index(final_model_name) if final_model_name in available_models else 0
    selector_columns = st.columns([1.2, 0.8])
    with selector_columns[0]:
        selected_model = st.selectbox(
            "Select model for class-level analysis",
            options=available_models,
            index=default_model_index,
        )
    with selector_columns[1]:
        if selected_model == final_model_name:
            st.markdown(
                '<span class="selector-badge">Final selected model</span>',
                unsafe_allow_html=True,
            )

    selected_model_metrics_df = get_selected_model_class_metrics_df(metadata, selected_model)
    confusion_insights = get_confusion_insights(metadata, selected_model)
    if selected_model_metrics_df.empty:
        st.warning("Selected model metadata is incomplete. Unable to render model-specific class analysis.")
        return

    st.markdown('<p class="section-title">Selected Model Class Performance</p>', unsafe_allow_html=True)
    performance_columns = st.columns([1.2, 1])
    with performance_columns[0]:
        if go is not None:
            figure = go.Figure()
            metric_colors = {
                "Precision": "#1D4ED8",
                "Recall": "#0891B2",
                "F1-score": "#7C3AED",
            }
            for metric_name in ["Precision", "Recall", "F1-score"]:
                figure.add_trace(
                    go.Bar(
                        x=selected_model_metrics_df["Class"],
                        y=selected_model_metrics_df[metric_name],
                        name=metric_name,
                        marker_color=metric_colors[metric_name],
                        text=[f"{value:.3f}" for value in selected_model_metrics_df[metric_name]],
                        textposition="auto",
                        hovertemplate="<b>%{x}</b><br>"
                        + f"{metric_name}: "
                        + "%{y:.3f}<extra></extra>",
                    )
                )

            figure.update_layout(
                barmode="group",
                bargap=0.18,
                bargroupgap=0.08,
                **get_plotly_layout(
                    f"Class Performance — {selected_model}",
                    height=380,
                    margin={"l": 35, "r": 20, "t": 85, "b": 40},
                    legend={
                        "orientation": "h",
                        "yanchor": "bottom",
                        "y": 1.02,
                        "xanchor": "center",
                        "x": 0.5,
                    },
                ),
            )
            figure.update_xaxes(title_text="")
            figure.update_yaxes(title_text="Score", range=[0, 1])
            st.plotly_chart(figure, use_container_width=True)
        else:
            fallback_df = selected_model_metrics_df.copy()
            for column in ["Precision", "Recall", "F1-score"]:
                fallback_df[column] = fallback_df[column].map(lambda value: f"{value:.3f}")
            st.dataframe(fallback_df, use_container_width=True, hide_index=True)

    with performance_columns[1]:
        render_support_vs_f1_scatter(selected_model_metrics_df, selected_model)

    display_df = selected_model_metrics_df.copy()
    for column in ["Precision", "Recall", "F1-score"]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.3f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown('<p class="section-title">Class-level Interpretation Cards</p>', unsafe_allow_html=True)
    interpretation_columns = st.columns(3)
    for column, class_name in zip(interpretation_columns, class_names):
        class_row = selected_model_metrics_df[selected_model_metrics_df["Class"] == class_name]
        if class_row.empty:
            continue
        metrics_row = class_row.iloc[0]
        with column:
            render_interpretation_card(
                class_name=class_name,
                precision=float(metrics_row["Precision"]),
                recall=float(metrics_row["Recall"]),
                f1_score=float(metrics_row["F1-score"]),
            )

    st.markdown('<p class="section-title">Selected Model Confusion Insights</p>', unsafe_allow_html=True)
    if confusion_insights is None:
        st.warning("Confusion matrix is not available for the selected model.")
    else:
        confusion_columns = st.columns([1.2, 1])
        with confusion_columns[0]:
            render_confusion_matrix(
                confusion_insights["matrix"],
                class_names,
                title=f"Confusion Matrix — {selected_model}",
            )

        matrix_df = pd.DataFrame(
            confusion_insights["matrix"],
            index=class_names,
            columns=class_names,
        )
        with confusion_columns[1]:
            st.dataframe(matrix_df, use_container_width=True)
            largest_confusion = confusion_insights["largest_confusion"]
            st.markdown(
                f"""
                <section class="metric-card">
                    <p class="metric-label">Most Confused Pair</p>
                    <p class="metric-value">{largest_confusion["true_class"]} → {largest_confusion["predicted_class"]}</p>
                    <p class="class-overview-meta">Count: {largest_confusion["count"]}</p>
                </section>
                """,
                unsafe_allow_html=True,
            )
            per_class_confusions = confusion_insights["per_class_confusions"]
            for class_name in class_names:
                insight = per_class_confusions.get(class_name, {})
                st.markdown(
                    f"""
                    <section class="metric-card">
                        <p class="metric-label">{class_name} Most Often Confused With</p>
                        <p class="metric-value">{insight.get("predicted_class", "N/A")}</p>
                        <p class="class-overview-meta">Count: {insight.get("count", 0)}</p>
                    </section>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown('<p class="section-title">Class Difficulty Ranking Across All Models</p>', unsafe_allow_html=True)
    difficulty_columns = st.columns([1.1, 1])
    with difficulty_columns[0]:
        render_class_difficulty_chart(average_class_f1_df)

    hardest_class = average_class_f1_df.iloc[0]["Class"]
    difficulty_table_df = average_class_f1_df.copy()
    difficulty_table_df["Rank"] = range(1, len(difficulty_table_df) + 1)
    difficulty_table_df["Interpretation"] = difficulty_table_df["Class"].map(
        {
            "CANDIDATE": "Most ambiguous boundary between planet-like and non-planet signals.",
            "CONFIRMED": "Generally easier because validated planets form a more stable pattern.",
            "FALSE POSITIVE": "Often easier because rejection patterns are stronger.",
        }
    )
    display_difficulty_df = difficulty_table_df[["Class", "Average F1", "Rank", "Interpretation"]].copy()
    display_difficulty_df["Average F1"] = display_difficulty_df["Average F1"].map(lambda value: f"{value:.3f}")
    with difficulty_columns[1]:
        st.dataframe(display_difficulty_df, use_container_width=True, hide_index=True)

    if hardest_class == "CANDIDATE":
        st.caption(
            "CANDIDATE is the hardest class across models, not only for the selected model. This suggests that the difficulty comes from the data/problem structure, not from one specific algorithm."
        )
    else:
        st.caption(f"{hardest_class} has the lowest average F1 across the evaluated models.")

    st.markdown('<p class="section-title">Why is CANDIDATE difficult?</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <section class="app-card">
            <p class="placeholder-text">
                CANDIDATE objects are not yet confirmed planets. Their feature patterns can look similar to both
                CONFIRMED exoplanets and FALSE POSITIVE signals. Because of this ambiguity, models usually achieve
                lower F1-score on CANDIDATE than on the other two classes.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-title">Final Takeaway</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <section class="takeaway-card">
            <p class="class-overview-title">Main Takeaway</p>
            <p class="placeholder-text">
                The main limitation of this task is not only the model architecture, but the ambiguity of the
                CANDIDATE class. Because CANDIDATE objects are not yet confirmed, their feature patterns can overlap
                with both CONFIRMED and FALSE POSITIVE examples. This explains why models usually perform strongly on
                CONFIRMED and FALSE POSITIVE, while CANDIDATE remains the hardest class.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
def render_feature_analysis(metadata: dict) -> None:
    st.title("Feature / Variable Analysis")
    st.markdown(
        """
        <section class="app-card">
            <p class="overview-subtitle">Feature distributions, class overlap, PCA projection, and final-model importance</p>
            <p class="placeholder-text">
                This view focuses on how individual features are distributed across CANDIDATE, CONFIRMED, and FALSE POSITIVE.
                If CANDIDATE overlaps strongly with the other two classes, that helps explain why it is often the hardest class.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    dataset_df = load_feature_dataset()
    if dataset_df is None:
        st.warning("Feature dataset is missing: `exoplanets_pull.csv`")
        return

    class_names = metadata.get("class_names", [])
    target_column = metadata.get("target_column", "koi_disposition")
    if not class_names or target_column not in dataset_df.columns:
        st.warning("Required class metadata or target column is missing for feature analysis.")
        return

    available_feature_options = get_numeric_feature_options(metadata, dataset_df)
    if not available_feature_options:
        st.warning("None of the requested feature columns are available in the dataset.")
        return

    default_index = (
        available_feature_options.index("koi_model_snr")
        if "koi_model_snr" in available_feature_options
        else 0
    )
    skewed_feature_suggestions = {
        "koi_model_snr",
        "koi_depth",
        "koi_prad",
        "koi_insol",
        "koi_srad",
        "koi_snr_depth_ratio",
        "koi_prad_srad_ratio",
        "koi_insol_teq_ratio",
    }

    control_columns = st.columns([1.1, 0.9, 0.8])
    with control_columns[0]:
        selected_feature = st.selectbox(
            "Select feature",
            options=available_feature_options,
            index=default_index,
        )
    if st.session_state.get("feature_analysis_last_feature") != selected_feature:
        st.session_state["feature_analysis_use_log_scale"] = selected_feature in skewed_feature_suggestions
        st.session_state["feature_analysis_last_feature"] = selected_feature
    with control_columns[1]:
        plot_type = st.radio(
            "Plot type",
            options=["Violin Plot", "Box Plot"],
            index=1,
            horizontal=True,
        )
    with control_columns[2]:
        use_log_scale = st.checkbox(
            "Use log scale for selected feature",
            key="feature_analysis_use_log_scale",
        )
    trim_outliers = st.checkbox(
        "Trim extreme outliers for plot",
        value=True,
        key="feature_analysis_trim_outliers",
    )
    if selected_feature in skewed_feature_suggestions:
        st.info("This feature is highly skewed. Try log scale for a clearer distribution view.")

    feature_df = (
        dataset_df[[target_column, selected_feature]]
        .dropna()
        .rename(columns={target_column: "Class"})
    )
    feature_df = feature_df[feature_df["Class"].isin(class_names)]

    if feature_df.empty:
        st.warning("No usable rows are available for the selected feature.")
        return

    render_feature_distribution_chart(
        feature_df=feature_df,
        selected_feature=selected_feature,
        plot_type=plot_type,
        class_names=class_names,
        use_log_scale=use_log_scale,
        trim_outliers=trim_outliers,
    )
    if trim_outliers:
        st.caption("Distribution plot is trimmed to the 1st–99th percentile for readability. Summary table uses full data.")

    st.markdown('<p class="section-title">Class Median Comparison</p>', unsafe_allow_html=True)
    summary_df = build_feature_summary_table(feature_df, selected_feature, class_names)
    render_class_median_chart(summary_df, selected_feature, class_names, use_log_scale)

    st.markdown('<p class="section-title">Feature Summary by Class</p>', unsafe_allow_html=True)
    display_summary_df = summary_df.copy()
    for column in ["Mean", "Median", "Std", "Min", "Max"]:
        display_summary_df[column] = display_summary_df[column].map(lambda value: f"{value:.3f}")
    st.dataframe(display_summary_df, use_container_width=True, hide_index=True)

    overlap_text = generate_feature_overlap_interpretation(feature_df, selected_feature, class_names)
    st.markdown(
        f"""
        <section class="app-card">
            <p class="class-overview-title">Automatic Overlap Interpretation</p>
            <p class="placeholder-text">{overlap_text}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-title">Bivariate Feature Relationship</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <section class="app-card">
            <p class="placeholder-text">
                This view compares two selected features directly. It can reveal whether classes separate in the original feature space without PCA compression.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    bivariate_feature_order = [
        "koi_model_snr",
        "koi_prad",
        "koi_depth",
        "koi_period",
        "koi_duration",
        "koi_teq",
        "koi_insol",
        "koi_srad",
    ]
    bivariate_feature_options = [
        feature for feature in bivariate_feature_order if feature in available_feature_options
    ] + [
        feature for feature in available_feature_options if feature not in bivariate_feature_order
    ]
    default_x_index = bivariate_feature_options.index("koi_model_snr") if "koi_model_snr" in bivariate_feature_options else 0
    default_y_index = bivariate_feature_options.index("koi_prad") if "koi_prad" in bivariate_feature_options else min(1, len(bivariate_feature_options) - 1)

    relationship_columns = st.columns(2)
    with relationship_columns[0]:
        x_feature = st.selectbox(
            "Select X feature",
            options=bivariate_feature_options,
            index=default_x_index,
        )
    with relationship_columns[1]:
        y_feature = st.selectbox(
            "Select Y feature",
            options=bivariate_feature_options,
            index=default_y_index,
        )

    if st.session_state.get("feature_analysis_last_x_feature") != x_feature:
        st.session_state["feature_analysis_use_log_x"] = is_highly_skewed_feature(x_feature)
        st.session_state["feature_analysis_last_x_feature"] = x_feature
    if st.session_state.get("feature_analysis_last_y_feature") != y_feature:
        st.session_state["feature_analysis_use_log_y"] = is_highly_skewed_feature(y_feature)
        st.session_state["feature_analysis_last_y_feature"] = y_feature

    relationship_control_columns = st.columns(3)
    with relationship_control_columns[0]:
        use_log_x = st.checkbox("Use log scale for X", key="feature_analysis_use_log_x")
    with relationship_control_columns[1]:
        use_log_y = st.checkbox("Use log scale for Y", key="feature_analysis_use_log_y")
    with relationship_control_columns[2]:
        trim_bivariate_outliers = st.checkbox(
            "Trim extreme outliers for scatter",
            value=True,
            key="feature_analysis_trim_bivariate_outliers",
        )

    if is_highly_skewed_feature(x_feature) or is_highly_skewed_feature(y_feature):
        st.info("One or both selected features are highly skewed. Log scale can make the class relationship easier to read.")

    scatter_columns = [target_column, x_feature, y_feature]
    optional_scatter_columns = [column for column in ["kepid", "kepoi_name"] if column in dataset_df.columns]
    scatter_df = (
        dataset_df[scatter_columns + optional_scatter_columns]
        .dropna(subset=[target_column, x_feature, y_feature])
        .rename(columns={target_column: "Class"})
    )
    scatter_df = scatter_df[scatter_df["Class"].isin(class_names)].copy()
    scatter_df["plot_x"] = apply_visual_log_transform(scatter_df[x_feature]) if use_log_x else scatter_df[x_feature].astype(float)
    scatter_df["plot_y"] = apply_visual_log_transform(scatter_df[y_feature]) if use_log_y else scatter_df[y_feature].astype(float)
    if trim_bivariate_outliers:
        scatter_df = trim_plot_outliers(scatter_df, [x_feature, y_feature])
        st.caption("Scatter plot is trimmed to the 1st–99th percentile on both selected axes for readability.")

    render_bivariate_scatter(
        scatter_df,
        x_feature=x_feature,
        y_feature=y_feature,
        class_names=class_names,
        use_log_x=use_log_x,
        use_log_y=use_log_y,
    )

    interpretation_messages: list[str] = []
    if not scatter_df.empty:
        class_stats: dict[str, dict[str, float]] = {}
        for class_name in class_names:
            class_slice = scatter_df[scatter_df["Class"] == class_name]
            if class_slice.empty:
                continue
            class_stats[class_name] = {
                "x_median": float(class_slice["plot_x"].median()),
                "y_median": float(class_slice["plot_y"].median()),
                "x_q1": float(class_slice["plot_x"].quantile(0.25)),
                "x_q3": float(class_slice["plot_x"].quantile(0.75)),
                "y_q1": float(class_slice["plot_y"].quantile(0.25)),
                "y_q3": float(class_slice["plot_y"].quantile(0.75)),
            }

        candidate_stats = class_stats.get("CANDIDATE")
        confirmed_stats = class_stats.get("CONFIRMED")
        false_positive_stats = class_stats.get("FALSE POSITIVE")
        if candidate_stats and confirmed_stats and false_positive_stats:
            overlap_x = (
                max(candidate_stats["x_q1"], confirmed_stats["x_q1"]) <= min(candidate_stats["x_q3"], confirmed_stats["x_q3"])
                and max(candidate_stats["x_q1"], false_positive_stats["x_q1"]) <= min(candidate_stats["x_q3"], false_positive_stats["x_q3"])
            )
            overlap_y = (
                max(candidate_stats["y_q1"], confirmed_stats["y_q1"]) <= min(candidate_stats["y_q3"], confirmed_stats["y_q3"])
                and max(candidate_stats["y_q1"], false_positive_stats["y_q1"]) <= min(candidate_stats["y_q3"], false_positive_stats["y_q3"])
            )
            if overlap_x or overlap_y:
                interpretation_messages.append(
                    "This scatter plot shows how two raw features relate to each other. When class-colored points overlap, these two features alone are not enough to fully separate the classes."
                )

            candidate_between_x = min(confirmed_stats["x_median"], false_positive_stats["x_median"]) <= candidate_stats["x_median"] <= max(confirmed_stats["x_median"], false_positive_stats["x_median"])
            candidate_between_y = min(confirmed_stats["y_median"], false_positive_stats["y_median"]) <= candidate_stats["y_median"] <= max(confirmed_stats["y_median"], false_positive_stats["y_median"])
            if candidate_between_x or candidate_between_y:
                interpretation_messages.append(
                    "CANDIDATE appears between CONFIRMED and FALSE POSITIVE on at least one selected feature axis, supporting its ambiguous class behavior."
                )

    if not interpretation_messages:
        interpretation_messages.append(
            "These two features show some class structure, but no simple two-feature view fully explains the overall decision boundaries."
        )

    st.markdown(
        f"""
        <section class="app-card">
            <p class="class-overview-title">Bivariate Interpretation</p>
            <p class="placeholder-text">{' '.join(interpretation_messages)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-title">PCA 2D Projection</p>', unsafe_allow_html=True)
    pca_control_columns = st.columns(4)
    with pca_control_columns[0]:
        pca_sample_size = st.slider("PCA sample size", min_value=500, max_value=3000, value=1500, step=100)
    with pca_control_columns[1]:
        pca_point_opacity = st.slider("PCA point opacity", min_value=0.2, max_value=0.9, value=0.45, step=0.05)
    with pca_control_columns[2]:
        pca_point_size = st.slider("PCA point size", min_value=3, max_value=9, value=5, step=1)
    with pca_control_columns[3]:
        focus_pca_view = st.checkbox("Focus PCA view on central 98%", value=True)

    pca_df = build_pca_dataframe(metadata, selected_feature, pca_sample_size)
    render_pca_scatter(
        pca_df,
        selected_feature,
        class_names,
        point_opacity=pca_point_opacity,
        point_size=pca_point_size,
        focus_central_view=focus_pca_view,
    )
    if focus_pca_view:
        st.caption("PCA axes are focused on the central 98% of points for readability.")
    st.markdown(
        """
        <section class="app-card">
            <p class="placeholder-text">
                PCA is used here only as an exploratory visualization. It compresses many numeric features into two
                components, so overlap in this plot suggests that classes may not be perfectly separable using a
                simple linear projection.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <section class="app-card">
            <p class="placeholder-text">
                The dense overlap in the central region suggests that the classes are not perfectly separable in a simple
                2D projection. This supports why CANDIDATE can be difficult to classify.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-title">Feature Correlation Heatmap</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <section class="app-card">
            <p class="placeholder-text">
                This heatmap shows relationships between numeric features. Highly correlated features may carry overlapping information.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    correlation_mode = st.radio(
        "Correlation feature set",
        options=["Top important features", "Selected feature neighborhood"],
        horizontal=True,
    )
    correlation_features = build_correlation_feature_set(metadata, dataset_df, correlation_mode, selected_feature)
    if len(correlation_features) < 2:
        st.warning("Not enough numeric features are available to build a correlation heatmap.")
    else:
        correlation_input_df = dataset_df[correlation_features].copy()
        imputer = SimpleImputer(strategy="median")
        imputed_correlation_df = pd.DataFrame(
            imputer.fit_transform(correlation_input_df),
            columns=correlation_features,
        )
        correlation_df = imputed_correlation_df.corr(method="pearson")
        render_correlation_heatmap(correlation_df)
        correlation_interpretation = generate_correlation_interpretation(correlation_df)
        st.markdown(
            f"""
            <section class="app-card">
                <p class="class-overview-title">Correlation Interpretation</p>
                <p class="placeholder-text">{correlation_interpretation}</p>
            </section>
            """,
            unsafe_allow_html=True,
    )

    st.markdown('<p class="section-title">Top Feature Importances — Final Model</p>', unsafe_allow_html=True)
    render_feature_importance_section(metadata)


def render_dictionary(metadata: dict) -> None:
    st.title("Dictionary")
    st.markdown(
        """
        <section class="app-card">
            <p class="placeholder-text">
                This page explains the main Kepler Object of Interest features used across prediction, model comparison,
                and feature analysis. Features marked with “(F.E)” are engineered during feature engineering and are used
                as additional model inputs.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    dictionary_df = build_feature_dictionary_df()
    total_features = len(dictionary_df)
    engineered_mask = dictionary_df["Display Feature"].str.contains(r"\(F\.E\)", regex=True, na=False)
    original_count = int((~engineered_mask).sum())
    engineered_count = int(engineered_mask.sum())
    category_count = int(dictionary_df["Category"].nunique())

    metric_columns = st.columns(4)
    metric_items = [
        ("Total Features", total_features),
        ("Original Features", original_count),
        ("Engineered Features", engineered_count),
        ("Categories", category_count),
    ]
    for column, (label, value) in zip(metric_columns, metric_items):
        with column:
            render_metric_card(label, value)

    filter_columns = st.columns([1.2, 0.9, 0.8])
    with filter_columns[0]:
        search_text = st.text_input("Search feature", value="")
    sorted_categories = sorted(dictionary_df["Category"].dropna().unique().tolist())
    with filter_columns[1]:
        category_filter = st.selectbox("Category filter", options=["All"] + sorted_categories, index=0)
    with filter_columns[2]:
        feature_type_filter = st.selectbox(
            "Feature type filter",
            options=["All", "Original only", "Engineered only"],
            index=0,
        )

    filtered_df = dictionary_df.copy()
    if search_text.strip():
        search_value = search_text.strip().lower()
        mask = pd.Series(False, index=filtered_df.index)
        for column in ["Display Feature", "Original Column", "Meaning", "Category", "Why it matters"]:
            mask = mask | filtered_df[column].str.lower().str.contains(search_value, na=False)
        filtered_df = filtered_df[mask]
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df["Category"] == category_filter]
    if feature_type_filter == "Original only":
        filtered_df = filtered_df[~filtered_df["Display Feature"].str.contains(r"\(F\.E\)", regex=True, na=False)]
    elif feature_type_filter == "Engineered only":
        filtered_df = filtered_df[filtered_df["Display Feature"].str.contains(r"\(F\.E\)", regex=True, na=False)]
    filtered_df = filtered_df.reset_index(drop=True)

    st.caption(f"Showing {len(filtered_df)} of {len(dictionary_df)} features")
    st.caption("Use the search and filters to find a feature quickly.")

    table_df = filtered_df[
        ["Display Feature", "Original Column", "Category", "Meaning", "Why it matters"]
    ].copy()
    column_config = None
    if hasattr(st, "column_config"):
        column_config = {
            "Display Feature": st.column_config.TextColumn("Display Feature", width="medium"),
            "Original Column": st.column_config.TextColumn("Original Column", width="medium"),
            "Category": st.column_config.TextColumn("Category", width="medium"),
            "Meaning": st.column_config.TextColumn("Meaning", width="large"),
            "Why it matters": st.column_config.TextColumn("Why it matters", width="large"),
        }
    st.dataframe(
        table_df,
        height=520,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
    )
    st.caption("Some feature descriptions are simplified for dashboard interpretation.")

    with st.expander("Category Guide", expanded=False):
        category_guide_df = pd.DataFrame(
            [
                {"Category": "Orbital property", "Meaning": "Timing and orbit-related measurements"},
                {"Category": "Transit property", "Meaning": "Brightness drop and transit-shape measurements"},
                {"Category": "Planet property", "Meaning": "Estimated physical properties of the candidate"},
                {"Category": "Planet/environment property", "Meaning": "Temperature and energy-related estimates"},
                {"Category": "Stellar property", "Meaning": "Host star characteristics"},
                {"Category": "Transit geometry", "Meaning": "Geometry of how the object crosses the star"},
                {"Category": "Signal quality", "Meaning": "Strength and reliability of the detected signal"},
                {"Category": "Catalog/identifier feature", "Meaning": "Dataset/catalog metadata used as input"},
                {"Category": "Engineered ratio feature", "Meaning": "Features created from existing columns during feature engineering"},
            ]
        )
        st.dataframe(category_guide_df, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="NASA Kepler Exoplanet Classification",
        page_icon="🪐",
        layout="wide",
    )
    inject_css()

    metadata = load_metadata()
    if metadata is None:
        st.error("Metadata file is missing: `models/metadata.joblib`")
        st.stop()

    models, missing_models = load_models(metadata)
    for model_name, path in missing_models:
        st.warning(f"Model file missing for {model_name}: `{path}`")

    render_global_header()

    overview_tab, prediction_tab, comparison_tab, class_tab, feature_tab, dictionary_tab = st.tabs(
        PAGE_TITLES
    )

    with overview_tab:
        render_overview(metadata, model_count=len(models))

    with prediction_tab:
        render_prediction(metadata, models)

    with comparison_tab:
        render_model_comparison(metadata)

    with class_tab:
        render_class_overview(metadata)

    with feature_tab:
        render_feature_analysis(metadata)

    with dictionary_tab:
        render_dictionary(metadata)


if __name__ == "__main__":
    main()
