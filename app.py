"""Web front end for the Mark Six pipeline — no install, no command line.

Deployed on Streamlit Community Cloud, this lets anyone with the link click
a button and get a prediction in their browser. Two modes:

- "Show prediction": reuses whatever model is already trained on this
  server (fast, seconds).
- "Refresh data & retrain": re-scrapes the latest draws and retrains from
  scratch (slow — several minutes — but keeps predictions current).
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

MODELS_DIR = ROOT / "data" / "models"

st.set_page_config(page_title="Mark Six Predictor", page_icon="🎱")
st.title("🎱 Mark Six Extra-Number Prediction")
st.caption(
    "Statistical/ML prediction for the next Hong Kong Mark Six extra number. "
    "Educational project — draws are independent, this is not a guarantee."
)


def _run(both: bool, retrain: bool) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if retrain:
            from scraper import scrape
            from preprocessing import preprocess
            from models import train

            print("=== Fetching latest draws ===")
            scrape()
            print("\n=== Building features ===")
            preprocess()
            print("\n=== Training model on full history (the slow part) ===")
            train("xgb", full=True)
        from predictor import predict

        print("\n=== Predicting next draw ===")
        predict(model_name="xgb", top_extra=10, top_main=10, both=both)
    return buf.getvalue()


has_model = (MODELS_DIR / "extra_xgb.joblib").exists()

both = st.checkbox(
    "Also predict the 6 main numbers (slower to retrain)",
    value=False,
    help="The project currently focuses on the extra number; main-number "
    "prediction is available but roughly doubles retrain time.",
)

col1, col2 = st.columns(2)
with col1:
    quick_clicked = st.button(
        "🔮 Show prediction",
        disabled=not has_model,
        help="Uses the model already trained on this server.",
    )
with col2:
    retrain_clicked = st.button(
        "🔄 Refresh data & retrain",
        help="Re-scrapes the latest draws and retrains from scratch. "
        "Takes several minutes — please be patient and don't close the tab.",
    )

if not has_model and not retrain_clicked:
    st.info(
        "No trained model on this server yet. Click **Refresh data & retrain** "
        "to build one — only needed once, then it's saved for next time. "
        "This takes several minutes, please wait."
    )

output: str | None = None
error: str | None = None

if retrain_clicked:
    with st.spinner(
        "Scraping latest draws and retraining — this can take 5+ minutes. "
        "Please keep this tab open."
    ):
        try:
            output = _run(both=both, retrain=True)
        except Exception as exc:  # noqa: BLE001 — surface any pipeline failure to the page
            error = str(exc)
elif quick_clicked:
    with st.spinner("Predicting…"):
        try:
            output = _run(both=both, retrain=False)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

if error:
    st.error(f"Something went wrong: {error}")
if output:
    st.code(output, language=None)
