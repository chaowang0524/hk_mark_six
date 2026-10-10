"""Web front end for the Mark Six pipeline — no install, no command line.

Deployed on a dedicated VPS, this lets anyone with the link click a button
to work the pipeline in their browser. Pages, switched with the buttons at
the top:

- "Preprocess": rebuilds data/features.csv from the current raw draws.
- "Train": trains the extra-number and main-number models from the current
  features.csv.
- "Predict": picks a trained model (if more than one is available) and
  predicts the next draw's extra number and 6 main numbers.
- "Data": password-protected. Shows the most recent draws and lets you add
  or remove one by hand — scraping is blocked from this server's network
  and from the local dev machine (see STATUS.md), so this is the only way
  new draws get into the dataset for now.
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

st.set_page_config(page_title="Mark Six Predictor", page_icon="🎱")

if "page" not in st.session_state:
    st.session_state.page = "preprocess"

nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
with nav_col1:
    if st.button(
        "⚙️ Preprocess",
        width="stretch",
        type="primary" if st.session_state.page == "preprocess" else "secondary",
    ):
        st.session_state.page = "preprocess"
with nav_col2:
    if st.button(
        "🏋️ Train",
        width="stretch",
        type="primary" if st.session_state.page == "train" else "secondary",
    ):
        st.session_state.page = "train"
with nav_col3:
    if st.button(
        "🔮 Predict",
        width="stretch",
        type="primary" if st.session_state.page == "predict" else "secondary",
    ):
        st.session_state.page = "predict"
with nav_col4:
    if st.button(
        "📋 Data",
        width="stretch",
        type="primary" if st.session_state.page == "data" else "secondary",
    ):
        st.session_state.page = "data"

st.divider()


def render_preprocess_page() -> None:
    st.title("⚙️ Build Features")
    st.caption(
        "Rebuilds data/features.csv from the current raw draws. Add new draws "
        "on the Data page first if there are any."
    )

    run_clicked = st.button(
        "⚙️ Run preprocessing",
        help="Takes a few seconds to a couple of minutes depending on history "
        "size — please keep this tab open.",
    )

    output: str | None = None
    error: str | None = None

    if run_clicked:
        with st.spinner("Preprocessing…"):
            from preprocessing import preprocess

            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    preprocess()
                output = buf.getvalue()
            except Exception as exc:  # noqa: BLE001 — surface any pipeline failure to the page
                error = str(exc)

    if error:
        st.error(f"Something went wrong: {error}")
    elif output:
        st.success("✅ Preprocess done")
        st.code(output, language=None)


_MODEL_LABELS = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lr": "Logistic Regression",
    "all": "All three",
}
_TARGET_LABELS = {"extra": "extra number", "main": "main numbers"}
_TARGET_CHOICE_LABELS = {
    "both": "Both (recommended)",
    "extra": "Extra number only",
    "main": "Main numbers only",
}
_TARGET_CHOICE_TARGETS = {
    "both": ("extra", "main"),
    "extra": ("extra",),
    "main": ("main",),
}


def render_train_page() -> None:
    st.title("🏋️ Train Models")
    st.caption(
        "Trains the extra-number and/or main-number models from the current "
        "data/features.csv. Run Preprocess first if the data has changed."
    )

    model_choice = st.radio(
        "Which model(s) to train?",
        options=["rf", "xgb", "lr", "all"],
        format_func=lambda m: _MODEL_LABELS[m],
        index=3,
        horizontal=True,
    )

    target_choice = st.radio(
        "Train which target?",
        options=["both", "extra", "main"],
        format_func=lambda t: _TARGET_CHOICE_LABELS[t],
        index=0,
        horizontal=True,
    )

    full = st.checkbox(
        "Use all data, no accuracy check",
        help="Normally some of the most recent draws are held back so we can "
        "check how accurate the model is. Turn this on to train on every "
        "draw instead — do this once you're happy with the accuracy, right "
        "before predicting.",
    )

    run_clicked = st.button("🏋️ Start training", type="primary")

    if run_clicked:
        from models import train, EXTRA_MODEL_REGISTRY

        names = (
            list(EXTRA_MODEL_REGISTRY.keys()) if model_choice == "all" else [model_choice]
        )
        targets = _TARGET_CHOICE_TARGETS[target_choice]
        total_steps = len(names) * len(targets)
        progress = {"done": 0}

        output: str | None = None
        error: str | None = None

        with st.status("Starting…", expanded=True) as status:

            def on_step(target: str, name: str) -> None:
                progress["done"] += 1
                status.update(
                    label=f"({progress['done']}/{total_steps}) Training "
                    f"{_TARGET_LABELS[target]} model: {_MODEL_LABELS[name]}…"
                )

            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    train(model_choice, full=full, targets=targets, on_step=on_step)
                output = buf.getvalue()
                status.update(label="Training complete", state="complete")
            except Exception as exc:  # noqa: BLE001 — surface any pipeline failure to the page
                error = str(exc)
                status.update(label="Training failed", state="error")

        if error:
            st.error(f"Something went wrong: {error}")
        elif output:
            st.success("✅ Training done")
            st.code(output, language=None)


_PREDICT_MODEL_LABELS = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lr": "Logistic Regression",
    "ensemble": "Ensemble (average of all trained)",
}


def render_predict_page() -> None:
    st.title("🔮 Predict Next Draw")
    st.caption(
        "Predicts the extra number and the 6 main numbers for the next draw "
        "from the currently trained models. Run Train first if you haven't, "
        "or if the data has changed since the last training."
    )

    from predictor import list_available_models

    available = list_available_models(both=True)
    if not available:
        st.error(
            "No trained models found for both targets yet — go to the Train "
            "page first."
        )
        return

    if len(available) == 1:
        model_choice = available[0]
        st.caption(f"Model: **{_PREDICT_MODEL_LABELS[model_choice]}** (only one trained)")
    else:
        model_choice = st.radio(
            "Which model?",
            options=available,
            format_func=lambda m: _PREDICT_MODEL_LABELS[m],
            horizontal=True,
        )

    override_date = st.checkbox(
        "Override the predicted draw date",
        help="By default the next Tue/Thu/Sat after the latest draw in the "
        "data is used. Turn this on if the real HKJC calendar deviates from "
        "that usual cadence.",
    )
    date_value: str | None = None
    if override_date:
        picked = st.date_input("Draw date")
        date_value = picked.isoformat()

    penalize_last = st.checkbox(
        "Penalize the last 3 extra numbers",
        help="Lowers the ranking of the 3 most-recently-drawn extra numbers, "
        "fading linearly (the most recent one discounted the most) — a "
        "heuristic, not model signal.",
    )

    run_clicked = st.button("🔮 Run prediction", type="primary")

    output: str | None = None
    error: str | None = None

    if run_clicked:
        from predictor import predict

        with st.spinner("Predicting…"):
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    predict(
                        model_name=model_choice,
                        both=True,
                        date=date_value,
                        penalize_last=0.5 if penalize_last else None,
                    )
                output = buf.getvalue()
            except Exception as exc:  # noqa: BLE001 — surface any pipeline failure to the page
                error = str(exc)

    if error:
        st.error(f"Something went wrong: {error}")
    elif output:
        st.success("✅ Prediction done")
        st.code(output, language=None)


def _check_password(entered: str) -> bool:
    try:
        correct = st.secrets["data_password"]
    except (KeyError, FileNotFoundError):
        st.error(
            "No `data_password` configured on this server "
            "(expected in `.streamlit/secrets.toml`)."
        )
        return False
    return entered == correct


def render_data_page() -> None:
    st.title("📋 Manage Draw Data")

    # st.rerun() below fires right after setting these, so the message has
    # to survive into the *next* run instead of being shown before the rerun.
    if "flash_success" in st.session_state:
        st.success(st.session_state.pop("flash_success"))
    if "flash_info" in st.session_state:
        st.info(st.session_state.pop("flash_info"))

    if not st.session_state.get("data_unlocked", False):
        st.caption(
            "Password-protected — this edits the live dataset used for predictions."
        )
        password = st.text_input("Password", type="password")
        if st.button("Unlock"):
            if _check_password(password):
                st.session_state.data_unlocked = True
                st.rerun()
            else:
                st.error("Wrong password.")
        return

    from data_entry import add_draw, delete_last, read_tail

    st.caption(
        "Scraping is currently blocked (see STATUS.md), so new draws are "
        "added here by hand."
    )

    st.subheader("Most recent draws")
    st.dataframe(read_tail(10), width="stretch", hide_index=True)

    st.subheader("Add a new draw")
    with st.form("add_draw_form", clear_on_submit=True):
        date = st.date_input("Draw date")
        mains = st.multiselect(
            "6 main numbers", options=list(range(1, 50)), max_selections=6
        )
        extra = st.number_input("Extra number", min_value=1, max_value=49, value=1, step=1)
        submitted = st.form_submit_button("Add draw")

    if submitted:
        try:
            draw_id = add_draw(date.isoformat(), mains, int(extra))
            st.session_state.flash_success = f"Added draw {draw_id}."
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.subheader("Undo")
    if st.session_state.get("confirm_delete", False):
        st.warning("Delete the most recent draw shown above? This can't be undone.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Yes, delete it"):
                removed = delete_last()
                st.session_state.confirm_delete = False
                if removed:
                    st.session_state.flash_success = (
                        f"Deleted draw {removed['draw']} ({removed['date']})."
                    )
                else:
                    st.session_state.flash_info = "Nothing to delete."
                st.rerun()
        with c2:
            if st.button("Cancel"):
                st.session_state.confirm_delete = False
                st.rerun()
    else:
        if st.button("🗑️ Delete most recent entry"):
            st.session_state.confirm_delete = True
            st.rerun()


if st.session_state.page == "data":
    render_data_page()
elif st.session_state.page == "train":
    render_train_page()
elif st.session_state.page == "predict":
    render_predict_page()
else:
    render_preprocess_page()
