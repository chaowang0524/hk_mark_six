"""Web front end for the Mark Six pipeline — no install, no command line.

Deployed on a dedicated VPS, this lets anyone with the link click a button
to work the pipeline in their browser. Two pages, switched with the buttons
at the top:

- "Preprocess": rebuilds data/features.csv from the current raw draws.
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

nav_col1, nav_col2 = st.columns(2)
with nav_col1:
    if st.button(
        "⚙️ Preprocess",
        width="stretch",
        type="primary" if st.session_state.page == "preprocess" else "secondary",
    ):
        st.session_state.page = "preprocess"
with nav_col2:
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
else:
    render_preprocess_page()
