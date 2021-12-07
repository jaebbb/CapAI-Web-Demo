import os

import streamlit as st

from app_page.classification import *
from app_page.detection import *
from utils.general import get_markdown


def main():
    st.set_page_config(layout="wide")
    readme_text = st.markdown(get_markdown("main.md"), unsafe_allow_html=True)

    st.sidebar.title("KVL Lab")
    st.sidebar.image("logo.png", use_column_width=True)
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox(
        "Choose the menu",
        ["Main", "🔍 Lesion detection", "🔍Transition classification", "Show the source code"],
    )
    
    if app_mode == "Main":
        st.sidebar.success("To continue select any menu.")
        os.system("./remove_cache.sh")
    elif app_mode == "🔍 Lesion detection":
        readme_text.empty()
        run_detection()
    elif app_mode == "🔍Transition classification":
        readme_text.empty()
        run_classification()
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_markdown("main.py", prefix="."))


if __name__ == "__main__":
    main()
