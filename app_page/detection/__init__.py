import streamlit as st
from utils.general import get_markdown

from .det_img import run_det_img
from .det_vid import run_det_vid


def run_detection():
    readme_text = st.markdown(get_markdown("lesion_detection.md"), unsafe_allow_html=True)
    
    st.sidebar.title("Task")    
    task_mode = st.sidebar.selectbox(
        "Choose the task", ["Introduction", "📷 Image", "📽️ Video"]
    )
    if task_mode == "Introduction":
        st.sidebar.success('To continue select any task.')
    elif task_mode == "📷 Image":
        readme_text.empty()
        run_det_img()
    elif task_mode == "📽️ Video":
        readme_text.empty()
        run_det_vid()
