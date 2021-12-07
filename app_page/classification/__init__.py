import streamlit as st
from utils.general import get_markdown

from .cls_img import run_cls_img
from .cls_vid import run_cls_default


def run_classification():
    readme_text = st.markdown(get_markdown("transition_classification.md"), unsafe_allow_html=True)
    
    st.sidebar.title("Task")    
    task_mode = st.sidebar.selectbox(
        "Choose the task", ["Introduction","📃 Default Testset", "📷 Upload Image"]
    )
    if task_mode == "Introduction":
        st.sidebar.success('To continue select any task.')
    elif task_mode == "📃 Default Testset":
        readme_text.empty()
        run_cls_default()
    elif task_mode == "📷 Upload Image":
        readme_text.empty()
        run_cls_img()
    
