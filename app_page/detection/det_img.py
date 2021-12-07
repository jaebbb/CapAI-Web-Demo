import os
import sys
from pathlib import Path

import cv2
import numpy as np
import nvidia_smi
import streamlit as st
import torch

load_model_list = {}


def run_det_img():
    # side bar
    model_type, conf_slider, iou_slider = frame_selector_ui()

    # model init
    model, device = load_model(model_name=model_type)

    # file upload
    uploaded_file = st.file_uploader("Upload a image", ["jpg", "jpeg", "png"])
    # dvide container into two parts
    _, col1, col2, _ = st.columns([1, 4, 4, 1])

    if uploaded_file is not None:  # inference
        bytes_data = uploaded_file.getvalue()
        decoded = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), -1)

        # load a image
        os.makedirs("data", exist_ok=True)
        img_path = f"data/{uploaded_file.name}"
        cv2.imwrite(img_path, decoded)
        img_org = cv2.imread(img_path)

        img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
        with col1:
            st.markdown('**<div align="center">Input image</div>**', unsafe_allow_html=True)
            st.image(img_org, use_column_width=True)  # display input image
        with col2:
            st.markdown('**<div align="center">Output image</div>**', unsafe_allow_html=True)

        ###############################################################################
        # inference results
        with torch.no_grad():
            if model_type == "yolov5":
                from models.yolov5 import yolov5

                # image preprocessing
                img = yolov5.preprocess_image(img_org, stride=int(model.stride.max()))

                # inferencem
                pred = model(img.to(device))[0]

                # resize
                rate = min(383 / img_org.shape[0], 383 / img_org.shape[1])
                if img_org.shape[0] < 383 or img_org.shape[1] < 383:
                    img_org = cv2.resize(
                        img_org,
                        (int(img_org.shape[1] * rate), int(img_org.shape[0] * rate)),
                        interpolation=cv2.INTER_LINEAR,
                    )

                img_bboxes = yolov5.draw_image_with_boxes(
                    img_org, pred, img.shape[2:], conf=conf_slider, iou=iou_slider
                )  # get bboxes and labels
            elif "swin" in model_type:
                from mmdet.apis import inference_detector

                # resize
                rate = min(512 / img_org.shape[0], 512 / img_org.shape[1])
                if img_org.shape[0] < 512 or img_org.shape[1] < 512:
                    img_org = cv2.resize(
                        img_org,
                        (int(img_org.shape[1] * rate), int(img_org.shape[0] * rate)),
                        interpolation=cv2.INTER_LINEAR,
                    )

                pred = inference_detector(model, img_org)
                img_bboxes = model.show_result(img_org, pred, score_thr=conf_slider)
            else:
                pass
        ###############################################################################

        with col2:
            st.image(img_bboxes, use_column_width=True)  # display input image

    elif uploaded_file is None:
        st.info("Check the Image format (e.g. jpg, jpeg, png)")


def frame_selector_ui():
    model_list = list(Path("models/weights").glob("*.pt"))
    model_list = sorted([str(model.name)[:-3] for model in model_list], reverse=True)

    st.sidebar.markdown("# Options")

    model_type = st.sidebar.selectbox("Select model", model_list, 0)

    conf_slider = st.sidebar.slider(
        "conf threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01
    )

    if model_type == "yolov5":
        iou_slider = st.sidebar.slider(
            "IoU threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.01
        )
    else:
        iou_slider = None

    return model_type, conf_slider, iou_slider


def load_model(model_name="yolov5", half=True):
    device = torch.device("cuda:0")

    if model_name in load_model_list.keys():
        return load_model_list[model_name], device

    path = "models/weights/" + model_name + ".pt"

    # for check GPU Memory
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    # load model
    MEGABYTES = 2.0 ** 20.0
    weights_warning, progress_bar = None, None
    used_memory_temp = 0
    try:
        weights_warning = st.warning("Loading %s..." % path)
        progress_bar = st.progress(0)

        # for import model library
        sys.path.append(f"./models")
        sys.path.append(f"./models/{model_name}")

        ###############################################################################
        if model_name == "yolov5":
            model = torch.load(path, map_location=device)["model"].float()
        elif "swin" in model_name:
            from mmcv import Config
            from mmdet.apis import init_detector

            config = f"models/{model_name}/{model_name}.py"
            classes = Config.fromfile(config).classes
            model = init_detector(config, path, device=device)
            model.CLASSES = classes
        else:
            pass
            # model = torch.load(path, map_location=device).float()
        ###############################################################################

        while True:
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            used_memory = info.used / MEGABYTES
            total_memory = info.free / MEGABYTES

            if used_memory_temp == used_memory:
                break

            used_memory_temp = used_memory

            # We perform animation by overwriting the elements.
            weights_warning.warning(
                "Loading %s... (%6.2f/%6.2f MB)" % (path, used_memory, total_memory)
            )
            progress_bar.progress(min(used_memory / total_memory, 1.0))

    finally:
        ###############################################################################
        if model_name == "yolov5":
            from models.yolov5 import yolov5

            # for inference
            model.eval()
            if half:
                model.half()
            # for warming up
            model(
                torch.zeros(1, 3, yolov5.IMG_SIZE, yolov5.IMG_SIZE)
                .to(device)
                .type_as(next(model.parameters()))
            )
        ###############################################################################

        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

    load_model_list[model_name] = model

    return model, device
