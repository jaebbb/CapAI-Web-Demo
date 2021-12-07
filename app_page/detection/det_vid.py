import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import nvidia_smi
import streamlit as st
import torch

load_model_list = {}
use_model_list = {}


def run_det_vid():
    # side bar
    model_type, conf_slider, iou_slider = frame_selector_ui()

    # model init
    model, device = load_model(model_name=model_type)

    # file upload
    uploaded_vid = st.file_uploader("Upload a video", ["mp4"])

    if uploaded_vid is not None:
        # get uploaded file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_vid.read())
        tvid_path = tfile.name
        vid_path = f"data/{uploaded_vid.name[:-4]}_inference.webm"

        # dvide container into two parts
        _, col1, col2, _ = st.columns([1, 4, 4, 1])
        with col1:
            st.markdown('**<div align="center">Input video</div>**', unsafe_allow_html=True)
            st.video(tvid_path)  # display input image
        with col2:
            st.markdown('**<div align="center">Output video</div>**', unsafe_allow_html=True)
            videobox = st.empty()

        # wait other inference
        while use_model_list[model_type]:
            time.sleep(0.1)
            videobox.warning("Model is in use!\n Please wait...")
        videobox.empty()

        # read video
        vid_org = cv2.VideoCapture(tvid_path)

        # create result video
        end_flag, frame_org = vid_org.read()
        total_frame = vid_org.get(7)
        fourcc = cv2.VideoWriter_fourcc(*"VP80")  # HTML5 - mp4v issue
        out_video = cv2.VideoWriter(
            vid_path, fourcc, 20, (frame_org.shape[1], frame_org.shape[0])
        )

        # inference
        use_model_list[model_type] = True  # semaphore
        try:
            with col2:
                inference_warning = st.warning("Inference...")
                progress_bar = st.progress(0)
            frame_count = 0
            while uploaded_vid is not None:
                frame_count += 1
                end_flag, frame_org = vid_org.read()
                if not end_flag:  # check vid end
                    break
                # inference by frame
                with torch.no_grad():
                    if model_type == "yolov5":
                        from models.yolov5 import yolov5

                        # image preprocessing
                        frame = yolov5.preprocess_image(frame_org, stride=int(model.stride.max()))

                        # inferencem
                        pred = model(frame.to(device))[0]

                        # resize
                        rate = min(383 / frame_org.shape[0], 383 / frame_org.shape[1])
                        if frame_org.shape[0] < 383 or frame_org.shape[1] < 383:
                            frame_org = cv2.resize(
                                frame_org,
                                (int(frame_org.shape[1] * rate), int(frame_org.shape[0] * rate)),
                                interpolation=cv2.INTER_LINEAR,
                            )

                        frame_bboxes = yolov5.draw_image_with_boxes(
                            frame_org, pred, frame.shape[2:], conf=conf_slider, iou=iou_slider
                        )  # get bboxes and labels
                    elif "swin" in model_type:
                        from mmdet.apis import inference_detector

                        # resize
                        rate = min(512 / frame_org.shape[0], 512 / frame_org.shape[1])
                        if frame_org.shape[0] < 512 or frame_org.shape[1] < 512:
                            frame_org = cv2.resize(
                                frame_org,
                                (int(frame_org.shape[1] * rate), int(frame_org.shape[0] * rate)),
                                interpolation=cv2.INTER_LINEAR,
                            )

                        pred = inference_detector(model, frame_org)
                        frame_bboxes = model.show_result(frame_org, pred, score_thr=conf_slider)
                    else:
                        pass
                # save frame in result video
                out_video.write(frame_bboxes)

                # display frame
                with col2:
                    inference_warning.warning("Inference... (%d/%d)" % (frame_count, total_frame))
                    progress_bar.progress(min(frame_count / total_frame, 1.0))
                    if frame_count % 10 == 0:
                        frame_bboxes = cv2.cvtColor(frame_bboxes, cv2.COLOR_BGR2RGB)
                        videobox.image(frame_bboxes)
        except:  # if error occurs
            inference_warning.empty()
            progress_bar.empty()
            videobox.empty()
            videobox.warning("Error occurs!\n Please try again.")
        else:
            # save result video
            # cv2.destroyAllWindows()
            out_video.release()

            with col2:
                inference_warning.empty()
                progress_bar.empty()
                videobox.video(vid_path)  # display result video
        finally:
            use_model_list[model_type] = False  # semaphore
    elif uploaded_vid is None:
        st.info("Check the Video format (e.g. mp4)")


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
    use_model_list[model_name] = False

    return model, device
