import io
import os
from typing import Any

import cv2
import torch

from vajra import Vajra
from vajra.utils import LOGGER
from vajra.checks import check_requirements
from vajra.utils.downloads import GITHUB_ASSETS_STEMS

torch.classes.__path__ = []

class Inference:
    def __init__(self, **kwargs: Any) -> None:
        check_requirements("streamlit>=1.29.0")
        import streamlit as st

        self.st = st
        self.source = None
        self.img_file_names = []
        self.enable_trk = False
        self.conf = 0.25
        self.iou = 0.45
        self.org_frame=None
        self.ann_frame=None
        self.vid_file_name=None
        self.selected_ind: list[int] = []
        self.model = None

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = None
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]
        
        LOGGER.info(f"Vayuvahana Technologies Solutions: {self.temp_dict}")
    
    def web_ui(self) -> None:
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""

        main_title_cfg = """<div><h1 style="color:#111F68; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;"> Vayuvahana Technologies Vajra Streamlit Application</h1></div>"""

        sub_title_cfg = """<div><h5 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif;
        margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam, videos, and images
        with the power of Vayuvahana Technologies VajraV1! 🚀</h5></div>"""

        self.st.set_page_config(page_title="Vayuvahana Technologies Streamlit App", layout="wide")
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self) -> None:
        with self.st.sidebar:
            logo = "https://github.com/NamanMakkar/VayuAI/vajra/assets/Vayuvahana_log.png"
            self.st.image(logo, width=250)
        
        self.st.sidebar.title("User Configuration")
        self.source = self.st.sidebar.selectbox(
            "Source",
            ("webcam", "video", "image"),
        )
        if self.source in ["webcam", "video"]:
            self.enable_trk = self.st.sidebar.radio("Enable Tracking", ("Yes", "No")) == "Yes"
        self.conf = float(
            self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01)
        )
        self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))

        if self.source != "image":
            col1, col2 = self.st.columns(2)
            self.org_frame = col1.empty()
            self.ann_frame = col2.empty()

    def source_upload(self) -> None:
        from vajra.dataset.utils import IMG_FORMATS, VID_FORMATS

        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Upload Video File", type=VID_FORMATS)
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())
                with open("vayuai.mp4", "wb") as out:
                    out.write(g.read())
                self.vid_file_name = "vayuai.mp4"
        elif self.source == "webcam":
            self.vid_file_name = 0
        elif self.source == "image":
            import tempfile

            if imgfiles := self.st.sidebar.file_uploader(
                "Upload Image Files", type=IMG_FORMATS, accept_multiple_files=True
            ):
                for imgfile in imgfiles:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{imgfile.name.split('.')[-1]}") as tf:
                        tf.write(imgfile.read())
                        self.img_file_names.append({"path": tf.name, "name": imgfile.name})

    def configure(self) -> None:
        M_ORD, T_ORD = ["vajra-v1-nano", "vajra-v1-small", "vajra-v1-medium", "vajra-v1-large", "vajra-v1-xlarge"], ["-det", "-seg", "-pose", "-obb", "-cls"]
        available_models = sorted(
            [
                x for x in GITHUB_ASSETS_STEMS if any(x.startswith(b) for b in M_ORD)
            ],
            key=lambda x: (M_ORD.index(x[:13]), T_ORD.index(x[:13].lower() or "")),
        )

        if self.model_path:
            available_models.insert(0, self.model_path)
        selected_model = self.st.sidebar.selectbox("Model", available_models)

        with self.st.spinner("Model is downloading..."):
            if selected_model.endswith((".pt", ".onnx", ".torchscript", ".engine")) or any(
                fmt in selected_model for fmt in ("openvino_model", "rknn_model")
            ):
                model_path = selected_model
            else:
                model_path = f"{selected_model.lower()}.pt"
            self.model = Vajra(model_path)
            class_names = list(self.model.names.values())
        self.st.success("Model loaded successfully!")

        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]

        if not isinstance(self.selected_ind, list):
            self.selected_ind = list(self.selected_ind)

    def image_inference(self) -> None:
        for img_info in self.img_file_names:
            img_path = img_info["path"]
            image = cv2.imread(img_path)
            if image is not None:
                self.st.markdown(f"### Processed: {img_info['name']}")
                col1, col2 = self.st.columns(2)
                with col1:
                    self.st.image(image, channels="BGR", caption="Original Shape")
                results = self.model(image, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                annotated_image = results[0].plot()
                with col2:
                    self.st.image(annotated_image, channels="BGR", caption="Predicted Image")
                try:
                    os.unlink(img_path)
                except FileNotFoundError:
                    pass
            else:
                self.st.error("Could not load the uploaded image.")

    def inference(self) -> None:
        self.web_ui()
        self.sidebar()
        self.source_upload()
        self.configure()

        if self.st.sidebar.button("Start"):
            if self.source == "image":
                if self.img_file_names:
                    self.image_inference()
                else:
                    self.st.info("Please upload an image file to perform inference.")
                return
            
            stop_button = self.st.sidebar.button("Stop")
            cap = cv2.VideoCapture(self.vid_file_name)
            if not cap.isOpened():
                self.st.error("Could not open webcam or video source.")
                return
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    self.st.warning("Failed to read frame from webcam. Please verify the webcam is connected properly.")
                    break
            
                if self.enable_trk:
                    results = self.model.track(
                        frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                    )
                else:
                    results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)

                annotated_frame = results[0].plot()

                if stop_button:
                    cap.release()
                    self.st.stop()
                
                self.org_frame.image(frame, channels="BGR", caption="Original Frame")
                self.ann_frame.image(annotated_frame, channels="BGR", caption="Predicted Frame")
        
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys

    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None
    Inference(model=model).inference()

