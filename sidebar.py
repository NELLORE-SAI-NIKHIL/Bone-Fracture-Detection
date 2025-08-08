import streamlit as st
import random

class Sidebar():
    def __init__(self) -> None:
        
        
        self.model_name = None
        self.confidence_threshold = None
        self.title_img = 'images\medical.jpg'
        
        self._titleimage()
        self._model()
        self._confidencethreshold()
        
        
    def _titleimage(self):
        st.sidebar.image(self.title_img)
        
    def _model(self):

        st.sidebar.markdown('## Choose model')

        self.model_name = st.sidebar.selectbox(
            label = 'Select Model',
            options = [
                'FastRCNN with ResNet',
                'VGG16',
                "Yolov11",
                "yolov10"
            ],
            index = 0,
            key = 'model_name'
        )

        
        
    def _confidencethreshold(self):
        
        ##st.sidebar.markdown('## Step 2: Set a Threshold')
        ##self.confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.00,1.00,0.5,0.01)
        self.confidence_threshold=0
        
        
        
        
        
        
        
        