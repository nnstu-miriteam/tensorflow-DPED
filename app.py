import gradio as gr
from test_image import process_image
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str, default='models')
path_to_model = parser.parse_args().model_path

models = [str(i.stem) for i in Path(path_to_model).glob('*iteration_*.ckpt.index')]
print(models)
class DPEDModel:
    def __init__(self):
        self.model = ''

    def load_model(self, model):
        self.model = model

    def process_image(self, img):
        image = process_image(img, str(path_to_model) + '/' + str(self.model))
        return image

with gr.Blocks() as app:
    dped = DPEDModel()
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(models, label="Model", value=models[0], interactive=True)
            
    model_dropdown.change(dped.load_model, inputs=model_dropdown)
    with gr.Row(equal_height=True):
        input_image = gr.Image(label="Input")
        output_image = gr.Image(label="Output")
        
    button = gr.Button("Process!", variant="primary")

    button.click(dped.process_image, inputs=input_image, outputs=output_image)
        
app.launch(server_name="127.0.0.1", server_port=7860, share=False)
