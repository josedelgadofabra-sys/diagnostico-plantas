import os

import gradio as gr
import tensorflow as tf
from huggingface_hub import hf_hub_download


model_repo = os.environ.get("HF_MODEL_REPO")
if not model_repo:
    raise RuntimeError("Set the HF_MODEL_REPO environment variable to your Hugging Face model repo.")

model_path = hf_hub_download(
    repo_id=model_repo,
    filename=os.environ.get("HF_MODEL_FILENAME", "modelo_diagnostico_plantas.keras"),
    token=os.environ.get("HF_TOKEN") or None,
)
model = tf.keras.models.load_model(model_path, compile=False)
image_height, image_width = (int(size) for size in model.input_shape[1:3])


def predict(image):
    image = image.convert("RGB").resize((image_width, image_height))
    image_batch = tf.keras.utils.img_to_array(image)[None, ...]
    probability_unhealthy = float(model.predict(image_batch, verbose=0)[0][0])
    return {
        "Healthy": 1.0 - probability_unhealthy,
        "Unhealthy": probability_unhealthy,
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Subir fotografía de la hoja"),
    outputs=gr.Label(num_top_classes=2, label="Resultado"),
    title="Diagnóstico de plantas con CNN",
    description="Clasifica la hoja como saludable o no saludable.",
)

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", "10000")),
)
