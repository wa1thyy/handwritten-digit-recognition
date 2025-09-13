import os, sys
import numpy as np
from PIL import Image, ImageOps
import gradio as gr

# подключаем src/*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.infer import load_model, predict

CKPT = "checkpoints/best.pt"

MODEL, DEVICE = None, None
MODEL_LOADED = False
if os.path.exists(CKPT):
    try:
        MODEL, DEVICE = load_model(CKPT)
        MODEL_LOADED = True
        print("✅ Model loaded from", CKPT)
    except Exception as e:
        print("❌ Failed to load model:", e)

def to_pil(x):
    """Универсально привести вход из Sketchpad/Image к PIL.Image."""
    if x is None:
        raise ValueError("Пустой ввод — нарисуй цифру или загрузи картинку!")

    if isinstance(x, Image.Image):
        return x

    if isinstance(x, dict):
        if "image" in x and x["image"] is not None:
            x = x["image"]
        elif "composite" in x and x["composite"] is not None:
            x = x["composite"]
        elif "layers" in x and x["layers"] is not None:
            x = x["layers"]

    if isinstance(x, np.ndarray):
        if x.dtype != np.uint8:
            x = (x * 255).clip(0, 255).astype("uint8")
        if x.ndim == 2:
            return Image.fromarray(x, mode="L")
        if x.ndim == 3:
            if x.shape[2] == 4:  # RGBA → RGB
                x = x[:, :, :3]
            return Image.fromarray(x)

    raise ValueError(f"Unsupported input type: {type(x)}")

def normalize_digit(img: Image.Image) -> Image.Image:
    """MNIST формат: белая цифра на чёрном фоне."""
    if img.mode != "L":
        img = img.convert("L")
    arr = np.array(img)
    if arr.mean() > 127:  # фон светлый → инвертируем
        img = ImageOps.invert(img)
    return img

def _predict_common(raw):
    if not MODEL_LOADED:
        raise RuntimeError("Модель не загружена. Сначала обучи её: python -m src.train")
    img = to_pil(raw)
    img = normalize_digit(img)
    pred, probs = predict(MODEL, DEVICE, img)
    probs_map = {str(i): float(probs[i]) for i in range(10)}
    return probs_map, int(pred)

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🏀 MNIST Digit Classifier")
    gr.Markdown("1. Обучи модель: `python -m src.train`  \n2. Запусти: `python -m app.app`")

    with gr.Tab("Draw"):
        canvas = gr.Sketchpad(canvas_size=(280, 280))  # numpy или dict
        btn1 = gr.Button("Predict")
        probs1 = gr.Label(num_top_classes=10)
        pred1 = gr.Number(label="Predicted digit", precision=0)
        btn1.click(fn=_predict_common, inputs=canvas, outputs=[probs1, pred1])

    with gr.Tab("Upload"):
        img = gr.Image(type="pil")
        btn2 = gr.Button("Predict")
        probs2 = gr.Label(num_top_classes=10)
        pred2 = gr.Number(label="Predicted digit", precision=0)
        btn2.click(fn=_predict_common, inputs=img, outputs=[probs2, pred2])

if __name__ == "__main__":
    demo.launch()
