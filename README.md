# MNIST Classifier + Gradio Demo

A minimal **computer vision** project: train a small CNN on **MNIST** and serve a **Gradio** demo where users can draw a digit or upload an image for prediction.

## Features
- ✅ Tiny CNN (PyTorch) with ~99% test accuracy after a few epochs
- ✅ Clean training loop with checkpointing (`checkpoints/best.pt`)
- ✅ Inference module for easy reuse
- ✅ **Gradio app** with a sketchpad (draw a digit) or image upload
- ✅ Reproducible: fixed seeds, requirements provided

## Quickstart (Local / Colab)
```bash
# 1) Create venv (optional) and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Train (downloads MNIST automatically)
python -m src.train --epochs 5 --batch-size 128 --lr 0.001

# 3) Launch demo
python app/app.py
```

### Gradio App
- **Canvas**: draw a digit (0–9) and press *Predict*.
- **Upload**: upload a 28x28 grayscale PNG/JPG of a digit.

- # Handwritten Digit Recognition

[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Open%20in%20Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/wa1thyy/handwritten-digit-recognition)

A minimal, portfolio-ready computer vision project: train a CNN on MNIST and serve a Gradio demo.


## Repo Structure
```
cv-mnist-gradio/
  src/
    model.py        # CNN
    train.py        # training script (saves best.pt)
    infer.py        # prediction helpers
  app/
    app.py          # Gradio UI
  checkpoints/      # saved models 
  assets/           # screenshots/GIFs 
  requirements.txt
  README.md
  .gitignore
```
## 📸 Demo
Example prediction of digit "8":

<img width="1280" height="719" alt="image" src="https://github.com/user-attachments/assets/92117b8f-4f11-4427-a1c2-f14542f74b74" />


## Results
- After 5 epochs on CPU, expect **~99%** test accuracy (±0.5%). On GPU: faster.

## Notes / Next Steps
- Add confusion matrix & misclassified samples to `assets/`.
- Swap model to **ResNet18** (transfer learning on EMNIST/CIFAR-10) for extra points.
- Deploy the app to **Hugging Face Spaces** or **Streamlit Cloud** and link here.

## License
MIT
