# Pneumonia Detection — Streamlit Deployment

This folder deploys your **VGG16_finetuned** model (VGG16_finetuned_final.keras
from `SAVE_DIR` in your Colab notebook) as a Streamlit web app with Grad-CAM.

## Files
- `app.py` — the Streamlit app (upload image → prediction + Grad-CAM overlay)
- `requirements.txt` — Python dependencies
- `packages.txt` — system libs needed by OpenCV on Streamlit Cloud

## Step 1 — Get the model file out of Google Drive

Since your model is saved on Google Drive, you have two options:

### Option A (simplest): Download and commit the file, if small enough
1. In Google Drive, go to `pneumonia_thesis_models/VGG16_finetuned_final.keras`
   and download it to your computer.
2. GitHub blocks files over 100MB in a normal push. If your file is under that,
   rename it to `model.keras` and put it in this same folder, then skip to Step 2.

### Option B (recommended — model is usually >100MB): Stream it from Drive at runtime
1. In Google Drive, right-click the model file → **Share** → **Anyone with the link**.
2. Copy the file ID from the share link:
   `https://drive.google.com/file/d/`**`THIS_PART_IS_THE_ID`**`/view`
3. Open `app.py` and paste it into:
   ```python
   GDRIVE_FILE_ID = "paste_your_id_here"
   ```
   The app will automatically download the model the first time it runs
   (using `gdown`), then reuse the cached copy.

## Step 2 — Push this folder to GitHub
```bash
cd pneumonia_app
git init
git add .
git commit -m "Pneumonia detection Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## Step 3 — Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app** → pick your repo → branch `main` → main file `app.py`.
3. Click **Deploy**. First deploy takes a few minutes (TensorFlow install + model download).

## Step 4 — Test
Upload a chest X-ray (`.jpg`/`.png`) from the Kaggle test set — you should see
the NORMAL/PNEUMONIA prediction, confidence, and a Grad-CAM heatmap overlay.

## Notes
- Model expects 224×224 RGB input, pixel values scaled to [0,1] — same
  preprocessing as training (`ImageDataGenerator(rescale=1.0/255)`).
- Class mapping: `0 = NORMAL`, `1 = PNEUMONIA` (matches `train_gen.class_indices`).
- `tensorflow-cpu` is used instead of `tensorflow` to keep the deploy lighter —
  fine for inference, you don't need GPU for single-image predictions.
- One thing worth fixing before pushing to GitHub: your notebook's Cell 1 has
  your Kaggle API token hardcoded in plain text — if this notebook is ever
  made public, revoke/regenerate that token on Kaggle first.
