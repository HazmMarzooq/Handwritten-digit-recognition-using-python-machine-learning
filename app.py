"""Streamlit UI: Image -> Grayscale -> 8x8 matrix -> 64 features -> MLP -> digit."""
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from model import load_or_train, predict_digit
from preprocessing import preprocess_image

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="🔢", layout="wide")


@st.cache_resource(show_spinner="Training the MLP on the digits dataset (first run only)...")
def get_model():
    return load_or_train()


def enlarge(arr, size=256):
    """Upscale with nearest-neighbour so individual pixels stay visible."""
    arr = np.asarray(arr, dtype=float)
    arr = (arr / max(arr.max(), 1) * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.Resampling.NEAREST)


bundle = get_model()
model = bundle["model"]

st.title("🔢 Handwritten Digit Recognition")
st.caption("Image → Grayscale → 8×8 Pixel Matrix → NumPy Array (64 features) → MLP → Digit (0–9)")

with st.sidebar:
    st.header("Model")
    st.write("scikit-learn `MLPClassifier` (128 → 64 hidden units)")
    st.metric("Test accuracy", f"{bundle['test_accuracy']:.1%}")
    st.write(f"Train samples: {bundle['n_train']}  \nTest samples: {bundle['n_test']}")
    st.header("Preprocessing options")
    auto_invert = st.checkbox("Auto-invert (dark ink on light paper)", value=True)
    crop = st.checkbox("Crop & centre the digit", value=True)

uploaded = st.file_uploader("Upload an image of ONE handwritten digit",
                            type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("Upload a PNG/JPG/JPEG with a single digit (0–9). "
            "Best results: a thick dark digit on a plain light background.")
    st.stop()

try:
    r = preprocess_image(uploaded, auto_invert=auto_invert, crop=crop)
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Could not read the image: {e}")
    st.stop()

# ---- Step 1-2: images -----------------------------------------------------
st.subheader("1. Image preprocessing")
c1, c2, c3, c4 = st.columns(4)
c1.image(r.original, caption="Original image", use_container_width=True)
c2.image(r.gray, caption=f"Grayscale{' (inverted)' if r.inverted else ''}",
         use_container_width=True)
c3.image(r.thresholded, caption=f"After thresholding (Otsu = {r.threshold})",
         use_container_width=True)
c4.image(enlarge(r.matrix), caption="Processed 8×8 image (enlarged)",
         use_container_width=True)

# ---- Step 3: numeric representation ---------------------------------------
st.subheader("2. Pixel matrix → NumPy array")
m1, m2 = st.columns([1, 1])
with m1:
    st.markdown("**8×8 pixel matrix** (0 = background, 16 = brightest ink)")
    st.dataframe(pd.DataFrame(r.matrix), use_container_width=True)
    st.caption(f"`matrix.shape = {r.matrix.shape}`")
with m2:
    st.markdown("**Flattened array** (64 features fed to the model)")
    st.code(np.array2string(r.features, separator=", ", max_line_width=60), language="python")
    st.caption(f"`features.shape = {r.features.shape}`")

# ---- Step 4: prediction ---------------------------------------------------
st.subheader("3. Prediction from the trained MLP")
digit, confidence, probs = predict_digit(model, r.features)
p1, p2 = st.columns([1, 2])
p1.metric("Predicted digit", digit)
p1.metric("Confidence", f"{confidence:.1%}")
if confidence < 0.6:
    p1.warning("Low confidence – try a clearer, thicker, centred digit.")
p2.markdown("**Probability for each digit**")
p2.bar_chart(pd.DataFrame({"probability": probs}, index=range(10)))
