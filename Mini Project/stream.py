import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import time

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="GreenLeaf AI | Plant Health",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "plant_disease_cnn7_best.keras"
IMG_SIZE = 224

LABELS = [
    "Apple - Apple Scab", "Apple - Black Rot", "Apple - Cedar Apple Rust", "Apple - Healthy",
    "Blueberry - Healthy", "Cherry - Powdery Mildew", "Cherry - Healthy",
    "Corn (Maize) - Cercospora Leaf Spot / Gray Leaf Spot", "Corn (Maize) - Common Rust",
    "Corn (Maize) - Northern Leaf Blight", "Corn (Maize) - Healthy",
    "Grape - Black Rot", "Grape - Black Measles (Esca)", "Grape - Leaf Blight (Isariopsis Leaf Spot)", "Grape - Healthy",
    "Orange - Citrus Greening (Huanglongbing)", "Peach - Bacterial Spot", "Peach - Healthy",
    "Bell Pepper - Bacterial Spot", "Bell Pepper - Healthy",
    "Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy",
    "Raspberry - Healthy", "Soybean - Healthy", "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch", "Strawberry - Healthy",
    "Tomato - Bacterial Spot", "Tomato - Early Blight", "Tomato - Late Blight",
    "Tomato - Leaf Mold", "Tomato - Septoria Leaf Spot", "Tomato - Spider Mites (Two-Spotted Spider Mite)",
    "Tomato - Target Spot", "Tomato - Yellow Leaf Curl Virus", "Tomato - Mosaic Virus", "Tomato - Healthy"
]

# -------------------- Styling --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --forest: #174C36;
    --green: #26734D;
    --mint: #E7F3EA;
    --ink: #182A22;
    --muted: #6B7D73;
    --line: #DCE7DE;
    --paper: #F6F8F5;
}
.stApp { background: var(--paper); color: var(--ink); font-family: 'DM Sans', sans-serif; }
[data-testid="stHeader"] { background: rgba(246,248,245,.92); }
.block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1440px; }
h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: var(--ink); letter-spacing: -0.035em; }
p, label, .stMarkdown { color: var(--ink); }
[data-testid="stSidebar"] { background: #173F30; border-right: 1px solid #244E3D; }
[data-testid="stSidebar"] * { color: #F4FAF5 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.10); border-radius: 10px; padding: .55rem .7rem; margin-bottom: .35rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.13); }
.brand-mark { width: 46px; height: 46px; border-radius: 14px; display:flex; align-items:center; justify-content:center; background:#DFF2E3; color:#174C36; font-size:25px; }
.brand-row { display:flex; align-items:center; gap:13px; margin:.4rem 0 2rem 0; }
.brand-name { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.35rem; color:white; letter-spacing:-.04em; }
.brand-tag { color:#C6DCCB; font-size:.78rem; margin-top:2px; }
.eyebrow { text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:700; color:#26734D; margin-bottom:.45rem; }
.hero { background:linear-gradient(115deg,#E5F2E7 0%,#F0F7F0 62%,#E8F1E8 100%); border:1px solid #D8E8D9; border-radius:24px; padding:2rem 2.1rem; margin-bottom:1.5rem; }
.hero-title { font-family:'Manrope',sans-serif; font-weight:800; letter-spacing:-.05em; font-size:clamp(1.8rem,3vw,2.65rem); line-height:1.12; color:#173F30; margin:0 0 .6rem 0; }
.hero-copy { font-size:1rem; line-height:1.7; color:#526A5B; max-width:700px; margin:0; }
.pill { display:inline-block; background:#D6EBDD; color:#1D6744; border-radius:999px; padding:.38rem .7rem; font-size:.75rem; font-weight:700; margin-bottom:1rem; }
.panel { background:#FFFFFF; border:1px solid var(--line); border-radius:18px; padding:1.25rem 1.35rem; box-shadow:0 6px 22px rgba(25,64,43,.045); }
.panel-title { font-family:'Manrope',sans-serif; font-size:1.12rem; font-weight:800; color:#1B3527; margin-bottom:.35rem; }
.panel-copy { color:#708075; font-size:.9rem; line-height:1.6; }
.metric-label { color:#6B7D73; font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { color:#174C36; font-family:'Manrope',sans-serif; font-size:1.5rem; font-weight:800; margin-top:.3rem; }
.result-box { background:#F0F7F1; border:1px solid #D6E8D9; border-radius:16px; padding:1.2rem; }
.result-kicker { font-size:.75rem; text-transform:uppercase; letter-spacing:.1em; font-weight:700; color:#558063; }
.result-name { font-family:'Manrope',sans-serif; font-size:1.35rem; font-weight:800; color:#173F30; line-height:1.3; margin:.45rem 0 .7rem; }
.small-note { font-size:.82rem; line-height:1.55; color:#708075; }
div.stButton > button { border-radius:10px; font-weight:700; border:1px solid #C9DCCB; min-height:2.7rem; }
div.stButton > button[kind="primary"] { background:#26734D; color:white; border-color:#26734D; }
div.stButton > button[kind="primary"]:hover { background:#174C36; border-color:#174C36; color:white; }
[data-testid="stFileUploader"] { background:white; border:1.5px dashed #B7D0BB; border-radius:14px; padding:.5rem; }
[data-testid="stProgress"] > div > div { background:#26734D; }
hr { border-color:#DFE8E0; }
footer { visibility:hidden; }
#MainMenu { visibility:hidden; }
@media (max-width: 800px) {
  .block-container { padding-top:1rem; }
  .hero { padding:1.35rem; border-radius:18px; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict(image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]
    idx = int(top_indices[0])
    return LABELS[idx], float(probabilities[idx] * 100), [
        (LABELS[int(i)], float(probabilities[int(i)] * 100)) for i in top_indices
    ]

def health_state(label):
    return "Healthy" if "healthy" in label.lower() else "Potential disease detected"

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model file '{MODEL_PATH}'. Make sure it is beside stream.py.")
    st.exception(exc)
    st.stop()

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("""
      <div class="brand-row">
        <div class="brand-mark">🌿</div>
        <div><div class="brand-name">GreenLeaf AI</div>
        <div class="brand-tag">Plant health intelligence</div></div>
      </div>
      """, unsafe_allow_html=True)
    st.markdown("### Workspace")
    mode = st.radio("Choose an analysis mode", ["Upload leaf image", "Live camera"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("<p style='color:#D3E5D7 !important;font-size:.88rem;line-height:1.65'>An image-based deep learning prototype for classifying 38 plant health and disease categories.</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='small-note' style='color:#C6DCCB !important'>MODEL<br><b style='color:white'>7-layer CNN</b><br><br>INPUT SIZE<br><b style='color:white'>224 × 224 RGB</b><br><br>OUTPUT CLASSES<br><b style='color:white'>38 categories</b></div>", unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown("""
<div class="hero">
  <div class="pill">DEEP LEARNING · PLANT HEALTH</div>
  <div class="hero-title">Understand your plant.<br>Protect what grows.</div>
  <p class="hero-copy">Upload a leaf image to explore the model's prediction, confidence score, and top alternative classifications.</p>
</div>
""", unsafe_allow_html=True)

# Summary metrics
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown('<div class="panel"><div class="metric-label">Model architecture</div><div class="metric-value">7-layer CNN</div><div class="small-note">Convolutional neural network</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="panel"><div class="metric-label">Supported classes</div><div class="metric-value">38</div><div class="small-note">Plant health categories</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="panel"><div class="metric-label">Image input</div><div class="metric-value">224 × 224</div><div class="small-note">RGB image preprocessing</div></div>', unsafe_allow_html=True)

st.write("")
if mode == "Upload leaf image":
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown('<div class="panel-title">01 · Leaf image</div><div class="panel-copy">Choose a clear image where the leaf is visible.</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload JPG or PNG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded:
            try:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption="Image submitted for analysis", use_container_width=True)
            except Exception:
                st.error("This image could not be opened. Please try a JPG or PNG file.")
                image = None
        else:
            image = None
            st.markdown('<div class="panel-copy" style="padding:1rem 0">Supported formats: JPG and PNG · Best with clear, well-lit leaf photographs.</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel-title">02 · Analysis results</div><div class="panel-copy">Predictions generated by the saved CNN model.</div>', unsafe_allow_html=True)
        if image is not None:
            with st.spinner("Analyzing leaf image…"):
                label, confidence, top3 = predict(image)
                time.sleep(0.25)
            state = health_state(label)
            st.markdown(f"""
              <div class="result-box">
                <div class="result-kicker">Primary prediction</div>
                <div class="result-name">{label}</div>
                <div class="metric-label">Model score · {confidence:.2f}%</div>
              </div>
            """, unsafe_allow_html=True)
            st.progress(min(max(confidence / 100, 0.0), 1.0))
            st.caption(f"Classification: {state}. The score is the model's softmax output, not a calibrated probability of real-world disease.")
            with st.expander("View top 3 model predictions", expanded=True):
                for rank, (candidate, score) in enumerate(top3, start=1):
                    st.write(f"**{rank}. {candidate}** — {score:.2f}%")
                    st.progress(min(max(score / 100, 0.0), 1.0))
        else:
            st.markdown('<div class="panel" style="background:#F8FBF8"><div class="panel-title">Ready for analysis</div><p class="panel-copy">Your prediction and top alternatives will appear here after you upload an image.</p></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="panel-title">Live camera analysis</div><div class="panel-copy">Use your device camera to analyze frames. Camera access depends on your browser and local environment.</div>', unsafe_allow_html=True)
    st.info("For a reliable live camera experience, this implementation uses Streamlit's camera input. Capture a frame to analyze it.")
    camera_image = st.camera_input("Capture a leaf image")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Captured leaf image", use_container_width=True)
        with st.spinner("Analyzing captured image…"):
            label, confidence, top3 = predict(image)
        st.markdown(f"""
          <div class="result-box">
            <div class="result-kicker">Primary prediction</div>
            <div class="result-name">{label}</div>
            <div class="metric-label">Model score · {confidence:.2f}%</div>
          </div>
        """, unsafe_allow_html=True)
        st.progress(min(max(confidence / 100, 0.0), 1.0))
        with st.expander("View top 3 model predictions", expanded=True):
            for rank, (candidate, score) in enumerate(top3, start=1):
                st.write(f"**{rank}. {candidate}** — {score:.2f}%")
                st.progress(min(max(score / 100, 0.0), 1.0))

st.markdown("---")
st.markdown("<div class='small-note'>GreenLeaf AI · Academic deep learning prototype. Predictions are model outputs and should not replace expert agricultural assessment. Performance on real-world field images may differ from performance on curated datasets.</div>", unsafe_allow_html=True)
