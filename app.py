import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SickleScan",
    page_icon="🩸",
    layout="wide"
)

# =========================================================
# CUSTOM STYLE
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 5% 5%, rgba(124,58,237,.18), transparent 30%),
        radial-gradient(circle at 95% 5%, rgba(219,39,119,.14), transparent 30%),
        #080a12;
}

.block-container {
    max-width: 1150px;
    padding-top: 25px;
    padding-bottom: 50px;
}

/* Headings */

h1 {
    font-weight: 850 !important;
}

h2, h3 {
    font-weight: 750 !important;
}

/* Hero */

.hero-text {
    text-align: center;
    padding: 35px 10px 20px;
}

.hero-subtitle {
    color: #a1a1aa;
    font-size: 18px;
    line-height: 1.6;
}

/* Info cards */

.info-card {
    background: rgba(24,27,40,.75);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 18px;
    padding: 22px;
    min-height: 155px;
}

.info-card h3 {
    margin-top: 5px;
}

.info-card p {
    color: #a1a1aa;
    line-height: 1.5;
}

/* Upload area */

.upload-box {
    background: rgba(24,27,40,.85);
    border: 1px solid rgba(192,132,252,.22);
    border-radius: 22px;
    padding: 25px;
}

/* Result boxes */

.result-normal {
    background: rgba(34,197,94,.10);
    border: 1px solid rgba(74,222,128,.30);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
}

.result-sickle {
    background: rgba(239,68,68,.10);
    border: 1px solid rgba(248,113,113,.35);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
}

.big-result {
    font-size: 32px;
    font-weight: 800;
}

.result-score {
    font-size: 36px;
    font-weight: 850;
    margin-top: 8px;
}

/* Footer */

.footer {
    text-align: center;
    color: #71717a;
    font-size: 13px;
    margin-top: 50px;
    padding-top: 25px;
    border-top: 1px solid rgba(255,255,255,.06);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_path = "/tmp/InceptionV3_UCL_Final.keras"

    if not os.path.exists(model_path):

        url = (
            "https://huggingface.co/"
            "abhijeetgour12/sickle-cell-inceptionv3/"
            "resolve/main/InceptionV3_UCL_Final.keras"
            "?download=true"
        )

        response = requests.get(
            url,
            stream=True
        )

        response.raise_for_status()

        with open(model_path, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    f.write(chunk)

    return tf.keras.models.load_model(model_path)


model = load_model()


# =========================================================
# GRAD-CAM
# =========================================================

def make_gradcam(image_array):

    backbone = model.get_layer("inception_v3")

    target_layer = None

    for layer in reversed(backbone.layers):

        try:
            if len(layer.output.shape) == 4:
                target_layer = layer
                break
        except:
            continue

    grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            target_layer.output,
            backbone.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, backbone_features = grad_model(
            image_array,
            training=False
        )

        x = model.get_layer(
            "global_average_pooling2d"
        )(backbone_features)

        x = model.get_layer(
            "dropout"
        )(x, training=False)

        prediction = model.get_layer(
            "dense"
        )(x)

        loss = prediction[:, 0]

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap /= (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


# =========================================================
# HEADER
# =========================================================

st.markdown("## 🩸 SICKLESCAN")

st.markdown("""
<div class="hero-text">

<h1>AI-Powered Blood Smear Analysis</h1>

<p class="hero-subtitle">
Explore how deep learning can be used to analyze blood-smear
images and classify them as Normal or Sickle Cell.
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# ABOUT SICKLE CELL
# =========================================================

st.markdown("## 🩸 What is Sickle Cell Disease?")

st.write(
    """
Sickle Cell Disease (SCD) is a genetic blood disorder in which
red blood cells can develop an abnormal sickle-like shape.
These changes can affect the normal movement of blood through
blood vessels.

Blood-smear images provide visual information about the
appearance of red blood cells. Deep-learning models can be
trained to identify visual patterns within these images.
"""
)


# =========================================================
# WHAT SICKLESCAN DOES
# =========================================================

st.markdown("## ✨ What SickleScan Does")

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown("""
    <div class="info-card">

    <h3>🔬 Image Analysis</h3>

    <p>
    Upload a blood-smear image and let the trained
    deep-learning model analyze its visual features.
    </p>

    </div>
    """, unsafe_allow_html=True)


with c2:

    st.markdown("""
    <div class="info-card">

    <h3>🧠 AI Classification</h3>

    <p>
    The InceptionV3 model generates a prediction for
    Normal or Sickle Cell along with a probability score.
    </p>

    </div>
    """, unsafe_allow_html=True)


with c3:

    st.markdown("""
    <div class="info-card">

    <h3>🔥 Visual Explanation</h3>

    <p>
    Grad-CAM highlights image regions that contributed
    to the model's prediction.
    </p>

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# MODEL PIPELINE
# =========================================================

st.markdown("## ⚙️ About the AI Model")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Architecture", "InceptionV3")

with col2:
    st.metric("Input", "224 × 224")

with col3:
    st.metric("Task", "Binary")

with col4:
    st.metric("Explainability", "Grad-CAM")


# =========================================================
# UPLOAD
# =========================================================

st.markdown("## 🔬 Analyze a Blood-Smear Image")

st.write(
    "Upload an image below to start the AI analysis."
)

uploaded_file = st.file_uploader(
    "Choose a blood-smear image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "tif",
        "tiff"
    ]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Same preprocessing used during training
    resized = image.resize(
        (224, 224)
    )

    image_array = np.array(
        resized
    ).astype("float32")

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    with st.spinner(
        "🧠 AI is analyzing the image..."
    ):

        probability = float(
            model.predict(
                image_array,
                verbose=0
            )[0][0]
        )

    normal_probability = 1 - probability


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown("## 📊 Analysis Result")

    if probability >= 0.5:

        confidence = probability * 100

        st.markdown(
            '<div class="result-sickle">',
            unsafe_allow_html=True
        )

        st.markdown("### 🔴 Sickle Cell")

        st.write("Model prediction")

        st.markdown(
            f'<div class="result-score">'
            f'{confidence:.1f}%'
            f'</div>',
            unsafe_allow_html=True
        )

        st.caption("Prediction score")

        st.markdown("</div>", unsafe_allow_html=True)

    else:

        confidence = normal_probability * 100

        st.markdown(
            '<div class="result-normal">',
            unsafe_allow_html=True
        )

        st.markdown("### 🟢 Normal")

        st.write("Model prediction")

        st.markdown(
            f'<div class="result-score">'
            f'{confidence:.1f}%'
            f'</div>',
            unsafe_allow_html=True
        )

        st.caption("Prediction score")

        st.markdown("</div>", unsafe_allow_html=True)


    # =====================================================
    # PROBABILITY
    # =====================================================

    st.markdown("### Prediction Breakdown")

    p1, p2 = st.columns(2)

    with p1:

        st.write("🟢 **Normal Probability**")

        st.progress(
            float(normal_probability)
        )

        st.write(
            f"{normal_probability * 100:.1f}%"
        )

    with p2:

        st.write("🔴 **Sickle Cell Probability**")

        st.progress(
            float(probability)
        )

        st.write(
            f"{probability * 100:.1f}%"
        )


    # =====================================================
    # IMAGE + GRAD CAM
    # =====================================================

    st.markdown("## 🔥 Model Explanation")

    st.write(
        "Grad-CAM provides a visual indication of the regions "
        "that influenced the model's prediction."
    )

    with st.spinner(
        "Generating Grad-CAM..."
    ):

        heatmap = make_gradcam(
            image_array
        )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("Grad-CAM Visualization")

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.imshow(image)

        ax.imshow(
            heatmap,
            cmap="jet",
            alpha=0.45,
            extent=(
                0,
                image.width,
                image.height,
                0
            )
        )

        ax.axis("off")

        st.pyplot(
            fig,
            clear_figure=True
        )


# =========================================================
# MORE INFORMATION
# =========================================================

st.markdown("## ℹ️ More About SickleScan")

with st.expander(
    "View model and project information"
):

    st.write(
        """
        **SickleScan** is an academic deep-learning research
        prototype for blood-smear image classification.

        **Model:** InceptionV3

        **Classification:** Normal vs Sickle Cell

        **Input:** RGB blood-smear image resized to 224 × 224 pixels

        **Explainability:** Grad-CAM

        **Purpose:** Academic and research demonstration
        """
    )

    st.warning(
        "This system is a research prototype and should not "
        "be used as a substitute for professional medical diagnosis."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

🩸 <b>SICKLESCAN</b><br>
AI-Powered Blood Smear Analysis<br><br>

Deep Learning Research Project • InceptionV3 • Grad-CAM

</div>
""", unsafe_allow_html=True)
