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
    page_title="Sickle Cell Detection",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0b0f19;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 35px 10px 20px 10px;
    text-align: center;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 18px;
    color: #aeb7c7;
    margin-bottom: 25px;
}

.card {
    background: #151a26;
    border: 1px solid #252c3b;
    border-radius: 18px;
    padding: 25px;
    margin-top: 15px;
}

.result-positive {
    background: rgba(220, 38, 38, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.45);
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}

.result-negative {
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.45);
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}

.result-title {
    font-size: 32px;
    font-weight: 800;
}

.confidence {
    font-size: 20px;
    margin-top: 8px;
}

.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-top: 35px;
    margin-bottom: 15px;
}

.info-box {
    background: #151a26;
    border-radius: 15px;
    padding: 20px;
    border: 1px solid #252c3b;
}

.footer {
    text-align: center;
    color: #7f899b;
    font-size: 13px;
    margin-top: 40px;
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

        response = requests.get(url, stream=True)
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
            shape = layer.output.shape

            if len(shape) == 4:
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
        tf.reduce_max(heatmap)
        + 1e-8
    )

    return heatmap.numpy()


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🩸 Sickle Cell Disease Detection
</div>

<div class="hero-subtitle">
Deep Learning-Based Classification from Blood-Smear Images
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# MODEL INFO
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "InceptionV3"
    )

with col2:
    st.metric(
        "Input Size",
        "224 × 224"
    )

with col3:
    st.metric(
        "Explainability",
        "Grad-CAM"
    )


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">Upload Blood-Smear Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Select a blood-smear image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "tif",
        "tiff"
    ],
    label_visibility="collapsed"
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
        "Analyzing blood-smear image..."
    ):

        probability = float(
            model.predict(
                image_array,
                verbose=0
            )[0][0]
        )

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    if probability >= 0.5:

        result = "Sickle Cell"
        confidence = probability * 100

        st.markdown(
            f"""
            <div class="result-positive">
                <div class="result-title">
                    🔴 Sickle Cell
                </div>
                <div class="confidence">
                    Model confidence: <b>{confidence:.2f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        result = "Normal"
        confidence = (
            1 - probability
        ) * 100

        st.markdown(
            f"""
            <div class="result-negative">
                <div class="result-title">
                    🟢 Normal
                </div>
                <div class="confidence">
                    Model confidence: <b>{confidence:.2f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Prediction Details</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write("Sickle Cell Probability")

        st.progress(
            min(probability, 1.0)
        )

        st.write(
            f"**{probability * 100:.2f}%**"
        )

    with col2:

        st.write("Normal Probability")

        normal_probability = (
            1 - probability
        )

        st.progress(
            min(normal_probability, 1.0)
        )

        st.write(
            f"**{normal_probability * 100:.2f}%**"
        )

    # -----------------------------------------------------
    # IMAGE + GRAD-CAM
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Explainability — Grad-CAM</div>',
        unsafe_allow_html=True
    )

    with st.spinner(
        "Generating Grad-CAM visualization..."
    ):

        heatmap = make_gradcam(
            image_array
        )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### Original Image"
        )

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.markdown(
            "### Grad-CAM Heatmap"
        )

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.imshow(
            image
        )

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

    st.info(
        "Grad-CAM highlights image regions that "
        "contributed to the model's prediction."
    )


# =========================================================
# ABOUT
# =========================================================

st.markdown(
    '<div class="section-title">About the Model</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">

<b>Architecture:</b> InceptionV3<br><br>

<b>Task:</b> Binary classification of blood-smear images<br><br>

<b>Input:</b> RGB blood-smear image resized to 224 × 224 pixels<br><br>

<b>Explainability:</b> Gradient-weighted Class Activation Mapping (Grad-CAM)

</div>
""", unsafe_allow_html=True)


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown(
    '<div class="section-title">Important</div>',
    unsafe_allow_html=True
)

st.warning(
    "This website is a research prototype. "
    "Its predictions should not be used as a substitute "
    "for professional medical diagnosis."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

Sickle Cell Disease Detection • Deep Learning Research Prototype

</div>
""", unsafe_allow_html=True)
