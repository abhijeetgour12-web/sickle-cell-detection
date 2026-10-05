import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
import os

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="SickleScan",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(139,92,246,0.18), transparent 28%),
        radial-gradient(circle at 90% 15%, rgba(236,72,153,0.14), transparent 25%),
        #090b13;
    color: #f8fafc;
}

.block-container {
    max-width: 1150px;
    padding-top: 25px;
}

/* Remove default top padding */
header {
    background: transparent !important;
}

/* Main title */
.logo {
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 1px;
}

.logo span {
    color: #c084fc;
}

/* Hero */
.hero {
    text-align: center;
    padding: 55px 10px 35px;
}

.hero-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 50px;
    background: rgba(168,85,247,0.13);
    border: 1px solid rgba(192,132,252,0.25);
    color: #d8b4fe;
    font-size: 14px;
    margin-bottom: 20px;
}

.hero h1 {
    font-size: 58px;
    line-height: 1.05;
    font-weight: 850;
    margin: 0;
    background: linear-gradient(
        90deg,
        #f8fafc,
        #d8b4fe,
        #f9a8d4
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #a1a1aa;
    font-size: 18px;
    margin-top: 18px;
}

/* Upload card */
.upload-card {
    background: rgba(24, 27, 40, 0.88);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 30px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.30);
}

/* Feature cards */
.feature {
    background: rgba(24, 27, 40, 0.72);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    height: 100%;
}

.feature-icon {
    font-size: 28px;
}

.feature-title {
    font-weight: 700;
    margin-top: 8px;
}

.feature-text {
    color: #9295a1;
    font-size: 13px;
    margin-top: 5px;
}

/* Result */
.result-card {
    border-radius: 22px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
}

.positive {
    background: rgba(239,68,68,0.10);
    border: 1px solid rgba(248,113,113,0.35);
}

.negative {
    background: rgba(34,197,94,0.09);
    border: 1px solid rgba(74,222,128,0.30);
}

.result-icon {
    font-size: 42px;
}

.result-name {
    font-size: 32px;
    font-weight: 800;
    margin-top: 5px;
}

.result-confidence {
    color: #a1a1aa;
    margin-top: 5px;
}

/* Section heading */
.section {
    font-size: 27px;
    font-weight: 800;
    margin-top: 42px;
    margin-bottom: 18px;
}

/* How it works */
.step {
    text-align: center;
    padding: 20px;
}

.step-number {
    width: 42px;
    height: 42px;
    line-height: 42px;
    border-radius: 50%;
    margin: auto;
    background: linear-gradient(
        135deg,
        #7c3aed,
        #db2777
    );
    font-weight: 800;
}

.step-title {
    font-weight: 700;
    margin-top: 12px;
}

.step-text {
    color: #8f93a1;
    font-size: 13px;
}

/* Info */
.info {
    background: rgba(24,27,40,0.70);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 24px;
}

.footer {
    text-align: center;
    color: #71717a;
    font-size: 13px;
    padding: 35px 0 15px;
}

.small-note {
    color: #71717a;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
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
# GRAD CAM
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
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


# =========================================================
# NAV
# =========================================================

st.markdown("""
<div class="logo">
🩸 <span>SICKLESCAN</span>
</div>
""", unsafe_allow_html=True)


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<div class="hero-badge">
✦ AI-POWERED BLOOD SMEAR ANALYSIS
</div>

<h1>
See What AI Sees.
</h1>

<p>
Upload a blood-smear image and explore an AI-based
Sickle Cell classification with visual explanation.
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# FEATURES
# =========================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="feature">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">Deep Learning</div>
        <div class="feature-text">
            Powered by an InceptionV3 model
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature">
        <div class="feature-icon">⚡</div>
        <div class="feature-title">Quick Analysis</div>
        <div class="feature-text">
            Get a prediction within seconds
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">AI Explanation</div>
        <div class="feature-text">
            Understand the model using Grad-CAM
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    '<div class="section">Analyze Your Image</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="upload-card">',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Drop your blood-smear image here",
    type=[
        "jpg",
        "jpeg",
        "png",
        "tif",
        "tiff"
    ]
)

st.markdown(
    '<div class="small-note">'
    'Supported formats: JPG, JPEG, PNG, TIFF'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# ANALYSIS
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

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

    with st.spinner("AI is analyzing your image..."):

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
        result_class = "positive"
        icon = "🔴"

    else:

        result = "Normal"
        confidence = (1 - probability) * 100
        result_class = "negative"
        icon = "🟢"

    st.markdown(
        '<div class="section">Your Result</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="result-card {result_class}">

            <div class="result-icon">
                {icon}
            </div>

            <div class="result-name">
                {result}
            </div>

            <div class="result-confidence">
                AI confidence
            </div>

            <div style="
                font-size:36px;
                font-weight:800;
                margin-top:5px;
            ">
                {confidence:.2f}%
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # PROBABILITY
    # -----------------------------------------------------

    st.markdown(
        '<div class="section">Prediction Breakdown</div>',
        unsafe_allow_html=True
    )

    normal_probability = 1 - probability

    p1, p2 = st.columns(2)

    with p1:

        st.write("🟢 Normal")

        st.progress(
            float(normal_probability)
        )

        st.write(
            f"{normal_probability * 100:.2f}%"
        )

    with p2:

        st.write("🔴 Sickle Cell")

        st.progress(
            float(probability)
        )

        st.write(
            f"{probability * 100:.2f}%"
        )


    # -----------------------------------------------------
    # GRAD CAM
    # -----------------------------------------------------

    st.markdown(
        '<div class="section">What Did the AI See?</div>',
        unsafe_allow_html=True
    )

    with st.spinner(
        "Generating AI attention map..."
    ):

        heatmap = make_gradcam(
            image_array
        )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🖼️ Original")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.markdown("### 🔥 AI Attention Map")

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


    st.info(
        "The Grad-CAM map highlights regions that "
        "contributed to the model's prediction."
    )


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown(
    '<div class="section">How It Works</div>',
    unsafe_allow_html=True
)

s1, s2, s3, s4 = st.columns(4)

steps = [
    ("1", "Upload", "Choose a blood-smear image"),
    ("2", "Analyze", "InceptionV3 processes the image"),
    ("3", "Predict", "The model generates a classification"),
    ("4", "Explain", "Grad-CAM shows model attention")
]

for col, (num, title, text) in zip(
    [s1, s2, s3, s4],
    steps
):

    with col:

        st.markdown(
            f"""
            <div class="step">

                <div class="step-number">
                    {num}
                </div>

                <div class="step-title">
                    {title}
                </div>

                <div class="step-text">
                    {text}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# MODEL INFO
# =========================================================

st.markdown(
    '<div class="section">Behind SickleScan</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info">

<b>Model:</b> InceptionV3<br><br>

<b>Classification:</b> Normal vs Sickle Cell<br><br>

<b>Input:</b> RGB blood-smear image resized to 224 × 224 pixels<br><br>

<b>Explainability:</b> Grad-CAM<br><br>

<b>Purpose:</b> Academic / research project

</div>
""", unsafe_allow_html=True)


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown(
    '<div class="section">Important</div>',
    unsafe_allow_html=True
)

st.warning(
    "SickleScan is a research prototype developed for "
    "academic purposes. Its output should not be used "
    "as a substitute for professional medical diagnosis."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

🩸 SICKLESCAN<br>
AI-powered Blood Smear Analysis<br><br>
Deep Learning Research Project • InceptionV3 • Grad-CAM

</div>
""", unsafe_allow_html=True)
