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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(124, 58, 237, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(219, 39, 119, 0.15),
            transparent 28%
        ),
        #080a12;
}

.block-container {
    max-width: 1100px;
    padding-top: 25px;
    padding-bottom: 50px;
}

/* Top logo */

.logo {
    font-size: 20px;
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

.hero h1 {
    font-size: 58px;
    font-weight: 850;
    margin-bottom: 10px;
    background: linear-gradient(
        90deg,
        #ffffff,
        #c084fc,
        #f9a8d4
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #a1a1aa;
    font-size: 18px;
}

/* Small badge */

.badge {
    display: inline-block;
    padding: 7px 15px;
    border-radius: 30px;
    background: rgba(168, 85, 247, 0.12);
    border: 1px solid rgba(192, 132, 252, 0.25);
    color: #d8b4fe;
    font-size: 13px;
    font-weight: 600;
}

/* Feature cards */

.feature-card {
    background: rgba(24, 27, 40, 0.8);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
}

.feature-icon {
    font-size: 28px;
}

.feature-title {
    font-weight: 700;
    margin-top: 8px;
}

.feature-text {
    color: #8f93a1;
    font-size: 13px;
    margin-top: 5px;
}

/* Section headings */

.section-title {
    font-size: 27px;
    font-weight: 800;
    margin-top: 40px;
    margin-bottom: 15px;
}

/* Result */

.result-positive {
    border: 1px solid rgba(248, 113, 113, 0.4);
    background: rgba(127, 29, 29, 0.18);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
}

.result-negative {
    border: 1px solid rgba(74, 222, 128, 0.35);
    background: rgba(20, 83, 45, 0.16);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
}

.result-icon {
    font-size: 42px;
}

.result-name {
    font-size: 30px;
    font-weight: 800;
}

.result-label {
    color: #a1a1aa;
    margin-top: 5px;
}

.result-score {
    font-size: 36px;
    font-weight: 850;
    margin-top: 3px;
}

/* Footer */

.footer {
    text-align: center;
    color: #71717a;
    font-size: 13px;
    margin-top: 50px;
    padding-top: 25px;
    border-top: 1px solid rgba(255,255,255,0.06);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
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

    return tf.keras.models.load_model(
        model_path
    )


model = load_model()


# =========================================================
# GRAD-CAM
# =========================================================

def make_gradcam(image_array):

    backbone = model.get_layer(
        "inception_v3"
    )

    target_layer = None

    for layer in reversed(
        backbone.layers
    ):

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
        )(
            x,
            training=False
        )

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

st.markdown(
    '<div class="logo">🩸 <span>SICKLESCAN</span></div>',
    unsafe_allow_html=True
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    '<div class="hero">'
    '<div class="badge">✦ AI-POWERED BLOOD SMEAR ANALYSIS</div>'
    '<h1>See What AI Sees.</h1>'
    '<p>'
    'Explore AI-based blood-smear classification '
    'with visual model explanation.'
    '</p>'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# FEATURES
# =========================================================

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(
        '<div class="feature-card">'
        '<div class="feature-icon">🧠</div>'
        '<div class="feature-title">Deep Learning</div>'
        '<div class="feature-text">'
        'Powered by InceptionV3'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        '<div class="feature-card">'
        '<div class="feature-icon">⚡</div>'
        '<div class="feature-title">Quick Analysis</div>'
        '<div class="feature-text">'
        'Prediction within seconds'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        '<div class="feature-card">'
        '<div class="feature-icon">🔍</div>'
        '<div class="feature-title">AI Explanation</div>'
        '<div class="feature-text">'
        'Visualized using Grad-CAM'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Analyze Your Image'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a blood-smear image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "tif",
        "tiff"
    ]
)


# =========================================================
# ANALYSIS
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Same preprocessing as training
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

    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    with st.spinner(
        "🧠 Analyzing your image..."
    ):

        probability = float(
            model.predict(
                image_array,
                verbose=0
            )[0][0]
        )

    normal_probability = 1 - probability

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Your Result'
        '</div>',
        unsafe_allow_html=True
    )

    if probability >= 0.5:

        result = "Sickle Cell"
        confidence = probability * 100

        st.markdown(
            '<div class="result-positive">'
            '<div class="result-icon">🔴</div>'
            '<div class="result-name">'
            'Sickle Cell'
            '</div>'
            '<div class="result-label">'
            'Model prediction'
            '</div>'
            '<div class="result-score">'
            f'{confidence:.1f}%'
            '</div>'
            '<div class="result-label">'
            'prediction score'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        result = "Normal"
        confidence = normal_probability * 100

        st.markdown(
            '<div class="result-negative">'
            '<div class="result-icon">🟢</div>'
            '<div class="result-name">'
            'Normal'
            '</div>'
            '<div class="result-label">'
            'Model prediction'
            '</div>'
            '<div class="result-score">'
            f'{confidence:.1f}%'
            '</div>'
            '<div class="result-label">'
            'prediction score'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # PROBABILITY
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Prediction Breakdown'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write("🟢 **Normal**")

        st.progress(
            float(normal_probability)
        )

        st.caption(
            f"{normal_probability * 100:.1f}%"
        )

    with col2:

        st.write("🔴 **Sickle Cell**")

        st.progress(
            float(probability)
        )

        st.caption(
            f"{probability * 100:.1f}%"
        )


    # -----------------------------------------------------
    # IMAGE PREVIEW
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'AI Explanation'
        '</div>',
        unsafe_allow_html=True
    )

    with st.spinner(
        "🔥 Generating Grad-CAM..."
    ):

        heatmap = make_gradcam(
            image_array
        )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🖼️ Original Image"
        )

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "🔥 Grad-CAM"
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
        "Grad-CAM highlights regions that contributed "
        "to the model's prediction."
    )


# =========================================================
# MORE ABOUT
# =========================================================

st.markdown(
    '<div class="section-title">'
    'More'
    '</div>',
    unsafe_allow_html=True
)

with st.expander(
    "ℹ️  More about SickleScan"
):

    st.markdown("""
### About SickleScan

SickleScan is an academic deep-learning research
prototype for blood-smear image classification.

### Model

- **Architecture:** InceptionV3
- **Task:** Normal vs Sickle Cell
- **Input:** RGB blood-smear image
- **Input size:** 224 × 224 pixels
- **Explainability:** Grad-CAM

### How the prediction works

The uploaded image is resized to the model's required
input size and processed by the trained InceptionV3 model.

The model generates a probability for the Sickle Cell
class. Grad-CAM is then used to visualize image regions
that contributed to the prediction.

### ⚠️ Important

This is an academic research prototype. The output
should not be used as a substitute for professional
medical diagnosis.
""")


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    '🩸 SICKLESCAN<br>'
    'AI-powered Blood Smear Analysis<br><br>'
    'Deep Learning Research Project • InceptionV3 • Grad-CAM'
    '</div>',
    unsafe_allow_html=True
)
