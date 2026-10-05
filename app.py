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

/* Main background */
.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(99,102,241,0.16), transparent 28%),
        radial-gradient(circle at 92% 8%, rgba(236,72,153,0.10), transparent 25%),
        linear-gradient(180deg, #070911 0%, #0b0e18 52%, #080a12 100%);
    color: #f8fafc;
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

[data-testid="stMainBlockContainer"] {
    padding-left: 2rem;
    padding-right: 2rem;
}


/* Keep Streamlit toolbar icons,
   remove black background/line */

[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
    box-shadow: none !important;
}


/* Main container */

.block-container {
    max-width: 1150px;
    padding-top: 25px;
    padding-bottom: 50px;
}


/* Logo */

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
    padding: 48px 10px 34px;
}

.hero h1 {
    font-size: clamp(38px, 5vw, 58px);
    font-weight: 850;
    margin-bottom: 12px;

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
    max-width: 760px;
    margin: auto;
    color: #a1a1aa;
    font-size: 18px;
    line-height: 1.6;
}


/* Section titles */

.section-title {
    font-size: 28px;
    font-weight: 800;
    margin-top: 38px;
    margin-bottom: 15px;
}


/* Information cards */
.info-card {
    transition: transform .2s ease, border-color .2s ease, background .2s ease;
}
    background: rgba(24, 27, 40, 0.78);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 22px;
    min-height: 165px;
}

.info-card h3 {
    margin-top: 5px;
}

.info-card p {
    color: #a1a1aa;
    line-height: 1.55;
}


/* Upload */

.upload-card {
    background: rgba(24,27,40,0.85);
    border: 1px solid rgba(192,132,252,0.22);
    border-radius: 22px;
    padding: 24px;
}


/* Results */

.result-normal {
    background: rgba(34,197,94,0.10);
    border: 1px solid rgba(74,222,128,0.30);
    border-radius: 20px;
    padding: 26px;
    text-align: center;
}

.result-sickle {
    background: rgba(239,68,68,0.10);
    border: 1px solid rgba(248,113,113,0.35);
    border-radius: 20px;
    padding: 26px;
    text-align: center;
}

.result-title {
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
    margin-top: 6px;
}


/* Login */

.auth-title {
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    margin-top: 20px;
}

.auth-subtitle {
    text-align: center;
    color: #a1a1aa;
    margin-bottom: 25px;
}

.auth-note {
    text-align: center;
    color: #71717a;
    font-size: 12px;
    margin-top: 18px;
}


/* Premium cards */
.info-card:hover {
    transform: translateY(-3px);
    border-color: rgba(192,132,252,0.28);
    background: rgba(30,34,50,0.90);
}

.upload-card {
    box-shadow: 0 18px 50px rgba(0,0,0,0.20);
}

[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed rgba(192,132,252,0.42) !important;
    background: rgba(255,255,255,0.025) !important;
    border-radius: 16px !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(244,114,182,0.65) !important;
}

.stButton > button {
    border-radius: 12px !important;
    border: 1px solid rgba(192,132,252,0.28) !important;
    background: linear-gradient(135deg, rgba(124,58,237,0.90), rgba(219,39,119,0.82)) !important;
    color: white !important;
    font-weight: 700 !important;
    min-height: 44px;
}

.stButton > button:hover {
    border-color: rgba(255,255,255,0.35) !important;
    transform: translateY(-1px);
}

[data-testid="stProgressBar"] > div > div {
    border-radius: 99px;
}

div[data-baseweb="tab-list"] {
    gap: 8px;
}

button[data-baseweb="tab"] {
    border-radius: 10px !important;
}

[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    background: rgba(24,27,40,0.55) !important;
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
# SUPABASE AUTHENTICATION
# =========================================================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]


def supabase_headers():

    return {
        "apikey": SUPABASE_KEY,
        "Content-Type": "application/json"
    }


def signup_user(email, password):

    return requests.post(
        f"{SUPABASE_URL}/auth/v1/signup",
        headers=supabase_headers(),
        json={
            "email": email,
            "password": password
        },
        timeout=30
    )


def login_user(email, password):

    return requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
        headers=supabase_headers(),
        json={
            "email": email,
            "password": password
        },
        timeout=30
    )


def logout_user():

    for key in [
        "authenticated",
        "user_email",
        "access_token",
        "refresh_token"
    ]:

        st.session_state.pop(
            key,
            None
        )


if "authenticated" not in st.session_state:

    st.session_state.authenticated = False


# =========================================================
# LOGIN / SIGNUP PAGE
# =========================================================

if not st.session_state.authenticated:

    st.markdown(
        "<h1 style='text-align:center; margin-top:70px;'>"
        "🩸 SICKLESCAN"
        "</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center; color:#a1a1aa;'>"
        "AI-Powered Blood Smear Analysis"
        "</p>",
        unsafe_allow_html=True
    )

    st.write("")

    login_tab, signup_tab = st.tabs(
        [
            "🔐 Login",
            "✨ Create Account"
        ]
    )


    # =====================================================
    # LOGIN
    # =====================================================

    with login_tab:

        st.markdown(
            "<h2 class='auth-title'>Welcome back</h2>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p class='auth-subtitle'>"
            "Sign in to continue to SickleScan"
            "</p>",
            unsafe_allow_html=True
        )

        with st.form("login_form"):

            email = st.text_input(
                "Email",
                placeholder="you@example.com"
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password"
            )

            login_clicked = st.form_submit_button(
                "Login",
                use_container_width=True
            )


        if login_clicked:

            if not email or not password:

                st.error(
                    "Please enter your email and password."
                )

            else:

                try:

                    response = login_user(
                        email.strip(),
                        password
                    )

                    if response.ok:

                        data = response.json()

                        st.session_state.authenticated = True

                        st.session_state.user_email = (
                            email.strip()
                        )

                        st.session_state.access_token = (
                            data.get("access_token")
                        )

                        st.session_state.refresh_token = (
                            data.get("refresh_token")
                        )

                        st.rerun()

                    else:

                        try:

                            error_data = response.json()

                            message = (
                                error_data.get("msg")
                                or error_data.get(
                                    "error_description"
                                )
                                or error_data.get(
                                    "message"
                                )
                                or "Invalid email or password."
                            )

                        except Exception:

                            message = (
                                "Invalid email or password."
                            )

                        st.error(message)

                except requests.RequestException:

                    st.error(
                        "Unable to connect to authentication service."
                    )


    # =====================================================
    # SIGN UP
    # =====================================================

    with signup_tab:

        st.markdown(
            "<h2 class='auth-title'>Create your account</h2>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p class='auth-subtitle'>"
            "Create an account to use SickleScan"
            "</p>",
            unsafe_allow_html=True
        )

        with st.form("signup_form"):

            new_email = st.text_input(
                "Email",
                placeholder="you@example.com",
                key="signup_email"
            )

            new_password = st.text_input(
                "Password",
                type="password",
                placeholder="Minimum 6 characters",
                key="signup_password"
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Re-enter your password",
                key="confirm_password"
            )

            signup_clicked = st.form_submit_button(
                "Create Account",
                use_container_width=True
            )


        if signup_clicked:

            if (
                not new_email
                or not new_password
                or not confirm_password
            ):

                st.error(
                    "Please fill in all fields."
                )

            elif len(new_password) < 6:

                st.error(
                    "Password must be at least 6 characters."
                )

            elif new_password != confirm_password:

                st.error(
                    "Passwords do not match."
                )

            else:

                try:

                    response = signup_user(
                        new_email.strip(),
                        new_password
                    )

                    if response.ok:

                        data = response.json()

                        if data.get("access_token"):

                            st.session_state.authenticated = True

                            st.session_state.user_email = (
                                new_email.strip()
                            )

                            st.session_state.access_token = (
                                data.get("access_token")
                            )

                            st.session_state.refresh_token = (
                                data.get("refresh_token")
                            )

                            st.rerun()

                        else:

                            st.success(
                                "Account created successfully!"
                            )

                            st.info(
                                "Check your email to confirm "
                                "your account, then log in."
                            )

                    else:

                        try:

                            error_data = response.json()

                            message = (
                                error_data.get("msg")
                                or error_data.get(
                                    "error_description"
                                )
                                or error_data.get(
                                    "message"
                                )
                                or "Could not create the account."
                            )

                        except Exception:

                            message = (
                                "Could not create the account."
                            )

                        st.error(message)

                except requests.RequestException:

                    st.error(
                        "Unable to connect to authentication service."
                    )


    st.markdown(
        "<p class='auth-note'>"
        "Secure authentication powered by Supabase"
        "</p>",
        unsafe_allow_html=True
    )

    st.stop()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_path = (
        "/tmp/InceptionV3_UCL_Final.keras"
    )

    if not os.path.exists(model_path):

        url = (
            "https://huggingface.co/"
            "abhijeetgour12/"
            "sickle-cell-inceptionv3/"
            "resolve/main/"
            "InceptionV3_UCL_Final.keras"
            "?download=true"
        )

        response = requests.get(
            url,
            stream=True
        )

        response.raise_for_status()

        with open(
            model_path,
            "wb"
        ) as f:

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

            if len(
                layer.output.shape
            ) == 4:

                target_layer = layer

                break

        except Exception:

            continue


    grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            target_layer.output,
            backbone.output
        ]
    )


    with tf.GradientTape() as tape:

        conv_outputs, backbone_features = (
            grad_model(
                image_array,
                training=False
            )
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

header_left, header_right = st.columns(
    [8, 1]
)


with header_left:

    st.markdown(
        '<div class="logo">'
        '🩸 <span>SICKLESCAN</span>'
        '</div>',
        unsafe_allow_html=True
    )


with header_right:

    if st.button(
        "Logout",
        use_container_width=True
    ):

        logout_user()

        st.rerun()


st.caption(
    "Signed in as "
    + st.session_state.get(
        "user_email",
        ""
    )
)

st.markdown(
    "<div style='text-align:center; margin-top:-8px; color:#71717a; font-size:12px;'>"
    "AI research prototype • Image-level classification"
    "</div>",
    unsafe_allow_html=True
)


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<h1>AI-Powered Blood Smear Analysis</h1>

<p>
Analyze a blood-smear image with SickleScan and explore
the model's prediction together with a visual Grad-CAM
explanation of the regions that influenced the result.
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# ABOUT SICKLE CELL
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🩸 What is Sickle Cell Disease?'
    '</div>',
    unsafe_allow_html=True
)


st.write(
    """
Sickle Cell Disease (SCD) is a genetic blood disorder in
which red blood cells can develop an abnormal sickle-like
shape. These changes can affect the normal movement of blood
through blood vessels.

Blood-smear images provide visual information about the
appearance of red blood cells. Deep-learning techniques can
be used to learn visual patterns from these images and assist
with image-level classification research.
"""
)


# =========================================================
# WHAT SICKLESCAN DOES
# =========================================================

st.markdown(
    '<div class="section-title">'
    '✨ What SickleScan Does'
    '</div>',
    unsafe_allow_html=True
)


c1, c2, c3 = st.columns(3)


with c1:

    st.markdown(
        """
<div class="info-card">

<h3>🔬 Image Analysis</h3>

<p>
Upload a blood-smear image and let the trained
deep-learning model analyze its visual features.
</p>

</div>
""",
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        """
<div class="info-card">

<h3>🧠 AI Classification</h3>

<p>
The InceptionV3 model produces a Normal or Sickle Cell
prediction together with a model probability score.
</p>

</div>
""",
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        """
<div class="info-card">

<h3>🔥 Visual Explanation</h3>

<p>
Grad-CAM highlights image regions that contributed
to the model's prediction.
</p>

</div>
""",
        unsafe_allow_html=True
    )


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🔬 Analyze a Blood-Smear Image'
    '</div>',
    unsafe_allow_html=True
)


st.write(
    "Upload an image below to start the AI analysis."
)


st.markdown(
    '<div class="upload-card">',
    unsafe_allow_html=True
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


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# ANALYSIS
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


    # =====================================================
    # PREDICTION
    # =====================================================

    with st.spinner(
        "🧠 AI is analyzing the image..."
    ):

        probability = float(
            model.predict(
                image_array,
                verbose=0
            )[0][0]
        )


    normal_probability = (
        1 - probability
    )


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Analysis Result'
        '</div>',
        unsafe_allow_html=True
    )


    if probability >= 0.5:

        confidence = (
            probability * 100
        )


        st.markdown(
            '<div class="result-sickle">',
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="result-title">'
            '🔴 Sickle Cell'
            '</div>',
            unsafe_allow_html=True
        )


        st.write(
            "Model prediction"
        )


        st.markdown(
            '<div class="result-score">'
            f'{confidence:.1f}%'
            '</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="result-label">'
            'prediction score'
            '</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    else:

        confidence = (
            normal_probability * 100
        )


        st.markdown(
            '<div class="result-normal">',
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="result-title">'
            '🟢 Normal'
            '</div>',
            unsafe_allow_html=True
        )


        st.write(
            "Model prediction"
        )


        st.markdown(
            '<div class="result-score">'
            f'{confidence:.1f}%'
            '</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="result-label">'
            'prediction score'
            '</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # =====================================================
    # PROBABILITY
    # =====================================================

    st.markdown(
        "### Prediction Breakdown"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            "🟢 **Normal Probability**"
        )


        st.progress(
            float(normal_probability)
        )


        st.caption(
            f"{normal_probability * 100:.1f}%"
        )


    with col2:

        st.write(
            "🔴 **Sickle Cell Probability**"
        )


        st.progress(
            float(probability)
        )


        st.caption(
            f"{probability * 100:.1f}%"
        )


    # =====================================================
    # GRAD-CAM
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🔥 Model Explanation'
        '</div>',
        unsafe_allow_html=True
    )


    st.write(
        "Grad-CAM provides a visual indication of the image "
        "regions that influenced the model's prediction."
    )


    with st.spinner(
        "Generating Grad-CAM..."
    ):

        heatmap = make_gradcam(
            image_array
        )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Original Image"
        )


        st.image(
            image,
            use_container_width=True
        )


    with col2:

        st.subheader(
            "Grad-CAM Visualization"
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


# =========================================================
# MORE ABOUT SICKLESCAN
# =========================================================

st.markdown(
    '<div class="section-title">'
    'ℹ️ More About SickleScan'
    '</div>',
    unsafe_allow_html=True
)


with st.expander(
    "View project and model information"
):

    st.markdown(
        """
### About the Project

**SickleScan** is an academic deep-learning research
prototype designed for blood-smear image classification.

### Model Information

- **Architecture:** InceptionV3
- **Task:** Normal vs Sickle Cell
- **Input:** RGB blood-smear image
- **Input Size:** 224 × 224 pixels
- **Explainability:** Grad-CAM

### Prediction

The uploaded image is resized to the model's required
input size and passed through the trained InceptionV3 model.
The output probability is used to classify the image as
Normal or Sickle Cell.

### Explainability

Grad-CAM is used to visualize regions of the image that
contributed to the model's prediction.

### ⚠️ Important

SickleScan is an academic research prototype. Its output
should not be used as a substitute for professional medical
diagnosis.
"""
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">

🩸 <b>SICKLESCAN</b><br>

AI-Powered Blood Smear Analysis<br><br>

Deep Learning Research Project • InceptionV3 • Grad-CAM

</div>
""",
    unsafe_allow_html=True
)
