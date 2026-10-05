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
        radial-gradient(circle at 8% 0%, rgba(124,58,237,.18), transparent 28%),
        radial-gradient(circle at 92% 8%, rgba(236,72,153,.12), transparent 25%),
        #080a12;
}

[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
    box-shadow: none !important;
}

.block-container {
    max-width: 1150px;
    padding-top: 28px;
    padding-bottom: 60px;
}

h1, h2, h3 {
    letter-spacing: -0.02em;
}

.stButton > button,
.stFormSubmitButton > button {
    border-radius: 12px !important;
    min-height: 44px !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed rgba(192,132,252,.45) !important;
    border-radius: 18px !important;
    background: rgba(255,255,255,.025) !important;
    min-height: 145px;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(244,114,182,.70) !important;
}

[data-testid="stExpander"] {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    background: rgba(20,23,36,.55) !important;
}

@media (max-width: 700px) {
    .block-container {
        padding-left: 14px;
        padding-right: 14px;
    }
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
        "<h1 style='text-align:center; margin-top:55px;'>🩸 SICKLESCAN</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center; color:#a1a1aa;'>"
        "AI-Powered Blood Smear Analysis"
        "</p>",
        unsafe_allow_html=True
    )

    login_tab, signup_tab = st.tabs(
        ["🔐 Login", "✨ Create Account"]
    )

    with login_tab:

        st.markdown("### Welcome back")
        st.caption("Sign in to continue to SickleScan")

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
                st.error("Please enter your email and password.")

            else:

                try:

                    response = login_user(
                        email.strip(),
                        password
                    )

                    if response.ok:

                        data = response.json()

                        st.session_state.authenticated = True
                        st.session_state.user_email = email.strip()
                        st.session_state.access_token = data.get("access_token")
                        st.session_state.refresh_token = data.get("refresh_token")

                        st.rerun()

                    else:

                        try:
                            error_data = response.json()

                            message = (
                                error_data.get("msg")
                                or error_data.get("error_description")
                                or error_data.get("message")
                                or "Invalid email or password."
                            )

                        except Exception:
                            message = "Invalid email or password."

                        st.error(message)

                except requests.RequestException:
                    st.error("Unable to connect to authentication service.")

    with signup_tab:

        st.markdown("### Create your account")
        st.caption("Create an account to use SickleScan")

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

            if not new_email or not new_password or not confirm_password:
                st.error("Please fill in all fields.")

            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters.")

            elif new_password != confirm_password:
                st.error("Passwords do not match.")

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
                            st.session_state.user_email = new_email.strip()
                            st.session_state.access_token = data.get("access_token")
                            st.session_state.refresh_token = data.get("refresh_token")

                            st.rerun()

                        else:

                            st.success("Account created successfully!")
                            st.info(
                                "Check your email to confirm your account, "
                                "then log in."
                            )

                    else:

                        try:

                            error_data = response.json()

                            message = (
                                error_data.get("msg")
                                or error_data.get("error_description")
                                or error_data.get("message")
                                or "Could not create the account."
                            )

                        except Exception:
                            message = "Could not create the account."

                        st.error(message)

                except requests.RequestException:
                    st.error("Unable to connect to authentication service.")

    st.caption("Secure authentication powered by Supabase")

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

header_left, header_right = st.columns([7, 2])

with header_left:
    st.markdown(
        "## 🩸 **SICKLESCAN**",
        unsafe_allow_html=False
    )

with header_right:
    if st.button("Logout", use_container_width=True):
        logout_user()
        st.rerun()

st.caption(
    "Signed in as "
    + st.session_state.get("user_email", "")
)

# =========================================================
# HERO
# =========================================================

st.markdown(
    "# AI-Powered Blood Smear Analysis"
)

st.markdown(
    "### 🧠 Deep Learning • Image Classification • Grad-CAM"
)

st.write(
    "Upload a blood-smear image and explore the model's "
    "prediction together with a visual explanation of the "
    "regions that influenced the result."
)

st.divider()

# =========================================================
# ABOUT SICKLE CELL
# =========================================================

st.markdown("## 🩸 What is Sickle Cell Disease?")

st.write(
    "Sickle Cell Disease (SCD) is a genetic blood disorder in "
    "which red blood cells can develop an abnormal sickle-like "
    "shape. These changes can affect the normal movement of blood "
    "through blood vessels."
)

st.write(
    "Blood-smear images provide visual information about the "
    "appearance of red blood cells. Deep-learning techniques can "
    "be used to learn visual patterns from these images and assist "
    "with image-level classification research."
)

# =========================================================
# WHAT SICKLESCAN DOES
# =========================================================

st.markdown("## ✨ How SickleScan Works")

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("### 🔬 Image Analysis")
        st.write(
            "Upload a blood-smear image and let the trained "
            "deep-learning model analyze its visual features."
        )

with c2:
    with st.container(border=True):
        st.markdown("### 🧠 AI Classification")
        st.write(
            "The InceptionV3 model produces a Normal or "
            "Sickle Cell prediction together with a probability score."
        )

with c3:
    with st.container(border=True):
        st.markdown("### 🔥 Visual Explanation")
        st.write(
            "Grad-CAM highlights image regions that contributed "
            "to the model's prediction."
        )

# =========================================================
# UPLOAD
# =========================================================

st.markdown("## 🔬 Analyze a Blood-Smear Image")

st.caption(
    "Upload a JPG, JPEG, PNG, TIFF or TIF image to start the analysis."
)

with st.container(border=True):

    st.markdown("### 📤 Upload your blood-smear image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "tif", "tiff"]
    )

# =========================================================
# ANALYSIS
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Same preprocessing used during training
    resized = image.resize((224, 224))

    image_array = np.array(resized).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)

    # =====================================================
    # PREDICTION
    # =====================================================

    with st.spinner("🧠 AI is analyzing the image..."):

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

        with st.container(border=True):
            st.markdown("### 🔴 Sickle Cell")
            st.caption("Model prediction")
            st.metric(
                "Prediction confidence",
                f"{confidence:.1f}%"
            )

    else:

        confidence = normal_probability * 100

        with st.container(border=True):
            st.markdown("### 🟢 Normal")
            st.caption("Model prediction")
            st.metric(
                "Prediction confidence",
                f"{confidence:.1f}%"
            )

    # =====================================================
    # PROBABILITY
    # =====================================================

    st.markdown("## 📈 Prediction Breakdown")

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.markdown("### 🟢 Normal Probability")

            st.progress(
                float(normal_probability)
            )

            st.metric(
                "Probability",
                f"{normal_probability * 100:.1f}%"
            )

    with col2:

        with st.container(border=True):

            st.markdown("### 🔴 Sickle Cell Probability")

            st.progress(
                float(probability)
            )

            st.metric(
                "Probability",
                f"{probability * 100:.1f}%"
            )

    # =====================================================
    # GRAD-CAM
    # =====================================================

    st.markdown("## 🔥 Model Explanation")

    st.write(
        "Grad-CAM provides a visual indication of the image "
        "regions that influenced the model's prediction."
    )

    with st.spinner("Generating Grad-CAM..."):

        heatmap = make_gradcam(image_array)

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.markdown("### Original Image")

            st.image(
                image,
                use_container_width=True
            )

    with col2:

        with st.container(border=True):

            st.markdown("### Grad-CAM Visualization")

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
# MORE ABOUT SICKLESCAN
# =========================================================

st.markdown("## ℹ️ More About SickleScan")

with st.expander("View project and model information"):

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

### Important

SickleScan is an academic research prototype.
Its output should not be used as a substitute for
professional medical diagnosis.
"""
    )

st.divider()

st.caption(
    "🩸 SICKLESCAN  •  AI-Powered Blood Smear Analysis  •  "
    "InceptionV3  •  Grad-CAM"
)
