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
# PREMIUM UI
# =========================================================

st.markdown("""
<style>

* {
    box-sizing: border-box;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%,
            rgba(124,58,237,.20), transparent 28%),
        radial-gradient(circle at 90% 8%,
            rgba(236,72,153,.14), transparent 26%),
        linear-gradient(180deg,#070911 0%,#0b0e18 55%,#070911 100%);
    color:#f8fafc;
}

[data-testid="stHeader"] {
    background:transparent !important;
    border-bottom:none !important;
    box-shadow:none !important;
}

.block-container {
    max-width:1180px;
    padding-top:28px;
    padding-bottom:70px;
}

/* ================= HEADER ================= */

.topbar {
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:8px 0 18px;
}

.brand {
    font-size:22px;
    font-weight:900;
    letter-spacing:1.5px;
}

.brand span {
    color:#c084fc;
}

.user-badge {
    display:inline-block;
    padding:8px 14px;
    border-radius:999px;
    background:rgba(255,255,255,.05);
    border:1px solid rgba(255,255,255,.08);
    color:#a1a1aa;
    font-size:12px;
}

/* ================= HERO ================= */

.hero-box {
    position:relative;
    overflow:hidden;
    text-align:center;
    padding:70px 30px 60px;
    margin:10px 0 35px;
    border:1px solid rgba(192,132,252,.16);
    border-radius:30px;
    background:
        linear-gradient(
            135deg,
            rgba(124,58,237,.12),
            rgba(219,39,119,.06)
        ),
        rgba(15,17,28,.75);
    box-shadow:0 30px 80px rgba(0,0,0,.25);
}

.hero-badge {
    display:inline-block;
    padding:7px 15px;
    border-radius:999px;
    background:rgba(192,132,252,.10);
    border:1px solid rgba(192,132,252,.25);
    color:#d8b4fe;
    font-size:12px;
    font-weight:700;
    letter-spacing:.8px;
    margin-bottom:18px;
}

.hero-box h1 {
    font-size:clamp(40px,6vw,68px);
    line-height:1.05;
    margin:0 0 18px;
    font-weight:900;
    background:linear-gradient(
        90deg,#fff,#c084fc,#f9a8d4
    );
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero-box p {
    max-width:720px;
    margin:auto;
    color:#a1a1aa;
    font-size:17px;
    line-height:1.7;
}

/* ================= SECTION ================= */

.section-heading {
    font-size:27px;
    font-weight:850;
    margin:40px 0 16px;
}

/* ================= FEATURE CARDS ================= */

.feature {
    min-height:185px;
    padding:25px;
    border-radius:20px;
    background:rgba(20,23,36,.78);
    border:1px solid rgba(255,255,255,.07);
    transition:.2s ease;
}

.feature:hover {
    transform:translateY(-4px);
    border-color:rgba(192,132,252,.30);
    background:rgba(27,30,45,.92);
}

.feature-icon {
    font-size:28px;
    margin-bottom:13px;
}

.feature h3 {
    margin:0 0 9px;
    font-size:18px;
}

.feature p {
    color:#9293a0;
    line-height:1.55;
    font-size:14px;
}

/* ================= UPLOAD ================= */

.upload-wrapper {
    padding:28px;
    border-radius:24px;
    background:
        linear-gradient(
            145deg,
            rgba(124,58,237,.10),
            rgba(24,27,40,.90)
        );
    border:1px solid rgba(192,132,252,.22);
    box-shadow:0 25px 65px rgba(0,0,0,.22);
}

.upload-title {
    font-size:21px;
    font-weight:800;
    margin-bottom:6px;
}

.upload-subtitle {
    color:#888a97;
    font-size:14px;
    margin-bottom:20px;
}

[data-testid="stFileUploaderDropzone"] {
    min-height:150px;
    border:1px dashed rgba(192,132,252,.45) !important;
    border-radius:18px !important;
    background:rgba(255,255,255,.025) !important;
}

/* ================= RESULT ================= */

.result-card {
    padding:35px 25px;
    text-align:center;
    border-radius:24px;
    margin:18px 0 28px;
}

.result-positive {
    background:
        linear-gradient(
            145deg,
            rgba(239,68,68,.14),
            rgba(127,29,29,.08)
        );
    border:1px solid rgba(248,113,113,.35);
}

.result-negative {
    background:
        linear-gradient(
            145deg,
            rgba(34,197,94,.14),
            rgba(20,83,45,.08)
        );
    border:1px solid rgba(74,222,128,.32);
}

.result-status {
    font-size:34px;
    font-weight:900;
    margin-bottom:8px;
}

.result-caption {
    color:#9293a0;
    font-size:14px;
}

.result-score {
    font-size:48px;
    font-weight:900;
    margin-top:10px;
}

/* ================= PROBABILITY ================= */

.prob-card {
    padding:22px;
    border-radius:18px;
    background:rgba(20,23,36,.72);
    border:1px solid rgba(255,255,255,.07);
}

/* ================= GRAD CAM ================= */

.cam-card {
    padding:18px;
    border-radius:22px;
    background:rgba(20,23,36,.78);
    border:1px solid rgba(255,255,255,.07);
}

.cam-label {
    text-align:center;
    font-weight:750;
    margin-bottom:12px;
    color:#e4e4e7;
}

/* ================= BUTTON ================= */

.stButton > button {
    min-height:45px !important;
    border-radius:12px !important;
    border:1px solid rgba(192,132,252,.25) !important;
    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #db2777
        ) !important;
    color:white !important;
    font-weight:750 !important;
}

.stButton > button:hover {
    transform:translateY(-1px);
    border-color:rgba(255,255,255,.35) !important;
}

/* ================= LOGIN ================= */

.login-shell {
    max-width:560px;
    margin:70px auto 0;
    padding:42px;
    border-radius:28px;
    background:
        linear-gradient(
            145deg,
            rgba(124,58,237,.09),
            rgba(20,23,36,.92)
        );
    border:1px solid rgba(255,255,255,.09);
    box-shadow:0 30px 90px rgba(0,0,0,.35);
}

.login-brand {
    text-align:center;
    font-size:32px;
    font-weight:900;
    letter-spacing:1px;
}

.login-brand span {
    color:#c084fc;
}

.login-tag {
    text-align:center;
    color:#8f909b;
    margin:9px 0 28px;
}

.auth-title {
    text-align:center;
    font-size:25px;
    font-weight:850;
}

.auth-subtitle {
    text-align:center;
    color:#858692;
    margin-bottom:20px;
}

.auth-note {
    text-align:center;
    color:#666874;
    font-size:12px;
    margin-top:20px;
}

/* ================= EXPANDER ================= */

[data-testid="stExpander"] {
    border-radius:18px !important;
    border:1px solid rgba(255,255,255,.08) !important;
    background:rgba(20,23,36,.55) !important;
}

/* ================= FOOTER ================= */

.footer {
    text-align:center;
    margin-top:60px;
    padding-top:25px;
    border-top:1px solid rgba(255,255,255,.06);
    color:#686a76;
    font-size:12px;
}

/* ================= MOBILE ================= */

@media (max-width:700px) {

    .block-container {
        padding-left:16px;
        padding-right:16px;
    }

    .hero-box {
        padding:48px 18px;
        border-radius:22px;
    }

    .hero-box p {
        font-size:15px;
    }

    .section-heading {
        font-size:23px;
    }

    .upload-wrapper {
        padding:18px;
    }

    .login-shell {
        margin-top:25px;
        padding:25px 18px;
    }

    .result-score {
        font-size:40px;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SUPABASE
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
        st.session_state.pop(key, None)


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


# =========================================================
# LOGIN PAGE
# =========================================================

if not st.session_state.authenticated:

    st.markdown("""
    <div class="login-shell">

        <div class="login-brand">
            🩸 <span>SICKLESCAN</span>
        </div>

        <div class="login-tag">
            AI-powered blood smear analysis
        </div>

    </div>
    """, unsafe_allow_html=True)

    login_tab, signup_tab = st.tabs(
        ["🔐 Login", "✨ Create Account"]
    )

    # ---------------- LOGIN ----------------

    with login_tab:

        st.markdown(
            "<h2 class='auth-title'>Welcome back</h2>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<p class='auth-subtitle'>"
            "Sign in to access your SickleScan dashboard"
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
                        st.session_state.user_email = email.strip()
                        st.session_state.access_token = data.get(
                            "access_token"
                        )
                        st.session_state.refresh_token = data.get(
                            "refresh_token"
                        )

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

                    st.error(
                        "Unable to connect to authentication service."
                    )

    # ---------------- SIGNUP ----------------

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

            if not new_email or not new_password or not confirm_password:

                st.error("Please fill in all fields.")

            elif len(new_password) < 6:

                st.error(
                    "Password must be at least 6 characters."
                )

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
                            st.session_state.access_token = data.get(
                                "access_token"
                            )
                            st.session_state.refresh_token = data.get(
                                "refresh_token"
                            )

                            st.rerun()

                        else:

                            st.success(
                                "Account created successfully!"
                            )

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
# MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_path = "/tmp/InceptionV3_UCL_Final.keras"

    if not os.path.exists(model_path):

        url = (
            "https://huggingface.co/"
            "abhijeetgour12/sickle-cell-inceptionv3/"
            "resolve/main/"
            "InceptionV3_UCL_Final.keras"
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

        conv_outputs, backbone_features = grad_model(
            image_array,
            training=False
        )

        x = model.get_layer(
            "global_average_pooling2d"
        )(backbone_features)

        x = model.get_layer("dropout")(
            x,
            training=False
        )

        prediction = model.get_layer("dense")(x)

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
# DASHBOARD HEADER
# =========================================================

left, right = st.columns([7, 2])

with left:

    st.markdown(
        """
        <div class="topbar">
            <div class="brand">
                🩸 <span>SICKLESCAN</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with right:

    if st.button(
        "Logout",
        use_container_width=True
    ):

        logout_user()
        st.rerun()


st.markdown(
    f"""
    <div style="
        text-align:right;
        color:#777985;
        font-size:12px;
        margin-top:-18px;
        margin-bottom:10px;
    ">
        Signed in as {st.session_state.get("user_email", "")}
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero-box">

        <div class="hero-badge">
            🧠 DEEP LEARNING • GRAD-CAM
        </div>

        <h1>
            AI-Powered Blood<br>
            Smear Analysis
        </h1>

        <p>
            Upload a blood-smear image and explore an
            AI-generated classification with a visual
            explanation of the regions influencing the result.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FEATURES
# =========================================================

st.markdown(
    "<div class='section-heading'>How SickleScan works</div>",
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(
        """
        <div class="feature">
            <div class="feature-icon">🔬</div>
            <h3>Image Analysis</h3>
            <p>
                Upload a blood-smear image and let the
                trained deep-learning model analyze its
                visual characteristics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        """
        <div class="feature">
            <div class="feature-icon">🧠</div>
            <h3>AI Classification</h3>
            <p>
                InceptionV3 produces an image-level
                Normal or Sickle Cell classification
                with a probability score.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        """
        <div class="feature">
            <div class="feature-icon">🔥</div>
            <h3>Visual Explanation</h3>
            <p>
                Grad-CAM highlights regions that
                contributed to the model's prediction,
                making the result easier to interpret.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    "<div class='section-heading'>Analyze an image</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="upload-wrapper">

        <div class="upload-title">
            📤 Upload Blood-Smear Image
        </div>

        <div class="upload-subtitle">
            JPG, JPEG, PNG, TIFF • Best results with clear
            blood-smear images
        </div>

    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Drop your image here or browse",
    type=[
        "jpg",
        "jpeg",
        "png",
        "tif",
        "tiff"
    ]
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


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

    # =====================================================
    # PREDICTION
    # =====================================================

    with st.spinner(
        "🧠 SickleScan is analyzing your image..."
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

    st.markdown(
        "<div class='section-heading'>Analysis Result</div>",
        unsafe_allow_html=True
    )

    if probability >= 0.5:

        confidence = probability * 100

        st.markdown(
            f"""
            <div class="result-card result-positive">

                <div class="result-status">
                    🔴 Sickle Cell
                </div>

                <div class="result-caption">
                    Model prediction
                </div>

                <div class="result-score">
                    {confidence:.1f}%
                </div>

                <div class="result-caption">
                    prediction confidence
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        confidence = normal_probability * 100

        st.markdown(
            f"""
            <div class="result-card result-negative">

                <div class="result-status">
                    🟢 Normal
                </div>

                <div class="result-caption">
                    Model prediction
                </div>

                <div class="result-score">
                    {confidence:.1f}%
                </div>

                <div class="result-caption">
                    prediction confidence
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # PROBABILITY
    # =====================================================

    st.markdown(
        "<div class='section-heading'>Prediction Breakdown</div>",
        unsafe_allow_html=True
    )

    p1, p2 = st.columns(2)

    with p1:

        st.markdown(
            "<div class='prob-card'>",
            unsafe_allow_html=True
        )

        st.write("🟢 **Normal Probability**")

        st.progress(
            float(normal_probability)
        )

        st.caption(
            f"{normal_probability * 100:.1f}%"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with p2:

        st.markdown(
            "<div class='prob-card'>",
            unsafe_allow_html=True
        )

        st.write("🔴 **Sickle Cell Probability**")

        st.progress(
            float(probability)
        )

        st.caption(
            f"{probability * 100:.1f}%"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # =====================================================
    # GRAD-CAM
    # =====================================================

    st.markdown(
        "<div class='section-heading'>Model Explanation</div>",
        unsafe_allow_html=True
    )

    st.write(
        "Grad-CAM highlights image regions that influenced "
        "the model's prediction."
    )

    with st.spinner(
        "🔥 Generating Grad-CAM visualization..."
    ):

        heatmap = make_gradcam(
            image_array
        )


    cam1, cam2 = st.columns(2)

    with cam1:

        st.markdown(
            "<div class='cam-card'>"
            "<div class='cam-label'>Original Image</div>",
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with cam2:

        st.markdown(
            "<div class='cam-card'>"
            "<div class='cam-label'>Grad-CAM Visualization</div>",
            unsafe_allow_html=True
        )

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
            clear_figure=True,
            use_container_width=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# =========================================================
# ABOUT
# =========================================================

st.markdown(
    "<div class='section-heading'>About SickleScan</div>",
    unsafe_allow_html=True
)

with st.expander(
    "ℹ️ View project and model information"
):

    st.markdown(
        """
### About the Project

**SickleScan** is an academic deep-learning research
prototype designed for blood-smear image classification.

### Model

- **Architecture:** InceptionV3
- **Task:** Normal vs Sickle Cell
- **Input:** RGB blood-smear image
- **Input Size:** 224 × 224 pixels
- **Explainability:** Grad-CAM

### Prediction

The uploaded image is resized to 224 × 224 pixels
and passed through the trained InceptionV3 model.

### Important

SickleScan is an academic research prototype.
Its output should **not** be used as a substitute
for professional medical diagnosis.
"""
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        🩸 <b>SICKLESCAN</b><br><br>

        AI-Powered Blood Smear Analysis<br>

        Deep Learning Research Project •
        InceptionV3 • Grad-CAM

    </div>
    """,
    unsafe_allow_html=True
)
