import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Sickle Cell Detection",
    page_icon="🩸",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("InceptionV3_UCL_Final.keras")

model = load_model()

# -----------------------------
# Grad-CAM
# -----------------------------
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
        outputs=[target_layer.output, backbone.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, backbone_features = grad_model(
            image_array,
            training=False
        )

        x = model.get_layer("global_average_pooling2d")(
            backbone_features
        )

        x = model.get_layer("dropout")(
            x,
            training=False
        )

        prediction = model.get_layer("dense")(x)

        loss = prediction[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


# -----------------------------
# UI
# -----------------------------
st.title("🩸 Sickle Cell Disease Detection")

st.write(
    "Upload a blood-smear image to obtain a prediction "
    "using the trained InceptionV3 model."
)

uploaded_file = st.file_uploader(
    "Upload Blood-Smear Image",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    # Same preprocessing used during model training
    img = image.resize((224, 224))
    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    probability = float(model.predict(img_array, verbose=0)[0][0])

    if probability >= 0.5:
        result = "Sickle Cell"
        confidence = probability * 100
    else:
        result = "Normal"
        confidence = (1 - probability) * 100

    st.subheader("Prediction")

    if result == "Sickle Cell":
        st.error(f"🔴 {result}")
    else:
        st.success(f"🟢 {result}")

    st.write(f"Confidence: **{confidence:.2f}%**")

    # -----------------------------
    # Grad-CAM
    # -----------------------------
    st.subheader("Grad-CAM Visualization")

    heatmap = make_gradcam(img_array)

    fig, ax = plt.subplots()

    ax.imshow(image)
    ax.imshow(
        heatmap,
        cmap="jet",
        alpha=0.4,
        extent=(
            0,
            image.width,
            image.height,
            0
        )
    )

    ax.axis("off")

    st.pyplot(fig)

    st.info(
        "Grad-CAM highlights image regions that contributed "
        "to the model prediction."
    )

st.markdown("---")

st.caption(
    "Research prototype only. This system is not intended "
    "for clinical diagnosis."
)
