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
