import streamlit as st
import requests
import base64

st.set_page_config(page_title="Natural Disaster Risk Predictor", page_icon="🌍")

st.title("🌍 Natural Disaster Risk Predictor")
st.markdown("Upload an image (e.g., flood, wildfire, earthquake) and detect one or more disaster categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict", type="primary"):
        with st.spinner("Analyzing image..."):
            bytes_data = uploaded_file.read()
            encoded = base64.b64encode(bytes_data).decode("utf-8")

            try:
                response = requests.post(
                    "http://localhost:8000/predict",
                    json={"image_base64": encoded}
                )

                if response.status_code == 200:
                    predictions = response.json().get("prediction", [])
                    if predictions:
                        if isinstance(predictions, list):
                            # Format labels nicely: Earthquake, Flood, etc.
                            formatted = ", ".join(pred.capitalize().replace("_", " ") for pred in predictions)
                            st.success(f"✅ Predicted Classes: **{formatted}**")
                        else:
                            # If it's a single string
                            st.success(f"✅ Predicted Class: **{predictions.capitalize()}**")
                    else:
                        st.warning("⚠️ No disaster type confidently detected.")
                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"❌ Could not connect to backend. Error: {e}")
