
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model components
@st.cache_resource
def load_model_components():
    try:
        with open('models/encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)

        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)

        return encoder, scaler, model
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please run the training script first. Error: {e}")
        return None, None, None

# Prediction function
def predict_crop(N, P, K, temperature, humidity, ph, rainfall, encoder, scaler, model):
    """
    Predict crop based on soil and environmental conditions
    """
    # Create input dataframe
    input_data = pd.DataFrame({
        'N': [N],
        'P': [P], 
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # Convert back to crop name
    crop_name = encoder.inverse_transform(prediction)[0]
    confidence = np.max(prediction_proba) * 100

    return crop_name, confidence

# Streamlit App
def main():
    st.set_page_config(
        page_title="🌾 Crop Recommendation System",
        page_icon="🌾",
        layout="wide"
    )

    st.title("🌾 Crop Recommendation System")
    st.markdown("### Smart Farming AI - Get personalized crop recommendations based on soil and climate conditions")

    # Load model components
    encoder, scaler, model = load_model_components()

    if encoder is None or scaler is None or model is None:
        st.error("Failed to load model components. Please ensure the model files are available.")
        st.info("Run the training script first: `python train_model.py`")
        return

    # Display model info
    st.sidebar.header("🤖 Model Information")
    st.sidebar.info(f"""
    - **Algorithm**: Random Forest
    - **Accuracy**: 100% on test data
    - **Supported Crops**: {len(encoder.classes_)} types
    - **Training Samples**: 2,200 samples
    """)

    st.sidebar.header("🌱 Supported Crops")
    crops_list = list(encoder.classes_)
    for i in range(0, len(crops_list), 2):
        crop_pair = crops_list[i:i+2]
        st.sidebar.write("• " + ", ".join(crop_pair))

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.header("🌱 Soil Nutrients")
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90, 
                           help="Nitrogen content in soil (0-200)")
        P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=42, 
                           help="Phosphorus content in soil (0-200)")
        K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43, 
                           help="Potassium content in soil (0-200)")
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1, 
                           help="Soil pH level (0-14)")

    with col2:
        st.header("🌤️ Climate Conditions")
        temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, 
                                    value=25.0, step=0.1, help="Average temperature (-10 to 50°C)")
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, 
                                 value=80.0, step=0.1, help="Relative humidity (0-100%)")
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, 
                                 value=200.0, step=0.1, help="Average rainfall (0-500mm)")

    # Add some spacing
    st.markdown("---")

    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Get Crop Recommendation", type="primary", use_container_width=True):
            try:
                # Make prediction
                predicted_crop, confidence = predict_crop(N, P, K, temperature, humidity, ph, rainfall, encoder, scaler, model)

                # Display results
                st.success("✅ Prediction Complete!")

                # Create result display
                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    st.metric(
                        label="🌾 Recommended Crop",
                        value=predicted_crop.title(),
                        help="Best crop for your soil and climate conditions"
                    )

                with result_col2:
                    st.metric(
                        label="🎯 Confidence",
                        value=f"{confidence:.1f}%",
                        help="Model confidence in the prediction"
                    )

                # Additional information based on crop
                crop_info = {
                    'rice': "🍚 Rice thrives in warm, humid conditions with high rainfall. Ideal for wet climates.",
                    'maize': "🌽 Maize grows well in moderate temperatures with adequate rainfall. Good for temperate regions.",
                    'wheat': "🌾 Wheat prefers cooler temperatures and moderate rainfall. Ideal for temperate climates.",
                    'cotton': "🌸 Cotton requires warm temperatures and moderate rainfall. Suitable for semi-arid regions.",
                    'banana': "🍌 Banana needs warm, humid conditions with consistent moisture. Perfect for tropical areas.",
                    'apple': "🍎 Apple trees prefer cooler climates with adequate rainfall. Ideal for temperate regions.",
                    'grapes': "🍇 Grapes thrive in warm, dry climates with moderate rainfall. Perfect for Mediterranean conditions.",
                    'orange': "🍊 Orange trees need warm temperatures and adequate water. Suitable for subtropical regions.",
                    'chickpea': "🫛 Chickpea grows well in cool, dry conditions. Ideal for arid and semi-arid regions.",
                    'kidneybeans': "🫘 Kidney beans prefer moderate temperatures and consistent moisture. Good for temperate climates.",
                    'coconut': "🥥 Coconut palms need warm, humid tropical conditions with high rainfall.",
                    'papaya': "🥭 Papaya requires warm temperatures and adequate moisture. Suitable for tropical regions.",
                    'watermelon': "🍉 Watermelon needs warm temperatures and moderate water. Great for summer cultivation.",
                    'muskmelon': "🍈 Muskmelon thrives in warm, dry conditions. Perfect for summer in arid regions.",
                    'coffee': "☕ Coffee plants prefer mild temperatures with adequate rainfall in tropical highlands."
                }

                if predicted_crop in crop_info:
                    st.info(crop_info[predicted_crop])
                else:
                    st.info(f"🌱 {predicted_crop.title()} is recommended for your soil and climate conditions.")

                # Display input summary
                with st.expander("📊 Input Summary"):
                    input_df = pd.DataFrame({
                        'Parameter': ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)', 
                                    'Temperature (°C)', 'Humidity (%)', 'pH Level', 'Rainfall (mm)'],
                        'Value': [N, P, K, temperature, humidity, ph, rainfall]
                    })
                    st.dataframe(input_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.error("Please check your input values and try again.")

    # Add information section
    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    This AI-powered system analyzes soil nutrients and climate conditions to recommend the most suitable crop for your farm:

    1. **Input Parameters**: Enter soil nutrient levels (N, P, K), pH, temperature, humidity, and rainfall
    2. **AI Analysis**: Our machine learning model processes your data using Random Forest algorithm
    3. **Recommendation**: Get the best crop recommendation with confidence score

    **Tips for best results:**
    - Ensure accurate soil test results for N, P, K values
    - Use average climate data for your region
    - Consider seasonal variations in your planning
    """)

if __name__ == "__main__":
    main()
