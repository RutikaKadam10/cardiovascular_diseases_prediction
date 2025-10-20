#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit Host Application for Cardiovascular Disease Prediction

This module provides a web interface for predicting cardiovascular disease
using both tabular patient data and medical image analysis.
"""

import os
import logging
import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from joblib import load
from PIL import Image, ImageOps
from streamlit_option_menu import option_menu

# =========================
# Image inference settings
# =========================
# Most TF/Keras models with Inception/Xception backbones use [-1, 1] normalization.
IMG_PREPROCESS = "minus1_to1"   # options: "minus1_to1" or "zero_to_one"
# Which index in a 2-class output corresponds to the Positive (disease) class?
IMG_POSITIVE_INDEX = 1          # set to 0 if your model's positive class is index 0

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('streamlit_app.log')]
)
logger = logging.getLogger('CardiovascularPredictionApp')

# ---------------------------
# TensorFlow availability
# ---------------------------
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow is available and imported successfully")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow is not available. Running in demonstration mode.")

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Paths
# ---------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
tb_model_path = os.path.join(base_dir, "saved_models", "tb_mdl.h5")
img_model_path = os.path.join(base_dir, "saved_models", "img_mdl.h5")
scaler_path    = os.path.join(base_dir, "saved_models", "scaler.joblib")

logger.info(f"Base directory: {base_dir}")
logger.info(f"Tabular model path: {tb_model_path}")
logger.info(f"Image model path: {img_model_path}")

# ---------------------------
# Input ranges (tabular)
# ---------------------------
input_ranges = {
    "age":      {"min": 20, "max": 100, "help": "Patient's age in years (20-100)", "unit": "years"},
    "sex":      {"min": 0, "max": 1,   "help": "Patient's gender (0: female, 1: male)", "unit": ""},
    "cp":       {"min": 0, "max": 3,   "help": "Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)", "unit": ""},
    "trestbps": {"min": 90, "max": 200,"help": "Resting blood pressure (90-200)", "unit": "mm Hg"},
    "chol":     {"min": 120,"max": 570,"help": "Serum cholesterol (120-570)", "unit": "mg/dl"},
    "fbs":      {"min": 0, "max": 1,   "help": "Fasting blood sugar > 120 mg/dl (0: false, 1: true)", "unit": ""},
    "restecg":  {"min": 0, "max": 2,   "help": "Resting ECG results (0: normal, 1: ST-T wave abnormality, 2: LV hypertrophy)", "unit": ""},
    "thalach":  {"min": 70,"max": 220, "help": "Maximum heart rate (70-220)", "unit": "bpm"},
    "exang":    {"min": 0, "max": 1,   "help": "Exercise induced angina (0: no, 1: yes)", "unit": ""},
    "oldpeak":  {"min": 0, "max": 6.5, "help": "ST depression (0-6.5)", "unit": "mm"},
    "slope":    {"min": 0, "max": 2,   "help": "Slope of ST segment (0: upsloping, 1: flat, 2: downsloping)", "unit": ""},
    "ca":       {"min": 0, "max": 4,   "help": "Number of major vessels colored by fluoroscopy (0-4)", "unit": "vessels"},
    "thal":     {"min": 0, "max": 3,   "help": "Thalassemia (0: normal, 1: fixed defect, 2: reversible defect, 3: unknown)", "unit": ""}
}

# ---------------------------
# Helpers (tabular)
# ---------------------------
def generate_sample_data():
    try:
        return {
            "age": random.randint(40, 75),
            "sex": random.randint(0, 1),
            "cp": random.randint(0, 3),
            "trestbps": random.randint(110, 160),
            "chol": random.randint(170, 300),
            "fbs": random.randint(0, 1),
            "restecg": random.randint(0, 2),
            "thalach": random.randint(120, 190),
            "exang": random.randint(0, 1),
            "oldpeak": round(random.uniform(0, 3.5), 1),
            "slope": random.randint(0, 2),
            "ca": random.randint(0, 3),
            "thal": random.randint(0, 2),
        }
    except Exception as e:
        logger.error(f"Error generating sample data: {str(e)}")
        return {}

def validate_input(value, input_type):
    try:
        if value is None or value == "":
            return False
        val = float(value) if input_type == "oldpeak" else int(float(value))
        return input_ranges[input_type]["min"] <= val <= input_ranges[input_type]["max"]
    except Exception:
        return False

@st.cache_resource
def load_tabular_model():
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available - can't load model")
        return None, None

    model, scaler = None, None
    try:
        logger.info(f"Loading tabular model from {tb_model_path}")
        model = tf.keras.models.load_model(tb_model_path)
        logger.info("Tabular model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading tabular model: {str(e)}")

    try:
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = load(scaler_path)
        logger.info("Scaler loaded successfully")
    except Exception as e:
        logger.warning(f"Scaler not loaded: {e}")

    return model, scaler

def predict_from_tabular_data(inputs, model=None, scaler=None):
    # Demo fallback
    if not TENSORFLOW_AVAILABLE or model is None:
        seed = sum([float(x) for x in inputs])
        random.seed(seed)
        prediction_value = random.uniform(0, 1) * 0.8  # bias towards negative
        if prediction_value > 0.5:
            return ("The person is having heart disease", prediction_value)
        return ("The person does not have any heart disease", 1.0 - prediction_value)

    try:
        x = np.array([float(v) for v in inputs], dtype=np.float32).reshape(1, -1)
        if scaler is not None:
            x = scaler.transform(x)
        pred = model.predict(x, verbose=0)
        p = float(np.squeeze(pred))
        if p > 0.5:
            return ("The person is having heart disease", p)
        return ("The person does not have any heart disease", 1.0 - p)
    except Exception as e:
        logger.error(f"Error during tabular prediction: {str(e)}")
        raise

# ---------------------------
# Helpers (image)
# ---------------------------
@st.cache_resource
def load_image_model():
    """Load the image-based prediction model, robust to Keras3/legacy H5."""
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available - can't load image model")
        return None
    try:
        if not os.path.exists(img_model_path):
            logger.error(f"Image model not found at: {img_model_path}")
            return None

        logger.info(f"Loading image model from {img_model_path}")
        try:
            model = tf.keras.models.load_model(img_model_path, compile=False)
        except Exception as e1:
            logger.warning(f"Standard load failed, retrying with safe_mode=False: {e1}")
            model = tf.keras.models.load_model(img_model_path, compile=False, safe_mode=False)

        logger.info(f"Image model loaded successfully. Input shape: {model.input_shape}")
        return model
    except Exception as e:
        logger.error(f"Error loading image model: {str(e)}", exc_info=True)
        return None

def _prepare_image_for_model(img: Image.Image, model):
    """Resize & normalize image according to model input shape and IMG_PREPROCESS."""
    try:
        _, H, W, C = model.input_shape  # NHWC
    except Exception:
        H, W, C = 299, 299, 3

    # Ensure expected channel count
    img = img.convert("L" if C == 1 else "RGB")

    # Resize (center-crop style)
    img = ImageOps.fit(img, (W, H), Image.LANCZOS)

    x = np.asarray(img, dtype=np.float32)
    if C == 1 and x.ndim == 2:
        x = x[..., np.newaxis]

    # ---- normalization (critical!) ----
    if IMG_PREPROCESS == "minus1_to1":
        x = (x / 255.0) * 2.0 - 1.0
    else:  # "zero_to_one"
        x = x / 255.0

    return np.expand_dims(x, axis=0)  # (1, H, W, C)

def predict_class(img, model=None):
    """
    Predict from image. Returns (prob_positive, raw_pred_array).
    Respects IMG_PREPROCESS and IMG_POSITIVE_INDEX.
    """
    # Demo fallback
    if not TENSORFLOW_AVAILABLE or model is None:
        arr = np.array(img.convert("RGB"))
        pred_value = float(np.clip(np.mean(arr)/255.0, 0.0, 1.0)) * 0.4 + 0.3  # ~[0.3,0.7]
        return pred_value, np.array([[1.0 - pred_value, pred_value]], dtype=np.float32)

    try:
        x = _prepare_image_for_model(img, model)
        pred = model.predict(x, verbose=0)

        # (1,1) sigmoid OR (1,2) two-class
        if pred.ndim == 2 and pred.shape[1] == 1:
            prob_pos = float(pred[0, 0])  # sigmoid already P(positive)
        elif pred.ndim == 2 and pred.shape[1] == 2:
            p = pred[0]
            # If looks like logits, softmax it
            if not (np.all(p >= 0) and np.all(p <= 1) and np.isclose(p.sum(), 1.0, atol=1e-3)):
                e = np.exp(p - np.max(p))
                p = e / e.sum()
            prob_pos = float(p[IMG_POSITIVE_INDEX])
        else:
            # Unexpected shape: use last value as logit
            z = float(pred.flatten()[-1])
            prob_pos = 1.0 / (1.0 + np.exp(-z))

        return prob_pos, pred
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return None, None

# ---------------------------
# UI Pages
# ---------------------------
def display_tabular_prediction_page():
    st.title('Healthy Heart?: Predict, Prevent, Protect')
    st.write("""
    This tool predicts the likelihood of heart disease based on patient clinical data.
    Fill in all the fields below with the patient's information, or use the 'Test with Sample Data' button to try the system.
    """)

    tb_model, tb_scaler = load_tabular_model()
    if not TENSORFLOW_AVAILABLE:
        st.warning("⚠️ TensorFlow not available: running in demo mode (simulated predictions).")
    elif tb_model is None:
        st.error("❌ Failed to load the prediction model. Please check the logs and model file.")
    else:
        st.info(f"Model ✅ | Scaler: {'✅' if tb_scaler is not None else '⚠️ missing (using raw inputs)'}")

    if 'form_data' not in st.session_state:
        st.session_state.form_data = {key: "" for key in input_ranges.keys()}

    with st.container():
        st.subheader("Quick Test")
        st.write("Click the button to auto-fill a realistic sample.")
        if st.button('Test with Sample Data'):
            st.session_state.form_data = generate_sample_data()
            st.success("✅ Sample data loaded.")

        st.markdown("---")
        st.subheader("Patient Information")

        input_valid = {}
        col1, col2, col3 = st.columns(3)

        def create_input(column, field_name, label):
            with column:
                info = input_ranges[field_name]
                lbl = f"{label} ({info['unit']})" if info['unit'] else label
                value = st.text_input(lbl, value=st.session_state.form_data[field_name], help=info['help'])
                is_valid = validate_input(value, field_name) if value else False
                input_valid[field_name] = is_valid
                if value:
                    st.write("✅ Valid input" if is_valid else f"❌ Invalid. Range: {info['min']}-{info['max']}")
                return value

        age      = create_input(col1, "age", "Age")
        sex      = create_input(col2, "sex", "Sex")
        cp       = create_input(col3, "cp", "Chest Pain Type")
        trestbps = create_input(col1, "trestbps", "Resting Blood Pressure")
        chol     = create_input(col2, "chol", "Serum Cholesterol")
        fbs      = create_input(col3, "fbs", "Fasting Blood Sugar > 120 mg/dl")
        restecg  = create_input(col1, "restecg", "Resting ECG Results")
        thalach  = create_input(col2, "thalach", "Maximum Heart Rate")
        exang    = create_input(col3, "exang", "Exercise Induced Angina")
        oldpeak  = create_input(col1, "oldpeak", "ST Depression")
        slope    = create_input(col2, "slope", "Slope of ST Segment")
        ca       = create_input(col3, "ca", "Number of Major Vessels")
        thal     = create_input(col1, "thal", "Thalassemia")

        st.session_state.form_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }

        invalid = [f for f, ok in input_valid.items() if not ok and st.session_state.form_data[f]]
        if invalid:
            st.warning(f"⚠️ Invalid values for: {', '.join(invalid)}")

        st.markdown("---")
        st.subheader("Prediction")

        if st.button('Predict Heart Disease Risk', use_container_width=True, type="primary"):
            if all(input_valid.values()):
                try:
                    with st.spinner('Running prediction...'):
                        inputs = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
                        result, confidence = predict_from_tabular_data(inputs, tb_model, tb_scaler)

                        st.markdown("### Results")
                        if "having heart disease" in result:
                            st.error(f"#### {result}")
                            risk_level = "High"
                        else:
                            st.success(f"#### {result}")
                            risk_level = "Low"

                        st.write(f"Prediction confidence: {confidence*100:.2f}%")
                        st.write(f"Risk level: {risk_level}")
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    logger.error(f"Prediction error: {str(e)}")
            else:
                st.error("❌ Please fill in all fields with valid values.")

def display_image_prediction_page():
    st.title('Heart Disease Prediction Using Medical Images')
    st.write("""
    This tool analyzes heart scan images to predict the likelihood of cardiovascular disease.
    Upload a clear image of a heart scan (JPG/PNG) to get a prediction.
    """)

    if not TENSORFLOW_AVAILABLE:
        st.warning("⚠️ TensorFlow not available: image predictions will be simulated.")

    model = load_image_model()
    if TENSORFLOW_AVAILABLE and model is None:
        st.error("❌ Failed to load the image model. Please check the logs and model file.")
        st.info(f"Expected at: {img_model_path}")

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Upload Image")
        file = st.file_uploader("Select a heart scan image file", type=["jpg", "jpeg", "png"])
        if file is None:
            st.info("Waiting for image upload…")
            st.image("https://via.placeholder.com/400x300?text=Sample+Heart+Scan", caption="Sample heart scan image")

    with col2:
        if file is not None:
            try:
                with st.spinner('Processing image...'):
                    test_image = Image.open(file)
                    st.image(test_image, caption="Uploaded Image", width=400)

                    prob_pos, raw = predict_class(test_image, model)
                    if prob_pos is not None:
                        result = 'Positive' if prob_pos >= 0.5 else 'Negative'
                        confidence = prob_pos if result == 'Positive' else (1.0 - prob_pos)

                        st.subheader("Prediction Result")
                        if result == 'Positive':
                            st.error("#### Cardiovascular Disease Detected")
                            risk_status = "High Risk"
                        else:
                            st.success("#### No Cardiovascular Disease Detected")
                            risk_status = "Low Risk"

                        st.write(f"Prediction: {result}")
                        st.write(f"Confidence: {confidence*100:.2f}%")
                        st.write(f"Risk Status: {risk_status}")

                        # Optional probability bars
                        neg = 1.0 - prob_pos
                        fig, ax = plt.subplots(figsize=(8, 3))
                        bars = ax.bar(['Negative', 'Positive'], [neg, prob_pos])
                        for b in bars:
                            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                                    f'{b.get_height()*100:.1f}%', ha='center', va='bottom')
                        ax.set_ylim(0, 1.0)
                        ax.set_ylabel('Probability')
                        ax.set_title('Prediction Probabilities')
                        st.pyplot(fig)
                    else:
                        st.error("❌ Failed to process the image. The model could not generate a prediction.")
            except Exception as e:
                st.error(f"❌ Error processing the image: {str(e)}")
                logger.error(f"Image processing error: {str(e)}")

# ---------------------------
# Main
# ---------------------------
def main():
    try:
        st.markdown("""
        <style>
        .main-header { font-size: 2.5rem; color: #1E88E5; }
        .sub-header  { font-size: 1.5rem; color: #424242; }
        .info-text   { font-size: 1rem;  color: #616161; }
        .centered    { text-align: center; }
        </style>
        """, unsafe_allow_html=True)

        with st.sidebar:
            st.markdown('<p class="main-header centered">Cardiovascular Disease Prediction System</p>', unsafe_allow_html=True)
            st.markdown('<p class="info-text">Choose a prediction method below</p>', unsafe_allow_html=True)

            selected = option_menu(
                menu_title=None,
                options=['Predict with Clinical Data', 'Predict with Heart Scan'],
                icons=['clipboard-data', 'image'],
                default_index=0
            )

            st.markdown("---")
            st.markdown('<p class="sub-header">About</p>', unsafe_allow_html=True)
            st.markdown("""
            This application uses deep learning models to predict cardiovascular disease risk
            using either clinical data or medical images. The models have been trained on validated
            datasets and provide risk assessments for educational purposes only.
            """)

        if selected == 'Predict with Clinical Data':
            display_tabular_prediction_page()
        else:
            display_image_prediction_page()

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
