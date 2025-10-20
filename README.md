🫀 Cardiovascular Disease Prediction Project

This project develops a deep learning–based system for predicting cardiovascular disease (CVD) using both structured clinical data and unstructured medical images.
It integrates Artificial Neural Networks (ANNs) for tabular data and Convolutional Neural Networks (CNNs) for image analysis, delivering a complete, dual-modal pipeline for disease risk assessment.

🔗 Repository: https://github.com/RutikaKadam10/cardiovascular_diseases_prediction.git

🌐 1. Project Overview

Cardiovascular diseases remain the leading cause of mortality worldwide. Early detection and prediction can significantly improve clinical outcomes.
This project combines traditional health indicators with imaging data to build a unified AI system capable of estimating the likelihood of CVD with high accuracy.

🔍 Objectives

Predict the presence of CVD from demographic, clinical, and lifestyle features.

Identify key predictive factors contributing most to disease risk.

Analyze heart-scan images to detect subtle structural patterns linked to cardiovascular abnormalities.

Provide an interactive, easy-to-use interface for model inference and visualization.

💻 2. Project Interface

The Cardiovascular Disease Prediction App provides an intuitive user interface for medical practitioners, researchers, and students.

Key Features

🧾 Clinical Data Input: Enter patient details such as age, cholesterol, blood pressure, etc.

🧠 Model-Driven Predictions: Backend integrates trained ANN and CNN models.

📊 Risk Visualization: Displays prediction probability and confidence levels.

🩺 Dual-Mode Inference: Supports both structured tabular data and heart-scan images.

🎯 3. Research Objectives
3.1 Objective 1: Predicting CVD Presence

Goal: Build accurate machine learning models that classify individuals as “having CVD” or “not having CVD” using clinical data.

Key Steps

Data Collection from medical repositories (UCI, Mendeley)

Preprocessing (missing data handling, normalization, outlier removal)

Feature Selection and Model Comparison

Evaluation using Accuracy, Precision, Recall, F1-Score, and ROC-AUC

3.2 Objective 2: Identifying Key Predictors

Goal: Identify critical features that most influence CVD prediction.
Insights from feature importance help clinicians understand which risk factors (age, cholesterol, blood pressure, etc.) have the strongest predictive power.

📁 4. Project Structure
CARDIOVASCULAR_DISEASES_PREDICTION/
├─ Code/
│  ├─ idmpcd.py                 # Tabular model (ANN)
│  ├─ idmcad_(images).py        # Image model (CNN)
│  └─ app.py                    # Streamlit interface
├─ dataset/
│  ├─ Tabular_data.csv          # Clinical dataset
│  └─ Test_images/              # Sample heart scans
├─ saved_models/
│  ├─ tb_mdl.h5                 # Trained tabular model
│  ├─ img_mdl.h5                # Trained image model
│  └─ scaler.joblib             # Preprocessing scaler
├─ checkpoints/                 # Model checkpoints
├─ visualizations/              # EDA and plots
├─ evaluation/                  # Confusion matrix, ROC, reports
├─ requirements.txt
└─ README.md

⚙️ 5. Environment Setup (with uv)

This project uses the uv virtual environment for fast, modern Python dependency management.

# Create a new uv environment
uv venv

# Activate the environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt


Alternatively, if using Conda:

conda env create --name cvd_env python==3.9
conda activate cvd_env

🔧 6. Deployment

To deploy this project locally, follow these steps:

Clone the repository

git clone https://github.com/RutikaKadam10/cardiovascular_diseases_prediction.git
cd cardiovascular_diseases_prediction


Install dependencies

pip install -r requirements.txt


(Optional) Create a virtual environment

uv venv
uv pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py


Open the provided local URL (usually http://localhost:8501) in your browser.

🧠 7. Training Workflow
Tabular Model (Code/idmpcd.py)

Loads Tabular_data.csv

Performs preprocessing, scaling, model training, and evaluation

Exports:

saved_models/tb_mdl.h5

saved_models/scaler.joblib

Image Model (Code/idmcad_(images).py)

Loads dataset with labeled image directories

Performs data augmentation, CNN training, and evaluation

Exports:

saved_models/img_mdl.h5

🧩 8. Tech Stack
Category	Tools
Language	Python
Data Analysis	Pandas, NumPy
Visualization	Matplotlib, Seaborn, Plotly, Bokeh
Deep Learning	TensorFlow, Keras, scikit-learn, FastAI, PyTorch Lightning
Frontend	Streamlit
Environment	uv (virtual environment), Conda
Logging & Tracking	TensorBoard, tqdm
📈 9. Results & Insights

High predictive accuracy achieved across both data types.

Cholesterol, Age, and Blood Pressure emerged as top predictors.

CNN visualization shows clear distinction between healthy and CVD-affected scans.

Streamlit UI provides dual-mode interaction for real-time inference.

🧭 10. Future Enhancements

Integrate Explainable AI (XAI): Grad-CAM, SHAP for interpretability.

Extend dataset diversity for better generalization.

Deploy containerized version using Docker + Azure/AWS.

Integrate live patient data pipelines for continuous learning.

👩‍💻 11. Author

Rutika Avinash Kadam
Graduate Research Assistant – Stony Brook Medicine
M.S. in Data Science, Stony Brook University

📧 rutikakadam2727@gmail.com

🔗 LinkedIn

💻 GitHub

⚖️ 12. Disclaimer

This project is for academic and research purposes only.
It is not intended for clinical use or medical decision-making.
Use responsibly and comply with healthcare data ethics standards.