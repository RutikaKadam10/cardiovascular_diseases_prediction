# 🫀 Cardiovascular Disease Prediction Project

This project aims to develop a **deep learning model** for predicting **Cardiovascular Disease (CVD)** by utilizing both **Artificial Neural Networks (ANN)** and **Convolutional Neural Networks (CNN)** to analyze **structured clinical data** and **unstructured medical images**.

---

## 🎨 1. Cardiovascular Disease Prediction Project Interface

This interface demonstrates how users interact with the system.

### ✨ Key Features:
- 🧾 **Intuitive data input fields** for easy and accurate user input.  
- 🧠 **Integrated machine learning models (ANN & CNN)** for back-end prediction.  
- 📊 **Clear visual representation** of results and prediction confidence.  
- 🔄 **Dual-mode inference:** Predict via **clinical data** or **medical heart scan images**.  
- 🖥️ **Built with Streamlit** for an interactive, responsive web interface.

---

### 🖼️ Project UI Snapshots

![Cardiovascular Disease Prediction Project Interface](figures/interface.png)  
![Cardiovascular Disease Prediction Project Interface](figures/interface2.png)

---

## ⚙️ 2. Use Case Diagram

The following diagram illustrates how users interact with the system.

![Use Case Diagram](figures/Use_Case_Diagram.png)

---

## 🔁 3. Project Implementation Workflow

The overall workflow of the project involves the following stages:

- **Data Collection**
- **Preprocessing & Feature Engineering**
- **Model Training (ANN & CNN)**
- **Evaluation & Visualization**
- **Deployment (Streamlit UI)**

![Project Implementation](figures/implementation.png)

---

## 🎯 4. Research Objectives

This research focuses on two fundamental questions related to cardiovascular disease (CVD) prediction.

### 🧩 **1. Can we accurately predict the presence of cardiovascular disease using clinical features?**

**Goal:**  
To determine whether machine learning models can effectively predict if a person has CVD based on clinical health indicators.

**Input Feature Categories:**
- **Demographic:** Age, gender  
- **Lifestyle:** Smoking, physical activity, alcohol intake  
- **Medical History:** Diabetes, hypertension, prior heart conditions  
- **Clinical Metrics:** Blood pressure, cholesterol, BMI, etc.

**Methodology Steps:**
1. **Data Collection** – Acquire labeled, comprehensive datasets (e.g., UCI, Mendeley).  
2. **Preprocessing** – Handle missing values, normalize scales, and clean data.  
3. **Feature Selection** – Identify the most impactful predictive attributes.  
4. **Model Evaluation** – Compare models (Logistic Regression, Decision Tree, Random Forest, ANN, CNN) using metrics such as **Accuracy, Precision, Recall, F1-score, ROC-AUC**.

---

### 💡 **2. What are the most important factors contributing to CVD prediction?**

**Objective:**  
To identify and rank the clinical and lifestyle factors most responsible for cardiovascular disease risk.

**Examples of Analytical Questions:**
- Are **cholesterol levels** more influential than **blood pressure**?  
- Does **age** contribute more to risk than **smoking** or **physical inactivity**?

**Outcome:**  
Feature-importance results guide clinicians toward high-impact diagnostic variables, improving both interpretability and medical decision-making.

---

### 🌍 **Significance of These Objectives**

- **For Healthcare:** Early detection and personalized care through predictive analytics.  
- **For Prevention:** Data-driven identification of high-risk populations.  
- **For Research:** Foundation for explainable AI in medical diagnostics.

---

## 📂 5. Project Structure

```
cardiovascular_diseases_prediction/
├─ Code/
│  ├─ idmpcd.py                # ANN for tabular/clinical data
│  ├─ idmcad_(images).py       # CNN for heart-scan images
│  └─ app.py                   # Streamlit front-end app
├─ dataset/
│  ├─ Tabular_data.csv         # Structured data file
│  └─ Test_images/             # Unstructured image data
├─ saved_models/
│  ├─ tb_mdl.h5                # Trained ANN model
│  ├─ img_mdl.h5               # Trained CNN model
│  └─ scaler.joblib            # Preprocessing scaler
├─ checkpoints/                # Saved model checkpoints
├─ evaluation/                 # Reports & confusion matrices
├─ visualizations/              # Data analysis & ROC curves
├─ requirements.txt
└─ README.md
```

---

## 🧠 6. Model Overview

### 🧾 **ANN – Tabular Data**
- Input: 13 clinical features (age, cholesterol, BP, etc.)  
- Architecture: Dense → BatchNorm → Dropout (multi-layer perceptron)  
- Optimizer: Adam | Loss: Binary Cross-Entropy  
- Metrics: Accuracy, Precision, Recall, AUC

### 🩺 **CNN – Image Data**
- Input: Heart scan images (RGB, 299×299)  
- Layers: Conv2D → MaxPooling → Dense → Softmax  
- Regularization: Data Augmentation + Dropout  
- Output: 2-class prediction (Healthy / CVD)

---

## 🧰 7. Tech Stack

| Category | Tools & Libraries |
|-----------|------------------|
| **Languages** | Python |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly, Bokeh |
| **Deep Learning** | TensorFlow, Keras, scikit-learn, FastAI, PyTorch Lightning |
| **Frontend** | Streamlit |
| **Environment** | `uv` Virtual Environment |
| **Tracking & Logging** | TensorBoard, tqdm |
| **Version Control** | Git & GitHub |
| **Development Tools** | Visual Studio Code, Google Colab |

---

## 🧭 8. Environment Setup & Dependencies

### Using **uv** (recommended)
```bash
# Create virtual environment
uv venv

# Activate environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install all dependencies
uv pip install -r requirements.txt
```

### Alternative (using Conda)
```bash
conda env create --name cvd_env python==3.9
conda activate cvd_env
```

---

## 🚀 9. Deployment

To deploy this project locally, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/RutikaKadam10/cardiovascular_diseases_prediction.git
   cd cardiovascular_diseases_prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Application**
   ```bash
   streamlit run app.py
   ```

4. **Access the app**
   - Open the local URL displayed in your terminal (e.g., `http://localhost:8501`).
   - Choose between **Clinical Data Prediction** or **Heart-Scan Prediction** in the sidebar.

---

## 📊 10. Evaluation Metrics & Results

- **Accuracy:** High overall predictive performance across both modalities  
- **Precision & Recall:** Balanced recall for early disease detection  
- **ROC-AUC:** Strong discrimination between positive and negative cases  
- **Feature Importance:** Age, cholesterol, and BP emerged as top indicators  

---

## 👩‍💻 12. Author

**Rutika Avinash Kadam**  
Graduate Research Assistant – Stony Brook Medicine  
M.S. Data Science, Stony Brook University  

📧 [rutikakadam2727@gmail.com](mailto:rutikakadam2727@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/rutika-avinash/)  
💻 [GitHub](https://github.com/RutikaKadam10)

---


