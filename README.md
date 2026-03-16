# Heart Disease Detection using Machine Learning

### UCI Heart Disease Dataset | Streamlit Deployment

## Overview

This project implements a **Machine Learning based Heart Disease Detection System** using the **UCI Heart Disease Dataset**. The application predicts whether a patient is likely to have heart disease based on several clinical parameters such as age, chest pain type, cholesterol level, resting blood pressure, and other medical indicators.

The project demonstrates a **complete end-to-end Machine Learning workflow**, including:

- Data acquisition from the UCI repository
- Data preprocessing and feature engineering
- Ensemble model training using **StackingClassifier**
- Model evaluation with multiple metrics
- Model persistence using `joblib`
- Deployment of the trained model using **Streamlit**

The application allows users to input medical attributes through an interactive web interface and receive a prediction about the likelihood of heart disease.

---

## Live Application

The app is deployed on Streamlit Cloud and the application can be accessed through a web interface where users can enter patient attributes and obtain predictions instantly.

App Link: https://uciheartdiseasedetection.streamlit.app/

---

## Dataset

The dataset used in this project is the **UCI Heart Disease Dataset**, a widely used dataset for research in medical machine learning.

It contains **303 patient records** with **13 clinical features**.

### Features Used

| Feature  | Description                                    |
| -------- | ---------------------------------------------- |
| age      | Age of the patient                             |
| sex      | Gender (1 = male, 0 = female)                  |
| cp       | Chest pain type                                |
| trestbps | Resting blood pressure                         |
| chol     | Serum cholesterol (mg/dl)                      |
| fbs      | Fasting blood sugar                            |
| restecg  | Resting electrocardiographic results           |
| thalach  | Maximum heart rate achieved                    |
| exang    | Exercise induced angina                        |
| oldpeak  | ST depression induced by exercise              |
| slope    | Slope of peak exercise ST segment              |
| ca       | Number of major vessels colored by fluoroscopy |
| thal     | Thalassemia status                             |

Target variable:

- **0 → No heart disease**
- **1 → Presence of heart disease**

The original dataset contains disease severity levels (0–4). In this project, it is converted into a **binary classification problem**.

---

## Machine Learning Pipeline

The model training pipeline consists of the following steps:

1. **Data Loading**
   - Dataset fetched directly from the UCI repository using `ucimlrepo`.

2. **Train-Test Split**
   - Stratified split to maintain class balance.

3. **Preprocessing**
   - Missing value handling using **SimpleImputer**
   - Feature scaling using **StandardScaler**
   - Implemented inside a **scikit-learn Pipeline**

4. **Model Architecture**

A **Stacking Ensemble Model** is used.

Base models:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree
- Random Forest

Final meta-model:

- **XGBoost Classifier**

This ensemble architecture improves predictive performance by combining the strengths of multiple algorithms.

5. **Evaluation Metrics**
   - Accuracy Score
   - Classification Report
   - Confusion Matrix

6. **Model Persistence**
   - Trained pipeline saved using `joblib`.

---

## Model Architecture

```
Input Features
      │
      ▼
Data Preprocessing
(SimpleImputer + StandardScaler)
      │
      ▼
Stacking Ensemble
 ├── KNN
 ├── Logistic Regression
 ├── Decision Tree
 └── Random Forest
      │
      ▼
Meta Model
   XGBoost
      │
      ▼
Final Prediction
```

---

## Streamlit Web Application

The project includes a **Streamlit-based web application** that provides a user-friendly interface for prediction.

Users can enter medical attributes such as:

- Age
- Sex
- Chest Pain Type
- Blood Pressure
- Cholesterol
- ECG Results
- Exercise Induced Angina
- Thalassemia Type

After entering the inputs, the application processes the data and returns a prediction indicating whether the patient is likely to have heart disease.

### Application Features

- Interactive UI with sliders and dropdown inputs
- Real-time prediction using trained ML model
- Clean and simple interface for user interaction
- Fast inference using cached model loading

---

## Project Structure

```
ml_classification_heart_disease_detection_app_2_UCI_dataset
│
├── training.ipynb        # Model training and evaluation
├── app.py                # Streamlit web application
├── pipeline.pkl          # Saved ML pipeline
├── col_names.pkl         # Feature column order
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/itspriyambhattacharya/ml_classification_heart_disease_detection_app_2_UCI_dataset.git
cd ml_classification_heart_disease_detection_app_2_UCI_dataset
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Run the Streamlit app locally:

```bash
streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

## Technologies Used

- Python
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

## Future Improvements

Possible improvements for this project include:

- ROC-AUC evaluation and visualization
- Hyperparameter tuning using GridSearchCV
- Model explainability using SHAP
- Probability-based risk scoring
- Improved UI with visualization dashboards

---

## Disclaimer

This project is developed for **educational and research purposes only**.
It should **not be used for medical diagnosis or clinical decision-making** without professional medical consultation.

---

## Author

**Priyam Bhattacharya**
M.Sc. Computer Science
University of Calcutta

GitHub:
https://github.com/itspriyambhattacharya
