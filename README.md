# Diabetes Prediction System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![GitHub Release](https://img.shields.io/github/v/release/mdsaad31/diabetes-prediction)
![Last Commit](https://img.shields.io/github/last-commit/mdsaad31/diabetes-prediction)
![Code Size](https://img.shields.io/github/languages/code-size/mdsaad31/diabetes-prediction)
![Repo Size](https://img.shields.io/github/repo-size/mdsaad31/diabetes-prediction)
![Open Issues](https://img.shields.io/github/issues/mdsaad31/diabetes-prediction)
![Contributors](https://img.shields.io/github/contributors/mdsaad31/diabetes-prediction)

A machine learning system that predicts diabetes risk with 85% accuracy, featuring an intuitive web interface with detailed risk analysis and personalized health recommendations.

## Features

- 🎯 **Accurate Predictions**: XGBoost model with 85% accuracy
- 💻 **User-Friendly Interface**: Clean Streamlit web app
- 📊 **Detailed Analysis**: Risk factors breakdown with visual indicators
- 💡 **Personalized Recommendations**: Actionable health advice
- 📱 **Responsive Design**: Works on desktop and mobile
- 🔍 **Model Interpretability**: Feature importance visualization

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Steps
1. Clone the repository:
```bash
git clone https://github.com/mdsaad31/diabetes-prediction.git
cd diabetes-prediction
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
streamlit run app/diabetes_app.py
```
The app will launch in your default browser at http://localhost:8501

## Usage
1. Fill in your health metrics in the web form
2. Get your risk assessment with probability score
3. Review detailed analysis of contributing factors
4. Follow personalized recommendations for risk reduction

## 📊 Model Performance

The trained XGBoost model achieved the following performance metrics on the test dataset:

| Metric     | Score  |
|------------|--------|
| Accuracy   | 85.2%  |
| AUC (ROC)  | 0.91   |
| Precision  | 0.83   |
| Recall     | 0.76   |
| F1 Score   | 0.79   |

## API Documentation
For developers wanting to integrate with the prediction model:
```bash
import joblib
import numpy as np
import os

# Load model and scaler using relative paths
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'diabetes_xgb_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'diabetes_scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Prepare input data (example)
input_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Standardize and predict
scaled_data = scaler.transform(input_data)
prediction = model.predict(scaled_data)
probability = model.predict_proba(scaled_data)[:, 1]
```

## 🤝 Contributing

We welcome contributions from the community to improve this project. To contribute:

1. **Fork** the repository  
2. **Clone** your forked repo:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction.git
   ```
3. **Create** a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Make your changes and commit them:
   ```bash
   git commit -m "Add your message"
   ```
5. **Push** to your branch:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request** and briefly describe your changes:
Please make sure your code follows the existing style and passes all tests. Refer to the code of conduct for respectful collaboration.

## 🙏 Acknowledgments

- Pima Indians Diabetes Dataset – for providing the data used
- Streamlit – for powering the interactive web interface
- XGBoost – for the machine learning model
- Scikit-learn – for preprocessing and evaluation support
- All open-source contributors and libraries that made this project possible

## 📬 Contact

If you have any questions, suggestions, or feedback, feel free to reach out:

**Md Saad**  
📧 Email: [mohammedsaad0462@gmail.com](mailto:mohammedsaad0462@gmail.com)  
💼 GitHub: [@mdsaad31](https://github.com/mdsaad31) 

---

![GitHub stars](https://img.shields.io/github/stars/mdsaad31/diabetes-prediction?style=social)
![GitHub forks](https://img.shields.io/github/forks/mdsaad31/diabetes-prediction?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/mdsaad31/diabetes-prediction?style=social)
