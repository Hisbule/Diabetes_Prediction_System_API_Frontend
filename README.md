# Diabetes_Prediction_System-_API_Frontend
The Diabetes Prediction System is a complete end-to-end machine learning project designed to predict the likelihood of diabetes based on medical attributes.
It includes:

A machine learning backend (Python, scikit-learn, FastAPI)

A frontend web interface for user interaction

A REST API to connect the frontend with the trained model

The system uses various ML algorithms and automatically selects the best-performing model for deployment.

## 🚀 Project Features

- ✅ Data preprocessing and feature scaling
- ✅ Model training using multiple ML algorithms
- ✅ Automatic selection of the best model
- ✅ RESTful API built with FastAPI
- ✅ Frontend web interface for easy predictions
- ✅ Scalable deployment (Docker + Render)



## 🧠 Machine Learning Models

The following supervised ML algorithms are trained and compared:

- Logistic Regression

- Random Forest Classifier

- Support Vector Machine (SVM)

The model with the highest accuracy is selected and saved for deployment.


## 📊 Dataset

- File: diabetes.csv

- Target Column: Outcome

- Description: Contains patient health data (e.g., glucose level, BMI, blood pressure) used to predict diabetes likelihood.

## ⚙️ How It Works

1. 📥 Data Loading: Load and split diabetes.csv into training and test sets.

2. 📊 Preprocessing: Standardize feature values using StandardScaler.

3. 🤖 Model Training: Train three ML models and evaluate them.

4. 🏆 Model Selection: Choose the best model based on test accuracy.

5. 💾 Saving: Save the trained model and scaler with joblib.

6. 🌐 Deployment: Serve predictions through a FastAPI backend and frontend UI.


## 🌐 Deployment Links

- 🌍 Frontend App: https://diabetify-api.onrender.com

- ⚙️ Backend API (Swagger Docs): https://diabetes-prediction-system-api-frontend.onrender.com/docs



## 📚 Technologies Used

- 🐍 Python 3.x

- 📊 Pandas, Scikit-learn

- ⚡ FastAPI

- 🐳 Docker

- ☁️ Render (Deployment)

## 🧑‍💻 Author

Hafiz Uddin Ahmed Adnan
🎓 CSE Undergraduate Student – University of Liberal Arts Bangladesh



## 🔮 Future Improvements

- 📈 Add deep learning models (e.g., ANN)

- 🔐 Implement user authentication

- 📊 Add a data visualization dashboard
