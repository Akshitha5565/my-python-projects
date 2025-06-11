Python 3.13.3 (tags/v3.13.3:6280bb5, Apr  8 2025, 14:47:33) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> import pickle
... import numpy as np
... import pandas as pd
... from tensorflow.keras.models import load_model
... from sklearn.preprocessing import StandardScaler
... 
... # Load pre-trained models
... diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))
... heart_model = pickle.load(open("models/heart_model.pkl", "rb"))
... cancer_model = pickle.load(open("models/cancer_model.pkl", "rb"))
... malaria_model = load_model("models/malaria_model.h5")  # CNN model for malaria
... 
... # Function for Diabetes Prediction
... def predict_diabetes(data):
...     scaler = StandardScaler()
...     data = scaler.fit_transform([data])
...     return diabetes_model.predict(data)[0]
... 
... # Function for Heart Disease Prediction
... def predict_heart_disease(data):
...     scaler = StandardScaler()
...     data = scaler.fit_transform([data])
...     return heart_model.predict(data)[0]
... 
... # Function for Breast Cancer Prediction
... def predict_cancer(data):
...     scaler = StandardScaler()
...     data = scaler.fit_transform([data])
...     return cancer_model.predict(data)[0]
... 
... # Function for Malaria Prediction (Image-based)
... def predict_malaria(image):
...     image = image.resize((64, 64))
...     image = np.array(image) / 255.0
...     image = image.reshape(1, 64, 64, 3)
...     prediction = malaria_model.predict(image)
