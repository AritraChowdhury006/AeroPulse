import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load trained model
clfr = joblib.load("heartmodel.pkl")

def preprocess(age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal):
    # Validate inputs
    inputs = [age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal]
    if '' in inputs or None in inputs:
        raise ValueError("❌ One or more fields are empty. Please fill out all inputs.")

    # Encode categorical values
    sex = 1 if sex.lower() == "male" else 0

    cp_map = {
        "Typical angina": 0,
        "Atypical angina": 1,
        "Non-anginal pain": 2,
        "Asymptomatic": 3
    }
    cp = cp_map.get(cp, 0)

    exang = 1 if exang == "Yes" else 0
    fbs = 1 if fbs == "Yes" else 0

    slope_map = {
        "Upsloping: better heart rate with excercise(uncommon)": 0,
        "Flatsloping: minimal change(typical healthy heart)": 1,
        "Downsloping: signs of unhealthy heart": 2
    }
    slope = slope_map.get(slope, 1)

    thal_map = {
        "fixed defect: used to be defect but ok now": 6,
        "reversable defect: no proper blood movement when excercising": 7,
        "normal": 2.31
    }
    thal = thal_map.get(thal, 2.31)

    restecg_map = {
        "Nothing to note": 0,
        "ST-T Wave abnormality": 1,
        "Possible or definite left ventricular hypertrophy": 2
    }
    restecg = restecg_map.get(restecg, 0)

    # Prepare input
    user_input = np.array([[age, sex, cp, trestbps, restecg, chol, fbs,
                            thalach, exang, oldpeak, slope, ca, thal]])

    # Load and scale training data
    df = pd.read_csv('heart.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler().fit(X_train)

    # Transform input
    user_input_scaled = scaler.transform(user_input)

    # Predict
    prediction = clfr.predict(user_input_scaled)
    return int(prediction)
