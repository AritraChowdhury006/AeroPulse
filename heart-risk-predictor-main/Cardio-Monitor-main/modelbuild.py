import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib

warnings.filterwarnings('ignore')

def load_csv_data(filepath="heart.csv"):
    try:
        df = pd.read_csv(filepath)
        df.drop_duplicates(inplace=True)
        required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
        scaler = MinMaxScaler()
        df[required_columns[:-1]] = scaler.fit_transform(df[required_columns[:-1]])
        print(f"✅ Loaded {len(df)} records from {filepath}")
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        raise

def bulidmodel():
    df = load_csv_data()
    y = df["target"]
    X = df.drop("target", axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"📊 Model accuracy: {accuracy:.4f}")
    joblib.dump(model, "heartmodel.pkl")
    print("✅ Model saved as heartmodel.pkl")

if __name__ == "__main__":
    bulidmodel()
