import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from data_preprocessing import preprocess_data

def train_model(train_path):
    # Load train data
    df = pd.read_csv(train_path)
    df = preprocess_data(df, is_train=True)

    # Features and target
    X = df[['SibSp','Pclass','Sex','Age','Parch']]
    y = df['Survived']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save trained model
    joblib.dump(model, "models/random_forest.pkl")

    return model, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = train_model("data/raw/train.csv")
    print("✅ Model trained and saved to models/random_forest.pkl")
