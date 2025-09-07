import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessing import preprocess_data

def evaluate_model(test_path):
    # Load data
    df = pd.read_csv("data/raw/train.csv")
    df = preprocess_data(df, is_train=True)

    X = df[['SibSp','Pclass','Sex','Age','Parch']]
    y = df['Survived']

    # Load model
    model = joblib.load("models/random_forest.pkl")

    # Predictions
    y_pred = model.predict(X)

    # Evaluation
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print("Classification Report:\n", classification_report(y, y_pred))

def generate_submission(test_path):
    df_test = pd.read_csv(test_path)
    df_test = preprocess_data(df_test, is_train=False)

    X_test_final = df_test[['SibSp','Pclass','Sex','Age','Parch']]

    model = joblib.load("models/random_forest.pkl")
    predictions = model.predict(X_test_final)

    submission = pd.DataFrame({
        'PassengerId': df_test['PassengerId'],
        'Survived': predictions
    })

    submission.to_csv("outputs/submission.csv", index=False)
    print("✅ Submission saved to outputs/submission.csv")

if __name__ == "__main__":
    evaluate_model("data/raw/train.csv")
    generate_submission("data/raw/test.csv")
