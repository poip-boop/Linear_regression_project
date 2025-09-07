import pandas as pd

def preprocess_data(df, is_train=True):
    """
    Preprocess Titanic dataset.
    """
    # Fill missing ages
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Encode Sex
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Drop Cabin
    if 'Cabin' in df.columns:
        df = df.drop('Cabin', axis=1)

    # Fill Embarked (if missing in train)
    if is_train and df['Embarked'].isnull().sum() > 0:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    return df
