import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.drop(['name'], axis=1)
    if 'age' in df.columns:
        df['age'] = df['age'].fillna(df['age'].median())
    if 'fare' in df.columns:
        df['fare'] = df['fare'].fillna(df['fare'].median())
    
    if 'sex' in df.columns:
        le = LabelEncoder()
        df['sex'] = le.fit_transform(df['sex'])
    return df

def load_and_split_data(url):
    df = pd.read_csv(url)
    df = preprocess_data(df)
    X = df.drop('survived', axis=1)
    y = df['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test