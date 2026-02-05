import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("Loan_Approval_Prediction/dataset/train.csv")

df.drop('Loan_ID' , axis=1 , inplace=True)

#   Missing Data handling   #

df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)

df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

#   Encoding    #
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)

#   X and Y    #
X = df.drop('Loan_Status', axis=1)
Y = df['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

#   Feature Scaling    #
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#   Random Forest Model    #
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, Y_train)

Y_pred_rf = rf.predict(X_test)

#print("Random Forest Accuracy:", accuracy_score(Y_test, Y_pred_rf))
#print(confusion_matrix(Y_test, Y_pred_rf))
#print(classification_report(Y_test, Y_pred_rf))

######        ######
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

joblib.dump(rf, os.path.join(BASE_DIR, "loan_model.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))

print("Model and scaler saved successfully")