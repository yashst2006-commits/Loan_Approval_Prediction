# 🏦 Loan Approval Prediction using Machine Learning

## 🎯 Objective
To predict whether a loan will be **approved or rejected** using historical loan application data and to compare multiple classification models in order to select the most suitable one for real-world banking scenarios.

---

## 📂 Dataset
- **Source:** Kaggle – Loan Prediction Problem Dataset  
- **File Used:** `train.csv`

### Features
- Gender  
- Married  
- Dependents  
- Education  
- Self_Employed  
- ApplicantIncome  
- CoapplicantIncome  
- LoanAmount  
- Loan_Amount_Term  
- Credit_History  
- Property_Area  

### Target Variable
- **Loan_Status**
  - `1` → Approved  
  - `0` → Rejected  

---

## 🤖 Models Implemented
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Random Forest Classifier  

---

## 📊 Model Evaluation
The models were evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## 📈 Model Comparison

| Model | Accuracy | False Approvals | False Rejections |
|------|----------|----------------|------------------|
| Logistic Regression | 0.86 | 16 | 1 |
| KNN | 0.84 | 17 | 3 |
| Random Forest | 0.85 | **14** | 5 |

## 🏆 Final Model Selection

### ✅ Random Forest Classifier

**Reason:**  
Although Logistic Regression achieved slightly higher accuracy, Random Forest was chosen as the final model because it produced the **lowest number of risky loan approvals**, which is crucial in financial applications.

> In banking systems, minimizing false approvals is more important than maximizing overall accuracy.

---

## 🌐 Streamlit Web Application (Live)

This project includes a fully deployed **Streamlit web application** that allows users to check loan eligibility in real time using a trained machine learning model.

### 🔗 Live Demo
👉 https://loan-approval-prediction-yashst2006.streamlit.app/


## 👤 Author
**Yashwanth S T**  
B.E. CSE (AI & ML) 