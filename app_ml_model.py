import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import numpy as np

st.title("📊 ML Model Trainer (Classification & Regression)")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Select target column
    target_col = st.selectbox("Select target column", df.columns)

    # Features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle missing values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col].fillna(X[col].mode()[0], inplace=True)
        else:
            X[col].fillna(X[col].median(), inplace=True)

    # Encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # Split data
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
    random_state = st.number_input("Random State", min_value=0, value=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Choose Task Type
    task_type = st.radio("Select Task Type", ["Classification", "Regression"])

    if task_type == "Classification":
        model_choice = st.selectbox("Choose a Classification Model", ["Logistic Regression", "Random Forest", "SVM"])
    else:
        model_choice = st.selectbox("Choose a Regression Model", ["Linear Regression", "Random Forest Regressor", "SVR"])

    if st.button("Train Model"):
        # Classification models
        if task_type == "Classification":
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_choice == "Random Forest":
                model = RandomForestClassifier()
            else:
                model = SVC()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("### Model Performance")
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
            st.write(f"**Precision:** {precision_score(y_test, y_pred, average='weighted'):.2f}")
            st.write(f"**Recall:** {recall_score(y_test, y_pred, average='weighted'):.2f}")
            st.write(f"**F1 Score:** {f1_score(y_test, y_pred, average='weighted'):.2f}")

            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.write("### Confusion Matrix")
            st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))

        # Regression models
        else:
            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Random Forest Regressor":
                model = RandomForestRegressor()
            else:
                model = SVR()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("### Model Performance")
            st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
            st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
            st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")