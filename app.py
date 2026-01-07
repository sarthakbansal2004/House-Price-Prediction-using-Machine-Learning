import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Title 
st.title(" House Price Prediction App")
st.write("Upload your dataset and compare Regression Models (Linear, Random Forest, XGBoost)")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(data.head())

    #  missing values and duplicates
    if st.checkbox("Show Missing Values"):
        st.write(data.isnull().sum())

    # Feature and Target split
    if "House_Price" not in data.columns:
        st.error("The dataset must include a column named 'House_Price'.")
    else:
        X = data.drop("House_Price", axis=1)
        y = data["House_Price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train_scaled, y_train)
        y_pred_rf = rf.predict(X_test_scaled)

        # XGBoost
        xgb = XGBRegressor(n_estimators=200, random_state=42)
        xgb.fit(X_train_scaled, y_train)
        y_pred_xgb = xgb.predict(X_test_scaled)

        # Metrics
        def eval_metrics(actual, predicted):
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            r2 = r2_score(actual, predicted)
            return mae, rmse, r2

        results = {
            "Linear Regression": eval_metrics(y_test, y_pred_lr),
            "Random Forest": eval_metrics(y_test, y_pred_rf),
            "XGBoost": eval_metrics(y_test, y_pred_xgb)
        }

        st.subheader(" Model Performance")
        results_df = pd.DataFrame(results, index=["MAE", "RMSE", "R²"])
        st.write(results_df)

        # Plot comparison
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred_lr, alpha=0.5, color="blue", label="Linear")
        ax.scatter(y_test, y_pred_rf, alpha=0.5, color="green", label="RF")
        ax.scatter(y_test, y_pred_xgb, alpha=0.5, color="red", label="XGB")
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.legend()
        st.pyplot(fig)

        # New Prediction
        st.subheader(" Predict New House Price")
        feature_values = []
        for col in X.columns:
            val = st.number_input(f"Enter value for {col}", value=float(X[col].mean()))
            feature_values.append(val)

        if st.button("Predict"):
            new_data = np.array([feature_values])
            new_scaled = scaler.transform(new_data)
            st.write("Linear Regression Prediction:", lr.predict(new_scaled)[0])
            st.write("Random Forest Prediction:", rf.predict(new_scaled)[0])
            st.write("XGBoost Prediction:", xgb.predict(new_scaled)[0])

        # Save
        joblib.dump(rf, "house_price_model.pkl")
        st.success("Random Forest model saved as 'house_price_model.pkl'")
