import secrets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import streamlit as st
import joblib

st.title('TensoraLabs - Regression')
st.write('Machine Learning Models without code.')

dataset = st.file_uploader("Choose a CSV File", type='.csv')

if dataset is not None:
    df = pd.read_csv(dataset)
    st.write(df.head())

    target = st.selectbox('Column to be predicted: ', df.columns)
    use_columns = st.multiselect('Column(s) to be used: ', [col for col in df.columns if col != target])

    if st.button('Train Model'):
        X = pd.get_dummies(df[use_columns])
        y = df[target]  

        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=4)

        model = LinearRegression().fit(train_x, train_y)
        predictions = model.predict(test_x)
        acc = r2_score(test_y, predictions)

        joblib.dump((model, list(X.columns)), 'trained_model.pkl')  

        if len(use_columns) == 1 and pd.api.types.is_numeric_dtype(df[use_columns[0]]):
            x_vals = train_x.squeeze()
            y_vals = train_y.squeeze()
            predicted_line = model.coef_[0] * x_vals + model.intercept_
            fig, ax = plt.subplots()
            ax.scatter(x_vals, y_vals, color='blue', label='Actual Data')
            ax.plot(x_vals, predicted_line, color='red', label='Fit Line')
            ax.set_xlabel(use_columns[0])
            ax.set_ylabel(target)
            ax.legend()
            st.pyplot(fig)

        st.write("Model's R² Score: ", np.round(acc, 2))
        if acc >= 0.5:
            st.success("✅ High Accuracy: Model predictions are stable and accurate.")
        else:
            st.warning("⚠️ Accuracy could be improved: Try tweaking features or cleaning data.")
        st.balloons()

    user_input = st.text_input('Enter values for prediction (comma-separated):')

    if user_input:
        try:
            input_list = [x.strip() for x in user_input.split(',')]
            input_df = pd.DataFrame([input_list], columns=use_columns)

            input_df = pd.get_dummies(input_df)

            model, trained_cols = joblib.load('trained_model.pkl')

            for col in trained_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[trained_cols]

            prediction = model.predict(input_df)
            st.write(f"Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
