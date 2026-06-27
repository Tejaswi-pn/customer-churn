import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import os


# Page Configuration


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

# Load Model & Preprocessor


@st.cache_resource
def load_artifacts():

    # Load Preprocessor
    preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

    # Load Model
    if os.path.exists("churn_model.keras"):
        model = tf.keras.models.load_model("churn_model.keras")

    elif os.path.exists("model.keras"):
        model = tf.keras.models.load_model("model.keras")

    else:
        raise FileNotFoundError(
            "Model file not found.\n"
            "Expected: churn_model.keras or model.keras"
        )

    return preprocessor, model


try:

    preprocessor, model = load_artifacts()
    loaded = True

except Exception as e:

    loaded = False
    error = str(e)


# Title


st.title("🏦 Customer Churn Prediction")

st.markdown(
    """
Predict whether a customer is likely to **leave the bank (Churn)**
or continue as an **active customer** using a trained Deep Learning model.
"""
)

if not loaded:

    st.error(error)
    st.stop()

st.divider()


# Customer Form


with st.form("customer_form"):

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650
        )

        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"]
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=50000.0
        )

        has_card = st.selectbox(
            "Has Credit Card",
            [0, 1]
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        tenure = st.slider(
            "Tenure",
            0,
            10,
            5
        )

        products = st.selectbox(
            "Number of Products",
            [1, 2, 3, 4]
        )

        active = st.selectbox(
            "Is Active Member",
            [0, 1]
        )

        salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=50000.0
        )

    threshold = st.slider(
        "Prediction Threshold",
        0.10,
        0.90,
        0.50,
        0.05
    )

    submitted = st.form_submit_button(
        "🔍 Predict Churn",
        use_container_width=True
    )


# Prediction


if submitted:

    # Feature Engineering
   

    balance_salary_ratio = balance / (salary + 1)

    is_zero_balance = int(balance == 0)

   
    # Create DataFrame
   

    customer = pd.DataFrame([{

        "CreditScore": credit_score,

        "Geography": geography,

        "Gender": gender,

        "Age": age,

        "Tenure": tenure,

        "Balance": balance,

        "NumOfProducts": products,

        "HasCrCard": has_card,

        "IsActiveMember": active,

        "EstimatedSalary": salary,

        "BalanceSalaryRatio": balance_salary_ratio,

        "IsZeroBalance": is_zero_balance

    }])

    try:

       
        # Transform Features
  

        X = preprocessor.transform(customer)

 
        # Predict
        

        probability = float(
            model.predict(X, verbose=0)[0][0]
        )

        prediction = int(probability >= threshold)

        confidence = round(
            max(probability, 1 - probability) * 100,
            2
        )

        st.divider()

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(

                "Churn Probability",

                f"{probability:.2%}"

            )

        with col2:

            st.metric(

                "Confidence",

                f"{confidence:.2f}%"

            )

        st.progress(probability)

    
        # Final Result
       

        if prediction == 1:

            st.error(

                f"""
🚨 Customer is likely to leave the bank.

Probability of Churn: **{probability:.2%}**
                """
            )

        else:

            st.success(

                f"""
✅ Customer is likely to stay with the bank.

Probability of Churn: **{probability:.2%}**
                """
            )

  
        # Customer Summary
        

        st.divider()

        st.subheader("Customer Details")

        st.dataframe(

            customer,

            use_container_width=True

        )

       
        # Engineered Features
        

        st.subheader("Engineered Features")

        engineered = pd.DataFrame({

            "Feature": [

                "BalanceSalaryRatio",

                "IsZeroBalance"

            ],

            "Value": [

                round(balance_salary_ratio, 4),

                is_zero_balance

            ]

        })

        st.dataframe(

            engineered,

            use_container_width=True

        )

    except Exception as e:

        st.error(

            f"Prediction Failed:\n\n{e}"

        )


# Footer


st.divider()

st.caption()
