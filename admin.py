import streamlit as st
import pandas as pd
import joblib
import random


# Load assets
@st.cache_data
def load_assets():
    try:
        model = joblib.load("rf_model.pkl")
        encoders = joblib.load("encoders.pkl")
        scaler = joblib.load("scaler.pkl")
        dataset = pd.read_csv("cleaned_dataset.csv")
        return model, encoders, scaler, dataset
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None, None

model, encoders, scaler, dataset = load_assets()

def get_fico_category(score):
    """Determine FICO credit category."""
    score_bins = [300, 579, 669, 739, 799, 850]
    score_labels = ['Fair', 'Good', 'Very Good', 'Excellent']
    for i in range(len(score_bins) - 1):
        if score_bins[i] <= score <= score_bins[i+1]:
            return score_labels[i]
    return None

def calculate_monthly_income(annual_income):
    return round(annual_income / 12, 2)

def calculate_dti(loan_amount, annual_income):
    if annual_income == 0:
        return 0.0
    return round((loan_amount / annual_income) * 100, 2)

def calculate_ins_inc(installment, monthly_income):
    if monthly_income == 0:
        return 0.0
    return round((installment / monthly_income) , 2)

def process_input(df):
    """Encode and scale input data."""
    cat_cols = [col for col in df.columns if col in encoders]
    num_cols = [col for col in df.columns if col not in cat_cols]

    for col in cat_cols:
        df[col] = df[col].astype(str)
        
        df[col] = encoders[col].transform(df[col]) if col in encoders else 0

    if num_cols:
        df[num_cols] = scaler.transform(df[num_cols])

    model_features = model.feature_names_in_
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    return df[model_features]

def predict_loan_status(sample_df):
    """Predict loan status."""
    expected_labels = sample_df['loan_status'].map(lambda x: 1 if x == "Default" else 0)
    df_processed = process_input(sample_df.drop(columns=['loan_status']))

    predictions = model.predict(df_processed)
    result_df = sample_df.copy()
    result_df['Predicted_Label'] = ['Fully Paid' if p == 0 else 'Default' for p in predictions]
    result_df['Expected_Label'] = ['Fully Paid' if e == 0 else 'Default' for e in expected_labels]
    result_df['Correctly_Classified'] = result_df['Predicted_Label'] == result_df['Expected_Label']

    return result_df

# Streamlit UI
st.title("Loan Default Prediction")

option = st.radio("Choose an option:", ("Use Existing Dataset", "Enter Custom Values"))
def highlight_correct_classification(val):
    if val:  # True means correctly classified
        return "background-color: lightgreen; font-weight: bold;"
    else:
        return "background-color: lightcoral; font-weight: bold;"

if option == "Use Existing Dataset":
    if st.button("Use Existing Dataset"):
        random.seed(42)
        sample_df = dataset.sample(n=10, random_state=random.randint(40, 100))
        result_df = predict_loan_status(sample_df)

        # Apply conditional formatting for correctness
        styled_df = result_df.style.applymap(highlight_correct_classification, subset=['Correctly_Classified'])

        st.write("Prediction Results:")
        st.dataframe(styled_df)

else:  
    st.subheader("Enter Custom Values")

    # Numerical inputs
    loan_amnt = st.number_input("Loan Amount", min_value=1000.0, step=500.0)
    annual_inc = st.number_input("Annual Income", min_value=1000.0, step=1000.0)
    installment = st.number_input("Installment Amount", min_value=100.0, step=50.0)
    emp_length = st.number_input("Employment Length", min_value=0, step=1)
    fico_range_high = st.number_input("FICO Score High", min_value=300, max_value=850, value=700)
    funded_amnt = loan_amnt
    int_rate = st.number_input("Interest Rate", min_value=0.0, max_value=100.0, value=10.0)

    # Categorical inputs
    term = st.selectbox("Term", [" 36 months", " 60 months"])
    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    purpose = st.selectbox("Purpose of Loan", ["Credit Card", "Car", "Small Business", "Debt Consolidation", 
                                               "Home Improvement", "Wedding", "Other"])

    # Derived calculations
    fico_category = get_fico_category(fico_range_high)
    monthly_inc = calculate_monthly_income(annual_inc)
    dti = calculate_dti(loan_amnt, annual_inc)
    ins_to_inc_ratio = calculate_ins_inc(installment, monthly_inc)

    # Display derived values
    st.write("Derived Values:")
    st.write(f"📌 **FICO Category:** {fico_category}")
    st.write(f"📌 **Monthly Income:** {monthly_inc}")
    st.write(f"📌 **DTI Ratio (%):** {dti}")
    st.write(f"📌 **Installment-to-Income Ratio (%):** {ins_to_inc_ratio}")

    # Create DataFrame for Prediction
    input_data = pd.DataFrame({
        'loan_amnt': [loan_amnt], 'installment': [installment], 'emp_length': [emp_length], 'term': [term], 
        'int_rate': [int_rate], 'home_ownership': [home_ownership], 'annual_inc': [annual_inc], 'purpose': [purpose],
        'dti': [dti], 'fico_range_high': [fico_range_high], 'funded_amnt': [funded_amnt], 'fico_category': [fico_category],
        'monthly_inc': [monthly_inc], 'installment_to_income_ratio_%': [ins_to_inc_ratio]
    })

    if st.button("Check Status"):
        processed_input = process_input(input_data)
        prediction = model.predict(processed_input)[0]
        predicted_label = "Fully Paid" if prediction == 0 else "Default"

        # Change color based on prediction
        if prediction == 0:
            st.markdown(
                f"<h2 style='color:green;'>✅ Predicted Loan Status: {predicted_label}</h2>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h2 style='color:red;'>❌ Predicted Loan Status: {predicted_label}</h2>", 
                unsafe_allow_html=True
            )

