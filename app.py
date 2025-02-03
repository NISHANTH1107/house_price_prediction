import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset for Label Encoding reference
data = pd.read_csv("india_housing_prices.csv")
df = pd.DataFrame(data)

# Encode categorical variables
le_dict = {}
for col in df.select_dtypes(include=['object']).columns:
    le_dict[col] = LabelEncoder()
    df[col] = le_dict[col].fit_transform(df[col])

# Load trained model
model = joblib.load("linear_regression_model.pkl")

# Streamlit UI
st.title("🏠 Indian House Price Prediction")

st.sidebar.header("Enter House Details")

# Initialize session state for input storage
if "user_input" not in st.session_state:
    st.session_state.user_input = {}

# Create a form for user inputs
with st.sidebar.form(key='input_form'):
    st.session_state.user_input = {
        "State": st.selectbox("State", le_dict["State"].classes_, key="state"),
        "City": st.selectbox("City", le_dict["City"].classes_, key="city"),
        "Locality": st.selectbox("Locality", le_dict["Locality"].classes_, key="locality"),
        "Property_Type": st.selectbox("Property Type", le_dict["Property_Type"].classes_, key="property_type"),
        "BHK": st.number_input("BHK", min_value=1, max_value=10, step=1, key="bhk"),
        "Size_in_SqFt": st.number_input("Size (Sq Ft)", min_value=300, max_value=10000, step=50, key="size"),
        "Year_Built": st.number_input("Year Built", min_value=1900, max_value=2025, step=1, key="year"),
        "Furnished_Status": st.selectbox("Furnished Status", le_dict["Furnished_Status"].classes_, key="furnished_status"),
        "Floor_No": st.number_input("Floor Number", min_value=0, max_value=50, step=1, key="floor_no"),
        "Total_Floors": st.number_input("Total Floors", min_value=1, max_value=50, step=1, key="total_floors"),
        "Nearby_Schools": st.slider("Nearby Schools", min_value=0, max_value=10, value=5),
        "Nearby_Hospitals": st.slider("Nearby Hospitals", min_value=0, max_value=10, value=5),
        "Public_Transport_Accessibility": st.selectbox("Public Transport", ["Low", "Medium", "High"], key="public_transport"),
        "Parking_Space": st.selectbox("Parking Space", ["Yes", "No"], key="parking_space"),
        "Security": st.selectbox("Security", ["Yes", "No"], key="security"),
        "Amenities": st.multiselect("Amenities", ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]),
        "Facing": st.selectbox("Facing", le_dict["Facing"].classes_, key="facing"),
        "Owner_Type": st.selectbox("Owner Type", le_dict["Owner_Type"].classes_, key="owner_type"),
        "Availability_Status": st.selectbox("Availability Status", le_dict["Availability_Status"].classes_, key="availability_status"),
    }

    # Calculate Age_of_Property AFTER Year_Built is set
    st.session_state.user_input["Age_of_Property"] = 2025 - st.session_state.user_input["Year_Built"]

    # Submit button
    submit_button = st.form_submit_button(label='Predict Price')

# Predict button (only processes when clicked)
if submit_button:
    input_df = pd.DataFrame([st.session_state.user_input])

    # Encode categorical values
    for col in input_df.select_dtypes(include=['object']).columns:
        if col != "Amenities":  # Handle Amenities separately
            input_df[col] = le_dict[col].transform(input_df[col])

    # Handle "Amenities" (convert to binary features)
    amenities_list = ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]
    for amenity in amenities_list:
        input_df[amenity] = input_df["Amenities"].apply(lambda x: 1 if amenity in x else 0)

    # Drop original "Amenities" column
    input_df.drop("Amenities", axis=1, inplace=True)

    # Ensure input_df matches training columns
    expected_features = [col for col in df.columns if col != "Price_in_Lakhs"]  # Exclude target column
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    # Debugging Prints
    print(f"Model expects {model.n_features_in_} features.")
    print(f"Input data has {input_df.shape[1]} features.")
    print(f"Final input columns: {input_df.columns.tolist()}")

    # Ensure input_data is defined before prediction
    if not input_df.empty:  # Check if input_df has data
        input_data = input_df.values  # Convert to NumPy array
        predicted_price = model.predict(input_data)[0]
        st.success(f"🏡 Estimated House Price: ₹ {round(predicted_price, 2)} Lakhs")
    else:
        st.error("⚠️ Error: No input data provided for prediction!")
