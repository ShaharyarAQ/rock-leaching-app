import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ------------------------------------------
# Load Models
# ------------------------------------------
vol_pkg = joblib.load("models/leachate_volume_model.joblib")
chem_pkg = joblib.load("models/leachate_chemistry_model.joblib")

vol_model = vol_pkg["model"]
chem_model = chem_pkg["model"]

INPUT_FEATURES = vol_pkg["input_features"]
CHEM_TARGETS = chem_pkg["target_names"]

# ------------------------------------------
# Page Title
# ------------------------------------------
st.title("🔬 Rock Leaching Prediction App")
st.write("Predict leachate volume and chemical composition after rainfall or snowfall events.")

# ------------------------------------------
# Sidebar – Rock Inputs
# ------------------------------------------
st.sidebar.header("Rock Properties")

rock_inputs = {}

for feature in INPUT_FEATURES:
    if feature in ["Type_event", "Acid"]:
        continue
    if feature.startswith("Corg_rock"):
        rock_inputs[feature] = st.sidebar.number_input(feature, value=0.05, format="%.4f")
    elif feature.endswith("_rock") or feature.endswith("_O") or feature.endswith("2O"):
        rock_inputs[feature] = st.sidebar.number_input(feature, value=1.0)
    elif feature in ["Cumulative_Water", "Cumulative_Acid"]:
        rock_inputs[feature] = st.sidebar.number_input(feature, value=0.0)
    elif feature == "Event_quantity":
        continue
    else:
        rock_inputs[feature] = st.sidebar.number_input(feature, value=1.0)

# ------------------------------------------
# Event Input Section
# ------------------------------------------
st.header("Event Configuration")

event_type = st.selectbox("Event Type", ["Rain", "Snow"])
event_quantity = st.number_input("Event Quantity", value=100)
acid_flag = st.selectbox("Acidic Event?", ["No", "Yes"])
temp = st.number_input("Temperature After Event", value=10.0)

# Convert into encoded values
Type_event = 0 if event_type == "Rain" else 1
Acid = 1 if acid_flag == "Yes" else 0

# ------------------------------------------
# Prediction Button
# ------------------------------------------
if st.button("Predict Leachate"):
    
    # Build input row
    input_dict = rock_inputs.copy()
    input_dict.update({
        "Type_event": Type_event,
        "Event_quantity": event_quantity,
        "Acid": Acid,
        "Temp": temp
    })
    
    # Ensure order matches model features
    X = [input_dict[f] for f in INPUT_FEATURES]
    X = np.array(X).reshape(1, -1)

    # Predict volume
    predicted_vol = vol_model.predict(X)[0]

    st.subheader("📌 Predicted Leachate Volume")
    st.success(f"{predicted_vol:.2f} ml")

    # Predict chemicals
    predicted_chem = chem_model.predict(X)[0]
    chem_results = {name: float(value) for name, value in zip(CHEM_TARGETS, predicted_chem)}

    st.subheader("🧪 Chemical Composition Prediction")
    st.write(pd.DataFrame([chem_results]))

    # --------------------------------------
    # Explanation Module (Simple & Clear)
    # --------------------------------------
    st.subheader("💡 Explanation (For Non-Experts)")
    st.info("""
    - The rock’s **chemical composition** influences how much material dissolves during rain or snow.  
    - **Event quantity** and **temperature** affect how much leachate is produced.  
    - Acidic events generally increase leaching by reacting with minerals.  
    - The model compares this event with historical patterns in similar rocks.  
    """)

    st.subheader("Feature Importance (Model Insight)")

    importances = vol_model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": INPUT_FEATURES,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(imp_df.set_index("Feature"))
