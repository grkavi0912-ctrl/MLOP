import streamlit as st
import requests
import json
import pandas as pd
import huggingface_hub import hf_hub_download
import jonlib

#Download and load the model
model_path = hf_hub_download(repo_id="grkavi0912/Tpro",filename="best_tour_model.joblib")
model = joblib.load(model_path)

#Streamlit UI for Tourism package prediction
st.title("Tourism Package Prediction")
st.write("Enter the details to predict the package price")

#User input
Age = st.number_input("Age",min_value=18,max_value=100)
Type_of_contact = st.selectbox("Type of Contact",["Direct","Call"])
City_Tier = st.selectbox("City Tier",[1,2,3])
Duration_of_Pitch = st.number_input("Duration of Pitch",min_value=1,max_value=365)
Occupation = st.selectbox("Occupation",["Self-employed","Salaried","Business"])
Gender = st.selectbox("Gender",["Male","Female"])
Number_of_Person_Visiting= st.number_input("Number of Person Traveling",min_value=1,max_value=10)
Number_of_Followups= st.number_input("Number of Followups",min_value=0,max_value=10)
Product_Pitched= st.selectbox("Product Pitched",["Basic","Standard","Premium"])
Preferred_Property_Star= st.number_input("Preferred Property Star",min_value=1,max_value=5)
Marital_Status= st.selectbox("Marital Status",["Married","Divorced","Single"])
Number_of_Trips= st.number_input("Number of Trips",min_value=1,max_value=10)
Passport= st.selectbox("Passport",["Yes","No"])
Pitch__Satisfaction_Score= st.number_input("Pitch Satisfaction Score",min_value=1,max_value=5)
Own_Car= st.selectbox("Own Car",["Yes","No"])
Number_of_Children= st.number_input("Number of Children",min_value=0,max_value=10)
Designation= st.selectbox("Designation",["Executive","Manager","Senior Manager","Associate","Director"])
Monthly_Income= st.number_input("Monthly Income",min_value=0,max_value=100000)

#Assemble input into DataFrame
input_data = pd.DataFrame([{
    "Age": [Age],
    "Type_of_Contact": [TypeofContact],
    "City_Tier": [CityTier],
    "Duration_of_Pitch": [DurationofPitch],
    "Occupation": [Occupation],
    "Gender": [Gender],
    "Number_of_Person_Visiting": [NumberofPersonVisiting],
    "Number_of_Followups": [NumberofFollowups],
    "Product_Pitched": [ProductPitched],
    "Preferred_Property_Star": [PreferredPropertyStar],
    "Marital_Status": [MaritalStatus],
    "NUmber_of_Trips": [NumberOfTrips],
    "Passport": [Passport],
    "Pitch_Satisfaction_Score": [PitchSatisfactionScore],
    "Own_Car": [OwnCar],
    "Number_of_Children": [NumberOfChildren],
    "Designation": [Designation],
    "Monthly_Income": [MonthlyIncome]

}])

if st.button("Predict"):
    #Make prediction
    prediction = model.predict(input_data)[0]
    result = "Tourism package predicted as " + str(prediction)
    st.subheader("Predicted Result:")
    st.success(f"The model predicts: **{result}**")








