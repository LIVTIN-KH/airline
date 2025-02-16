import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import io

# Load sample.csv to get column names
file_path = "sample.csv"
df_sample = pd.read_csv(file_path)

# Define categorical and numerical columns
categorical_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
numerical_features = [col for col in df_sample.columns if col not in categorical_features]

# Define preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
preprocessor.fit(df_sample)  # Fit on sample data


# Define model architecture
class SatisfactionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)


# Load trained model
input_dim = preprocessor.transform(df_sample).shape[1]  # Get transformed input size
model = SatisfactionModel(input_dim)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Streamlit UI styling
st.set_page_config(page_title="Airline Satisfaction Prediction", page_icon="✈️", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 10%;
        background-color: #007BFF;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 8px;
    }
    .block-container {
        width: 60%;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)

st.title("✈️ Airline Passenger Satisfaction Prediction")
st.write("Fill in the passenger details below and get a prediction of their satisfaction level.")

# Create a radio button to select prediction mode (individual or batch)
mode = st.radio("Select Prediction Mode", ["Individual Prediction", "Batch Prediction"])

if mode == "Individual Prediction":
    # Create input fields in two columns for individual prediction
    cols = st.columns(2)
    data = {}
    for i, col in enumerate(df_sample.columns):
        with cols[i % 2]:  # Distribute inputs across two columns
            if col in categorical_features:
                data[col] = st.selectbox(f"**{col}**", df_sample[col].unique())
            else:
                data[col] = st.number_input(f"**{col}**", value=float(df_sample[col].mean()))

    # Predict button
    st.markdown("---")
    if st.button("🚀 Predict Satisfaction"):
        input_df = pd.DataFrame([data])
        input_processed = preprocessor.transform(input_df)
        input_tensor = torch.tensor(input_processed, dtype=torch.float32)
        output = model(input_tensor).item()
        prediction = "😃 Satisfied" if torch.sigmoid(torch.tensor(output)) > 0.5 else "😐 Neutral or Dissatisfied"
        st.success(f"**Predicted Satisfaction: {prediction}**")

elif mode == "Batch Prediction":
    # File Upload for Bulk Prediction
    st.markdown("---")
    st.subheader("📂 Upload a CSV file for batch prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        df_processed = preprocessor.transform(df_uploaded)
        input_tensor = torch.tensor(df_processed, dtype=torch.float32)
        outputs = model(input_tensor).detach().numpy().flatten()
        predictions = ["Satisfied" if torch.sigmoid(torch.tensor(o)) > 0.5 else "Neutral or Dissatisfied" for o in
                       outputs]
        df_uploaded["Predicted Satisfaction"] = predictions

        # Display results
        st.write("### Prediction Results:")
        st.dataframe(df_uploaded)

        # File format selection for download
        file_format = st.radio("Select File Format", ["CSV", "Excel"])

        # Prepare data for download based on selected format
        if file_format == "CSV":
            csv = df_uploaded.to_csv(index=False).encode('utf-8')
            download_button = st.download_button("Download File", csv, "predictions.csv", "text/csv")
        elif file_format == "Excel":
            xlsx_buffer = io.BytesIO()
            df_uploaded.to_excel(xlsx_buffer, index=False, engine='xlsxwriter')
            xlsx_data = xlsx_buffer.getvalue()
            download_button = st.download_button("Download File", xlsx_data, "predictions.xlsx",
                                                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
