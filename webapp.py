import streamlit as st
import joblib
import pandas as pd

# 1. Page Configuration (Browser tab name aur icon)
st.set_page_config(page_title="AI Industrial Health Guard", page_icon="🤖")

# 2. Model load karein
model = joblib.load('predictive_maintenance_model.pkl')

# 3. Custom Styling (CSS) takay interface attractive lagay
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007BFF;
        color: white;
        font-weight: bold;
    }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 4. Header Section
st.title("🛡️ AI Industrial Health Guard")
st.subheader("Smart Predictive Maintenance Dashboard")
st.write("Monitor machine health in real-time using advanced AI analysis.")
st.divider()

# 5. Sidebar for Info
with st.sidebar:
    st.header("About Project")
    st.info("Ye AI model machine ke sensors ka data read kar ke batata hy ke machine fail hone wali hy ya nahi.")
    st.write("**Developer:** Kamran Ahmad")

# 6. Input Section (Columns mein divide kiya hy takay look achi aye)
st.markdown("### 📊 Sensor Data Input")
col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input("🌡️ Air Temperature [K]", value=300.0, step=0.1)
    proc_temp = st.number_input("🔥 Process Temperature [K]", value=310.0, step=0.1)
    rpm = st.number_input("⚙️ Rotational Speed [rpm]", value=1500.0, step=10.0)

with col2:
    torque = st.number_input("🔧 Torque [Nm]", value=40.0, step=1.0)
    tool_wear = st.number_input("⏳ Tool Wear [min]", value=5.0, step=1.0)

st.write("") # Khali jagah

# 7. Prediction Logic
if st.button("🚀 Analyze Machine Health"):
    input_data = pd.DataFrame([[air_temp, proc_temp, rpm, torque, tool_wear]], 
                              columns=['Air temperature [K]', 'Process temperature [K]', 
                                       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
    
    # Spinner takay user ko lage ke AI soch raha hy
    with st.spinner('Analyzing sensor data...'):
        prediction = model.predict(input_data)
    
    st.divider()
    
    if prediction[0] == 1:
        st.error("### 🚨 ALERT: Machine Failure Predicted!")
        st.warning("Immediate maintenance required to avoid breakdown.")
    else:
        st.success("### ✅ STATUS: Machine is Healthy")
        st.balloons() # Thori khushi ke liye balloons!