import joblib
import pandas as pd

# 1. Saved model ko load karein
# Ensure predictive_maintenance_model.pkl is inside the 'code' folder
model = joblib.load('predictive_maintenance_model.pkl')

print("--- Machine Health Checker ---")
print("Kindly enter the following details to check machine status:")

# 2. User se 1-1 kar ke data mangna
air_temp = float(input("1. Air Temperature [K] (e.g., 300): "))
proc_temp = float(input("2. Process Temperature [K] (e.g., 310): "))
rpm = float(input("3. Rotational Speed [rpm] (e.g., 1500): "))
torque = float(input("4. Torque [Nm] (e.g., 40): "))
tool_wear = float(input("5. Tool Wear [min] (e.g., 5): "))

# 3. Input ko DataFrame mein convert karna
input_data = pd.DataFrame([[air_temp, proc_temp, rpm, torque, tool_wear]], 
                          columns=['Air temperature [K]', 'Process temperature [K]', 
                                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])

# 4. Prediction karna
result = model.predict(input_data)

# 5. Final Output dikhana
print("\n" + "="*30)
if result[0] == 1:
    print("RESULT: WARNING! Machine Failure Detected.")
else:
    print("RESULT: Everything is Fine. Machine is Healthy.")
print("="*30)