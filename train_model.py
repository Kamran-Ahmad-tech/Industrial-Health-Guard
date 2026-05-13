import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load Dataset
# Machine sensor data ko load kar rahe hain
df = pd.read_csv('ai4i2020.csv')

# 2. Data Cleaning
# Faltu ID columns aur specific failure types ko delete kar rahe hain takay model confuse na ho
drop_cols = ['UDI', 'Product ID', 'Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
df_final = df.drop(drop_cols, axis=1)

# 3. Features (X) aur Target (y) ko alag karna
# 'Machine failure' hamara target hy, baqi 5 sensors hamare inputs (X) hain
X = df_final.drop('Machine failure', axis=1)
y = df_final['Machine failure']

# 4. Split Data
# 80% data training ke liye aur 20% testing ke liye alag kar liya
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training
# Random Forest Classifier use kar rahe hain jo aik powerful decision tree algorithm hy
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Accuracy Check
# Sir ko dikhane ke liye ke model kitna acha perform kar raha hy
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 7. Save Model
# Train hua model file mein save kar rahe hain takay Streamlit app mein use ho sakay
joblib.dump(model, 'predictive_maintenance_model.pkl')
print("Model saved as 'predictive_maintenance_model.pkl'")