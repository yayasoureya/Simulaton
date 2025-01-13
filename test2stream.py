import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib

matplotlib.use('Agg')  # Use the non-interactive backend

# Function to display SHAP plots in Streamlit
def show_shap_plot(plot_function, *args, **kwargs):
    fig, ax = plt.subplots()
    plot_function(*args, show=False, **kwargs)  # Generate the SHAP plot
    st.pyplot(fig)  # Display the plot in Streamlit

# 1. Data simulation
np.random.seed(42)
n_samples = 1000
n_features = 7  # Sensors: temperature, motion, brightness, etc.

# Generate sensor data
X = np.random.rand(n_samples, n_features) * 10  # Sensors with values between 0 and 10
y = np.random.choice(['Light ON', 'Light OFF', 'Open Door', 'Nothing', 'Turn ON AC', 'Turn OFF AC'], size=n_samples)

# Convert to DataFrame with new sensors
columns = ['Sensor_Temperature', 'Sensor_Motion', 'Sensor_Brightness', 'Sensor_Humidity', 'Sensor_Sound', 'Sensor_LivingRoom_Presence', 'Sensor_Bedroom_Presence']
df = pd.DataFrame(X, columns=columns)
df['Action'] = y

# Add an estimate of presence in each room (Living Room vs Bedroom)
df['Time_in_LivingRoom'] = np.random.rand(n_samples) * 10  # Estimated time spent in the living room (in minutes)
df['Time_in_Bedroom'] = np.random.rand(n_samples) * 10  # Estimated time spent in the bedroom (in minutes)

# Split the data into train/test
X_train, X_test, y_train, y_test = train_test_split(df[columns + ['Time_in_LivingRoom', 'Time_in_Bedroom']], df['Action'], test_size=0.2, random_state=0)

# 2. Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Predict on new data
predictions = model.predict(X_test)

# Display predictions
st.write(f"Predictions: {predictions[:5]}")

# 4. Interpretation with SHAP
explainer = shap.TreeExplainer(model)
#X_test_limited = X_test.iloc[:, :6]
shap_values = explainer.shap_values(X_test)

# Global visualization: feature importance
st.write("### Feature Importance")
# Check the number of classes in shap_values
st.write(f"Number of SHAP classes: {len(shap_values)}")

# Verify consistency between columns and X_test
# Include the additional columns in the check
expected_columns = columns + ['Time_in_LivingRoom', 'Time_in_Bedroom']
assert len(expected_columns) == X_test.shape[1], "Column names and data do not match."

# Fix SHAP plot (choose an existing class, e.g., 0)
# SHAP visualization: include additional columns
shap.summary_plot(shap_values[1], X_test[expected_columns], feature_names=expected_columns)
show_shap_plot(shap.summary_plot, shap_values[1], X_test[expected_columns], feature_names=expected_columns)

# Local visualization: explanation for a specific sample
# Check the dimensions of shap_values and X_test
assert shap_values[0].shape[1] == len(expected_columns), "Mismatch between SHAP values and columns."

# Handle SHAP values size
if shap_values[0].shape[1] != len(expected_columns):
    st.warning("Generated SHAP values do not match the expected number of columns.")
    explanation = shap.Explanation(
        shap_values[0][0][:len(expected_columns)],
        base_values=explainer.expected_value[0],
        data=X_test.iloc[0].values[:len(expected_columns)],
        feature_names=expected_columns
    )
else:
    explanation = shap.Explanation(
        shap_values[0][0],
        base_values=explainer.expected_value[0],
        data=X_test.iloc[0].values,
        feature_names=expected_columns
    )

st.write("### Local Explanation for a Specific Sample")
# Convert base_values to float, if necessary
if isinstance(explanation.base_values, pd.Series):
    explanation.base_values = float(explanation.base_values.iloc[0])
    
shap.plots.waterfall(explanation)
show_shap_plot(shap.plots.waterfall, explanation)

# 5. Line chart (sensor variations)
st.write("### Sensor Variations Over Time")
plt.figure(figsize=(10, 6))
for i, column in enumerate(columns):
    plt.plot(df.index[:100], df[column][:100], label=column)
plt.title('Sensor Variations Over Time')
plt.xlabel('Index')
plt.ylabel('Sensor Values')
plt.legend(loc='upper right')
plt.grid(True)
st.pyplot(plt)

# 6. Pie chart: Distribution of sensor values
st.write("### Distribution of Average Sensor Values")
plt.figure(figsize=(7, 7))
sensor_means = df[columns].mean()
plt.pie(sensor_means, labels=sensor_means.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Average Sensor Values')
plt.axis('equal')  # For a perfect circle
st.pyplot(plt)

# 7. Future predictions chart
st.write("### Future Predictions Over Time")
future_predictions = model.predict(X_test[:100])  # Predictions on the first 100 samples
plt.figure(figsize=(10, 6))
plt.plot(np.arange(100), future_predictions, marker='o', linestyle='-', color='b', label='Predictions')
plt.title('Future Predictions Over Time')
plt.xlabel('Index')
plt.ylabel('Prediction')
plt.legend(loc='upper left')
plt.grid(True)
st.pyplot(plt)

# 8. User psychological summary
def generate_psychological_profile(df):
    profile = []
    
    if df['Sensor_Temperature'].mean() > 20:
        profile.append("User prefers temperatures below 20°C.")
    else:
        profile.append("User prefers temperatures above 20°C.")
    
    if df['Sensor_LivingRoom_Presence'].mean() < df['Sensor_Bedroom_Presence'].mean():
        profile.append("User prefers spending time in the bedroom.")
    else:
        profile.append("User prefers spending time in the living room.")
    
    if df['Sensor_Brightness'].mean() < 5:
        profile.append("User prefers staying in darkness rather than light.")
    else:
        profile.append("User prefers bright environments.")
    
    return " ".join(profile)

profile = generate_psychological_profile(df)
st.write("### User Psychological Profile")
st.write(profile)
