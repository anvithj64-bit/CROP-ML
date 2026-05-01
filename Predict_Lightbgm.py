import joblib
import numpy as np

model = joblib.load('predictor/ml_models/lightgbm_model.pkl')
le_region = joblib.load('predictor/ml_models/le_region.pkl')
le_soil = joblib.load('predictor/ml_models/le_soil.pkl')
le_crop = joblib.load('predictor/ml_models/le_crop.pkl')
le_weather = joblib.load('predictor/ml_models/le_weather.pkl')

print("LightGBM Model loaded!")

# =============================================
# 👇 CHANGE THESE VALUES TO YOUR FARM DATA
# =============================================

my_region = "South"
my_soil = "Clay"
my_crop = "Rice"
my_weather = "Sunny"
my_rainfall = 100
my_temperature = 28
my_fertilizer = 0
my_irrigation = 0
my_days_to_harvest = 140

# =============================================

region = le_region.transform([my_region])[0]
soil = le_soil.transform([my_soil])[0]
crop = le_crop.transform([my_crop])[0]
weather = le_weather.transform([my_weather])[0]

features = np.array([[region, soil, crop, my_rainfall,
                       my_temperature, my_fertilizer, my_irrigation,
                       weather, my_days_to_harvest]])

prediction = model.predict(features)[0]

if prediction < 2:
    risk = " High Risk"
elif prediction < 5:
    risk = "Medium Risk"
else:
    risk = "Low Risk"

print(f"\n{'='*40}")
print(f"LIGHTGBM PREDICTION RESULTS")
print(f"{'='*40}")
print(f"Crop        : {my_crop}")
print(f"Region      : {my_region}")
print(f"Soil Type   : {my_soil}")
print(f"Weather     : {my_weather}")
print(f"Rainfall    : {my_rainfall} mm")
print(f"Temperature : {my_temperature}°C")
print(f"Fertilizer  : {'Yes' if my_fertilizer else 'No'}")
print(f"Irrigation  : {'Yes' if my_irrigation else 'No'}")
print(f"Days        : {my_days_to_harvest}")
print(f"{'='*40}")
print(f"Predicted Yield : {round(float(prediction), 2)} tonnes/hectare")
print(f"Risk Level      : {risk}")
print(f"{'='*40}")
