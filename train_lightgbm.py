import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import joblib
import os

# Load dataset
df = pd.read_csv(r'C:\Users\Admin\Desktop\crop_yield.csv')
print("Loaded", len(df), "rows")
print("Columns:", df.columns.tolist())

df = df.dropna()

# Convert text to numbers
le_region = LabelEncoder()
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_weather = LabelEncoder()

df['Region_enc'] = le_region.fit_transform(df['Region'])
df['Soil_enc'] = le_soil.fit_transform(df['Soil_Type'])
df['Crop_enc'] = le_crop.fit_transform(df['Crop'])
df['Weather_enc'] = le_weather.fit_transform(df['Weather_Condition'])
df['Fertilizer_enc'] = df['Fertilizer_Used'].astype(int)
df['Irrigation_enc'] = df['Irrigation_Used'].astype(int)

# Define inputs and output
features = ['Region_enc', 'Soil_enc', 'Crop_enc', 'Rainfall_mm',
            'Temperature_Celsius', 'Fertilizer_enc', 'Irrigation_enc',
            'Weather_enc', 'Days_to_Harvest']
target = 'Yield_tons_per_hectare'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Train LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
print("\nTraining started...")
model.fit(X_train, y_train)
print("Training done!")

# Test accuracy
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nAccuracy Results:")
print(f"MAE: {mae:.4f} → predictions off by {mae:.2f} tonnes on average")
print(f"R2 Score: {r2:.4f} → {r2*100:.1f}% accuracy")

# Save model
os.makedirs('predictor/ml_models', exist_ok=True)
joblib.dump(model, 'predictor/ml_models/lightgbm_model.pkl')
joblib.dump(le_region, 'predictor/ml_models/le_region.pkl')
joblib.dump(le_soil, 'predictor/ml_models/le_soil.pkl')
joblib.dump(le_crop, 'predictor/ml_models/le_crop.pkl')
joblib.dump(le_weather, 'predictor/ml_models/le_weather.pkl')

print("\nModel saved!")
print("Regions:", le_region.classes_.tolist())
print("Soils:", le_soil.classes_.tolist())
print("Crops:", le_crop.classes_.tolist())
print("Weather:", le_weather.classes_.tolist())
