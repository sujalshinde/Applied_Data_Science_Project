from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_map = {1: "Normal", 2: "Suspect", 3: "Pathological"}

# Feature Names
feature_names = [
    'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
    'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
    'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability',
    'mean_value_of_long_term_variability', 'histogram_width', 'histogram_min',
    'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes',
    'histogram_mode', 'histogram_mean', 'histogram_median',
    'histogram_variance', 'histogram_tendency'
]

# Example Baseline Samples
baseline_samples = {
    "Healthy Sample": [120.0, 0.005, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5,
                       0.0, 0.5, 50, 60, 160, 1, 0, 150, 140, 140, 10, 1],
    "Suspect Sample": [100.0, 0.001, 0.004, 0.002, 0.01, 0.0, 0.0, 1.0, 0.3,
                       10.0, 0.6, 30, 50, 140, 3, 0, 120, 110, 105, 12, -1],
    "Pathological Sample": [80.0, 0.0, 0.001, 0.003, 0.02, 0.01, 0.01, 2.0, 0.2,
                            20.0, 0.4, 40, 30, 110, 5, 1, 100, 90, 85, 20, -1]
}

@app.route('/')
def index():
    return render_template("index.html", feature_names=feature_names, baselines=baseline_samples)

@app.route('/predict', methods=['POST'])
def predict():
    # If CSV is uploaded
    if 'csv_file' in request.files and request.files['csv_file'].filename != '':
        file = request.files['csv_file']
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        df = pd.read_csv(filepath)

        if "fetal_health" in df.columns:
            df = df.drop("fetal_health", axis=1)

        scaled = scaler.transform(df)
        predictions = model.predict(scaled)
        df["Prediction"] = [label_map.get(p, "Unknown") for p in predictions]

        return render_template("result.html", tables=[df.to_html(classes='table table-sm table-striped', index=False)])

    # Manual Input Prediction
    try:
        input_values = []
        for f in feature_names:
            val = request.form.get(f)
            if val is None or val.strip() == '':
                return f"❌ Missing or invalid input for field: {f}"
            try:
                input_values.append(float(val))
            except ValueError:
                return f"❌ Invalid numeric value for field: {f} → {val}"

        X_input = np.array(input_values).reshape(1, -1)
        X_scaled = scaler.transform(X_input)
        prediction = model.predict(X_scaled)[0]
        label = label_map.get(prediction, "Unknown")

        result_df = pd.DataFrame([input_values], columns=feature_names)
        result_df["Prediction"] = label

        return render_template("result.html", tables=[result_df.to_html(classes='table table-sm table-striped', index=False)])
    except Exception as e:
        return f"❌ Error processing input: {e}"

if __name__ == '__main__':
    app.run(debug=True)
