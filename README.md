1. FetalAI: Predict and Monitor Fetal Health Using Machine Learning
  A clinical decision support system that uses cardiotocography (CTG) data and machine learning models to *predict fetal health conditions* as Normal, Suspect, or Pathological. Developed to assist doctors and expecting parents by automating and visualizing fetal health predictions.

2. Overview
*FetalAI* analyzes fetal health using *21 physiological features* from *CTG data. It uses supervised learning models trained on the **UCI fetal_health dataset. It helps detect early signs of fetal distress using a **Flask-based UI*, providing:

- 📤 CSV Upload
- 📝 Manual Form Input
- ✅ Baseline Sample Predictions
- 🔍 Visualization of Results


3. Dataset Description
- *Source*: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Fetal+Health)  
- *File*: fetal_health.csv  
- *Records*: ~2,100 samples  
- *Target Variable*: fetal_health  
  - 1 → Normal  
  - 2 → Suspect  
  - 3 → Pathological  

4. Feature Descriptions
| Feature                          | Description                                                |
|----------------------------------|------------------------------------------------------------|
| baseline value                   | Baseline fetal heart rate (FHR)                            |
| accelerations                    | Number of accelerations per second                         |
| fetal_movement                   | Number of fetal movements per second                       |
| uterine_contractions             | Frequency of uterine contractions                          |
| light_decelerations              | Number of light decelerations                              |
| severe_decelerations             | Number of severe decelerations                             |
| prolongued_decelerations        | Number of prolonged decelerations                          |
| abnormal_short_term_variability | Duration of abnormal STV (Short Term Variability)         |
| mean_value_of_short_term_variability | Average STV value                                |
| percentage_of_time_with_abnormal_long_term_variability | % time with abnormal LTV |
| mean_value_of_long_term_variability | Average LTV value                                 |
| histogram_width                  | Width of the FHR histogram                                 |
| histogram_min                    | Minimum histogram value                                    |
| histogram_max                    | Maximum histogram value                                    |
| histogram_number_of_peaks       | Number of peaks in the histogram                           |
| histogram_number_of_zeroes      | Number of zero crossings                                   |
| histogram_mode                   | Most frequent histogram value                              |
| histogram_mean                   | Mean value of histogram                                    |
| histogram_median                 | Median histogram value                                     |
| histogram_variance               | Variance in histogram data                                 |
| histogram_tendency               | Direction of histogram trend (rising/falling)             |


5. Model Pipeline
    1. *📥 Load Dataset*  
    2. *🧹 Clean Missing Values*  
    3. *🧬 Balance Dataset* using *SMOTE*  
    4. *🔀 Split Data* into Train/Test (80:20)  
    5. *⚖ Scale Features* using StandardScaler  
    6. *🧠 Train Models*  
   - Random Forest  
   - Decision Tree  
   - Logistic Regression  
   - K-Nearest Neighbors  
    7. *🏆 Select Best Model* based on accuracy  
    8. *💾 Save Artifacts* (model.pkl, scaler.pkl)  


6. Model Accuracy & Selection
| Model               | Accuracy (approx.) |
|--------------------|--------------------|
| *Random Forest*   | 0.98 (Best)      |
| Decision Tree       | 0.93               |
| Logistic Regression | 0.91               |
| KNN                 | 0.88               |


7. Web Interface
The web app provides:

- *CSV Upload*: For batch predictions  
- *Manual Form Input*: With dynamic baseline filling  
- *Result Table*: With Bootstrap styling  
- *Tabs*: Home | Upload CSV | Manual Input | Contact

8. Routes
| Route        | Description                      |
|--------------|----------------------------------|
| /          | Homepage with tabs               |
| /predict   | POST route for both CSV & Form   |

---

9. File Structure

fetal_health_project/
│
├── fetal_health.csv                  # Dataset
├── model.pkl                         # Best ML model
├── scaler.pkl                        # Feature scaler
├── model_accuracy_comparison.png     # Accuracy comparison chart
│
├── app.py                            # Flask backend
├── templates/
│   ├── index.html                    # Web UI
│   └── result.html                   # Results display
├── static/
│   └── style.css (optional)          # Custom styles
├── uploads/
│   └── (User-uploaded CSVs)


---

10. How to Run Locally

### 🧰 Requirements

bash
pip install pandas numpy scikit-learn imbalanced-learn flask joblib matplotlib


### ▶️ Run the App

bash
python app.py


### 🌐 Open in Browser

bash
http://127.0.0.1:5000/


---

11. Sample Predictions

Use preloaded test cases in the **manual input tab**:

### 🟢 Healthy Sample

json
[120.0, 0.005, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 50, 60, 160, 1, 0, 150, 140, 140, 10, 1]


### 🟡 Suspect Sample

json
[100.0, 0.001, 0.004, 0.002, 0.01, 0.0, 0.0, 1.0, 0.3, 10.0, 0.6, 30, 50, 140, 3, 0, 120, 110, 105, 12, -1]


### 🔴 Pathological Sample

json
[80.0, 0.0, 0.001, 0.003, 0.02, 0.01, 0.01, 2.0, 0.2, 20.0, 0.4, 40, 30, 110, 5, 1, 100, 90, 85, 20, -1]
```
