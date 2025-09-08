from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained pipeline (Scaler + LogisticRegression)
with open("logistic_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values in correct order
        iq = float(request.form['IQ'])
        prev_sem = float(request.form['Prev_Sem_Result'])
        cgpa = float(request.form['CGPA'])
        academic_perf = int(request.form['Academic_Performance'])
        internship = int(request.form['Internship_Experience'])  # 1 for Yes, 0 for No
        extra_curr = int(request.form['Extra_Curricular_Score'])
        comm_skills = int(request.form['Communication_Skills'])
        projects = int(request.form['Projects_Completed'])

        # Make prediction
        features = np.array([[iq, prev_sem, cgpa, academic_perf,
                              internship, extra_curr, comm_skills, projects]])
        
        prediction = pipeline.predict(features)[0]
        probability = pipeline.predict_proba(features)[0][1] * 100  # probability of being placed

        result = "Placed 🎉" if prediction == 1 else "Not Placed 😞"

        return render_template('index.html',
                               prediction_text=f'Prediction: {result}',
                               probability_text=f'Chance of Placement: {probability:.2f}%')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
