from flask import Flask, render_template, request
import joblib
import pandas as pd

def add_weighted_score(df):
    weights = {
        "CGPA": 0.4,
        "Communication_Skills": 0.3,
        "Projects_Completed": 0.2,
        "Internship_Experience": 0.1
    }
    df = df.copy()
    df["Weighted_Score"] = (
        df["CGPA"] * weights["CGPA"] +
        df["Communication_Skills"] * weights["Communication_Skills"] +
        df["Projects_Completed"] * weights["Projects_Completed"] +
        df["Internship_Experience"] * weights["Internship_Experience"]
    )
    return df

with open("logistic_new_model.pkl", "rb") as f:
    model = joblib.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        iq = float(request.form["iq"])
        prev_sem = float(request.form["prev_sem"])
        cgpa = float(request.form["cgpa"])
        academic_perf = float(request.form["academic_perf"])
        internship = int(request.form["internship"])
        extra_curricular = float(request.form["extra_curricular"])
        communication = float(request.form["communication"])
        projects = int(request.form["projects"])

        student = pd.DataFrame([{
            "IQ": iq,
            "Prev_Sem_Result": prev_sem,
            "CGPA": cgpa,
            "Academic_Performance": academic_perf,
            "Internship_Experience": internship,
            "Extra_Curricular_Score": extra_curricular,
            "Communication_Skills": communication,
            "Projects_Completed": projects
        }])

        # Apply feature engineering
        student = add_weighted_score(student)

        # Predict
        prediction = model.predict(student)[0]
        probability = model.predict_proba(student)[0][1] * 100

        result = "Placed ✅" if prediction == 1 else "Not Placed ❌"

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {result} (Probability: {probability:.2f}%)"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)