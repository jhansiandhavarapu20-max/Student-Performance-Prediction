
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("student_performance_model.pkl")

print("STUDENT PERFORMANCE PREDICTION")
print("--------------------------------")

# Get student details from user
study_hours = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance: "))
previous_marks = float(input("Enter Previous Marks: "))
assignment_score = float(input("Enter Assignment Score: "))
internal_marks = float(input("Enter Internal Marks: "))
sleep_hours = float(input("Enter Sleep Hours: "))
internet_usage = float(input("Enter Internet Usage: "))
class_participation = float(input("Enter Class Participation: "))
backlogs = int(input("Enter Number of Backlogs: "))
lab_score = float(input("Enter Lab Score: "))
project_score = float(input("Enter Project Score: "))
learning_hours = float(input("Enter Learning Hours: "))

# Prepare student data
new_student = pd.DataFrame([{
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Previous_Marks": previous_marks,
    "Assignment_Score": assignment_score,
    "Internal_Marks": internal_marks,
    "Sleep_Hours": sleep_hours,
    "Internet_Usage": internet_usage,
    "Class_Participation": class_participation,
    "Backlogs": backlogs,
    "Lab_Score": lab_score,
    "Project_Score": project_score,
    "Learning_Hours": learning_hours
}])

# Predict final marks
prediction = model.predict(new_student)

print("\nPredicted Final Marks:", round(prediction[0], 2))
