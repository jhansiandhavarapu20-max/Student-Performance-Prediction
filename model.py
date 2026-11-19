import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("dataset.csv")

# Separate input features and target
X = data.drop("Final_Marks", axis=1)
y = data["Final_Marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)
# Save the trained model
joblib.dump(model, "student_performance_model.pkl")

print("Model saved successfully!")

print("MODEL TRAINING COMPLETED!")

# Predict test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

# Predict a new student's marks
new_student = pd.DataFrame([{
    "Study_Hours": 6,
    "Attendance": 85,
    "Previous_Marks": 75,
    "Assignment_Score": 80,
    "Internal_Marks": 78,
    "Sleep_Hours": 7,
    "Internet_Usage": 3,
    "Class_Participation": 8,
    "Backlogs": 0,
    "Lab_Score": 85,
    "Project_Score": 90,
    "Learning_Hours": 2
}])

predicted_marks = model.predict(new_student)

print("\nSTUDENT PERFORMANCE PREDICTION")
print("Predicted Final Marks:", round(predicted_marks[0], 2))



import matplotlib.pyplot as plt

# Actual vs Predicted Marks
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Final Marks")
plt.ylabel("Predicted Final Marks")
plt.title("Student Performance Prediction")

plt.plot([0, 100], [0, 100], linestyle="--")

plt.grid(True)
plt.savefig("prediction_graph.png")
plt.show()
