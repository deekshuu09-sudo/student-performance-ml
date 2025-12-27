import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("data.csv")

# Features and target
X = data[['study_hours', 'attendance']]
y = data['score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# User input
study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
prediction = model.predict([[study_hours, attendance]])
print("Predicted Score:", round(prediction[0], 2))