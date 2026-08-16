# Student Performance Prediction using Machine Learning

A machine learning project that predicts a student's exam score based on study hours and attendance percentage.

## Project Overview

The project uses Linear Regression to estimate student exam performance from historical student data.

The model takes the following inputs:

- Study hours
- Attendance percentage

It then predicts the expected exam score.

## Problem Statement

Student performance can be influenced by factors such as study time and attendance. This project explores how machine learning can be used to estimate exam scores based on these factors.

## Machine Learning Approach

1. Load the student performance dataset using Pandas.
2. Prepare the input features and target variable.
3. Split the dataset into training and testing sets.
4. Train a Linear Regression model using Scikit-learn.
5. Generate predictions for the test data.
6. Evaluate the model using Mean Squared Error.
7. Accept user input and generate an estimated exam score.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Linear Regression
- Git & GitHub

## Project Structure

```text
student-performance-ml/
├── student_performance.py
├── data.csv
├── requirements.txt
└── README.md
