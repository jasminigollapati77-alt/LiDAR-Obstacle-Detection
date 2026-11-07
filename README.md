
# LiDAR Obstacle Detection and Risk Analysis

## Project Overview
A machine learning based LiDAR obstacle detection dashboard.

## Technologies
- Python
- ESP32
- TF-Luna LiDAR
- Random Forest Classifier
- Streamlit
- Plotly
- Pandas
- NumPy

## Features
- Obstacle and Clear predictions
- Manual sensor readings
- Historical CSV upload
- Distance-based risk analysis
- Critical, High, Medium and Low risk levels
- Interactive graphs
- Servo angle analysis
- Downloadable risk reports

## Dataset
The project uses previously recorded LiDAR readings.

Columns:
timestamp, run_id, time_s, distance_m,
distance_cm, signal, servo_angle, valid

## Run Locally

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Disclaimer
This is an educational demonstration system.
It is not a safety-certified obstacle detection system.
