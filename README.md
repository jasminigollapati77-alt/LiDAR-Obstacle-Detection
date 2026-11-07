# LiDAR-Obstacle-Detection
A LiDAR-based obstacle detection system using ESP32 and TF-Luna sensor with Random Forest machine learning. Collects distance, signal strength, and servo-angle data to classify Clear and Obstacle conditions, with an interactive Streamlit dashboard for real-time-style visualization, sensor data analysis, and prediction results.
🚗 LiDAR-Based Obstacle Detection Using Machine Learning

A machine learning project built using ESP32, TF-Luna LiDAR, Python, Random Forest, and Streamlit to measure distances, identify obstacles, classify surrounding areas as Clear or Obstacle, and visualize sensor readings through an interactive web dashboard.

Features

Real-time distance measurement using TF-Luna LiDAR

ESP32-based sensor data collection

Servo motor-based scanning at different angles

LED status indication

Serial communication between ESP32 and Python

Automatic collection and storage of sensor readings in CSV format

Machine learning-based obstacle classification

Random Forest classification model

Clear and Obstacle prediction

Risk-level identification based on measured distance

Interactive Streamlit dashboard

Distance and signal-strength visualizations

Servo-angle-based scanning visualization

CSV file upload and analysis

Downloadable obstacle analysis reports

Technologies

Python

ESP32

TF-Luna LiDAR Sensor

Random Forest

Scikit-learn

Pandas

NumPy

Streamlit

Plotly

PySerial

CSV

Hardware Components

Component

Purpose

ESP32

Main microcontroller

TF-Luna LiDAR

Measures distance to objects

Servo Motor

Changes the LiDAR scanning angle

LED

Provides status indication

USB Cable

ESP32 programming and serial communication

System Configuration

Component

ESP32 Pin

TF-Luna RX

GPIO 16

TF-Luna TX

GPIO 17

Servo Signal

GPIO 18

LED

GPIO 2

The ESP32 reads distance information from the TF-Luna sensor and sends the readings through serial communication to the Python data collection program.

Dataset

The project uses sensor readings collected from the ESP32 and TF-Luna LiDAR.

Dataset Features

Feature

Description

timestamp

Date and time of reading

run_id

Sensor collection session ID

time_s

Elapsed collection time

distance_m

Measured distance in meters

distance_cm

Measured distance in centimeters

signal

LiDAR signal strength

servo_angle

Servo scanning angle

valid

Indicates whether the reading is valid

Dataset Summary

Total collected readings: 28,486

Valid readings used for labeling: 28,077

Collection sessions: 3

Classification Labels

Clear

Obstacle

The labels are derived using a distance threshold of 0.50 meters:

Clear: Distance greater than 0.50 m

Obstacle: Distance less than or equal to 0.50 m

Note: These are threshold-derived labels, not independently verified obstacle annotations.

Machine Learning Model

A Random Forest Classifier is trained to classify sensor readings into Clear and Obstacle categories.

Input Features

Distance in meters

Servo scanning angle

LiDAR signal strength

Model Output

Clear

Obstacle

Model Evaluation

The model is evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

The earlier session-based evaluation achieved 100% accuracy. Since the labels are generated using a distance threshold, this result measures how well the model reproduces those labels and does not establish real-world obstacle detection reliability.

Streamlit Dashboard

The project includes an interactive Streamlit application that provides:

Sensor data analysis

Clear and Obstacle predictions

Prediction confidence

Risk-level visualization

Distance distribution graphs

Signal-strength analysis

Servo-angle scanning visualization

CSV upload and filtering

Downloadable analysis reports

The dashboard can analyze previously collected CSV files without requiring the ESP32 or LiDAR hardware to be connected.
