# Forex Price Forecasting Using Recurrent Neural Networks (RNNs)

This project focuses on the application of Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) networks, to forecast foreign exchange (Forex) prices. RNNs have the capability to remember sequences, making them ideal for time-series data like Forex rates.

![RNN Architecture](forex_euro_us.png)

## Table of Contents
1. [Libraries and Tools](#libraries-and-tools)
2. [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
3. [Model Architecture](#model-architecture)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Interactive Data Visualization](#interactive-data-visualization)
6. [Code Implementation](#code-implementation)

## Libraries and Tools

- `numpy`: For array manipulations and mathematical operations.
- `pandas`: For data ingestion, manipulation, and dataframes.
- `matplotlib`: For generating static plots and data visualizations.
- `plotly`: For creating interactive charts and figures.
- `scikit-learn`: For preprocessing data and scaling features.
- `tensorflow`: For designing, training, and evaluating the neural network.

## Data Preprocessing and Feature Engineering

### Data Source and Attributes

The dataset spans from 2020 to 2023, capturing various aspects of Forex rates such as the opening, highest, lowest, and closing prices over different intervals.

### Data Loading

Data is read into Pandas dataframes and initial inspections are performed to understand the shape, size, and data types of the attributes.

### Data Scaling

A feature scaling step normalizes the feature set using Min-Max scaling, ensuring that the values lie in a similar range. This is critical for the efficient training of neural networks.

### Time-Series Transformation

The data is then transformed into sequences that represent the Forex rates at different time intervals. Each sequence consists of 60 time steps, which act as the features for training the model.

## Model Architecture

The neural network used in this project is a specific type of RNN known as Long Short-Term Memory (LSTM). It consists of three LSTM layers followed by a dense output layer.

### Layer Details

- The first LSTM layer has 50 units and returns sequences to match the input shape.
- The second LSTM layer also has 50 units and returns sequences.
- The third LSTM layer has 50 units but does not return sequences.
- The dense layer has one unit, corresponding to the output feature, which is the Forex rate at the next time interval.

## Evaluation Metrics

### Mean Squared Error

The model's performance is assessed using the Mean Squared Error (MSE) which is a measure of the average of the squares of the errors between the predicted and actual values.

## Interactive Data Visualization

Post-training, the model's predictions are plotted alongside the actual data points using Plotly, providing an interactive way to assess model performance. The Plotly graph supports zooming, panning, and hovering over data points, allowing for a thorough examination of the model's accuracy.

## Code Implementation

To run the code, a well-structured CSV file with Forex rates is needed. The `file_path` in the code should be updated to point to this file. All required Python packages can be installed using pip:

```bash
pip install numpy pandas matplotlib plotly scikit-learn tensorflow
