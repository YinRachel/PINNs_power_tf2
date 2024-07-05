# Physics-Informed Neural Networks for Power Systems

This project implements a physics-informed neural network (PINN) to model and predict the behavior of power systems using TensorFlow, based on:
G. Misyris, A. Venzke, S. Chatzivasileiadis, " Physics-Informed Neural Networks for Power Systems", 2019. Available online: https://arxiv.org/abs/1911.03737

## Features

- Implementation of a custom TensorFlow model class to encapsulate the PINN architecture.
- Use of Latin Hypercube Sampling (LHS) for efficient sampling of the domain.
- Integration of differential equations directly into the loss functions to enforce physical laws.
- Visualization of prediction results against actual data to evaluate model performance.

## Requirements

To run this project, you need the following packages:
- TensorFlow
- NumPy
- SciPy
- Matplotlib
- pandas
- pyDOE

Ensure you have the latest version of these packages to avoid compatibility issues.

## Data for training
The model expects a .mat file containing the variables t, x1, and usol1, representing time, delta that connnected with this Bus(delta_j), and the solution(delta_k), respectively. Adjust the script accordingly if your data format differs.

## Outputs
The script outputs:

A plot comparing the predicted values to the actual data.
An L2 error metric printed to the console.
Optionally, predictions can be saved to an Excel file for further analysis.
