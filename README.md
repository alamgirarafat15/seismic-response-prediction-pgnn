# Seismic Response Prediction Using Machine Learning and PGNN

This repository contains the computational codes developed for the prediction of earthquake-induced structural responses using machine learning and a Physics-Guided Neural Network (PGNN).

## Project Overview

The study develops and compares three machine-learning approaches for predicting structural seismic responses:

- Random Forest (RF)
- Artificial Neural Network (ANN)
- Physics-Guided Neural Network (PGNN)

The prediction targets are:

- Maximum Story Drift
- Maximum Roof Displacement

## Dataset Generation

A simulation-based dataset containing 1,000 structural–ground-motion cases was developed using:

- 50 synthetic reinforced-concrete (RC) building structures
- 20 earthquake ground motions
- OpenSeesPy-based nonlinear time-history analysis (NLTHA)

The resulting NLTHA simulations were used to obtain the structural response quantities required for machine-learning model development.

## Methodology

The computational workflow includes:

1. Synthetic RC building generation
2. Earthquake ground-motion preparation
3. NLTHA using OpenSeesPy
4. Structural response extraction
5. Data preprocessing and feature preparation
6. RF, ANN, and PGNN model development
7. Model evaluation
8. SHAP-based feature interpretation of the PGNN

The dataset was divided into training, validation, and independent test sets using a 70%, 15%, and 15% split, respectively.

## Physics-Guided Neural Network

The PGNN incorporates reduced-order, physics-based information as supplementary guidance during model training. The physics-guided component uses selected structural and ground-motion characteristics to provide additional information related to structural dynamic response.

## Explainable AI

SHapley Additive exPlanations (SHAP) was used to interpret the PGNN predictions and examine the contribution of the input features to:

- Maximum Story Drift
- Maximum Roof Displacement

## Repository Structure

```text
Seismic-Response-Prediction-PGNN/
│
├── OpenSeesPy/
│   └── OpenSeesPy and NLTHA codes
│
├── Models/
│   ├── RF/
│   ├── ANN/
│   └── PGNN/
│
├── SHAP/
│   └── SHAP analysis code
│
├── Data/
│   └── Dataset-related files
│
└── README.md