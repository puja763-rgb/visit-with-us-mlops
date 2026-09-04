# Visit With Us - Wellness Tourism Package Prediction

## Project Objective

This project develops an automated machine learning solution to predict
whether a customer is likely to purchase the Wellness Tourism Package.

## Machine Learning Workflow

1. Data registration
2. Data validation
3. Data cleaning
4. Train-test split
5. Data preprocessing
6. Random Forest model
7. Hyperparameter tuning
8. Model evaluation
9. Model serialization
10. Streamlit deployment
11. GitHub Actions CI/CD

## Model

Random Forest Classifier with hyperparameter tuning using GridSearchCV.

## Target

ProdTaken

- 0 = Customer did not purchase
- 1 = Customer purchased

## Deployment

The Streamlit application accepts customer information and provides
the predicted probability of purchasing the package.

## MLOps

GitHub Actions automatically:

- validates the dataset
- prepares the data
- trains the model
- evaluates the model
- saves the trained model
- uploads workflow artifacts
- commits the updated model to the repository

## Repository Structure

data/
models/
artifacts/
src/
app/
.github/workflows/