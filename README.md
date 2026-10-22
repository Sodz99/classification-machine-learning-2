# Classification Projects on Machine Learning for Beginners - 2

Understanding various Ensemble techniques and implementing them to predict license status for the given business.

## Overview
In machine learning, Classification is one of the most widely used techniques with various applications. For sentiment analysis, spam detection, risk assessment, churn prediction, and medical diagnosis classification have served as a very simple yet powerful method. In this project, we aim to give you hands-on experience and theoretical explanations of various ensemble techniques. You can find the first project of this series here.

## Data
The dataset used is a licensed dataset. It contains information about 86K different businesses over various features. The target variable is the status of the license, which has five different categories.

_Data is not included in this repository._

## Approach
1. Data Description
2. Exploratory Data Analysis
3. Data Cleaning a. Missing Value imputation b. Outlier Detection
4. Data Imbalance
5. Data Encoding
6. Model Building a. Random Forest b. AdaBoost c. XGBoost
7. Feature importance
8. Hyperparameter tuning a. Random search optimization b. Grid search optimization c. Bayesian optimization

Modular code overview

Once you unzip the modular_code.zip file, you can find the following folders within it.
1. input
2. src
3. output
4. lib
1. The input folder contains the data that we have for analysis. In our case, it contains Licence_Data.csv.
2. The src folder is the heart of the project. This folder contains all the modularized code for all the above steps in a modularized manner.

The model_selection.py and preprocessing.py files contain all the functions in a modularized manner, which are appropriately named. These python functions are then called inside the run.py file. The requirements.txt file has all the required libraries with respective versions. Kindly install the file by using the command pip install -r requirements.txt

3. The output folder contains an excel file for classification metrics scores of each model.
4. The lib folder is a reference folder. It contains the original ipython notebook that we saw in the videos.

## Tech stack
➔ Language: Python ➔ Libraries: pandas, scikit_learn, category_encoders, numpy, os, seaborn, matplotlib, hyperopt, xgboost

## How to run
```bash
pip install -r multi_class_classification_2/src/requirements.txt
python multi_class_classification_2/src/run.py
```
Notebooks: [`License Status Multi Class Classification.ipynb`](multi_class_classification_2/lib/License%20Status%20Multi%20Class%20Classification.ipynb)

## Project structure
```
.gitignore
README.md
multi_class_classification_2/
  lib/
    License Status Multi Class Classification.ipynb
  src/
    model_selection.py
    preprocessing.py
    requirements.txt
    run.py
```

## Credits
Based on a ProjectPro guided project.
