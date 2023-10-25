# Binary Model With Kedro

## Overview

This is a Kedro project, which was generated using `kedro 0.18.11`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

The flow is about building a model to predict a dataset with binary outcomes. The example used in here is [divorce dataset](https://archive.ics.uci.edu/dataset/497/divorce+predictors+data+set). 

The flow can be used for any others dataset with binary outcomes, by modifing the `data` path in `conf/base/catalog.yml` and target `y` in  `conf/base/parameters.yml`

This project aims to predict a binary outcome using a step-by-step flow that includes data cleansing, data exploratory analysis, feature selection, model training, and model validation. The following sections describe each step in detail.

## Getting Started

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
pip install -r src/requirements.txt
```

### How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

### How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, go to the `.coveragerc` file.

### Run with Docker

```
$ docker build .
$ docker run -i -t IMAGE_ID /bin/bash
```

## Overall flow in summary

![kedro_viz_img](https://github.com/kang20006/binary-model-kedro/blob/cee857a8e887038d945ee98e0835a35b9aca51e7/kedro-pipeline.png)

This can be generated using

```
kedro viz
```
## Steps
### Data Cleansing

The data cleansing step focuses on preparing the dataset for analysis by removing any inconsistencies, or outliers that may affect the quality of the predictions. This step may involve:

- One-hot encoding: Converting categorical variables into binary features using one-hot encoding.
- Resampling: Balancing imbalanced datasets using techniques such as SMOTE (Synthetic Minority Over-sampling Technique) and Tomek links.
- Outlier treatment: Addressing outliers that can skew the analysis using the percentile-based method.

### Data Exploratory Analysis

Data exploratory analysis involves understanding the dataset's characteristics, identifying patterns, and gaining insights that can guide feature selection and model development. This step may include:

- Descriptive statistics: Calculating summary statistics, such as mean, median, standard deviation, percentile, missing count, and max min.
- All reports will generated in an excel file.

## Feature Selection

Feature selection aims to identify the most relevant and informative features for the prediction task. This step helps reduce dimensionality and improve model performance. The feature selection process includes the following steps:

1. IV-WOE (Information Value and Weight of Evidence): Calculate the IV and WOE for each feature to assess their predictive power and relationship with the target variable.
2. Random Forest: Use a random forest model to determine feature importance based on the Gini index.
3. Combine Selection: Combine the results from IV-WOE and random forest feature importance to identify the top features.
4. Correlation Threshold: Remove highly correlated features by setting a correlation threshold and eliminating one feature from each highly correlated pair.
5. VIF (Variance Inflation Factor): Calculate the VIF for the remaining features and remove features with high multicollinearity.

### Model Training

In this step, various machine learning models are trained using the selected features and the labeled data. The model training process includes the following steps:

1. Basic Logistic Model: Build a basic logistic regression model using the selected features and evaluate its performance.
2. Backward Selection: Perform backward selection to eliminate less significant features based on statistical tests (accuracy).
3. Forward Selection: Perform forward selection to iteratively add the most significant features based on statistical tests (accuracy).

### Model Validation

Model validation assesses the performance and generalizability of the trained models using appropriate evaluation metrics. This step helps understand how well the model performs on unseen data and allows for model comparison. The model validation process includes the following steps:

1. 4-Fold Cross-Validation: Perform 4-fold cross-validation to obtain performance estimates on different subsets of the data.
2. Evaluation Metrics: Calculate metrics such as accuracy, precision, recall, and F1 score for each model and feature selection technique.
3. Logistic Regression Hyperparameter Tuning: Fine-tune the hyperparameters of the logistic regression model using techniques random search.
4. Model Comparison: Compare the performance of the basic logistic model, logistic regression with hyperparameter tuning, backward selection, and forward selection to identify the most effective approach.
5. Propose final model.

## Output Datasets

- models : `data/06_models`
- reports : `data/08_reporting`

## Parameters 

All parameters can be modified in `conf/base/parameters`
