# California Housing Value Prediction
## CSCI 6201 Lab 2

**Project type:** Regression / supervised machine learning  
**Dataset:** California Housing dataset provided by scikit-learn

## Team Members
- Venkat Saketh Chowdary Maddineni

## Project Overview
This project investigates how housing characteristics, median income, and geographic information relate to median house values across California districts. We will first explore the data and train a baseline regression model, then improve the analysis for the Final Project.

**Important limitation:** This dataset is derived from the 1990 California census. It is useful for learning predictive modeling and studying relationships, but it should not be presented as a source of current housing prices.

## Dataset
The dataset contains 20,640 observations and 8 input features.

Features:
- `MedInc`: median income in a block group
- `HouseAge`: median house age
- `AveRooms`: average number of rooms
- `AveBedrms`: average number of bedrooms
- `Population`: block-group population
- `AveOccup`: average household occupancy
- `Latitude`: geographic latitude
- `Longitude`: geographic longitude

Target:
- `MedHouseVal`: median house value, expressed in units of $100,000

Dataset documentation: https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

The dataset is loaded using `fetch_california_housing(as_frame=True)` from `sklearn.datasets`; the dataset is downloaded by scikit-learn when needed and is not manually uploaded to Blackboard.

## Research Questions
1. Relationship Analysis: How strongly is median income associated with median house value in California, and what patterns can be observed through exploratory data analysis and data visualization?
2. Linear Regression Performance: How accurately can a Linear Regression model predict median house values using the available dataset features, and which features contribute most to the predictions?
3. Model Comparison: Does a Decision Tree Regressor improve prediction performance compared with Linear Regression when evaluated using the same training and testing datasets?
4. Prediction Error Analysis: What types of prediction errors do the models make, and are there patterns in their errors across different house-value ranges?

## Current Lab 2 Work
The `analysis.py` script:
1. Loads the dataset and reports its dimensions, feature names, descriptive statistics, minimum and maximum values, and missing-value counts.
2. Creates two visualizations:
   - `income_vs_house_value.png`: median income versus median house value.
   - `house_value_distribution.png`: distribution of median house values.
3. Splits the data into training and testing sets.
4. Trains Linear Regression and a Decision Tree Regressor.
5. Reports Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).
6. Saves generated charts, statistics, and model results in `outputs/`.

## Visualization Interpretation
After running the script, review both charts and replace the notes below with observations supported by your actual figures.

- **Income vs. house value:** Check whether higher median income tends to correspond to higher median house value. Describe any non-linear pattern, spread, or ceiling effect you observe.
- **House-value distribution:** Describe the shape of the distribution and whether many observations appear near the maximum recorded value.

## Preliminary Model Evaluation
The script reports:
- **MAE:** average absolute prediction error in target units.
- **MSE:** average squared prediction error in squared target units.
- **R²:** proportion of target variation explained by the model on the test set, relative to predicting the test-set mean.

Because the target is in units of $100,000, an MAE of 0.5 would correspond to $50,000 in the dataset's target scale. This is an illustration of the units, not a result claimed for this project.

## Next Steps for Final Project

### 1. What Is Working So Far?

At the current Lab 2 checkpoint, the project focuses on analyzing the California Housing Dataset from scikit-learn and investigating factors associated with median house values.

The initial implementation is intended to establish the data-loading and regression-analysis workflow.

- **Dataset:** California Housing Dataset, loaded using scikit-learn.
- **Research focus:** Investigating the relationship between median income and median house value.
- **Predictive modeling:** Using Linear Regression as a baseline model and a Decision Tree Regressor for comparison.
- **Evaluation approach:** Using R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to evaluate predictive performance, once the models have been executed.
- **Project organization:** Maintaining the Python analysis code and supporting project materials in the GitHub repository.

The current checkpoint establishes the foundation for evaluating regression models and identifying opportunities for improvement. Actual model performance and completed implementation steps will be documented after verifying the code execution and results.

### 2. What Still Needs Work?

Several areas require further implementation, testing, and analysis before the final project is complete.

- **Model evaluation:** Run the baseline models and record their R², MAE, and RMSE scores on the same test dataset.
- **Model comparison:** Determine whether the Decision Tree Regressor performs better than Linear Regression and investigate possible reasons for any differences.
- **Feature analysis:** Examine correlations between the available features and the target variable, particularly median income and median house value.
- **Prediction errors:** Analyze residuals to identify patterns, systematic errors, and differences in prediction accuracy across house-value ranges.
- **Geographic analysis:** Investigate whether latitude and longitude reveal geographic patterns in housing values or prediction errors.
- **Target-value limitation:** Examine the upper-value cap in the target variable and consider how it may affect model evaluation and predictions.
- **Reproducibility:** Ensure that the data-processing steps, train-test split, evaluation metrics, and reported results are consistent and reproducible.

### 3. What Will We Improve?

The following improvements are planned for the next stages of the project.

1. **Compare additional regression models:** Extend the comparison to include Random Forest Regression and evaluate whether an ensemble model improves predictive performance over Linear Regression and Decision Tree Regression.

2. **Tune model hyperparameters:** Use cross-validation and appropriate hyperparameter search methods to identify better model configurations while reducing the risk of overfitting.

3. **Analyze feature importance:** Investigate which features contribute most to the predictions and assess the relationship between median income and median house value.

4. **Examine residuals and prediction errors:** Use residual plots and error summaries to identify systematic prediction errors, outliers, and differences in performance across target-value ranges.

5. **Investigate geographic patterns:** Explore how location-related features, including latitude and longitude, relate to housing values and model errors. Consider geographic visualizations where appropriate.

6. **Investigate the target-value cap:** Determine how the dataset's upper-value limit affects the distribution of the target variable, model predictions, and evaluation metrics.

7. **Evaluate preprocessing and feature engineering:** Test whether feature scaling, transformations, or derived features improve performance. Compare these approaches using consistent evaluation procedures and avoid introducing data leakage.

8. **Document and communicate results:** Summarize model performance in comparison tables and visualizations, explain the trade-offs between models, and identify the most suitable approach based on the evaluation results.

### 4. What Feedback Would Help?

We would appreciate instructor feedback on the following questions:

- Is comparing Linear Regression, Decision Tree Regression, and Random Forest Regression an appropriate scope for the final project?
- Are R², MAE, and RMSE sufficient evaluation metrics for comparing the regression models, or would you recommend additional metrics?
- Would geographic error analysis and investigation of the target variable's upper-value cap provide meaningful depth to the project?
- Is using cross-validation for hyperparameter tuning, followed by evaluation on a held-out test set, an appropriate methodology for this dataset?

Our goal is to develop a reproducible regression-analysis workflow, compare model performance fairly, understand prediction errors, and explain the limitations of housing-value prediction using the available features.

## How to Run
Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the analysis from this project folder:

```bash
python analysis.py
```

The first run needs an internet connection so scikit-learn can download the dataset. The script creates an `outputs/` folder with the charts, descriptive statistics, missing-value report, and model comparison.

## Team Contributions
- Venkat Saketh Chowdary Maddineni

Selected the California Housing Dataset from scikit-learn for the project.

Defined research questions focused on the relationship between median income and median house value and the predictive performance of regression models.

Created and organized the GitHub repository for the project.

Set up the project folder and Python development environment.

Prepared the initial project structure for data loading and analysis.

Documented the dataset source, loading method, and project objectives.

Identified Linear Regression and Decision Tree Regression as the initial models for comparison.

Identified R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) as evaluation metrics for the planned model comparison.
