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
1. How strongly is median income associated with median house value?
2. How accurately can a Linear Regression model predict median house values using the available features?
3. Does a Decision Tree Regressor improve prediction performance compared with Linear Regression?

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
1. What is working so far?
   - [After running the code, summarize what loads and runs successfully.]
2. What still needs work?
   - [Describe any issues, limitations, or areas that need further analysis.]
3. What will we improve?
   - Compare additional models, such as Random Forest Regressor.
   - Tune model parameters using cross-validation.
   - Examine feature importance and residual/error patterns.
   - Investigate geographic patterns and the target's upper-value cap.
   - Consider whether scaling, transformations, or feature engineering improve results.
4. What feedback would help?
   - [Add a specific question for the instructor.]

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
Record each member's actual work here. Every team member should make a meaningful contribution to the repository and understand the code they submit.

- Venkat Saketh Chowdary Maddineni
