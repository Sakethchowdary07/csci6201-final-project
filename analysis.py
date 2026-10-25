"""
CSCI 6201 Lab 2 — California Housing Dataset
Run: python analysis.py
Generated charts and metrics are saved in the outputs/ folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    # 1. Load the dataset
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target  # Target is in units of $100,000.

    df = X.copy()
    df["MedHouseVal"] = y

    print("\n=== DATASET OVERVIEW ===")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of input features: {X.shape[1]}")
    print("\nFeature names:")
    for name in X.columns:
        print(f" - {name}")
    print("\nTarget: MedHouseVal (median house value, in units of $100,000)")
    print("\nFirst five rows:")
    print(df.head())

    # 2. Exploratory data analysis
    print("\n=== DATA SHAPE ===")
    print(df.shape)

    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(df.describe().round(3))

    print("\n=== MINIMUM VALUES ===")
    print(df.min().round(3))

    print("\n=== MAXIMUM VALUES ===")
    print(df.max().round(3))

    print("\n=== MISSING VALUES BY COLUMN ===")
    print(df.isnull().sum())

    # Save a compact summary for the project repository.
    df.describe().round(3).to_csv(OUTPUT_DIR / "descriptive_statistics.csv")
    df.isnull().sum().rename("missing_values").to_csv(
        OUTPUT_DIR / "missing_values.csv"
    )

    # 3. Visualization 1: median income vs median house value
    plt.figure(figsize=(8, 5))
    plt.scatter(
        df["MedInc"],
        df["MedHouseVal"],
        alpha=0.35,
        s=12,
    )
    plt.title("Median Income vs. Median House Value")
    plt.xlabel("Median income (dataset units)")
    plt.ylabel("Median house value (units of $100,000)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "income_vs_house_value.png", dpi=200)
    plt.close()

    # 4. Visualization 2: distribution of the target
    plt.figure(figsize=(8, 5))
    plt.hist(df["MedHouseVal"], bins=35, edgecolor="black")
    plt.title("Distribution of Median House Values")
    plt.xlabel("Median house value (units of $100,000)")
    plt.ylabel("Number of districts")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "house_value_distribution.png", dpi=200)
    plt.close()

    print("\n=== VISUALIZATION NOTES ===")
    print(
        "1. Income vs. value: inspect whether higher median income generally "
        "corresponds to higher median house value. The relationship may not "
        "be perfectly linear, and the target has an upper cap in this dataset."
    )
    print(
        "2. Target histogram: inspect the spread and skew of median house values. "
        "Look for a concentration of observations near the upper end."
    )

    # 5. Train/test split and baseline models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=8, random_state=42
        ),
    }

    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results.append(
            {
                "Model": model_name,
                "MAE (target units)": mae,
                "MSE (target units squared)": mse,
                "R2": r2,
            }
        )

        print(f"\n=== {model_name} ===")
        print(f"MAE: {mae:.4f} target units")
        print(f"MSE: {mse:.4f} target units squared")
        print(f"R-squared: {r2:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)

    print("\n=== MODEL COMPARISON ===")
    print(results_df.round(4).to_string(index=False))
    print(f"\nSaved charts and output files in: {OUTPUT_DIR.resolve()}")
    print(
        "\nReminder: the target is measured in units of $100,000 and the data "
        "comes from the 1990 census. Do not interpret it as current market prices."
    )

if __name__ == "__main__":
    main()
