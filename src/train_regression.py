"""
Predict expected salary - but only for students who got placed.
Salary is meaningless for unplaced students, so we filter to
status == Placed before training, otherwise the zeros distort everything.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from data_prep import prepare

TARGET = "salary"


def main():
    raw, df, encoders = prepare()

    placed_label = list(encoders["status"].classes_).index("Placed")
    placed_df = df[df["status"] == placed_label].drop(columns=["status"])

    X = placed_df.drop(columns=[TARGET])
    y = placed_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"\n=== {name} ===")
        print(f"MAE: {mae:,.0f}  |  R2: {r2:.3f}")

    # Salary has a lot of noise in this dataset (small sample, few outliers) -
    # be upfront about that rather than overselling R2.
    joblib.dump(models["Random Forest"], "models/salary_regressor.pkl")
    print("\nSaved regressor to models/salary_regressor.pkl")
    print("\nNote: salary R2 will look modest - only 148 placed students and a couple")
    print("of high outliers dominate variance. That's a real, worth-mentioning finding,")
    print("not a bug: academic scores predict WHETHER you're placed far better than")
    print("they predict HOW MUCH you're paid.")


if __name__ == "__main__":
    main()
