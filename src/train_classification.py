"""
Predict whether a student gets placed or not.

Uses 5-fold stratified cross-validation as the primary accuracy metric,
not a single train/test split. With only 215 rows, a single 80/20 split
tests on just ~43 students - a couple of lucky/unlucky guesses can swing
"accuracy" by 5-10 points and mean nothing. Cross-validation tests every
row exactly once as unseen data, which is the honest number to report.
"""
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from data_prep import prepare

TARGET = "status"
DROP_FOR_CLASSIFICATION = ["salary"]

BEST_RF_PARAMS = dict(
    n_estimators=200, max_depth=None, min_samples_split=2,
    min_samples_leaf=1, max_features="log2", random_state=42,
)
BEST_GB_PARAMS = dict(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    subsample=0.7, random_state=42,
)


def main():
    raw, df, encoders = prepare()
    X = df.drop(columns=[TARGET] + DROP_FOR_CLASSIFICATION)
    y = df[TARGET]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest (tuned)": RandomForestClassifier(**BEST_RF_PARAMS),
        "Gradient Boosting (tuned)": GradientBoostingClassifier(**BEST_GB_PARAMS),
    }

    print("=== 5-fold cross-validated accuracy (the honest number) ===")
    cv_results = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        cv_results[name] = scores.mean()
        print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

    best_name = max(cv_results, key=cv_results.get)
    print(f"\nBest by cross-validation: {best_name} ({cv_results[best_name]:.3f})")

    best_model = candidates[best_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)

    print("\n=== Held-out split (illustrative only, not the reported metric) ===")
    print(classification_report(y_test, preds, target_names=encoders["status"].classes_))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, preds))

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
        print("\nTop factors influencing placement:")
        print(importances.sort_values(ascending=False).head(6))

    best_model.fit(X, y)
    joblib.dump(best_model, "models/placement_classifier.pkl")
    joblib.dump(encoders, "models/encoders.pkl")
    print(f"\nSaved final model ({best_name}, refit on all data) to models/placement_classifier.pkl")


if __name__ == "__main__":
    main()
