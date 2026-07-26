"""
Predict whether a student gets placed or not.
Compares Logistic Regression vs Random Forest, and reports
which features matter most - the actual insight recruiters/
interviewers will ask about, not just the accuracy number.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data_prep import prepare

TARGET = "status"
DROP_FOR_CLASSIFICATION = ["salary"]  # salary leaks the answer - only known post-placement


def main():
    raw, df, encoders = prepare()

    X = df.drop(columns=[TARGET] + DROP_FOR_CLASSIFICATION)
    y = df[TARGET]  # 1 = Placed, 0 = Not Placed (see encoders["status"].classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = (model, acc, preds)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test, preds, target_names=encoders["status"].classes_))

    # Keep the better performer
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_acc, best_preds = results[best_name]
    print(f"\nBest model: {best_name} ({best_acc:.3f} accuracy)")

    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, best_preds))

    # Feature importance (only meaningful for the Random Forest)
    if best_name == "Random Forest":
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
        print("\nTop factors influencing placement:")
        print(importances.sort_values(ascending=False).head(6))

    joblib.dump(best_model, "models/placement_classifier.pkl")
    joblib.dump(encoders, "models/encoders.pkl")
    print("\nSaved best model to models/placement_classifier.pkl")


if __name__ == "__main__":
    main()
