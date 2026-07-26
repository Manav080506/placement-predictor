"""
Data preparation for the Campus Placement & Salary Predictor.
Loads the raw CSV, cleans it, and encodes categorical variables
for both the classification (placement) and regression (salary) tasks.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

RAW_PATH = "data/Placement_Data_Full_Class.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # sl_no is just a row index - drop it
    df = df.drop(columns=["sl_no"])

    # Students with no salary simply weren't placed - fill with 0
    # rather than dropping them, since "not placed" is a real, useful class.
    df["salary"] = df["salary"].fillna(0)

    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encodes categorical columns and returns both the encoded
    dataframe and the fitted encoders (needed later to decode predictions
    or encode new incoming data at inference time).
    """
    df = df.copy()
    categorical_cols = [
        "gender", "ssc_b", "hsc_b", "hsc_s",
        "degree_t", "workex", "specialisation", "status",
    ]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


def prepare(path: str = RAW_PATH):
    df = load_raw(path)
    df = clean(df)
    df_encoded, encoders = encode_categoricals(df)
    return df, df_encoded, encoders


if __name__ == "__main__":
    raw, encoded, encoders = prepare()
    print("Raw shape:", raw.shape)
    print("Placed / Not placed counts:")
    print(raw["status"].value_counts())
    encoded.to_csv("outputs/placement_processed.csv", index=False)
    print("Saved cleaned + encoded data to outputs/placement_processed.csv")
