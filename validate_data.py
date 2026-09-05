
import pandas as pd

DATA_PATH = "data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]

df = pd.read_csv(DATA_PATH)

actual_columns = [
    col for col in df.columns
    if col != "Unnamed: 0"
]

missing_columns = set(EXPECTED_COLUMNS) - set(actual_columns)

if missing_columns:
    raise ValueError(
        f"Missing expected columns: {missing_columns}"
    )

print("DATA VALIDATION PASSED")
print("Rows:", df.shape[0])
print("Columns:", len(actual_columns))
print("\nMissing values:")
print(df.isnull().sum())
print("\nTarget distribution:")
print(df["ProdTaken"].value_counts())
