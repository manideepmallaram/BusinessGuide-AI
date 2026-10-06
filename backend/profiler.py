import pandas as pd


def profile_dataset(df: pd.DataFrame):
    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string", "str"]
    ).columns.tolist()

    date_columns = []

    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            date_columns.append(column)

    column_profiles = {}

    for column in df.columns:
        column_profiles[column] = {
            "data_type": str(df[column].dtype),
            "missing_values": int(df[column].isna().sum()),
            "unique_values": int(df[column].nunique()),
        }

    numerical_statistics = {}

    for column in numerical_columns:
        numerical_statistics[column] = {
            "mean": float(df[column].mean()),
            "median": float(df[column].median()),
            "minimum": float(df[column].min()),
            "maximum": float(df[column].max()),
        }

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns,
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_profiles": column_profiles,
        "numerical_statistics": numerical_statistics,
    }