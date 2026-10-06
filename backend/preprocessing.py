import pandas as pd


def detect_date_columns(df: pd.DataFrame):
    date_columns = []

    for column in df.columns:

        # Already a datetime column
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            date_columns.append(column)
            continue

        # Check string-like columns
        if pd.api.types.is_string_dtype(df[column]):
            converted = pd.to_datetime(
                df[column],
                errors="coerce",
                format="mixed"
            )

            valid_ratio = converted.notna().mean()

            if valid_ratio >= 0.8:
                date_columns.append(column)

    return date_columns


def preprocess_dataset(df: pd.DataFrame):

    df = df.copy()

    # Clean column names
    df.columns = [
        str(col).lstrip("\ufeff").strip().replace(" ", "_")
        for col in df.columns
    ]

    # Remove completely empty rows and columns
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    # Remove duplicate rows
    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # Detect and convert date columns
    date_columns = detect_date_columns(df)

    for column in date_columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

    # Handle missing values
    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string", "str"]
    ).columns.tolist()

    for column in numerical_columns:
        if df[column].isna().any():
            df[column] = df[column].fillna(
                df[column].median()
            )

    for column in categorical_columns:
        if df[column].isna().any():
            df[column] = df[column].fillna(
                "Unknown"
            )

    return {
        "rows_after_preprocessing": len(df),
        "columns_after_preprocessing": len(df.columns),
        "date_columns": date_columns,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "duplicates_removed": duplicates_removed,
        "missing_values_after_preprocessing": int(
            df.isna().sum().sum()
        ),
        "data": df
    }