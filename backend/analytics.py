import pandas as pd


def analyze_dataset(df: pd.DataFrame):

    result = {}

    # --------------------------------
    # Detect column types dynamically
    # --------------------------------

    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    date_columns = [
        column
        for column in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[column])
    ]

    # --------------------------------
    # Dataset Overview
    # --------------------------------

    result["dataset_overview"] = {
        "rows": len(df),
        "columns": len(df.columns),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns
    }

    # --------------------------------
    # Numerical Statistics
    # --------------------------------

    numerical_statistics = {}

    for column in numerical_columns:

        numerical_statistics[column] = {
            "sum": float(df[column].sum()),
            "mean": float(df[column].mean()),
            "median": float(df[column].median()),
            "minimum": float(df[column].min()),
            "maximum": float(df[column].max())
        }

    result["numerical_statistics"] = numerical_statistics

    # --------------------------------
    # Categorical Analysis
    # --------------------------------

    categorical_analysis = {}

    for column in categorical_columns:

        categorical_analysis[column] = {
            "unique_values": int(df[column].nunique()),
            "top_values": (
                df[column]
                .value_counts()
                .head(10)
                .to_dict()
            )
        }

    result["categorical_analysis"] = categorical_analysis

    # --------------------------------
    # Categorical vs Numerical
    # --------------------------------

    grouped_analysis = {}

    for categorical_column in categorical_columns:

        grouped_analysis[categorical_column] = {}

        for numerical_column in numerical_columns:

            grouped = (
                df.groupby(
                    categorical_column,
                    observed=True
                )[numerical_column]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )

            grouped_analysis[categorical_column][
                numerical_column
            ] = {
                str(key): float(value)
                for key, value in grouped.items()
            }

    result["grouped_analysis"] = grouped_analysis

    # --------------------------------
    # Date vs Numerical Trends
    # --------------------------------

    trends = {}

    for date_column in date_columns:

        trends[date_column] = {}

        for numerical_column in numerical_columns:

            trend = (
                df.groupby(
                    date_column,
                    observed=True
                )[numerical_column]
                .sum()
                .sort_index()
            )

            # Prevent extremely large responses
            if len(trend) > 100:

                trend = trend.tail(100)

            trends[date_column][numerical_column] = {
                str(date): float(value)
                for date, value in trend.items()
            }

    result["trends"] = trends

    return result