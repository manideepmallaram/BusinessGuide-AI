import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



# ============================================================
# CONSTANTS
# ============================================================

RANDOM_STATE = 42
N_ESTIMATORS = 50

# Maximum number of unique categorical values allowed
# before a column is considered too high-cardinality.
HIGH_CARDINALITY_LIMIT = 500

# A column with this ratio of unique values is suspicious
# when its name also indicates an identifier.
HIGH_CARDINALITY_RATIO = 0.80


# ============================================================
# IDENTIFIER DETECTION
# ============================================================

def is_identifier_column(column_name, series):
    """
    Detect columns that are identifiers rather than meaningful
    business prediction targets.
    """

    name = str(column_name).strip().lower()

    identifier_names = {
        "id",
        "key",
        "customer_id",
        "customer_key",
        "customerid",
        "customerkey",
        "user_id",
        "user_key",
        "userid",
        "userkey",
        "product_id",
        "product_key",
        "productid",
        "productkey",
        "order_id",
        "order_key",
        "orderid",
        "orderkey",
        "transaction_id",
        "transaction_key",
        "transactionid",
        "transactionkey"
    }

    if name in identifier_names:
        return True

    if name.endswith("_id") or name.endswith("_key"):
        return True

    if name.startswith("id_") or name.startswith("key_"):
        return True

    # Very high-cardinality text columns are usually identifiers.
    if pd.api.types.is_string_dtype(series):
        unique_count = series.nunique(dropna=True)
        total_count = max(len(series), 1)
        unique_ratio = unique_count / total_count

        # A column is only considered an identifier by ratio if there are enough
        # samples and unique entries to avoid false positives on small/sample datasets
        if total_count >= 20 and unique_count > 15 and unique_ratio >= 0.90:
            return True

    return False


def detect_id_like_columns(df: pd.DataFrame):
    """
    Detect columns that are likely identifiers.

    Examples:
        customer_id
        customer_key
        order_id
        product_id
        cst_id
        cst_key
    """

    id_like_columns = []

    if len(df) == 0:
        return id_like_columns

    for column in df.columns:

        if is_identifier_column(
            column,
            df[column]
        ):
            id_like_columns.append(column)
            continue

        unique_count = df[column].nunique(
            dropna=True
        )

        unique_ratio = unique_count / len(df)

        # Only treat a column as ID-like by cardinality when
        # it is clearly almost unique and is numeric.
        if (
            pd.api.types.is_numeric_dtype(df[column])
            and unique_count > 1000
            and unique_ratio >= HIGH_CARDINALITY_RATIO
        ):
            id_like_columns.append(column)

    return id_like_columns


# ============================================================
# TARGET CANDIDATES
# ============================================================

def detect_target_candidates(df: pd.DataFrame):

    candidates = []

    for column in df.columns:

        series = df[column]

        # ----------------------------------------------------
        # Ignore identifier columns
        # ----------------------------------------------------

        if is_identifier_column(
            column,
            series
        ):
            continue

        # ----------------------------------------------------
        # Ignore datetime columns as direct targets
        # ----------------------------------------------------

        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        # ----------------------------------------------------
        # Ignore columns with too many missing values
        # ----------------------------------------------------

        missing_ratio = series.isna().mean()

        if missing_ratio > 0.30:
            continue

        # ----------------------------------------------------
        # Ignore columns with only one unique value
        # ----------------------------------------------------

        unique_count = series.nunique(
            dropna=True
        )

        if unique_count <= 1:
            continue

        # ----------------------------------------------------
        # Numeric target
        # ----------------------------------------------------

        if pd.api.types.is_numeric_dtype(series):

            candidates.append({
                "column": column,
                "problem_type": "regression",
                "data_type": str(series.dtype),
                "unique_values": int(unique_count),
                "missing_ratio": float(missing_ratio)
            })

        # ----------------------------------------------------
        # Categorical target
        # ----------------------------------------------------

        elif (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):

            # Avoid extremely high-cardinality text columns.
            if unique_count > HIGH_CARDINALITY_LIMIT:
                continue

            candidates.append({
                "column": column,
                "problem_type": "classification",
                "data_type": str(series.dtype),
                "unique_values": int(unique_count),
                "missing_ratio": float(missing_ratio)
            })

    return candidates


# ============================================================
# SCORE TARGET CANDIDATES
# ============================================================

def score_target_candidates(df: pd.DataFrame):

    candidates = detect_target_candidates(df)

    scored_candidates = []

    total_rows = max(len(df), 1)

    for candidate in candidates:

        column = candidate["column"]

        unique_values = df[column].nunique(
            dropna=True
        )

        missing_ratio = df[column].isna().mean()

        unique_ratio = unique_values / total_rows

        score = 0

        # ----------------------------------------------------
        # Prefer reasonable cardinality
        # ----------------------------------------------------

        if 2 <= unique_values <= 20:
            score += 4

        elif 21 <= unique_values <= 100:
            score += 3

        elif unique_values <= 500:
            score += 2

        # ----------------------------------------------------
        # Penalize extremely high cardinality
        # ----------------------------------------------------

        if unique_ratio > 0.80:
            score -= 5

        elif unique_ratio > 0.50:
            score -= 3

        # ----------------------------------------------------
        # Penalize missing data
        # ----------------------------------------------------

        if missing_ratio == 0:
            score += 2

        elif missing_ratio < 0.10:
            score += 1

        scored_candidates.append({
            "column": column,
            "problem_type": candidate["problem_type"],
            "unique_values": int(unique_values),
            "missing_ratio": float(missing_ratio),
            "score": score
        })

    scored_candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored_candidates


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    df: pd.DataFrame,
    target_column: str
):

    df = df.copy()

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset"
        )

    # Never allow an identifier to become a target.
    if is_identifier_column(
        target_column,
        df[target_column]
    ):
        raise ValueError(
            f"Target column '{target_column}' is an identifier "
            "and cannot be used for prediction."
        )

    # --------------------------------------------------------
    # Remove rows where target is missing
    # --------------------------------------------------------

    df = df.dropna(
        subset=[target_column]
    )

    if df.empty:
        raise ValueError(
            "No rows remain after removing missing target values."
        )

    # --------------------------------------------------------
    # Separate target
    # --------------------------------------------------------

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    # --------------------------------------------------------
    # Remove ID-like columns
    # --------------------------------------------------------

    id_like_columns = detect_id_like_columns(X)

    if id_like_columns:
        X = X.drop(
            columns=id_like_columns,
            errors="ignore"
        )

    # --------------------------------------------------------
    # Handle datetime columns
    # --------------------------------------------------------

    datetime_columns = []

    for column in X.columns:

        if pd.api.types.is_datetime64_any_dtype(
            X[column]
        ):
            datetime_columns.append(column)

    for column in datetime_columns:

        X[f"{column}_year"] = X[column].dt.year
        X[f"{column}_month"] = X[column].dt.month
        X[f"{column}_day"] = X[column].dt.day

    # Remove original datetime columns.
    if datetime_columns:
        X = X.drop(
            columns=datetime_columns
        )

    # --------------------------------------------------------
    # Detect numerical columns
    # --------------------------------------------------------

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # --------------------------------------------------------
    # Detect categorical columns
    # --------------------------------------------------------

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool", "string", "str"]
    ).columns.tolist()

    # --------------------------------------------------------
    # Numerical pipeline
    # --------------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Column transformer
    # --------------------------------------------------------

    transformers = []

    if numerical_columns:
        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    if not transformers:
        raise ValueError(
            "No usable feature columns are available for training."
        )

    # --------------------------------------------------------
    # Build preprocessor
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return (
        X,
        y,
        preprocessor,
        numerical_columns,
        categorical_columns,
        id_like_columns
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str
):

    (
        X,
        y,
        preprocessor,
        numerical_columns,
        categorical_columns,
        id_like_columns
    ) = prepare_features(
        df,
        target_column
    )

    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    if problem_type == "classification":

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif problem_type == "regression":

        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    else:

        raise ValueError(
            "problem_type must be either "
            "'classification' or 'regression'"
        )

    # --------------------------------------------------------
    # Complete ML pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    pipeline.fit(
        X,
        y
    )

    # --------------------------------------------------------
    # Actual features used for training
    # --------------------------------------------------------

    feature_columns = X.columns.tolist()

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "pipeline": pipeline,
        "target_column": target_column,
        "problem_type": problem_type,
        "feature_columns": feature_columns,

        # Names expected by upload.py
        "numerical_features": numerical_columns,
        "categorical_features": categorical_columns,

        # Internal/useful aliases
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,

        "excluded_id_columns": id_like_columns,
        "training_rows": len(X)
    }


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str
):

    (
        X,
        y,
        preprocessor,
        numerical_columns,
        categorical_columns,
        id_like_columns
    ) = prepare_features(
        df,
        target_column
    )

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    if problem_type == "classification":

        # Stratification requires at least two samples in
        # every class. Fall back to a normal split if that
        # condition is not satisfied.
        class_counts = y.value_counts()

        if len(class_counts) > 1 and class_counts.min() >= 2:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=RANDOM_STATE,
                stratify=y
            )

        else:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=RANDOM_STATE
            )

    elif problem_type == "regression":

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE
        )

    else:

        raise ValueError(
            "problem_type must be either "
            "'classification' or 'regression'"
        )

    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    if problem_type == "classification":

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    else:

        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    if problem_type == "classification":

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        metrics_dict = {
            "Accuracy": float(accuracy)
        }

        try:
            f1 = float(f1_score(y_test, predictions, average="weighted", zero_division=0))
            prec = float(precision_score(y_test, predictions, average="weighted", zero_division=0))
            rec = float(recall_score(y_test, predictions, average="weighted", zero_division=0))
            metrics_dict["F1"] = f1
            metrics_dict["Precision"] = prec
            metrics_dict["Recall"] = rec
        except Exception:
            f1, prec, rec = None, None, None

        return {
            "problem_type": problem_type,
            "accuracy": float(accuracy),
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
            "metrics": metrics_dict,
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "target_column": target_column,
            "excluded_id_columns": id_like_columns
        }

    # --------------------------------------------------------
    # Regression metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    return {
        "problem_type": problem_type,
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2_score": float(r2),
        "metrics": {
            "R2": float(r2),
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse)
        },
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "target_column": target_column,
        "excluded_id_columns": id_like_columns
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str
):

    (
        X,
        y,
        preprocessor,
        numerical_columns,
        categorical_columns,
        id_like_columns
    ) = prepare_features(
        df,
        target_column
    )

    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    if problem_type == "classification":

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif problem_type == "regression":

        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    else:

        raise ValueError(
            "problem_type must be either "
            "'classification' or 'regression'"
        )

    # --------------------------------------------------------
    # Fit preprocessor
    # --------------------------------------------------------

    X_transformed = preprocessor.fit_transform(
        X
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model.fit(
        X_transformed,
        y
    )

    # --------------------------------------------------------
    # Get transformed feature names
    # --------------------------------------------------------

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

    except Exception:

        feature_names = [
            f"feature_{i}"
            for i in range(
                len(model.feature_importances_)
            )
        ]

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importances = model.feature_importances_

    importance_data = []

    for feature_name, importance in zip(
        feature_names,
        importances
    ):

        importance_data.append({
            "feature": str(feature_name),
            "importance": float(importance)
        })

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    importance_data.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    return {
        "target_column": target_column,
        "problem_type": problem_type,
        "feature_importance": importance_data,
        "excluded_id_columns": id_like_columns
    }
