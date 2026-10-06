import logging
import pandas as pd

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from profiler import profile_dataset
from preprocessing import preprocess_dataset
from analytics import analyze_dataset

from ml_engine import (
    detect_target_candidates,
    score_target_candidates,
    train_model,
    evaluate_model,
    get_feature_importance
)

from ai_engine import select_business_target, chat_with_assistant

logger = logging.getLogger("businessguide")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


router = APIRouter()


# ============================================================
# GLOBAL STATE
# ============================================================

latest_dataset = None
processed_dataset = None
preprocessing_cache = None
analytics_cache = None
target_candidates_cache = None
ranked_targets_cache = None
ai_target_cache = None
evaluation_cache = None
feature_importance_cache = None
prediction_schema_cache = None
latest_prediction = None

trained_model = None
trained_target = None
trained_problem_type = None
trained_feature_columns = None


# ============================================================
# CACHED ML/DATA HELPERS
# ============================================================

def require_dataset():
    if latest_dataset is None:
        raise HTTPException(
            status_code=404,
            detail="No dataset has been uploaded yet"
        )


def get_processed_dataset():
    global processed_dataset
    global preprocessing_cache

    require_dataset()

    if processed_dataset is None:
        preprocessing_cache = preprocess_dataset(latest_dataset)
        processed_dataset = preprocessing_cache["data"]
        logger.info(
            "[PIPELINE - PREPROCESSING] Preprocessed dataset | Input shape: (%d, %d) -> Cleaned shape: (%d, %d) | Removed duplicates: %d | Date columns: %s",
            len(latest_dataset),
            len(latest_dataset.columns),
            len(processed_dataset),
            len(processed_dataset.columns),
            preprocessing_cache.get("duplicates_removed", 0),
            preprocessing_cache.get("date_columns", [])
        )

    return processed_dataset


def get_analytics_cached():
    global analytics_cache

    processed_df = get_processed_dataset()

    if analytics_cache is None:
        analytics_cache = analyze_dataset(processed_df)
        logger.info(
            "[PIPELINE - ANALYTICS] Generated dataset analytics | Numerical columns: %d | Categorical columns: %d",
            len(analytics_cache.get("numerical_statistics", {})),
            len(analytics_cache.get("categorical_analysis", {}))
        )

    return analytics_cache


def get_candidates_cached():
    global target_candidates_cache

    processed_df = get_processed_dataset()

    if target_candidates_cache is None:
        target_candidates_cache = detect_target_candidates(processed_df)
        logger.info(
            "[PIPELINE - TARGET DISCOVERY] Discovered %d candidate targets: %s",
            len(target_candidates_cache),
            [c.get("column") for c in target_candidates_cache]
        )

    return target_candidates_cache


def get_ranked_targets_cached():
    global ranked_targets_cache

    processed_df = get_processed_dataset()

    if ranked_targets_cache is None:
        ranked_targets_cache = score_target_candidates(processed_df)
        logger.info(
            "[PIPELINE - TARGET DISCOVERY] Ranked %d candidate targets: %s",
            len(ranked_targets_cache),
            [f"{r.get('column')} (score: {r.get('score')})" for r in ranked_targets_cache]
        )

    return ranked_targets_cache


def get_ai_target_cached():
    global ai_target_cache

    analytics_result = get_analytics_cached()
    target_candidates = get_candidates_cached()

    if not target_candidates:
        logger.warning("[PIPELINE - AI TARGET RECOMMENDATION] No target candidates available to recommend from.")
        raise HTTPException(
            status_code=400,
            detail="No suitable ML target candidates found."
        )

    if ai_target_cache is None:
        ai_target_cache = select_business_target(
            analytics_result["dataset_overview"],
            analytics_result["numerical_statistics"],
            analytics_result["categorical_analysis"],
            target_candidates
        )
        logger.info(
            "[PIPELINE - AI TARGET RECOMMENDATION] Target selected via Groq | Recommended Target: '%s' | Problem Type: '%s' | Confidence: %.2f | Rationale: %s",
            ai_target_cache.get("recommended_target"),
            ai_target_cache.get("problem_type"),
            float(ai_target_cache.get("confidence", 0)),
            ai_target_cache.get("reason", "")[:120]
        )

    return ai_target_cache


# ============================================================
# DATASET UPLOAD
# ============================================================

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...)
):

    global latest_dataset

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected"
        )

    filename = file.filename.lower()

    if not (
        filename.endswith(".csv")
        or filename.endswith(".xlsx")
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only CSV and Excel (.xlsx) "
                "files are supported"
            )
        )

    try:

        if filename.endswith(".csv"):

            df = pd.read_csv(file.file)

        else:

            df = pd.read_excel(file.file)

        if df.empty:

            raise HTTPException(
                status_code=400,
                detail="Uploaded dataset is empty"
            )

        latest_dataset = df

        # A new dataset invalidates the previous model
        global processed_dataset
        global preprocessing_cache
        global analytics_cache
        global target_candidates_cache
        global ranked_targets_cache
        global ai_target_cache
        global evaluation_cache
        global feature_importance_cache
        global prediction_schema_cache
        global latest_prediction
        global trained_model
        global trained_target
        global trained_problem_type
        global trained_feature_columns

        processed_dataset = None
        preprocessing_cache = None
        analytics_cache = None
        target_candidates_cache = None
        ranked_targets_cache = None
        ai_target_cache = None
        evaluation_cache = None
        feature_importance_cache = None
        prediction_schema_cache = None
        latest_prediction = None
        trained_model = None
        trained_target = None
        trained_problem_type = None
        trained_feature_columns = None

        logger.info(
            "[PIPELINE - UPLOAD] Successfully uploaded '%s' | Rows: %d | Columns: %d | Columns: %s",
            file.filename,
            len(df),
            len(df.columns),
            df.columns.tolist()
        )
        logger.info("[PIPELINE - UPLOAD] Reset all downstream pipeline state and caches.")

        return {
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist()
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not read the dataset: {str(e)}"
            )
        )


# ============================================================
# PROFILE
# ============================================================

@router.get("/profile")
async def get_profile():

    if latest_dataset is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "No dataset has been uploaded yet"
            )
        )

    logger.info(
        "[PIPELINE - PROFILE] Profiling dataset with %d rows and %d columns",
        len(latest_dataset),
        len(latest_dataset.columns)
    )

    return profile_dataset(
        latest_dataset
    )


# ============================================================
# PREPROCESSING
# ============================================================

@router.get("/preprocess")
async def preprocess():
    require_dataset()

    processed_df = get_processed_dataset()

    result = dict(preprocessing_cache)
    result["data"] = processed_df.head(100).to_dict(orient="records")
    result["preview_rows"] = min(100, len(processed_df))
    result["data_total_rows"] = len(processed_df)

    return result


# ============================================================
# ANALYTICS
# ============================================================

@router.get("/analytics")
async def get_analytics():
    require_dataset()
    return get_analytics_cached()


# ============================================================
# ML TARGET CANDIDATES
# ============================================================

@router.get("/ml/candidates")
async def get_ml_candidates():
    require_dataset()
    return {"target_candidates": get_candidates_cached()}


# ============================================================
# RANKED TARGETS
# ============================================================

@router.get("/ml/targets")
async def get_ranked_targets():
    require_dataset()
    return {"ranked_targets": get_ranked_targets_cached()}


# ============================================================
# AI TARGET SELECTION
# ============================================================

@router.get("/ml/ai-target")
async def get_ai_target():
    require_dataset()
    return {"ai_target_selection": get_ai_target_cached()}


# ============================================================
# ML TRAINING
# ============================================================

@router.get("/ml/train")
async def train_ml_model():

    global trained_model
    global trained_target
    global trained_problem_type
    global trained_feature_columns
    global evaluation_cache
    global feature_importance_cache
    global prediction_schema_cache

    if latest_dataset is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "No dataset has been uploaded yet"
            )
        )

    try:

        # ----------------------------------------------------
        # Reuse cached dataset analysis and AI decision
        # ----------------------------------------------------

        processed_df = get_processed_dataset()
        analytics_result = get_analytics_cached()
        target_candidates = get_candidates_cached()

        if not target_candidates:
            raise HTTPException(
                status_code=400,
                detail="No suitable ML target candidates found."
            )

        ai_result = get_ai_target_cached()

        target_column = ai_result["recommended_target"]
        problem_type = ai_result["problem_type"]

        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        training_result = train_model(
            processed_df,
            target_column,
            problem_type
        )

        # ----------------------------------------------------
        # Store trained model
        # ----------------------------------------------------

        trained_model = training_result[
            "pipeline"
        ]

        trained_target = target_column

        trained_problem_type = problem_type

        trained_feature_columns = training_result[
            "feature_columns"
        ]

        evaluation_cache = None
        feature_importance_cache = None
        prediction_schema_cache = None

        logger.info(
            "[PIPELINE - MODEL TRAINING] Training Random Forest pipeline | Target: '%s' | Problem Type: '%s' | Features (%d): %s | Train rows: %d | Test rows: %d",
            target_column,
            problem_type,
            len(trained_feature_columns),
            trained_feature_columns,
            training_result["training_rows"],
            len(processed_df) - training_result["training_rows"]
        )

        # ----------------------------------------------------
        # Return training information
        # ----------------------------------------------------

        return {
            "ai_decision": {
                "target": target_column,
                "problem_type": problem_type,
                "confidence": ai_result[
                    "confidence"
                ],
                "reason": ai_result[
                    "reason"
                ]
            },

            "training": {
                "training_rows": training_result[
                    "training_rows"
                ],

                "feature_columns": training_result[
                    "feature_columns"
                ],

                "numerical_features": training_result[
                    "numerical_features"
                ],

                "categorical_features": training_result[
                    "categorical_features"
                ],

                "model": "Random Forest",

                "model_status": (
                    "trained_and_stored"
                )
            }
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Training failed: "
                f"{type(e).__name__}: {str(e)}"
            )
        )


# ============================================================
# ML EVALUATION
# ============================================================

@router.get("/ml/evaluate")
async def evaluate_ml_model():
    global evaluation_cache

    require_dataset()

    if trained_model is None:
        raise HTTPException(
            status_code=400,
            detail="Train the ML model first using /ml/train."
        )

    try:
        if evaluation_cache is None:
            processed_df = get_processed_dataset()
            evaluation_cache = evaluate_model(
                processed_df,
                trained_target,
                trained_problem_type
            )
            logger.info(
                "[PIPELINE - MODEL EVALUATION] Evaluated model on test set | Target: '%s' (%s) | Metrics: %s",
                trained_target,
                trained_problem_type,
                evaluation_cache.get("metrics", evaluation_cache)
            )

        return {
            "ai_decision": get_ai_target_cached(),
            "evaluation": evaluation_cache
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {type(e).__name__}: {str(e)}"
        )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

@router.get("/ml/feature-importance")
async def feature_importance():
    global feature_importance_cache

    require_dataset()

    if trained_model is None:
        raise HTTPException(
            status_code=400,
            detail="Train the ML model first using /ml/train."
        )

    try:
        if feature_importance_cache is None:
            model = trained_model.named_steps["model"]
            preprocessor = trained_model.named_steps["preprocessor"]
            feature_names = preprocessor.get_feature_names_out()
            importances = model.feature_importances_

            importance_data = [
                {
                    "feature": name,
                    "importance": float(importance)
                }
                for name, importance in zip(feature_names, importances)
            ]

            importance_data.sort(
                key=lambda x: x["importance"],
                reverse=True
            )

            total_importance = sum(
                item["importance"] for item in importance_data
            )

            for item in importance_data:
                item["percentage"] = round(
                    (item["importance"] / total_importance) * 100, 2
                ) if total_importance > 0 else 0.0

            feature_importance_cache = {
                "target": trained_target,
                "problem_type": trained_problem_type,
                "features": importance_data
            }

            logger.info(
                "[PIPELINE - FEATURE IMPORTANCE] Computed feature importances | Top drivers: %s",
                [(f["feature"], f"{f.get('percentage')}%") for f in importance_data[:5]]
            )

        return {
            "ai_decision": get_ai_target_cached(),
            "feature_importance": feature_importance_cache
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature importance failed: {type(e).__name__}: {str(e)}"
        )


# ============================================================
# PREDICTION SCHEMA
# ============================================================

@router.get("/ml/prediction-schema")
async def prediction_schema():
    global prediction_schema_cache

    require_dataset()

    if trained_model is None:
        raise HTTPException(
            status_code=400,
            detail="Train the ML model first using /ml/train."
        )

    try:
        if prediction_schema_cache is None:
            processed_df = get_processed_dataset()
            target_column = trained_target
            ai_result = get_ai_target_cached()
            prediction_features = []

            for column in processed_df.columns:
                if column == target_column:
                    continue

                series = processed_df[column]

                if pd.api.types.is_numeric_dtype(series):
                    prediction_features.append({
                        "column": column,
                        "type": "numerical",
                        "required": True,
                        "minimum": float(series.min()),
                        "maximum": float(series.max()),
                        "mean": float(series.mean())
                    })

                elif pd.api.types.is_datetime64_any_dtype(series):
                    prediction_features.append({
                        "column": column,
                        "type": "date",
                        "required": True
                    })

                else:
                    values = (
                        series.dropna()
                        .astype(str)
                        .value_counts()
                        .head(100)
                        .index
                        .tolist()
                    )
                    prediction_features.append({
                        "column": column,
                        "type": "categorical",
                        "required": True,
                        "allowed_values": sorted(values)
                    })

            prediction_schema_cache = {
                "target": target_column,
                "problem_type": ai_result["problem_type"],
                "confidence": ai_result["confidence"],
                "features": prediction_features
            }

        return prediction_schema_cache

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction schema failed: {type(e).__name__}: {str(e)}"
        )


# ============================================================
# ML PREDICTION
# ============================================================

@router.post("/ml/predict")
async def predict_ml(data: dict):

    if trained_model is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "No trained model is available. "
                "Run /ml/train first."
            )
        )

    if not isinstance(data, dict):

        raise HTTPException(
            status_code=400,
            detail=(
                "Prediction input must be "
                "a JSON object."
            )
        )

    input_data = data.get("data")

    if input_data is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "Request must contain "
                "a 'data' object."
            )
        )

    if not isinstance(input_data, dict):

        raise HTTPException(
            status_code=400,
            detail=(
                "'data' must be a JSON object."
            )
        )

    try:

        # ----------------------------------------------------
        # Convert input to DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [input_data]
        )

        # ----------------------------------------------------
        # Validate required feature columns
        # ----------------------------------------------------

        missing_columns = [
            column
            for column in trained_feature_columns
            if column not in input_df.columns
        ]

        # ----------------------------------------------------
        # Date handling
        #
        # Training converts:
        #
        # Date
        #   ↓
        # Date_year
        # Date_month
        # Date_day
        #
        # Prediction must perform the same transformation.
        # ----------------------------------------------------

        if "Date" in input_df.columns:

            date_value = pd.to_datetime(
                input_df["Date"],
                errors="coerce"
            )

            if date_value.isna().any():

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Invalid date value "
                        "provided for 'Date'."
                    )
                )

            input_df["Date_year"] = (
                date_value.dt.year
            )

            input_df["Date_month"] = (
                date_value.dt.month
            )

            input_df["Date_day"] = (
                date_value.dt.day
            )

            input_df = input_df.drop(
                columns=["Date"]
            )

            # Recalculate missing columns after
            # date transformation
            missing_columns = [
                column
                for column in trained_feature_columns
                if column not in input_df.columns
            ]

        if missing_columns:

            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        "Required prediction "
                        "features are missing."
                    ),
                    "missing_columns": missing_columns
                }
            )

        # ----------------------------------------------------
        # Keep exactly the features used during training
        # ----------------------------------------------------

        input_df = input_df[
            trained_feature_columns
        ]

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        prediction = trained_model.predict(
            input_df
        )

        prediction_value = prediction[0]

        # Convert NumPy scalar to native Python value
        if hasattr(
            prediction_value,
            "item"
        ):

            prediction_value = (
                prediction_value.item()
            )

        result = {
            "target": trained_target,
            "problem_type": trained_problem_type,
            "prediction": prediction_value,
            "input_features": input_data
        }

        global latest_prediction
        latest_prediction = result

        logger.info(
            "[PIPELINE - PREDICTION] Executed scenario prediction | Target: '%s' | Predicted Value: %s | Inputs: %s",
            trained_target,
            prediction_value,
            input_data
        )

        return result

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Prediction failed: "
                f"{type(e).__name__}: {str(e)}"
            )
        )
# ============================================================
# BUSINESS INSIGHTS
# ============================================================

@router.get("/business/insights")
async def get_business_insights():

    global evaluation_cache
    global feature_importance_cache

    if latest_dataset is None:
        raise HTTPException(
            status_code=404,
            detail="No dataset has been uploaded yet"
        )

    if trained_model is None:
        raise HTTPException(
            status_code=400,
            detail="Train the ML model first using /ml/train."
        )

    if trained_target is None:
        raise HTTPException(
            status_code=400,
            detail="No trained target is available."
        )

    try:

        # ----------------------------------------------------
        # Preprocess dataset
        # ----------------------------------------------------

        preprocessing_result = preprocess_dataset(
            latest_dataset
        )

        processed_df = preprocessing_result["data"]

        # ----------------------------------------------------
        # Dataset analytics
        # ----------------------------------------------------

        analytics_result = analyze_dataset(
            processed_df
        )

        # ----------------------------------------------------
        # Model evaluation
        #
        # This is LOCAL ML computation.
        # No Groq call happens here.
        # ----------------------------------------------------

        evaluation_result = evaluation_cache

        if evaluation_result is None:
            evaluation_result = evaluate_model(
                processed_df,
                trained_target,
                trained_problem_type
            )
            evaluation_cache = evaluation_result

        # ----------------------------------------------------
        # Feature importance
        #
        # Also LOCAL.
        # ----------------------------------------------------

        importance_result = feature_importance_cache

        if importance_result is None:
            model = trained_model.named_steps["model"]
            preprocessor = trained_model.named_steps["preprocessor"]
            feature_names = preprocessor.get_feature_names_out()
            importances = model.feature_importances_
            importance_data = [
                {"feature": name, "importance": float(importance)}
                for name, importance in zip(feature_names, importances)
            ]
            importance_data.sort(key=lambda x: x["importance"], reverse=True)
            total_importance = sum(x["importance"] for x in importance_data)
            for item in importance_data:
                item["percentage"] = round(
                    (item["importance"] / total_importance) * 100, 2
                ) if total_importance > 0 else 0.0
            importance_result = {
                "target": trained_target,
                "problem_type": trained_problem_type,
                "features": importance_data
            }
            feature_importance_cache = importance_result

        # ----------------------------------------------------
        # Extract top feature
        # ----------------------------------------------------

        features = importance_result.get(
            "features",
            []
        )

        top_features = sorted(
            features,
            key=lambda x: x.get(
                "importance",
                0
            ),
            reverse=True
        )

        key_drivers = top_features[:5]

        # ----------------------------------------------------
        # Target statistics
        #
        # Dynamically obtained from uploaded dataset.
        # No "Profit" or "Revenue" hardcoding.
        # ----------------------------------------------------

        target_statistics = None

        if trained_target in processed_df.columns:

            target_series = processed_df[
                trained_target
            ]

            if pd.api.types.is_numeric_dtype(
                target_series
            ):

                target_statistics = {
                    "sum": float(
                        target_series.sum()
                    ),
                    "mean": float(
                        target_series.mean()
                    ),
                    "median": float(
                        target_series.median()
                    ),
                    "minimum": float(
                        target_series.min()
                    ),
                    "maximum": float(
                        target_series.max()
                    )
                }

        # ----------------------------------------------------
        # Build local insights
        # ----------------------------------------------------

        insights = []

        if key_drivers:

            top_driver = key_drivers[0]

            insights.append({
                "type": "key_driver",
                "feature": top_driver[
                    "feature"
                ],
                "importance": top_driver[
                    "importance"
                ],
                "message": (
                    f"{top_driver['feature']} "
                    "is the strongest predictive "
                    "feature in the trained model."
                )
            })

        # ----------------------------------------------------
        # Model performance insight
        # ----------------------------------------------------

        metrics = evaluation_result.get(
            "metrics",
            {}
        )

        if trained_problem_type == "regression":

            r2 = metrics.get("R2")

            if r2 is not None:

                insights.append({
                    "type": "model_performance",
                    "metric": "R2",
                    "value": r2,
                    "message": (
                        f"The regression model explains "
                        f"approximately {r2 * 100:.2f}% "
                        "of the variance in the target "
                        "on the evaluation data."
                    )
                })

        elif trained_problem_type == "classification":

            accuracy = metrics.get(
                "Accuracy"
            )

            if accuracy is not None:

                insights.append({
                    "type": "model_performance",
                    "metric": "Accuracy",
                    "value": accuracy,
                    "message": (
                        f"The classification model "
                        f"achieved {accuracy * 100:.2f}% "
                        "accuracy on the evaluation data."
                    )
                })

        # ----------------------------------------------------
        # Dataset-level target insight
        # ----------------------------------------------------

        if target_statistics is not None:

            insights.append({
                "type": "target_statistics",
                "target": trained_target,
                "mean": target_statistics[
                    "mean"
                ],
                "minimum": target_statistics[
                    "minimum"
                ],
                "maximum": target_statistics[
                    "maximum"
                ],
                "message": (
                    f"{trained_target} ranges from "
                    f"{target_statistics['minimum']:.2f} "
                    f"to "
                    f"{target_statistics['maximum']:.2f} "
                    "in the uploaded dataset."
                )
            })

        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        logger.info(
            "[PIPELINE - BUSINESS INSIGHTS] Generated %d executive insights for target '%s' | Primary driver: %s",
            len(insights),
            trained_target,
            key_drivers[0]["feature"] if key_drivers else "None"
        )

        return {
            "target": trained_target,

            "problem_type": trained_problem_type,

            "dataset": {
                "rows": len(processed_df),
                "columns": len(
                    processed_df.columns
                )
            },

            "target_statistics": target_statistics,

            "model_evaluation": evaluation_result,

            "key_drivers": key_drivers,

            "insights": insights,

            "ai_status": (
                "AI interpretation not called yet"
            )
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Business insights failed: "
                f"{type(e).__name__}: {str(e)}"
            )
        )


# ============================================================
# AI ASSISTANT CHAT CONTEXT & ENDPOINT
# ============================================================

def build_chat_context() -> str:
    global latest_dataset, processed_dataset, preprocessing_cache, analytics_cache
    global target_candidates_cache, ranked_targets_cache, ai_target_cache
    global trained_model, trained_target, trained_problem_type, trained_feature_columns
    global evaluation_cache, feature_importance_cache, latest_prediction

    if latest_dataset is None:
        return (
            "NO DATASET STATUS: No dataset has been uploaded into BusinessGuide AI yet. "
            "Please instruct the user to upload a CSV or Excel dataset first."
        )

    lines = []
    lines.append("=== 1. CURRENT UPLOADED DATASET ===")
    lines.append(f"- Total Rows: {len(latest_dataset)}")
    lines.append(f"- Total Columns: {len(latest_dataset.columns)}")
    lines.append(f"- Column Names: {', '.join(str(c) for c in latest_dataset.columns)}")
    
    dtypes_dict = {str(col): str(dtype) for col, dtype in latest_dataset.dtypes.items()}
    lines.append(f"- Data Types: {dtypes_dict}")
    
    missing_dict = {
        str(col): int(cnt)
        for col, cnt in latest_dataset.isnull().sum().items()
        if cnt > 0
    }
    lines.append(f"- Missing Values: {missing_dict if missing_dict else 'None (Clean)'}")

    lines.append("\n=== 2. DATASET STATISTICS & ANALYTICS ===")
    numeric_cols = latest_dataset.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        lines.append(f"- Numeric Columns: {', '.join(numeric_cols)}")
        stats = {}
        for col in numeric_cols[:8]:
            s = latest_dataset[col].dropna()
            if not s.empty:
                stats[col] = {
                    "mean": round(float(s.mean()), 2),
                    "min": round(float(s.min()), 2),
                    "max": round(float(s.max()), 2),
                    "std": round(float(s.std()), 2) if len(s) > 1 else 0.0
                }
        lines.append(f"- Numerical Summary Stats: {stats}")
    
    cat_cols = [c for c in latest_dataset.columns if c not in numeric_cols]
    if cat_cols:
        cat_samples = {}
        for col in cat_cols[:6]:
            uniq = latest_dataset[col].dropna().unique()
            cat_samples[col] = {
                "unique_values_count": len(uniq),
                "samples": [str(v) for v in uniq[:5]]
            }
        lines.append(f"- Categorical Columns: {cat_samples}")

    lines.append("\n=== 3. TARGET CANDIDATES & RANKING ===")
    if target_candidates_cache is not None:
        if isinstance(target_candidates_cache, dict):
            candidates = target_candidates_cache.get("candidates", target_candidates_cache.get("target_candidates", []))
        elif isinstance(target_candidates_cache, list):
            candidates = target_candidates_cache
        else:
            candidates = []
        if candidates:
            cand_repr = [
                f"{c.get('column')} (Problem: {c.get('problem_type')}, Score: {c.get('score', 'N/A')})"
                for c in candidates if isinstance(c, dict)
            ]
            lines.append(f"- Detected Candidate Targets: {', '.join(cand_repr)}")
        else:
            lines.append("- Detected Candidate Targets: None found meeting criteria.")
    else:
        lines.append("- Target candidates have not been evaluated yet.")

    lines.append("\n=== 4. AI TARGET RECOMMENDATION ===")
    if ai_target_cache is not None and isinstance(ai_target_cache, dict):
        rec_info = ai_target_cache.get("ai_target_selection", ai_target_cache) if isinstance(ai_target_cache.get("ai_target_selection"), dict) else ai_target_cache
        lines.append(f"- Recommended Target: {rec_info.get('recommended_target')}")
        lines.append(f"- Recommended Problem Type: {rec_info.get('problem_type')}")
        lines.append(f"- Recommendation Confidence: {rec_info.get('confidence')}")
        lines.append(f"- Recommendation Rationale: {rec_info.get('reason')}")
    else:
        lines.append("- AI target recommendation has not been executed yet.")

    lines.append("\n=== 5. MACHINE LEARNING MODEL STATUS ===")
    if trained_model is not None:
        lines.append(f"- Active Trained Target: {trained_target}")
        lines.append(f"- Problem Type: {trained_problem_type}")
        model_name = "RandomForestRegressor" if trained_problem_type == "regression" else "RandomForestClassifier"
        lines.append(f"- Model Algorithm: {model_name} (Scikit-Learn Pipeline with median/most-frequent imputer & one-hot encoder)")
        lines.append(f"- Feature Columns ({len(trained_feature_columns)}): {', '.join(trained_feature_columns)}")
    else:
        lines.append("- ML model has NOT been trained yet. If asked about model accuracy or predictions, inform the user they must train the model first.")

    lines.append("\n=== 6. MODEL EVALUATION METRICS ===")
    if evaluation_cache is not None:
        if isinstance(evaluation_cache, dict):
            eval_dict = evaluation_cache.get("evaluation", evaluation_cache) if isinstance(evaluation_cache.get("evaluation"), dict) else evaluation_cache
            metrics = eval_dict.get("metrics", eval_dict)
        else:
            metrics = evaluation_cache
        lines.append(f"- Performance Metrics: {metrics}")
    else:
        lines.append("- Model evaluation has NOT been performed yet.")

    lines.append("\n=== 7. FEATURE IMPORTANCE & DRIVERS ===")
    if feature_importance_cache is not None:
        if isinstance(feature_importance_cache, dict):
            fi_dict = feature_importance_cache.get("feature_importance", feature_importance_cache) if isinstance(feature_importance_cache.get("feature_importance"), dict) else feature_importance_cache
            feats = fi_dict.get("features", []) if isinstance(fi_dict, dict) else []
        elif isinstance(feature_importance_cache, list):
            feats = feature_importance_cache
        else:
            feats = []
        if feats:
            top_feats = [
                f"{f.get('feature')}: {f.get('percentage', round(f.get('importance', 0)*100, 2))}% importance"
                for f in feats[:6] if isinstance(f, dict)
            ]
            lines.append(f"- Top Feature Drivers: {', '.join(top_feats)}")
        else:
            lines.append("- Feature importances list is empty.")
    else:
        lines.append("- Feature importance has NOT been calculated yet.")

    lines.append("\n=== 8. LATEST PREDICTION SIMULATOR RUN ===")
    if latest_prediction is not None:
        lines.append(f"- Simulated Target: {latest_prediction.get('target')}")
        lines.append(f"- Simulated Outcome / Prediction: {latest_prediction.get('prediction')}")
        lines.append(f"- Input Features used in simulation: {latest_prediction.get('input_features')}")
    else:
        lines.append("- No scenario prediction simulation has been executed yet.")

    return "\n".join(lines)


@router.post("/chat")
async def chat_assistant(data: dict):
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=400,
            detail="Request body must be a JSON object."
        )

    raw_message = data.get("message")
    if raw_message is None or not str(raw_message).strip():
        raise HTTPException(
            status_code=400,
            detail="Question/message cannot be empty."
        )

    user_message = str(raw_message).strip()
    history = data.get("history", [])
    if not isinstance(history, list):
        history = []

    try:
        context_str = build_chat_context()
        logger.info(
            "[PIPELINE - AI CHAT] User query: '%s' | History turns: %d | Context size: %d chars",
            user_message[:80],
            len(history),
            len(context_str)
        )
        ai_response = chat_with_assistant(
            user_message=user_message,
            history=history,
            context_str=context_str
        )
        logger.info(
            "[PIPELINE - AI CHAT] Groq response completed (%d chars)",
            len(ai_response)
        )
        return {
            "response": ai_response,
            "status": "success"
        }
    except Exception as e:
        logger.error("[PIPELINE - AI CHAT] Failed to generate response: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"AI Assistant error: {str(e)}"
        )


# ============================================================
# DIAGNOSTIC OBSERVABILITY ENDPOINTS
# ============================================================

@router.get("/debug/status")
async def debug_status():
    """
    Diagnostic status endpoint exposing pipeline readiness and current execution state.
    Does NOT expose secrets or sensitive raw dataset records.
    """
    has_dataset = latest_dataset is not None
    dims = (
        {"rows": len(latest_dataset), "columns": len(latest_dataset.columns)}
        if has_dataset else None
    )

    cand_count = 0
    if target_candidates_cache is not None:
        if isinstance(target_candidates_cache, list):
            cand_count = len(target_candidates_cache)
        elif isinstance(target_candidates_cache, dict):
            cand_count = len(target_candidates_cache.get("target_candidates", []))

    return {
        "status": "healthy",
        "dataset_loaded": has_dataset,
        "dataset_dimensions": dims,
        "preprocessing_completed": processed_dataset is not None,
        "analytics_completed": analytics_cache is not None,
        "target_discovery_completed": target_candidates_cache is not None,
        "candidate_count": cand_count,
        "ai_target_available": ai_target_cache is not None,
        "model_trained": trained_model is not None,
        "evaluation_available": evaluation_cache is not None,
        "feature_importance_available": feature_importance_cache is not None,
        "prediction_available": latest_prediction is not None,
        "chat_ready": True
    }


@router.get("/debug/pipeline")
async def debug_pipeline():
    """
    Diagnostic pipeline inspection endpoint returning the computed state of every stage.
    """
    return {
        "upload_stage": {
            "loaded": latest_dataset is not None,
            "rows": len(latest_dataset) if latest_dataset is not None else 0,
            "columns": len(latest_dataset.columns) if latest_dataset is not None else 0,
            "column_names": latest_dataset.columns.tolist() if latest_dataset is not None else []
        },
        "preprocessing_stage": {
            "completed": processed_dataset is not None,
            "cleaned_rows": len(processed_dataset) if processed_dataset is not None else 0,
            "cleaned_columns": len(processed_dataset.columns) if processed_dataset is not None else 0
        },
        "analytics_stage": {
            "completed": analytics_cache is not None,
            "numerical_columns_analyzed": len(analytics_cache.get("numerical_statistics", {})) if analytics_cache else 0,
            "categorical_columns_analyzed": len(analytics_cache.get("categorical_analysis", {})) if analytics_cache else 0
        },
        "target_discovery_stage": {
            "completed": target_candidates_cache is not None,
            "candidates": [
                {
                    "column": c.get("column"),
                    "problem_type": c.get("problem_type"),
                    "score": c.get("score")
                }
                for c in (target_candidates_cache if isinstance(target_candidates_cache, list) else [])
            ]
        },
        "ai_target_stage": {
            "completed": ai_target_cache is not None,
            "recommended_target": ai_target_cache.get("recommended_target") if ai_target_cache else None,
            "problem_type": ai_target_cache.get("problem_type") if ai_target_cache else None,
            "confidence": ai_target_cache.get("confidence") if ai_target_cache else None
        },
        "training_stage": {
            "model_trained": trained_model is not None,
            "target": trained_target,
            "problem_type": trained_problem_type,
            "algorithm": "RandomForest" if trained_model else None,
            "feature_columns": trained_feature_columns
        },
        "evaluation_stage": {
            "completed": evaluation_cache is not None,
            "metrics": evaluation_cache.get("metrics") if evaluation_cache else None
        },
        "feature_importance_stage": {
            "completed": feature_importance_cache is not None,
            "top_drivers": [
                {
                    "feature": f.get("feature"),
                    "percentage": f.get("percentage")
                }
                for f in (feature_importance_cache.get("features", [])[:5] if feature_importance_cache else [])
            ]
        },
        "prediction_stage": {
            "latest_prediction_available": latest_prediction is not None,
            "target": latest_prediction.get("target") if latest_prediction else None,
            "predicted_value": latest_prediction.get("prediction") if latest_prediction else None
        }
    }
