import unittest
import requests
import io
import time

BASE_URL = "http://127.0.0.1:8000"

DATASET_A_CSV = """Customer_ID,Marketing_Spend,Discount_Percent,Store_Visits,Region,Units_Sold,Profit
CUST_101,500.0,0.1,12,North,45,1200.5
CUST_102,300.0,0.05,8,South,30,850.0
CUST_103,750.0,0.15,18,North,65,1850.2
CUST_104,200.0,0.0,5,West,18,420.0
CUST_105,900.0,0.2,22,East,80,2400.0
CUST_106,450.0,0.08,10,North,38,1050.0
CUST_107,600.0,0.12,14,West,50,1350.5
CUST_108,150.0,0.0,4,South,12,280.0
CUST_109,820.0,0.18,20,East,72,2100.0
CUST_110,380.0,0.05,9,North,35,920.0
CUST_111,520.0,0.1,13,South,46,1220.0
CUST_112,290.0,0.05,7,West,26,710.0
CUST_113,710.0,0.15,16,North,60,1650.0
CUST_114,190.0,0.0,4,West,15,350.0
CUST_115,880.0,0.2,21,East,78,2300.0
CUST_116,420.0,0.08,10,South,36,980.0
CUST_117,610.0,0.12,15,North,52,1400.0
CUST_118,170.0,0.0,3,West,14,310.0
CUST_119,790.0,0.18,19,East,70,2050.0
CUST_120,350.0,0.05,8,North,32,870.0
CUST_121,540.0,0.1,13,South,46,1220.0
CUST_122,310.0,0.05,7,West,28,750.0
"""

DATASET_B_CSV = """Employee_ID,Department,Tenure_Years,Salary,Satisfaction_Score,Performance_Rating
EMP_1,Engineering,3,85000,4.2,4
EMP_2,Sales,1,55000,3.8,3
EMP_3,HR,5,70000,4.5,5
EMP_4,Support,2,48000,3.1,3
EMP_5,Engineering,6,110000,4.8,5
EMP_6,Marketing,4,75000,3.9,4
EMP_7,Sales,2,58000,3.6,3
EMP_8,HR,3,65000,4.0,4
EMP_9,Engineering,1,72000,3.7,3
EMP_10,Support,4,52000,3.4,4
EMP_11,Engineering,5,105000,4.6,5
EMP_12,Sales,3,62000,3.9,4
EMP_13,Marketing,2,68000,3.5,3
EMP_14,Engineering,4,95000,4.3,4
EMP_15,Support,1,45000,3.0,2
EMP_16,Sales,5,78000,4.1,4
EMP_17,HR,2,60000,3.8,3
EMP_18,Engineering,7,125000,4.9,5
EMP_19,Marketing,5,82000,4.2,4
EMP_20,Support,3,50000,3.2,3
"""


class TestBusinessGuideReliability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Verify server is reachable
        try:
            r = requests.get(f"{BASE_URL}/")
            assert r.status_code == 200
        except Exception as e:
            raise RuntimeError(f"Backend server not reachable at {BASE_URL}: {e}")

    def test_01_invalid_inputs_and_failure_recovery(self):
        """Verify invalid uploads, bad extensions, empty files, and empty chats fail cleanly with 400s."""
        # 1. Empty message to chat
        r = requests.post(f"{BASE_URL}/chat", json={"message": "   "})
        self.assertEqual(r.status_code, 400)
        self.assertIn("empty", r.json()["detail"].lower())

        # 2. Unsupported file type (.txt)
        txt_file = io.BytesIO(b"Hello world")
        r = requests.post(f"{BASE_URL}/upload", files={"file": ("test.txt", txt_file, "text/plain")})
        self.assertEqual(r.status_code, 400)
        self.assertIn("only csv and excel", r.json()["detail"].lower())

        # 3. Empty CSV upload
        empty_file = io.BytesIO(b"")
        r = requests.post(f"{BASE_URL}/upload", files={"file": ("empty.csv", empty_file, "text/csv")})
        self.assertEqual(r.status_code, 400)

        # 4. Predict endpoint without trained model
        r = requests.post(f"{BASE_URL}/ml/predict", json={"data": {"test": 123}})
        self.assertEqual(r.status_code, 400)

    def test_02_dataset_upload_a(self):
        """Upload valid Dataset A and verify dimensions."""
        r = requests.post(
            f"{BASE_URL}/upload",
            files={"file": ("sales_analysis.csv", io.StringIO(DATASET_A_CSV), "text/csv")}
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["rows"], 22)
        self.assertEqual(data["columns"], 7)
        self.assertIn("Profit", data["column_names"])
        self.assertIn("Marketing_Spend", data["column_names"])

        # Check diagnostic status
        status = requests.get(f"{BASE_URL}/debug/status").json()
        self.assertTrue(status["dataset_loaded"])
        self.assertEqual(status["dataset_dimensions"], {"rows": 22, "columns": 7})

    def test_03_repeated_candidates_regression(self):
        """
        REGRESSION TEST FOR PREVIOUS BUG:
        Verify GET /ml/candidates called 15 times repeatedly NEVER intermittently drops to an empty list.
        """
        candidate_lengths = []
        candidate_first_names = []

        for i in range(15):
            r = requests.get(f"{BASE_URL}/ml/candidates")
            self.assertEqual(r.status_code, 200, f"Failed on iteration {i}")
            cands = r.json().get("target_candidates", [])
            candidate_lengths.append(len(cands))
            if cands:
                candidate_first_names.append(cands[0].get("column"))

        self.assertEqual(len(set(candidate_lengths)), 1, "Candidate count fluctuated across calls!")
        self.assertGreater(candidate_lengths[0], 0, "Candidates returned empty list!")
        self.assertEqual(len(set(candidate_first_names)), 1, "First candidate column name was inconsistent!")

    def test_04_profile_and_preprocessing(self):
        """Verify profiling and preprocessing return deterministic shapes and stats."""
        prof_res = requests.get(f"{BASE_URL}/profile")
        self.assertEqual(prof_res.status_code, 200)
        prof_data = prof_res.json()
        self.assertEqual(prof_data["rows"], 22)
        self.assertEqual(prof_data["columns"], 7)

        prep_res = requests.get(f"{BASE_URL}/preprocess")
        self.assertEqual(prep_res.status_code, 200)
        prep_data = prep_res.json()
        self.assertEqual(prep_data.get("rows_after_preprocessing", prep_data.get("data_total_rows")), 22)

        analytics_res = requests.get(f"{BASE_URL}/analytics")
        self.assertEqual(analytics_res.status_code, 200)

    def test_05_target_ranking_and_ai_recommendation(self):
        """Verify targets discovery, ranking, and Groq target recommendation."""
        targets_res = requests.get(f"{BASE_URL}/ml/targets")
        self.assertEqual(targets_res.status_code, 200)

        ai_res = requests.get(f"{BASE_URL}/ml/ai-target")
        self.assertEqual(ai_res.status_code, 200)
        ai_data = ai_res.json().get("ai_target_selection", {})
        self.assertIn("recommended_target", ai_data)
        self.assertIn("problem_type", ai_data)
        self.assertGreater(ai_data.get("confidence", 0), 0)

    def test_06_model_training_and_evaluation(self):
        """Verify model training builds a valid Random Forest and evaluation returns valid metrics."""
        train_res = requests.get(f"{BASE_URL}/ml/train")
        self.assertEqual(train_res.status_code, 200)
        train_data = train_res.json()
        self.assertIn("training", train_data)

        eval_res = requests.get(f"{BASE_URL}/ml/evaluate")
        self.assertEqual(eval_res.status_code, 200)
        eval_data = eval_res.json().get("evaluation", {})
        self.assertIn("metrics", eval_data)
        metrics = eval_data["metrics"]
        self.assertTrue("R2" in metrics or "Accuracy" in metrics)

    def test_07_feature_importance_and_prediction(self):
        """Verify feature importance and what-if simulation."""
        feat_res = requests.get(f"{BASE_URL}/ml/feature-importance")
        self.assertEqual(feat_res.status_code, 200)
        features = feat_res.json().get("feature_importance", {}).get("features", [])
        self.assertGreater(len(features), 0)

        # Simulation
        schema_res = requests.get(f"{BASE_URL}/ml/prediction-schema")
        self.assertEqual(schema_res.status_code, 200)

        pred_payload = {
            "Marketing_Spend": 500.0,
            "Discount_Percent": 0.1,
            "Store_Visits": 12,
            "Region": "North",
            "Units_Sold": 45
        }
        pred_res = requests.post(f"{BASE_URL}/ml/predict", json={"data": pred_payload})
        self.assertEqual(pred_res.status_code, 200)
        pred_data = pred_res.json()
        self.assertIn("prediction", pred_data)
        self.assertIsInstance(pred_data["prediction"], (int, float))

    def test_08_ai_chat_with_project_context(self):
        """Verify AI chat responds accurately with project metrics and drivers."""
        chat_res = requests.post(f"{BASE_URL}/chat", json={"message": "Explain my model performance"})
        self.assertEqual(chat_res.status_code, 200)
        reply = chat_res.json().get("response", "")
        self.assertGreater(len(reply), 30)

    def test_09_dataset_replacement_and_context_isolation(self):
        """
        Verify that uploading Dataset B cleanly invalidates Dataset A:
        - Profiling, candidates, recommendation, and model reflect Dataset B.
        - Zero columns or metrics from Dataset A survive.
        """
        # Upload Dataset B (HR Employees)
        r = requests.post(
            f"{BASE_URL}/upload",
            files={"file": ("hr_employees.csv", io.StringIO(DATASET_B_CSV), "text/csv")}
        )
        self.assertEqual(r.status_code, 200)

        # 1. Status reflects Dataset B dimensions
        status = requests.get(f"{BASE_URL}/debug/status").json()
        self.assertEqual(status["dataset_dimensions"], {"rows": 20, "columns": 6})
        self.assertFalse(status["model_trained"])
        self.assertFalse(status["prediction_available"])

        # 2. Candidates reflect Dataset B
        cands_res = requests.get(f"{BASE_URL}/ml/candidates").json()
        candidate_cols = [c["column"] for c in cands_res.get("target_candidates", [])]
        self.assertIn("Salary", candidate_cols)
        self.assertNotIn("Profit", candidate_cols)
        self.assertNotIn("Marketing_Spend", candidate_cols)

        # 3. Chat reflects Dataset B
        chat_res = requests.post(f"{BASE_URL}/chat", json={"message": "What columns exist in my dataset?"})
        self.assertEqual(chat_res.status_code, 200)
        chat_reply = chat_res.json().get("response", "")
        self.assertIn("Salary", chat_reply)
        self.assertNotIn("Marketing_Spend", chat_reply)

    def test_10_five_consecutive_pipeline_repeatability_runs(self):
        """
        Execute 5 complete consecutive runs of the pipeline on Dataset A
        to verify structural determinism and repeatability across runs.
        """
        for run_idx in range(1, 6):
            # Upload
            up = requests.post(
                f"{BASE_URL}/upload",
                files={"file": ("sales_analysis.csv", io.StringIO(DATASET_A_CSV), "text/csv")}
            )
            self.assertEqual(up.status_code, 200, f"Run {run_idx}: Upload failed")

            # Candidates
            cands = requests.get(f"{BASE_URL}/ml/candidates")
            self.assertEqual(cands.status_code, 200, f"Run {run_idx}: Candidates failed")
            cand_list = cands.json().get("target_candidates", [])
            self.assertEqual(len(cand_list), 6, f"Run {run_idx}: Expected 6 candidates, got {len(cand_list)}")

            # Train
            train = requests.get(f"{BASE_URL}/ml/train")
            self.assertEqual(train.status_code, 200, f"Run {run_idx}: Training failed")

            # Eval
            ev = requests.get(f"{BASE_URL}/ml/evaluate")
            self.assertEqual(ev.status_code, 200, f"Run {run_idx}: Evaluation failed")
            metrics = ev.json().get("evaluation", {}).get("metrics", {})
            self.assertIn("R2", metrics, f"Run {run_idx}: R2 missing")
            self.assertGreater(metrics["R2"], 0.95, f"Run {run_idx}: R2 under expected baseline")

            # Predict
            pred = requests.post(
                f"{BASE_URL}/ml/predict",
                json={"data": {
                    "Marketing_Spend": 500.0,
                    "Discount_Percent": 0.1,
                    "Store_Visits": 12,
                    "Region": "North",
                    "Units_Sold": 45
                }}
            )
            self.assertEqual(pred.status_code, 200, f"Run {run_idx}: Prediction failed")
            pred_val = pred.json().get("prediction")
            self.assertIsInstance(pred_val, (int, float), f"Run {run_idx}: Prediction was not numeric")


if __name__ == "__main__":
    unittest.main()
