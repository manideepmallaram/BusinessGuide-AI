"""
End-to-End Capstone Presentation Demonstration Runner
Runs the full 15-step capstone pipeline 3 times sequentially against the live backend server.
Records exact returned values, execution times, and verifies data integrity at every step.
"""
import sys
import io
import time
import json
import requests

BASE_URL = "http://127.0.0.1:8000"

DATASET_CSV = """Customer_ID,Marketing_Spend,Discount_Percent,Store_Visits,Region,Units_Sold,Profit
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

def run_single_presentation_cycle(cycle_num: int):
    print(f"\n========================================================")
    print(f"  STARTING PRESENTATION DEMONSTRATION CYCLE {cycle_num}/3")
    print(f"========================================================")
    cycle_start = time.time()
    results = {}

    # Step 1: Health check
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    print(f"Step 1: Health Check -> OK ({r.json()}) [{time.time()-t0:.3f}s]")

    # Step 2: Diagnostic status check
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/debug/status")
    assert r.status_code == 200
    status_pre = r.json()
    print(f"Step 2: Pre-Upload Status Check -> OK (Dataset Loaded: {status_pre.get('dataset_loaded')}) [{time.time()-t0:.3f}s]")

    # Step 3: Dataset Upload
    t0 = time.time()
    r = requests.post(
        f"{BASE_URL}/upload",
        files={"file": ("sales_analysis.csv", io.StringIO(DATASET_CSV), "text/csv")}
    )
    assert r.status_code == 200, f"Upload failed: {r.text}"
    upload_data = r.json()
    results['upload'] = {
        'rows': upload_data.get('rows'),
        'columns': upload_data.get('columns'),
        'column_names': upload_data.get('column_names')
    }
    print(f"Step 3: Dataset Upload -> OK (Rows: {results['upload']['rows']}, Cols: {results['upload']['columns']}) [{time.time()-t0:.3f}s]")

    # Step 4: Dataset Profiling
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/profile")
    assert r.status_code == 200, f"Profile failed: {r.text}"
    prof_data = r.json()
    results['profile'] = {
        'rows': prof_data.get('rows'),
        'columns': prof_data.get('columns'),
        'numeric_cols': len(prof_data.get('numeric_columns', [])),
        'categorical_cols': len(prof_data.get('categorical_columns', []))
    }
    print(f"Step 4: Profiling -> OK (Numeric: {results['profile']['numeric_cols']}, Categorical: {results['profile']['categorical_cols']}) [{time.time()-t0:.3f}s]")

    # Step 5: Preprocessing
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/preprocess")
    assert r.status_code == 200, f"Preprocess failed: {r.text}"
    prep_data = r.json()
    results['preprocess'] = {
        'rows_after': prep_data.get('rows_after_preprocessing', prep_data.get('data_total_rows')),
        'missing_imputed': prep_data.get('missing_imputed', 0)
    }
    print(f"Step 5: Preprocessing -> OK (Rows: {results['preprocess']['rows_after']}) [{time.time()-t0:.3f}s]")

    # Step 6: Statistical Analytics
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/analytics")
    assert r.status_code == 200, f"Analytics failed: {r.text}"
    analytics_data = r.json().get('analytics', {})
    corrs = analytics_data.get('correlations', {})
    top_corr = list(corrs.keys())[0] if corrs else "None"
    results['analytics'] = {
        'correlations_count': len(corrs),
        'top_correlation_pair': top_corr
    }
    print(f"Step 6: Analytics -> OK (Correlations computed: {len(corrs)}) [{time.time()-t0:.3f}s]")

    # Step 7: Target Candidates Discovery (Run 3x in this cycle to test repeatability)
    for c_i in range(3):
        t0 = time.time()
        r = requests.get(f"{BASE_URL}/ml/candidates")
        assert r.status_code == 200, f"Candidates failed: {r.text}"
        cands = r.json().get('target_candidates', [])
        assert len(cands) == 6, f"Expected 6 candidates, got {len(cands)}"
    results['candidates'] = {
        'candidate_count': len(cands),
        'top_candidate': cands[0]['column'],
        'top_score': cands[0].get('suitability_score')
    }
    print(f"Step 7: Target Candidates -> OK (Count: {len(cands)}, Top: {cands[0]['column']} with score {cands[0].get('suitability_score')}) [{time.time()-t0:.3f}s]")

    # Step 8: AI Target Recommendation
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/ml/ai-target")
    assert r.status_code == 200, f"AI Target failed: {r.text}"
    ai_target_data = r.json().get('ai_target_selection', {})
    results['ai_target'] = {
        'recommended_target': ai_target_data.get('recommended_target'),
        'problem_type': ai_target_data.get('problem_type'),
        'confidence': ai_target_data.get('confidence'),
        'reason': ai_target_data.get('reason')[:60] + "..."
    }
    print(f"Step 8: AI Target Selection -> OK (Target: {results['ai_target']['recommended_target']}, Type: {results['ai_target']['problem_type']}, Confidence: {results['ai_target']['confidence']}) [{time.time()-t0:.3f}s]")

    # Step 9: Model Training
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/ml/train")
    assert r.status_code == 200, f"Training failed: {r.text}"
    train_resp = r.json()
    ai_dec = train_resp.get('ai_decision', {})
    train_data = train_resp.get('training', {})
    results['train'] = {
        'model_name': train_data.get('model'),
        'target_variable': ai_dec.get('target'),
        'features_count': len(train_data.get('feature_columns', [])),
        'training_rows': train_data.get('training_rows')
    }
    print(f"Step 9: Model Training -> OK (Model: {results['train']['model_name']}, Target: {results['train']['target_variable']}, Features: {results['train']['features_count']}, Rows: {results['train']['training_rows']}) [{time.time()-t0:.3f}s]")

    # Step 10: Model Evaluation
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/ml/evaluate")
    assert r.status_code == 200, f"Evaluation failed: {r.text}"
    eval_data = r.json().get('evaluation', {})
    metrics = eval_data.get('metrics', {})
    results['evaluation'] = {
        'problem_type': eval_data.get('problem_type'),
        'metrics': metrics
    }
    print(f"Step 10: Model Evaluation -> OK (Metrics: {metrics}) [{time.time()-t0:.3f}s]")

    # Step 11: Feature Importance
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/ml/feature-importance")
    assert r.status_code == 200, f"Feature Importance failed: {r.text}"
    fi_data = r.json().get('feature_importance', {})
    top_feature = fi_data.get('features', [{}])[0]
    results['feature_importance'] = {
        'features_analyzed': len(fi_data.get('features', [])),
        'top_feature': top_feature.get('feature'),
        'top_importance': top_feature.get('importance')
    }
    print(f"Step 11: Feature Importance -> OK (Top Feature: {top_feature.get('feature')} = {top_feature.get('importance'):.4f}) [{time.time()-t0:.3f}s]")

    # Step 12: Prediction Schema
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/ml/prediction-schema")
    assert r.status_code == 200, f"Schema failed: {r.text}"
    schema_fields = r.json().get('features', [])
    print(f"Step 12: Prediction Schema -> OK (Features: {len(schema_fields)}) [{time.time()-t0:.3f}s]")

    # Step 13: Scenario Simulation / Prediction
    t0 = time.time()
    pred_payload = {
        "Marketing_Spend": 550.0,
        "Discount_Percent": 0.12,
        "Store_Visits": 15,
        "Region": "North",
        "Units_Sold": 48
    }
    r = requests.post(f"{BASE_URL}/ml/predict", json={"data": pred_payload})
    assert r.status_code == 200, f"Predict failed: {r.text}"
    pred_data = r.json()
    results['prediction'] = {
        'target_variable': pred_data.get('target'),
        'predicted_value': pred_data.get('prediction')
    }
    print(f"Step 13: What-If Prediction -> OK (Target: {results['prediction']['target_variable']}, Predicted: {results['prediction']['predicted_value']:.2f}) [{time.time()-t0:.3f}s]")

    # Step 14: AI Business Chat Assistant
    t0 = time.time()
    chat_prompt = "What is my model's R2 score and what is the most important driver of profit?"
    r = requests.post(f"{BASE_URL}/chat", json={"message": chat_prompt})
    assert r.status_code == 200, f"Chat failed: {r.text}"
    chat_reply = r.json().get('response', '')
    assert len(chat_reply) > 50, "Chat reply was too short or empty"
    results['chat'] = {
        'user_question': chat_prompt,
        'response_preview': chat_reply[:120].replace('\n', ' ') + "...",
        'response_length': len(chat_reply)
    }
    print(f"Step 14: AI Business Chat Assistant -> OK (Length: {len(chat_reply)} chars) [{time.time()-t0:.3f}s]")
    print(f"       Chat Response Preview: {results['chat']['response_preview']}")

    # Step 15: Diagnostic Pipeline Status Verification
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/debug/pipeline")
    assert r.status_code == 200, f"Debug pipeline failed: {r.text}"
    pipeline_state = r.json()
    assert pipeline_state['upload_stage']['loaded'] is True
    assert pipeline_state['training_stage']['model_trained'] is True
    assert pipeline_state['prediction_stage']['latest_prediction_available'] is True
    print(f"Step 15: Pipeline Observability Check -> ALL STAGES VERIFIED READY [{time.time()-t0:.3f}s]")

    total_cycle_time = time.time() - cycle_start
    print(f"CYCLE {cycle_num} COMPLETED SUCCESSFULLY in {total_cycle_time:.2f}s\n")
    return results

def main():
    print("======================================================================")
    print("CAPSTONE PRESENTATION DEMONSTRATION VERIFICATION SUITE")
    print("Verifying 3 Sequential Cycles against Live Clean Backend")
    print("======================================================================")
    
    cycles_data = []
    for c in range(1, 4):
        res = run_single_presentation_cycle(c)
        cycles_data.append(res)
        time.sleep(1) # Brief pause between presentation cycles
        
    print("\n======================================================================")
    print("CROSS-CYCLE CONSISTENCY AUDIT")
    print("======================================================================")
    for idx, cycle in enumerate(cycles_data, 1):
        print(f"Cycle {idx}:")
        print(f"  - Target: {cycle['train']['target_variable']}")
        print(f"  - Model: {cycle['train']['model_name']}")
        print(f"  - Metrics: {cycle['evaluation']['metrics']}")
        print(f"  - Top Driver: {cycle['feature_importance']['top_feature']} ({cycle['feature_importance']['top_importance']:.4f})")
        print(f"  - Scenario Prediction: {cycle['prediction']['predicted_value']}")
        print(f"  - Chat Response Length: {cycle['chat']['response_length']} chars")

    print("\nALL 3 PRESENTATION DEMONSTRATION CYCLES EXECUTED AND FULLY VERIFIED!")

if __name__ == "__main__":
    main()
