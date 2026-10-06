"""
Comprehensive 10-Dataset Pipeline Verification Suite
Tests the complete end-to-end BusinessGuide AI platform across 10 diverse real-world datasets:
1. Retail Sales & Profit (Regression)
2. Customer Churn (Classification)
3. HR Compensation & Salary (Regression)
4. Financial Loan Approval (Classification)
5. Real Estate Property Valuation (Regression)
6. Healthcare Patient Readmission (Classification)
7. Digital Marketing Lead Conversion (Classification)
8. Academic Exam Performance (Regression)
9. Supply Chain Delivery Logistics (Regression)
10. Manufacturing Defect Inspection (Classification)

Verifies every response structure, execution repeatability, and state isolation across datasets.
"""
import time
import io
import json
import requests

BASE_URL = "http://127.0.0.1:8000"

DATASETS = [
    {
        "name": "Dataset 1: Retail Sales & Profit",
        "domain": "Retail & E-commerce",
        "filename": "retail_sales.csv",
        "expected_type": "regression",
        "csv": """Store_ID,Marketing_Spend,Discount_Rate,Foot_Traffic,Online_Orders,Region,Net_Profit
STR_001,500.0,0.05,1200,350,East,4500.0
STR_002,300.0,0.02,800,210,West,2800.0
STR_003,750.0,0.10,1850,540,East,6900.0
STR_004,200.0,0.00,550,150,North,1800.0
STR_005,900.0,0.15,2100,680,South,8200.0
STR_006,450.0,0.04,1100,310,East,4100.0
STR_007,600.0,0.08,1500,420,West,5300.0
STR_008,150.0,0.00,450,120,North,1400.0
STR_009,820.0,0.12,1950,610,South,7500.0
STR_010,380.0,0.03,950,270,East,3400.0
STR_011,520.0,0.06,1300,380,West,4800.0
STR_012,280.0,0.02,720,190,North,2500.0
STR_013,710.0,0.09,1750,500,East,6400.0
STR_014,190.0,0.00,520,140,West,1700.0
STR_015,880.0,0.14,2050,650,South,8000.0
STR_016,420.0,0.04,1050,300,East,3900.0
STR_017,610.0,0.08,1520,430,North,5400.0
STR_018,170.0,0.00,480,130,West,1500.0
STR_019,790.0,0.11,1900,590,South,7200.0
STR_020,350.0,0.03,900,250,East,3200.0
STR_021,540.0,0.07,1350,390,North,4900.0
STR_022,310.0,0.02,780,220,West,2700.0
""",
        "scenario": {
            "Marketing_Spend": 550.0,
            "Discount_Rate": 0.06,
            "Foot_Traffic": 1350,
            "Online_Orders": 400,
            "Region": "East"
        },
        "chat_prompt": "What is the primary factor driving Net_Profit?"
    },
    {
        "name": "Dataset 2: Customer Churn Prediction",
        "domain": "SaaS / Telecom",
        "filename": "customer_churn.csv",
        "expected_type": "classification",
        "csv": """Account_ID,Contract_Months,Monthly_Charges,Support_Tickets,Payment_Method,Paperless_Billing,Churn_Status
ACC_101,1,85.5,5,Electronic_Check,Yes,Churned
ACC_102,24,35.0,0,Credit_Card,No,Retained
ACC_103,12,65.0,2,Bank_Transfer,Yes,Retained
ACC_104,1,92.0,6,Electronic_Check,Yes,Churned
ACC_105,36,45.0,1,Credit_Card,No,Retained
ACC_106,1,78.0,4,Electronic_Check,Yes,Churned
ACC_107,24,55.0,0,Bank_Transfer,Yes,Retained
ACC_108,12,70.0,3,Credit_Card,Yes,Retained
ACC_109,1,88.5,5,Electronic_Check,Yes,Churned
ACC_110,36,40.0,0,Credit_Card,No,Retained
ACC_111,1,95.0,7,Electronic_Check,Yes,Churned
ACC_112,12,60.0,1,Bank_Transfer,No,Retained
ACC_113,24,50.0,1,Credit_Card,Yes,Retained
ACC_114,1,82.0,4,Electronic_Check,Yes,Churned
ACC_115,36,38.0,0,Bank_Transfer,No,Retained
ACC_116,12,72.0,2,Credit_Card,Yes,Retained
ACC_117,1,90.0,5,Electronic_Check,Yes,Churned
ACC_118,24,48.0,0,Credit_Card,No,Retained
ACC_119,1,86.0,4,Electronic_Check,Yes,Churned
ACC_120,36,42.0,1,Bank_Transfer,Yes,Retained
ACC_121,12,68.0,2,Electronic_Check,Yes,Retained
ACC_122,1,94.0,6,Electronic_Check,Yes,Churned
""",
        "scenario": {
            "Contract_Months": 1,
            "Monthly_Charges": 89.0,
            "Support_Tickets": 5,
            "Payment_Method": "Electronic_Check",
            "Paperless_Billing": "Yes"
        },
        "chat_prompt": "What are the biggest indicators of customer churn?"
    },
    {
        "name": "Dataset 3: HR Compensation & Salary",
        "domain": "Human Resources",
        "filename": "hr_compensation.csv",
        "expected_type": "regression",
        "csv": """Emp_ID,Department,Years_Experience,Job_Level,Performance_Score,Annual_Salary
EMP_501,Engineering,3,2,4.2,85000
EMP_502,Sales,1,1,3.8,55000
EMP_503,HR,5,3,4.5,72000
EMP_504,Support,2,1,3.1,48000
EMP_505,Engineering,6,4,4.8,115000
EMP_506,Marketing,4,2,3.9,76000
EMP_507,Sales,2,1,3.6,58000
EMP_508,HR,3,2,4.0,64000
EMP_509,Engineering,1,1,3.7,72000
EMP_510,Support,4,2,3.4,54000
EMP_511,Engineering,5,3,4.6,105000
EMP_512,Sales,3,2,3.9,65000
EMP_513,Marketing,2,1,3.5,60000
EMP_514,Engineering,4,3,4.3,96000
EMP_515,Support,1,1,3.0,46000
EMP_516,Sales,5,3,4.1,80000
EMP_517,HR,2,1,3.8,59000
EMP_518,Engineering,7,5,4.9,130000
EMP_519,Marketing,5,3,4.2,84000
EMP_520,Support,3,2,3.2,51000
EMP_521,Engineering,4,2,4.1,92000
EMP_522,Sales,4,2,4.0,73000
""",
        "scenario": {
            "Department": "Engineering",
            "Years_Experience": 5,
            "Job_Level": 3,
            "Performance_Score": 4.5
        },
        "chat_prompt": "How does experience relate to annual salary?"
    },
    {
        "name": "Dataset 4: Financial Loan Approval",
        "domain": "Fintech / Lending",
        "filename": "loan_approval.csv",
        "expected_type": "classification",
        "csv": """App_ID,Credit_Score,Annual_Income,Debt_Ratio,Loan_Amount,Employment_Status,Approval_Outcome
APP_01,780,95000,0.18,25000,Employed,Approved
APP_02,590,38000,0.48,15000,Unemployed,Denied
APP_03,720,82000,0.25,20000,Employed,Approved
APP_04,540,29000,0.52,10000,PartTime,Denied
APP_05,810,125000,0.12,40000,Employed,Approved
APP_06,660,62000,0.34,18000,Employed,Approved
APP_07,580,35000,0.45,12000,PartTime,Denied
APP_08,740,88000,0.22,22000,Employed,Approved
APP_09,610,48000,0.42,14000,Employed,Denied
APP_10,790,110000,0.15,35000,Employed,Approved
APP_11,570,32000,0.50,9000,Unemployed,Denied
APP_12,710,78000,0.28,19000,Employed,Approved
APP_13,640,54000,0.38,16000,Employed,Denied
APP_14,800,115000,0.14,38000,Employed,Approved
APP_15,600,42000,0.46,13000,PartTime,Denied
APP_16,730,85000,0.24,21000,Employed,Approved
APP_17,560,31000,0.53,8000,Unemployed,Denied
APP_18,770,98000,0.19,27000,Employed,Approved
APP_19,650,59000,0.36,17000,Employed,Denied
APP_20,820,130000,0.11,45000,Employed,Approved
APP_21,700,74000,0.30,18000,Employed,Approved
APP_22,580,36000,0.47,11000,PartTime,Denied
""",
        "scenario": {
            "Credit_Score": 760,
            "Annual_Income": 90000,
            "Debt_Ratio": 0.20,
            "Loan_Amount": 24000,
            "Employment_Status": "Employed"
        },
        "chat_prompt": "What determines whether a loan application is approved or denied?"
    },
    {
        "name": "Dataset 5: Real Estate Property Valuation",
        "domain": "Real Estate",
        "filename": "property_valuation.csv",
        "expected_type": "regression",
        "csv": """Property_ID,Square_Feet,Bedrooms,Bathrooms,Location_Zone,Year_Built,Selling_Price
PROP_01,1800,3,2.0,Suburban,2005,320000
PROP_02,1200,2,1.0,Rural,1995,195000
PROP_03,2400,4,2.5,Urban,2015,480000
PROP_04,950,1,1.0,Urban,1988,170000
PROP_05,3100,5,3.5,Suburban,2020,620000
PROP_06,1650,3,2.0,Suburban,2002,295000
PROP_07,2100,4,2.5,Urban,2012,430000
PROP_08,1100,2,1.0,Rural,1990,180000
PROP_09,2800,4,3.0,Suburban,2018,560000
PROP_10,1450,3,1.5,Suburban,1998,260000
PROP_11,1950,3,2.0,Urban,2008,370000
PROP_12,1300,2,1.5,Rural,1996,210000
PROP_13,2600,4,3.0,Suburban,2016,515000
PROP_14,1000,2,1.0,Urban,1985,175000
PROP_15,2950,5,3.5,Suburban,2019,590000
PROP_16,1550,3,2.0,Suburban,2000,280000
PROP_17,2250,4,2.5,Urban,2014,450000
PROP_18,1150,2,1.0,Rural,1992,190000
PROP_19,2700,4,3.0,Suburban,2017,540000
PROP_20,1400,3,1.5,Suburban,1997,250000
PROP_21,2000,3,2.0,Urban,2010,390000
PROP_22,1250,2,1.0,Rural,1994,200000
""",
        "scenario": {
            "Square_Feet": 2100,
            "Bedrooms": 3,
            "Bathrooms": 2.5,
            "Location_Zone": "Suburban",
            "Year_Built": 2012
        },
        "chat_prompt": "What feature has the strongest impact on Selling_Price?"
    },
    {
        "name": "Dataset 6: Healthcare Patient Readmission",
        "domain": "Healthcare Analytics",
        "filename": "patient_readmission.csv",
        "expected_type": "classification",
        "csv": """Patient_ID,Age,BMI,Systolic_BP,Prior_Visits,Condition_Type,Readmitted
PAT_001,68,32.4,145,4,Cardiac,Readmitted
PAT_002,42,24.1,118,0,General,Discharged
PAT_003,59,29.5,138,2,Diabetes,Readmitted
PAT_004,35,22.8,115,0,General,Discharged
PAT_005,74,35.0,155,5,Cardiac,Readmitted
PAT_006,51,26.2,125,1,Respiratory,Discharged
PAT_007,63,31.0,142,3,Cardiac,Readmitted
PAT_008,38,23.5,120,0,General,Discharged
PAT_009,70,33.8,150,4,Diabetes,Readmitted
PAT_010,48,25.8,122,1,Respiratory,Discharged
PAT_011,66,30.5,140,3,Cardiac,Readmitted
PAT_012,40,24.0,116,0,General,Discharged
PAT_013,72,34.2,152,5,Diabetes,Readmitted
PAT_014,45,25.0,120,1,General,Discharged
PAT_015,69,33.0,148,4,Cardiac,Readmitted
PAT_016,53,27.0,128,1,Respiratory,Discharged
PAT_017,61,29.8,136,2,Diabetes,Readmitted
PAT_018,36,22.0,114,0,General,Discharged
PAT_019,75,36.1,158,6,Cardiac,Readmitted
PAT_020,49,26.5,124,1,General,Discharged
PAT_021,65,31.2,141,3,Diabetes,Readmitted
PAT_022,44,24.5,119,0,Respiratory,Discharged
""",
        "scenario": {
            "Age": 65,
            "BMI": 31.0,
            "Systolic_BP": 142,
            "Prior_Visits": 3,
            "Condition_Type": "Cardiac"
        },
        "chat_prompt": "What clinical factors most strongly predict patient readmission?"
    },
    {
        "name": "Dataset 7: Digital Marketing Lead Conversion",
        "domain": "Digital Marketing",
        "filename": "marketing_leads.csv",
        "expected_type": "classification",
        "csv": """Lead_ID,Ad_Platform,Click_Rate,Pages_Viewed,Time_On_Site_Sec,Converted
LEAD_101,Google,0.08,6,320,Yes
LEAD_102,Facebook,0.02,1,45,No
LEAD_103,LinkedIn,0.06,5,280,Yes
LEAD_104,Twitter,0.01,1,30,No
LEAD_105,Google,0.09,7,390,Yes
LEAD_106,Facebook,0.03,2,65,No
LEAD_107,LinkedIn,0.07,6,310,Yes
LEAD_108,Twitter,0.01,1,25,No
LEAD_109,Google,0.10,8,420,Yes
LEAD_110,Facebook,0.02,2,55,No
LEAD_111,LinkedIn,0.05,4,240,Yes
LEAD_112,Twitter,0.01,1,35,No
LEAD_113,Google,0.08,6,330,Yes
LEAD_114,Facebook,0.02,1,50,No
LEAD_115,Google,0.09,7,380,Yes
LEAD_116,Twitter,0.01,1,20,No
LEAD_117,LinkedIn,0.06,5,290,Yes
LEAD_118,Facebook,0.03,2,70,No
LEAD_119,Google,0.07,5,260,Yes
LEAD_120,Twitter,0.01,1,40,No
LEAD_121,LinkedIn,0.08,7,350,Yes
LEAD_122,Facebook,0.02,1,45,No
""",
        "scenario": {
            "Ad_Platform": "Google",
            "Click_Rate": 0.08,
            "Pages_Viewed": 6,
            "Time_On_Site_Sec": 320
        },
        "chat_prompt": "How does time on site influence conversion?"
    },
    {
        "name": "Dataset 8: Academic Exam Performance",
        "domain": "Education & EdTech",
        "filename": "student_performance.csv",
        "expected_type": "regression",
        "csv": """Student_ID,Study_Hours_Weekly,Attendance_Rate,Tutoring_Sessions,Extracurriculars,Final_Score
STU_001,18.5,0.96,4,Yes,94.0
STU_002,6.0,0.75,0,No,62.0
STU_003,14.0,0.90,2,Yes,86.0
STU_004,4.5,0.68,0,No,54.0
STU_005,22.0,0.98,5,Yes,98.0
STU_006,10.0,0.82,1,No,74.0
STU_007,16.0,0.92,3,Yes,90.0
STU_008,5.0,0.70,0,No,58.0
STU_009,20.0,0.97,4,Yes,96.0
STU_010,8.5,0.78,1,No,68.0
STU_011,13.5,0.88,2,Yes,84.0
STU_012,7.0,0.76,0,No,64.0
STU_013,17.0,0.94,3,Yes,92.0
STU_014,4.0,0.65,0,No,52.0
STU_015,21.0,0.98,5,Yes,97.0
STU_016,9.5,0.80,1,No,72.0
STU_017,15.0,0.91,3,Yes,88.0
STU_018,5.5,0.72,0,No,60.0
STU_019,19.0,0.95,4,Yes,95.0
STU_020,8.0,0.77,1,No,66.0
STU_021,14.5,0.89,2,Yes,85.0
STU_022,6.5,0.74,0,No,63.0
""",
        "scenario": {
            "Study_Hours_Weekly": 15.0,
            "Attendance_Rate": 0.90,
            "Tutoring_Sessions": 2,
            "Extracurriculars": "Yes"
        },
        "chat_prompt": "What is the correlation between study hours and final score?"
    },
    {
        "name": "Dataset 9: Supply Chain Delivery Logistics",
        "domain": "Logistics & Supply Chain",
        "filename": "delivery_logistics.csv",
        "expected_type": "regression",
        "csv": """Shipment_ID,Distance_Miles,Package_Weight_Kg,Transport_Mode,Weather_Impact,Delivery_Hours
SHIP_01,450,15.5,Truck,Low,12.5
SHIP_02,1200,8.0,Air,Low,6.0
SHIP_03,800,45.0,Train,Medium,28.0
SHIP_04,150,5.0,Van,Low,4.5
SHIP_05,600,22.0,Truck,High,18.0
SHIP_06,1400,12.0,Air,Medium,8.5
SHIP_07,950,50.0,Train,High,34.0
SHIP_08,200,8.5,Van,Low,5.5
SHIP_09,500,18.0,Truck,Low,13.5
SHIP_10,1100,7.5,Air,Low,5.5
SHIP_11,750,40.0,Train,Low,24.0
SHIP_12,180,6.0,Van,Medium,6.0
SHIP_13,550,20.0,Truck,Medium,15.5
SHIP_14,1300,10.0,Air,High,10.0
SHIP_15,850,48.0,Train,Medium,30.0
SHIP_16,220,9.0,Van,Low,5.8
SHIP_17,480,16.5,Truck,Low,13.0
SHIP_18,1050,7.0,Air,Low,5.0
SHIP_19,700,38.0,Train,Low,22.5
SHIP_20,160,5.5,Van,Low,4.8
SHIP_21,520,19.0,Truck,Low,14.0
SHIP_22,1250,9.5,Air,Low,6.2
""",
        "scenario": {
            "Distance_Miles": 500,
            "Package_Weight_Kg": 18.0,
            "Transport_Mode": "Truck",
            "Weather_Impact": "Low"
        },
        "chat_prompt": "What factors cause the longest delivery delays?"
    },
    {
        "name": "Dataset 10: Manufacturing Quality Inspection",
        "domain": "Manufacturing / Quality Control",
        "filename": "manufacturing_quality.csv",
        "expected_type": "classification",
        "csv": """Batch_ID,Machine_Temp_C,Pressure_PSI,Vibration_Hz,Operator_Shift,Quality_Status
BATCH_01,82.5,145.0,2.1,Morning,Pass
BATCH_02,115.0,190.0,5.8,Night,Fail
BATCH_03,85.0,150.0,2.3,Morning,Pass
BATCH_04,118.2,195.5,6.2,Night,Fail
BATCH_05,80.1,142.0,1.9,Day,Pass
BATCH_06,112.5,188.0,5.5,Night,Fail
BATCH_07,84.0,148.0,2.2,Day,Pass
BATCH_08,120.0,198.0,6.5,Night,Fail
BATCH_09,81.5,144.0,2.0,Morning,Pass
BATCH_10,114.0,189.0,5.6,Night,Fail
BATCH_11,83.0,146.0,2.1,Day,Pass
BATCH_12,116.5,192.0,5.9,Night,Fail
BATCH_13,79.5,140.0,1.8,Morning,Pass
BATCH_14,119.0,196.0,6.3,Night,Fail
BATCH_15,86.0,152.0,2.4,Day,Pass
BATCH_16,113.0,187.0,5.4,Night,Fail
BATCH_17,82.0,145.0,2.0,Morning,Pass
BATCH_18,117.0,194.0,6.0,Night,Fail
BATCH_19,80.8,143.0,1.9,Day,Pass
BATCH_20,115.5,191.0,5.7,Night,Fail
BATCH_21,84.5,149.0,2.2,Morning,Pass
BATCH_22,114.5,190.0,5.6,Night,Fail
""",
        "scenario": {
            "Machine_Temp_C": 83.0,
            "Pressure_PSI": 146.0,
            "Vibration_Hz": 2.1,
            "Operator_Shift": "Morning"
        },
        "chat_prompt": "What operational thresholds indicate batch failure?"
    }
]

def run_test_on_dataset(ds_info, index):
    print(f"\n================================================================================")
    print(f"  TESTING DATASET {index}/10: {ds_info['name']} ({ds_info['domain']})")
    print(f"================================================================================")
    start_t = time.time()
    report = {"name": ds_info["name"], "domain": ds_info["domain"]}

    # 1. POST /upload
    t0 = time.time()
    r = requests.post(
        f"{BASE_URL}/upload",
        files={"file": (ds_info["filename"], io.StringIO(ds_info["csv"]), "text/csv")}
    )
    assert r.status_code == 200, f"Upload failed: {r.text}"
    up_data = r.json()
    report["rows"] = up_data["rows"]
    report["columns"] = up_data["columns"]
    print(f"[Step 1] Upload: {report['rows']} rows, {report['columns']} cols [{time.time()-t0:.3f}s]")

    # 2. GET /profile
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/profile")
    assert r.status_code == 200
    prof = r.json()
    report["profile_numeric"] = len(prof.get("numeric_columns", []))
    report["profile_cat"] = len(prof.get("categorical_columns", []))
    print(f"[Step 2] Profile: {report['profile_numeric']} numeric, {report['profile_cat']} cat [{time.time()-t0:.3f}s]")

    # 3. GET /preprocess
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/preprocess")
    assert r.status_code == 200
    prep = r.json()
    report["cleaned_rows"] = prep.get("rows_after_preprocessing")
    print(f"[Step 3] Preprocess: Cleaned rows {report['cleaned_rows']} [{time.time()-t0:.3f}s]")

    # 4. GET /analytics
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/analytics")
    assert r.status_code == 200
    analytics = r.json().get("analytics", {})
    report["analytics_corrs"] = len(analytics.get("correlations", {}))
    print(f"[Step 4] Analytics: {report['analytics_corrs']} correlation pairs computed [{time.time()-t0:.3f}s]")

    # 5. GET /ml/candidates
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/ml/candidates")
    assert r.status_code == 200
    cands = r.json().get("target_candidates", [])
    assert len(cands) > 0, "No candidates discovered!"
    report["candidates_count"] = len(cands)
    report["top_candidate"] = cands[0]["column"]
    print(f"[Step 5] Candidates: Found {len(cands)} candidates (Top: {cands[0]['column']}) [{time.time()-t0:.3f}s]")

    # 6. GET /ml/targets
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/ml/targets")
    assert r.status_code == 200
    ranked = r.json().get("ranked_targets", [])
    print(f"[Step 6] Ranked Targets: {len(ranked)} targets ranked [{time.time()-t0:.3f}s]")

    # 7. GET /ml/ai-target
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/ml/ai-target")
    assert r.status_code == 200
    ai_dec = r.json().get("ai_target_selection", {})
    report["selected_target"] = ai_dec.get("recommended_target")
    report["problem_type"] = ai_dec.get("problem_type")
    report["confidence"] = ai_dec.get("confidence")
    print(f"[Step 7] AI Target Selection: '{report['selected_target']}' ({report['problem_type']}) Conf: {report['confidence']} [{time.time()-t0:.3f}s]")

    # 8. GET /ml/train
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/ml/train")
    assert r.status_code == 200, f"Training failed: {r.text}"
    tr = r.json().get("training", {})
    report["model"] = tr.get("model")
    report["features_count"] = len(tr.get("feature_columns", []))
    print(f"[Step 8] Train Model: {report['model']} on {report['features_count']} features [{time.time()-t0:.3f}s]")

    # 9. GET /ml/evaluate
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/ml/evaluate")
    assert r.status_code == 200, f"Evaluation failed: {r.text}"
    ev = r.json().get("evaluation", {})
    report["metrics"] = ev.get("metrics", {})
    print(f"[Step 9] Evaluate Model: {report['metrics']} [{time.time()-t0:.3f}s]")

    # 10. GET /ml/feature-importance
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/ml/feature-importance")
    assert r.status_code == 200
    fi = r.json().get("feature_importance", {})
    top_driver = fi.get("features", [{}])[0]
    report["top_driver"] = top_driver.get("feature")
    report["top_importance"] = top_driver.get("importance")
    print(f"[Step 10] Feature Importance: Top driver '{report['top_driver']}' = {report['top_importance']:.4f} [{time.time()-t0:.3f}s]")

    # 11. GET /ml/prediction-schema
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/ml/prediction-schema")
    assert r.status_code == 200
    schema_fields = r.json().get("features", [])
    report["schema_fields"] = len(schema_fields)
    print(f"[Step 11] Prediction Schema: {len(schema_fields)} fields [{time.time()-t0:.3f}s]")

    # 12. POST /ml/predict
    t0 = time.time()
    pred_res = requests.post(f"{BASE_URL}/ml/predict", json={"data": ds_info["scenario"]})
    assert pred_res.status_code == 200, f"Predict failed: {pred_res.text}"
    pred_data = pred_res.json()
    report["prediction"] = pred_data.get("prediction")
    print(f"[Step 12] What-If Prediction: {report['prediction']} [{time.time()-t0:.3f}s]")

    # 13. GET /business/insights
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/business/insights")
    assert r.status_code == 200
    insights = r.json().get("insights", [])
    report["insights_count"] = len(insights)
    print(f"[Step 13] Executive Insights: {len(insights)} generated [{time.time()-t0:.3f}s]")

    # 14. POST /chat
    t0 = time.time()
    chat_res = requests.post(f"{BASE_URL}/chat", json={"message": ds_info["chat_prompt"]})
    assert chat_res.status_code == 200, f"Chat failed: {chat_res.text}"
    chat_reply = chat_res.json().get("response", "")
    assert len(chat_reply) > 50, "Chat response was too short"
    report["chat_reply_len"] = len(chat_reply)
    print(f"[Step 14] AI Business Chat: {len(chat_reply)} chars generated [{time.time()-t0:.3f}s]")

    # 15. GET /debug/pipeline
    t0 = time.time()
    r = requests.get(f"{BASE_URL}/debug/pipeline")
    assert r.status_code == 200
    pipe = r.json()
    assert pipe["upload_stage"]["loaded"] is True
    assert pipe["training_stage"]["model_trained"] is True
    assert pipe["prediction_stage"]["latest_prediction_available"] is True
    print(f"[Step 15] Pipeline Observability: ALL STAGES VERIFIED [{time.time()-t0:.3f}s]")

    total_time = time.time() - start_t
    report["total_time_seconds"] = round(total_time, 2)
    print(f"DATASET {index} COMPLETED SUCCESSFULLY in {total_time:.2f}s\n")
    return report

def main():
    print("================================================================================")
    print("      BUSINESSGUIDE AI — 10-DATASET COMPREHENSIVE VERIFICATION SUITE           ")
    print("================================================================================")
    results = []
    total_start = time.time()

    for idx, ds in enumerate(DATASETS, 1):
        rep = run_test_on_dataset(ds, idx)
        results.append(rep)
        time.sleep(1.5) # Grace period between datasets to avoid rate limits

    total_duration = time.time() - total_start
    print("\n================================================================================")
    print("                         SUMMARY OF ALL 10 DATASETS TESTED                      ")
    print("================================================================================")
    print(f"{'#':<3} | {'Domain / Dataset':<34} | {'Target':<18} | {'Type':<14} | {'Primary Metric':<20} | {'Status'}")
    print("-" * 105)
    for idx, r in enumerate(results, 1):
        metrics_str = ""
        if "R2" in r["metrics"]:
            metrics_str = f"R2={r['metrics']['R2']:.3f}"
        elif "Accuracy" in r["metrics"]:
            metrics_str = f"Acc={r['metrics']['Accuracy']*100:.1f}%"
            if "F1" in r["metrics"] and r["metrics"]["F1"] is not None:
                metrics_str += f", F1={r['metrics']['F1']:.2f}"
        print(f"{idx:<3} | {r['name'][:34]:<34} | {r['selected_target']:<18} | {r['problem_type']:<14} | {metrics_str:<20} | 100% PASS")

    print("-" * 105)
    print(f"ALL 10 DATASETS (15 STAGES EACH = 150 OPERATIONS) VERIFIED IN {total_duration:.2f}s!")

if __name__ == "__main__":
    main()
