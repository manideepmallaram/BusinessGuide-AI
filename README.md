# BusinessGuide AI — AI-Powered Business Decision Support System

An enterprise-ready AI Decision Support System designed to ingest business datasets, automatically discover and recommend prediction targets, train and evaluate scikit-learn machine learning pipelines, compute key business drivers, simulate what-if scenarios, and engage in real-time conversational business intelligence powered by Groq LLM integration.

---

## Key Capabilities

1. **Autonomous Target Discovery & Ranking**: Evaluates dataset columns based on business suitability, cardinality, and entropy.
2. **AI Target Recommendation**: Uses Groq LLM intelligence to determine whether regression or classification is appropriate, with statistical fallback protection.
3. **Automated ML Pipelines**: Trains Scikit-Learn Random Forest Regressor/Classifier models with preprocessing, median imputation, and one-hot encoding.
4. **Rigorous Validation**: Computes $R^2$, MAE, MSE, and RMSE for regression; Accuracy, weighted F1-Score, Precision, and Recall for classification.
5. **Driver Analysis & What-If Simulator**: Measures feature importance percentages and provides real-time scenario simulation.
6. **Conversational AI Assistant**: Ingests dataset profile and live ML model performance into conversational context for business querying.
7. **Production Observability**: Full structured logging across 11 pipeline stages with diagnostic status endpoints (`/debug/status` and `/debug/pipeline`).

---

## Tech Stack

- **Backend**: FastAPI, Uvicorn, Pandas, Scikit-Learn, Groq SDK, Python 3.11+
- **Frontend**: React 19, Vite, Lucide Icons, Vanilla CSS Design System
- **LLM Engine**: Groq Cloud API (`openai/gpt-oss-20b` / Llama-3) with backoff retry

---

## Getting Started Locally

### 1. Backend Setup
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env # Add your GROQ_API_KEY
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173/` in your browser.

---

## Automated Verification Suite

Run the 10-dataset multi-domain verification suite (150 pipeline operations):
```bash
cd backend
python tests/verify_10_datasets.py
```

Run the unit test regression suite:
```bash
cd backend
python -m unittest discover -s tests -p "test_reliability.py"
```

---

## Deployment

Refer to [`DEPLOYMENT.md`](DEPLOYMENT.md) for detailed cloud deployment guides on Render, Railway, Vercel, and Docker.
