import { useState, useRef, useEffect } from "react";
import {
  Upload,
  FileSpreadsheet,
  CheckCircle2,
  Loader2,
  Database,
  Columns3,
  AlertTriangle,
  Copy,
  BarChart3,
  Brain,
  Sparkles,
  Target,
  Play,
  Gauge,
  TrendingUp,
  Sliders,
  Lightbulb,
  ArrowRight,
  RefreshCw,
  MessageSquare,
  Send,
  Bot,
  User,
  Trash2,
} from "lucide-react";
import "./index.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const SUGGESTED_PROMPTS = [
  "Summarize my dataset",
  "Explain the recommended target",
  "What are the strongest business drivers?",
  "Explain my model performance",
  "What should I focus on?",
];

function renderInlineMarkdown(str) {
  const parts = str.split(/(\*\*.*?\*\*|`.*?`)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={i}>{part.slice(1, -1)}</code>;
    }
    return part;
  });
}

function formatAiMessage(text) {
  if (!text) return null;
  const lines = text.split("\n");
  const elements = [];
  let currentList = [];

  const flushList = (key) => {
    if (currentList.length > 0) {
      elements.push(
        <ul key={key}>
          {currentList.map((item, idx) => (
            <li key={idx}>{renderInlineMarkdown(item)}</li>
          ))}
        </ul>
      );
      currentList = [];
    }
  };

  lines.forEach((line, index) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushList(`list-before-${index}`);
      return;
    }
    if (trimmed.startsWith("- ") || trimmed.startsWith("* ") || trimmed.startsWith("• ")) {
      currentList.push(trimmed.substring(2));
    } else {
      flushList(`list-before-${index}`);
      if (trimmed.startsWith("### ")) {
        elements.push(
          <h4 key={`h4-${index}`} style={{ margin: "10px 0 6px 0", color: "#a5b4fc" }}>
            {renderInlineMarkdown(trimmed.substring(4))}
          </h4>
        );
      } else if (trimmed.startsWith("## ")) {
        elements.push(
          <h3 key={`h3-${index}`} style={{ margin: "12px 0 6px 0", color: "#c084fc" }}>
            {renderInlineMarkdown(trimmed.substring(3))}
          </h3>
        );
      } else {
        elements.push(<p key={`p-${index}`}>{renderInlineMarkdown(trimmed)}</p>);
      }
    }
  });

  flushList("list-final");
  return elements;
}

function formatValue(value) {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return "—";
    }
    return value.toLocaleString(undefined, {
      maximumFractionDigits: 2,
    });
  }
  return String(value);
}

function formatColumnName(column) {
  return String(column)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Data Pipeline States
  const [profile, setProfile] = useState(null);
  const [preprocessing, setPreprocessing] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [candidates, setCandidates] = useState(null);
  const [rankedTargets, setRankedTargets] = useState(null);

  // AI & ML States
  const [aiTarget, setAiTarget] = useState(null);
  const [loadingAI, setLoadingAI] = useState(false);

  const [training, setTraining] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [loadingTrain, setLoadingTrain] = useState(false);

  // Prediction & Insights States
  const [predictionSchema, setPredictionSchema] = useState(null);
  const [predictionInput, setPredictionInput] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  const [loadingPredict, setLoadingPredict] = useState(false);

  const [businessInsights, setBusinessInsights] = useState(null);

  // AI Assistant Chat States
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [loadingChat, setLoadingChat] = useState(false);
  const [chatError, setChatError] = useState("");
  const chatEndRef = useRef(null);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollTop = chatEndRef.current.scrollHeight;
    }
  }, [chatMessages, loadingChat]);

  // ============================================================
  // RESET ALL WORKFLOW STATE
  // ============================================================
  const resetWorkflow = () => {
    setProfile(null);
    setPreprocessing(null);
    setAnalytics(null);
    setCandidates(null);
    setRankedTargets(null);
    setAiTarget(null);
    setTraining(null);
    setEvaluation(null);
    setFeatureImportance(null);
    setPredictionSchema(null);
    setPredictionInput({});
    setPredictionResult(null);
    setBusinessInsights(null);
    setChatMessages([]);
    setChatInput("");
    setChatError("");
    setError("");
  };

  // ============================================================
  // FILE SELECTION
  // ============================================================
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;
    setFile(selectedFile);
    resetWorkflow();
  };

  // ============================================================
  // UPLOAD & INITIAL EXPLORATION
  // ============================================================
  const handleUpload = async () => {
    if (!file || loading) return;

    setLoading(true);
    resetWorkflow();

    try {
      // 1. Upload
      const formData = new FormData();
      formData.append("file", file);

      const uploadRes = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!uploadRes.ok) {
        const data = await uploadRes.json().catch(() => null);
        throw new Error(data?.detail || "Dataset upload failed.");
      }

      // 2. Profile
      const profileRes = await fetch(`${API_URL}/profile`);
      if (!profileRes.ok) {
        const data = await profileRes.json().catch(() => null);
        throw new Error(data?.detail || "Dataset profiling failed.");
      }
      setProfile(await profileRes.json());

      // 3. Preprocess
      const preprocessRes = await fetch(`${API_URL}/preprocess`);
      if (!preprocessRes.ok) {
        const data = await preprocessRes.json().catch(() => null);
        throw new Error(data?.detail || "Dataset preprocessing failed.");
      }
      setPreprocessing(await preprocessRes.json());

      // 4. Analytics
      const analyticsRes = await fetch(`${API_URL}/analytics`);
      if (!analyticsRes.ok) {
        const data = await analyticsRes.json().catch(() => null);
        throw new Error(data?.detail || "Dataset analytics failed.");
      }
      setAnalytics(await analyticsRes.json());

      // 5. ML Candidates & Ranked Targets
      const candidatesRes = await fetch(`${API_URL}/ml/candidates`);
      if (!candidatesRes.ok) {
        const data = await candidatesRes.json().catch(() => null);
        throw new Error(data?.detail || "Target discovery failed.");
      }
      const cData = await candidatesRes.json();
      setCandidates(cData.target_candidates || []);

      const rankedRes = await fetch(`${API_URL}/ml/targets`);
      if (rankedRes.ok) {
        const rData = await rankedRes.json();
        setRankedTargets(rData.ranked_targets || []);
      }
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to process dataset.");
    } finally {
      setLoading(false);
    }
  };

  // ============================================================
  // AI TARGET SELECTION
  // ============================================================
  const analyzePredictionTarget = async () => {
    if (loadingAI) return;
    setLoadingAI(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/ml/ai-target`);
      if (!response.ok) {
        const data = await response.json().catch(() => null);
        throw new Error(data?.detail || "AI target analysis failed.");
      }

      const result = await response.json();
      if (!result || !result.ai_target_selection) {
        throw new Error("Invalid response from AI engine.");
      }

      setAiTarget(result.ai_target_selection);
    } catch (err) {
      console.error(err);
      setError(err.message || "AI target analysis failed.");
    } finally {
      setLoadingAI(false);
    }
  };

  // ============================================================
  // MODEL TRAINING & EVALUATION
  // ============================================================
  const handleTrainModel = async () => {
    if (loadingTrain) return;
    setLoadingTrain(true);
    setError("");

    try {
      // 1. Train Model
      const trainRes = await fetch(`${API_URL}/ml/train`);
      if (!trainRes.ok) {
        const data = await trainRes.json().catch(() => null);
        throw new Error(data?.detail || "Model training failed.");
      }
      const trainData = await trainRes.json();
      setTraining(trainData.training);

      // 2. Evaluate Model
      const evalRes = await fetch(`${API_URL}/ml/evaluate`);
      if (!evalRes.ok) {
        const data = await evalRes.json().catch(() => null);
        throw new Error(data?.detail || "Model evaluation failed.");
      }
      const evalData = await evalRes.json();
      setEvaluation(evalData.evaluation);

      // 3. Feature Importance
      const featRes = await fetch(`${API_URL}/ml/feature-importance`);
      if (!featRes.ok) {
        const data = await featRes.json().catch(() => null);
        throw new Error(data?.detail || "Feature importance computation failed.");
      }
      const featData = await featRes.json();
      setFeatureImportance(featData.feature_importance);

      // 4. Prediction Schema
      const schemaRes = await fetch(`${API_URL}/ml/prediction-schema`);
      if (!schemaRes.ok) {
        const data = await schemaRes.json().catch(() => null);
        throw new Error(data?.detail || "Prediction schema generation failed.");
      }
      const schemaData = await schemaRes.json();
      setPredictionSchema(schemaData);

      // Pre-fill initial prediction inputs with sensible defaults
      const initialInputs = {};
      (schemaData.features || []).forEach((f) => {
        if (f.type === "numerical") {
          initialInputs[f.column] = f.mean !== undefined ? Math.round(f.mean * 100) / 100 : 0;
        } else if (f.type === "categorical" && f.allowed_values?.length > 0) {
          initialInputs[f.column] = f.allowed_values[0];
        } else if (f.type === "date") {
          initialInputs[f.column] = new Date().toISOString().split("T")[0];
        }
      });
      setPredictionInput(initialInputs);

      // 5. Business Insights
      const insightsRes = await fetch(`${API_URL}/business/insights`);
      if (!insightsRes.ok) {
        const data = await insightsRes.json().catch(() => null);
        throw new Error(data?.detail || "Business insights generation failed.");
      }
      const insightsData = await insightsRes.json();
      setBusinessInsights(insightsData);
    } catch (err) {
      console.error(err);
      setError(err.message || "Model training or evaluation failed.");
    } finally {
      setLoadingTrain(false);
    }
  };

  // ============================================================
  // RUN PREDICTION
  // ============================================================
  const handlePredict = async (e) => {
    e?.preventDefault();
    setLoadingPredict(true);
    setPredictionResult(null);
    setError("");

    try {
      // Cast numeric inputs
      const formattedData = { ...predictionInput };
      if (predictionSchema?.features) {
        predictionSchema.features.forEach((f) => {
          if (f.type === "numerical" && formattedData[f.column] !== undefined) {
            formattedData[f.column] = Number(formattedData[f.column]);
          }
        });
      }

      const response = await fetch(`${API_URL}/ml/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: formattedData }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => null);
        throw new Error(data?.detail || "Prediction request failed.");
      }

      const result = await response.json();
      setPredictionResult(result);
    } catch (err) {
      console.error(err);
      setError(err.message || "Prediction execution failed.");
    } finally {
      setLoadingPredict(false);
    }
  };

  // ============================================================
  // AI ASSISTANT CHAT HANDLER
  // ============================================================
  const handleSendMessage = async (customPrompt) => {
    const text = (typeof customPrompt === "string" ? customPrompt : chatInput).trim();
    if (!text || loadingChat) return;

    setChatError("");
    setChatInput("");
    const newHistory = [...chatMessages, { role: "user", content: text }];
    setChatMessages(newHistory);
    setLoadingChat(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: chatMessages.slice(-6),
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || "Chat request failed.");
      }

      const data = await response.json();
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.response },
      ]);
    } catch (err) {
      console.error(err);
      setChatError(err.message || "Failed to communicate with AI Assistant.");
    } finally {
      setLoadingChat(false);
    }
  };

  return (
    <div className="app">
      {/* NAVBAR */}
      <header className="navbar">
        <div className="brand">
          <div className="brand-icon">
            <Brain size={21} />
          </div>
          <div>
            <h2>BusinessGuide AI</h2>
            <span>Business Decision Support System</span>
          </div>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          System Ready
        </div>
      </header>

      <main className="main-content">
        {/* HERO */}
        <section className="hero">
          <p className="eyebrow">INTELLIGENT BUSINESS ANALYTICS</p>
          <h1>
            Turn your business data
            <br />
            into <span>better decisions.</span>
          </h1>
          <p className="subtitle">
            Upload your business dataset and let BusinessGuide AI analyze,
            profile, predict outcomes, and guide your strategic decisions.
          </p>
        </section>

        {/* WORKFLOW TRACKER */}
        {profile && (
          <div className="workflow-steps">
            <span className={`step-pill ${profile ? "completed" : "active"}`}>
              <CheckCircle2 size={14} /> 1. Upload & Profile
            </span>
            <span className={`step-pill ${preprocessing ? "completed" : "active"}`}>
              <CheckCircle2 size={14} /> 2. Preprocessing
            </span>
            <span className={`step-pill ${candidates?.length ? "completed" : "active"}`}>
              <CheckCircle2 size={14} /> 3. Target Discovery
            </span>
            <span className={`step-pill ${aiTarget ? "completed" : "active"}`}>
              {aiTarget ? <CheckCircle2 size={14} /> : <Target size={14} />} 4. AI Recommendation
            </span>
            <span className={`step-pill ${training ? "completed" : aiTarget ? "active" : ""}`}>
              {training ? <CheckCircle2 size={14} /> : <Gauge size={14} />} 5. ML Pipeline
            </span>
            <span className={`step-pill ${businessInsights ? "completed" : training ? "active" : ""}`}>
              {businessInsights ? <CheckCircle2 size={14} /> : <Lightbulb size={14} />} 6. Insights & Prediction
            </span>
            <span
              className={`step-pill ${chatMessages.length > 0 ? "completed" : "active"}`}
              style={{ cursor: "pointer" }}
              onClick={() => {
                document.getElementById("ai-assistant")?.scrollIntoView({ behavior: "smooth" });
              }}
            >
              <MessageSquare size={14} /> 7. AI Assistant
            </span>
          </div>
        )}

        {/* UPLOAD CARD */}
        <section className="upload-card">
          <div className="upload-icon">
            <Upload size={28} />
          </div>
          <h2>Upload Business Dataset</h2>
          <p>Supports clean CSV or Excel (.xlsx) files with structured business records.</p>

          <div style={{ marginTop: "24px" }}>
            <input
              type="file"
              id="file-upload"
              accept=".csv,.xlsx"
              style={{ display: "none" }}
              onChange={handleFileChange}
            />
            <label htmlFor="file-upload" className="upload-button" style={{ display: "inline-flex", cursor: "pointer" }}>
              <FileSpreadsheet size={18} />
              {file ? file.name : "Choose CSV / Excel File"}
            </label>
          </div>

          {file && (
            <div style={{ marginTop: "18px" }}>
              <button
                className="ai-button"
                onClick={handleUpload}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <Loader2 size={18} className="spin" />
                    Analyzing Dataset...
                  </>
                ) : (
                  <>
                    <Sparkles size={18} />
                    Process & Analyze Dataset
                  </>
                )}
              </button>
            </div>
          )}

          {error && (
            <div style={{ marginTop: "20px", padding: "14px", borderRadius: "10px", background: "#3b151a", color: "#fca5a5", fontSize: "14px", display: "flex", alignItems: "center", gap: "10px", justifyContent: "center" }}>
              <AlertTriangle size={18} />
              <span>{error}</span>
            </div>
          )}
        </section>

        {/* DATASET PROFILE OVERVIEW */}
        {profile && (
          <section className="profile-section">
            <div className="section-heading">
              <p className="eyebrow">DATASET INTELLIGENCE</p>
              <h2>Dataset Profile & Health</h2>
              <p>Comprehensive overview of dataset volume, dimensions, and statistical distributions.</p>
            </div>

            <div className="profile-summary">
              <div className="summary-card">
                <Database size={20} className="summary-icon" />
                <span className="summary-label">Total Rows</span>
                <span className="summary-value">{formatValue(profile.rows)}</span>
              </div>
              <div className="summary-card">
                <Columns3 size={20} className="summary-icon" />
                <span className="summary-label">Columns</span>
                <span className="summary-value">{formatValue(profile.columns)}</span>
              </div>
              <div className="summary-card">
                <AlertTriangle size={20} className="summary-icon" />
                <span className="summary-label">Missing Values</span>
                <span className="summary-value">{formatValue(profile.missing_values_total)}</span>
              </div>
              <div className="summary-card">
                <Copy size={20} className="summary-icon" />
                <span className="summary-label">Duplicate Rows</span>
                <span className="summary-value">{formatValue(profile.duplicate_rows)}</span>
              </div>
            </div>

            {/* PREPROCESSING SUMMARY */}
            {preprocessing && (
              <div className="profile-card" style={{ marginTop: "24px" }}>
                <div className="analytics-header">
                  <CheckCircle2 size={24} style={{ color: "#4ade80" }} />
                  <div>
                    <h3>Automated Preprocessing Health</h3>
                    <p>Cleaning, normalization, date detection, and null imputation results.</p>
                  </div>
                </div>

                <div className="category-grid" style={{ marginTop: "16px" }}>
                  <div className="category-item">
                    <div className="category-name">Cleaned Active Rows</div>
                    <div className="category-count">{formatValue(preprocessing.rows_after_preprocessing)}</div>
                  </div>
                  <div className="category-item">
                    <div className="category-name">Detected Date Columns</div>
                    <div className="category-count">{preprocessing.date_columns?.length || 0}</div>
                  </div>
                  <div className="category-item">
                    <div className="category-name">Duplicates Removed</div>
                    <div className="category-count">{formatValue(preprocessing.duplicates_removed)}</div>
                  </div>
                </div>
              </div>
            )}

            {/* COLUMN TYPES TABLE */}
            <div className="profile-card" style={{ marginTop: "24px" }}>
              <h3>Column Schemas & Profiles</h3>
              <div className="candidates-table-wrap">
                <table className="candidates-table">
                  <thead>
                    <tr>
                      <th>Column Name</th>
                      <th>Data Type</th>
                      <th>Unique Values</th>
                      <th>Missing Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(profile.column_profiles || {}).map(([col, meta]) => (
                      <tr key={col}>
                        <td style={{ fontWeight: 600 }}>{formatColumnName(col)}</td>
                        <td>
                          <span className={`badge-tag ${meta.data_type.includes("int") || meta.data_type.includes("float") ? "badge-blue" : "badge-purple"}`}>
                            {meta.data_type}
                          </span>
                        </td>
                        <td>{formatValue(meta.unique_values)}</td>
                        <td>{formatValue(meta.missing_values)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}

        {/* TARGET DISCOVERY & RANKED TARGETS */}
        {candidates && candidates.length > 0 && (
          <section className="profile-section">
            <div className="section-heading">
              <p className="eyebrow">AUTOMATED TARGET DISCOVERY</p>
              <h2>Detected ML Prediction Targets</h2>
              <p>Candidate outcome columns automatically filtered, validated, and ranked for machine learning.</p>
            </div>

            <div className="profile-card">
              <div className="analytics-header">
                <Target size={24} />
                <div>
                  <h3>Prediction Target Candidates ({candidates.length})</h3>
                  <p>Columns identified with sufficient variance, business relevance, and valid outcome properties.</p>
                </div>
              </div>

              <div className="candidates-table-wrap">
                <table className="candidates-table">
                  <thead>
                    <tr>
                      <th>Candidate Column</th>
                      <th>Recommended ML Problem</th>
                      <th>Data Type</th>
                      <th>Unique Values</th>
                      <th>Missing Ratio</th>
                    </tr>
                  </thead>
                  <tbody>
                    {candidates.map((cand) => (
                      <tr key={cand.column}>
                        <td style={{ fontWeight: 700 }}>{formatColumnName(cand.column)}</td>
                        <td>
                          <span className={`badge-tag ${cand.problem_type === "regression" ? "badge-blue" : "badge-green"}`}>
                            {cand.problem_type}
                          </span>
                        </td>
                        <td><code>{cand.data_type}</code></td>
                        <td>{formatValue(cand.unique_values)}</td>
                        <td>{(cand.missing_ratio * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}

        {/* AI TARGET RECOMMENDATION */}
        {profile && candidates && candidates.length > 0 && (
          <section className="decision-section">
            <div className="section-heading">
              <p className="eyebrow">GROQ LLM DECISION INTELLIGENCE</p>
              <h2>AI Target Recommendation</h2>
              <p>BusinessGuide AI evaluates all candidate business outcomes to select the primary goal.</p>
            </div>

            <div className="decision-card">
              <div className="decision-icon">
                <Brain size={28} />
              </div>

              <div className="decision-content">
                {!aiTarget ? (
                  <div>
                    <h3>Evaluate Prediction Goal</h3>
                    <p>
                      Leverage LLM reasoning with dataset statistics to select the most
                      impactful business outcome and machine learning formulation.
                    </p>
                    <button
                      className="ai-button"
                      onClick={analyzePredictionTarget}
                      disabled={loadingAI}
                    >
                      {loadingAI ? (
                        <>
                          <Loader2 size={18} className="spin" />
                          AI is reasoning...
                        </>
                      ) : (
                        <>
                          <Sparkles size={18} />
                          Analyze Target with AI
                        </>
                      )}
                    </button>
                  </div>
                ) : (
                  <div>
                    <div className="decision-result-header">
                      <div>
                        <span className="result-label">RECOMMENDED BUSINESS TARGET</span>
                        <h3>{formatColumnName(aiTarget.recommended_target)}</h3>
                      </div>
                      <span className="confidence-badge">
                        {Math.round(Number(aiTarget.confidence) * 100)}% Confidence
                      </span>
                    </div>

                    <div className="decision-details">
                      <div>
                        <span>FORMULATION</span>
                        <strong>{aiTarget.problem_type}</strong>
                      </div>
                      <div>
                        <span>BUSINESS RATIONALE</span>
                        <p>{aiTarget.reason}</p>
                      </div>
                    </div>

                    {/* TRAIN MODEL ACTION */}
                    <div style={{ marginTop: "28px", paddingTop: "20px", borderTop: "1px solid #252a35", display: "flex", alignItems: "center", gap: "16px", flexWrap: "wrap" }}>
                      <button
                        className="ai-button"
                        onClick={handleTrainModel}
                        disabled={loadingTrain}
                      >
                        {loadingTrain ? (
                          <>
                            <Loader2 size={18} className="spin" />
                            Training Pipeline...
                          </>
                        ) : (
                          <>
                            <Play size={18} />
                            Train {formatColumnName(aiTarget.recommended_target)} Model
                          </>
                        )}
                      </button>
                      <span style={{ fontSize: "13px", color: "#8f96a4" }}>
                        Builds scikit-learn Random Forest pipeline with automatic encoding & imputation.
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {/* MODEL EVALUATION & FEATURE IMPORTANCE */}
        {training && evaluation && (
          <section className="profile-section">
            <div className="section-heading">
              <p className="eyebrow">MODEL PERFORMANCE & DRIVERS</p>
              <h2>Model Evaluation & Key Drivers</h2>
              <p>Quantitative validation metrics and business feature importances from the trained pipeline.</p>
            </div>

            {/* METRICS */}
            <div className="metrics-grid">
              {evaluation.problem_type === "regression" ? (
                <>
                  <div className="metric-card">
                    <span className="metric-lbl">R² Variance Score</span>
                    <div className="metric-val">{(evaluation.r2_score * 100).toFixed(1)}%</div>
                    <div className="metric-sub">Variance explained</div>
                  </div>
                  <div className="metric-card">
                    <span className="metric-lbl">Mean Absolute Error</span>
                    <div className="metric-val">{formatValue(evaluation.mae)}</div>
                    <div className="metric-sub">Avg prediction error</div>
                  </div>
                  <div className="metric-card">
                    <span className="metric-lbl">Root Mean Sq Error</span>
                    <div className="metric-val">{formatValue(evaluation.rmse)}</div>
                    <div className="metric-sub">Standard deviation error</div>
                  </div>
                </>
              ) : (
                <>
                  <div className="metric-card">
                    <span className="metric-lbl">Model Accuracy</span>
                    <div className="metric-val">{(evaluation.accuracy * 100).toFixed(1)}%</div>
                    <div className="metric-sub">Validation test accuracy</div>
                  </div>
                  {evaluation.f1_score !== undefined && evaluation.f1_score !== null && (
                    <div className="metric-card">
                      <span className="metric-lbl">F1 Score</span>
                      <div className="metric-val">{(evaluation.f1_score * 100).toFixed(1)}%</div>
                      <div className="metric-sub">Weighted harmonic mean</div>
                    </div>
                  )}
                  {evaluation.precision !== undefined && evaluation.precision !== null && (
                    <div className="metric-card">
                      <span className="metric-lbl">Precision</span>
                      <div className="metric-val">{(evaluation.precision * 100).toFixed(1)}%</div>
                      <div className="metric-sub">Positive predictive value</div>
                    </div>
                  )}
                </>
              )}
              <div className="metric-card">
                <span className="metric-lbl">Training / Test Rows</span>
                <div className="metric-val">{evaluation.training_rows} / {evaluation.testing_rows}</div>
                <div className="metric-sub">80/20 train-test split</div>
              </div>
            </div>

            {/* FEATURE IMPORTANCE LIST */}
            {featureImportance && featureImportance.features?.length > 0 && (
              <div className="profile-card" style={{ marginTop: "24px" }}>
                <div className="analytics-header">
                  <TrendingUp size={24} style={{ color: "#8d7cff" }} />
                  <div>
                    <h3>Primary Business Drivers (Feature Importance)</h3>
                    <p>The strongest predictive factors driving {formatColumnName(evaluation.target_column)}.</p>
                  </div>
                </div>

                <div className="importance-list">
                  {featureImportance.features.map((item) => (
                    <div key={item.feature} className="importance-item">
                      <div className="importance-header">
                        <span className="importance-feature-name">
                          {formatColumnName(item.feature.replace(/^numerical__|^categorical__/, ""))}
                        </span>
                        <span className="importance-percentage">{item.percentage}%</span>
                      </div>
                      <div className="importance-bar-bg">
                        <div
                          className="importance-bar-fill"
                          style={{ width: `${Math.max(item.percentage, 2)}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}

        {/* INTERACTIVE WHAT-IF SIMULATOR */}
        {training && predictionSchema && (
          <section className="profile-section">
            <div className="section-heading">
              <p className="eyebrow">DECISION SUPPORT SIMULATOR</p>
              <h2>What-If Scenario Predictor</h2>
              <p>Simulate different business conditions and generate real-time model predictions.</p>
            </div>

            <div className="profile-card">
              <form onSubmit={handlePredict} className="prediction-form">
                <div className="form-grid">
                  {(predictionSchema.features || []).map((f) => (
                    <div key={f.column} className="form-group">
                      <label className="form-label">{formatColumnName(f.column)}</label>

                      {f.type === "numerical" && (
                        <>
                          <input
                            type="number"
                            step="any"
                            className="form-input"
                            value={predictionInput[f.column] ?? ""}
                            onChange={(e) =>
                              setPredictionInput({
                                ...predictionInput,
                                [f.column]: e.target.value,
                              })
                            }
                            required
                          />
                          <span className="form-helper">
                            Range: {formatValue(f.minimum)} - {formatValue(f.maximum)}
                          </span>
                        </>
                      )}

                      {f.type === "categorical" && (
                        <select
                          className="form-select"
                          value={predictionInput[f.column] ?? ""}
                          onChange={(e) =>
                            setPredictionInput({
                              ...predictionInput,
                              [f.column]: e.target.value,
                            })
                          }
                          required
                        >
                          {(f.allowed_values || []).map((val) => (
                            <option key={val} value={val}>
                              {val}
                            </option>
                          ))}
                        </select>
                      )}

                      {f.type === "date" && (
                        <input
                          type="date"
                          className="form-input"
                          value={predictionInput[f.column] ?? ""}
                          onChange={(e) =>
                            setPredictionInput({
                              ...predictionInput,
                              [f.column]: e.target.value,
                            })
                          }
                          required
                        />
                      )}
                    </div>
                  ))}
                </div>

                <div style={{ marginTop: "24px" }}>
                  <button
                    type="submit"
                    className="ai-button"
                    disabled={loadingPredict}
                  >
                    {loadingPredict ? (
                      <>
                        <Loader2 size={18} className="spin" />
                        Calculating Outcome...
                      </>
                    ) : (
                      <>
                        <Sliders size={18} />
                        Run What-If Prediction
                      </>
                    )}
                  </button>
                </div>
              </form>

              {/* PREDICTION RESULT CARD */}
              {predictionResult && (
                <div className="prediction-result-card">
                  <div>
                    <span className="prediction-result-label">
                      PREDICTED {formatColumnName(predictionResult.target).toUpperCase()}
                    </span>
                    <div className="prediction-result-value">
                      {formatValue(predictionResult.prediction)}
                    </div>
                  </div>
                  <span className="confidence-badge">
                    Target: {formatColumnName(predictionResult.target)} ({predictionResult.problem_type})
                  </span>
                </div>
              )}
            </div>
          </section>
        )}

        {/* EXECUTIVE BUSINESS INSIGHTS */}
        {businessInsights && businessInsights.insights?.length > 0 && (
          <section className="profile-section">
            <div className="section-heading">
              <p className="eyebrow">STRATEGIC SYNTHESIS</p>
              <h2>Executive Business Insights</h2>
              <p>Automated conclusions extracted from dataset statistics, model drivers, and validation metrics.</p>
            </div>

            <div className="insights-grid">
              {businessInsights.insights.map((ins, idx) => (
                <div key={idx} className="insight-card">
                  <div className="insight-card-header">
                    <Lightbulb size={20} style={{ color: "#ecc94b" }} />
                    <span className="badge-tag badge-amber">{formatColumnName(ins.type)}</span>
                  </div>
                  <p>{ins.message}</p>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* ============================================================ */}
        {/* BUSINESSGUIDE AI ASSISTANT CHAT                              */}
        {/* ============================================================ */}
        <section id="ai-assistant" className="chat-assistant-section">
          <div className="section-heading">
            <p className="eyebrow">CONVERSATIONAL INTELLIGENCE</p>
            <h2>BusinessGuide AI Assistant</h2>
            <p>
              Ask questions about your uploaded dataset, target candidates, model performance,
              feature drivers, or strategic conclusions.
            </p>
          </div>

          <div className="chat-card">
            {/* CHAT HEADER */}
            <div className="chat-header">
              <div className="chat-header-title">
                <div className="brand-icon" style={{ width: "32px", height: "32px" }}>
                  <Brain size={18} />
                </div>
                <div>
                  <h3>BusinessGuide AI Assistant</h3>
                </div>
                <span className="chat-online-badge">
                  <span className="chat-online-dot"></span>
                  Online & Context-Aware
                </span>
              </div>

              {chatMessages.length > 0 && (
                <button
                  type="button"
                  className="chat-clear-btn"
                  onClick={() => setChatMessages([])}
                  title="Clear conversation history"
                >
                  <Trash2 size={13} />
                  Clear History
                </button>
              )}
            </div>

            {/* CHAT MESSAGES AREA */}
            <div className="chat-messages-area" ref={chatEndRef}>
              {chatMessages.length === 0 ? (
                <div className="chat-welcome-state">
                  <div className="chat-welcome-icon">
                    <Sparkles size={28} />
                  </div>
                  <h4>Ask BusinessGuide AI</h4>
                  <p>
                    {profile
                      ? "I am fully synced with your active dataset, target rankings, and ML models. Ask me anything to unpack your business insights."
                      : "Upload a CSV or Excel dataset above to unlock context-aware explanations, or ask me how BusinessGuide AI operates."}
                  </p>

                  <div className="chat-suggested-title">Suggested Prompts</div>
                  <div className="chat-prompts-grid">
                    {SUGGESTED_PROMPTS.map((promptText, idx) => (
                      <button
                        key={idx}
                        type="button"
                        className="prompt-pill"
                        onClick={() => handleSendMessage(promptText)}
                        disabled={loadingChat}
                      >
                        <Sparkles size={13} style={{ color: "#a855f7" }} />
                        {promptText}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                chatMessages.map((msg, index) => (
                  <div
                    key={index}
                    className={`chat-bubble-container ${msg.role}`}
                  >
                    <div
                      className={`chat-avatar ${
                        msg.role === "user" ? "user-avatar" : "assistant-avatar"
                      }`}
                    >
                      {msg.role === "user" ? <User size={16} /> : <Bot size={16} />}
                    </div>
                    <div
                      className={`chat-bubble ${
                        msg.role === "user" ? "user-bubble" : "assistant-bubble"
                      }`}
                    >
                      {msg.role === "user" ? (
                        msg.content
                      ) : (
                        formatAiMessage(msg.content)
                      )}
                    </div>
                  </div>
                ))
              )}

              {loadingChat && (
                <div className="chat-bubble-container assistant">
                  <div className="chat-avatar assistant-avatar">
                    <Bot size={16} />
                  </div>
                  <div className="chat-loading-bubble">
                    <Loader2 size={16} className="spin" />
                    <span>BusinessGuide AI is analyzing project context...</span>
                  </div>
                </div>
              )}

              {chatError && (
                <div className="chat-error-alert">
                  <AlertTriangle size={16} />
                  <span>{chatError}</span>
                </div>
              )}
            </div>

            {/* CHAT FOOTER & INPUT */}
            <div className="chat-footer">
              {chatMessages.length > 0 && (
                <div className="chat-footer-prompts">
                  <span style={{ fontSize: "11px", color: "#64748b", fontWeight: 600, marginRight: "4px" }}>
                    Quick:
                  </span>
                  {SUGGESTED_PROMPTS.slice(0, 4).map((promptText, idx) => (
                    <button
                      key={idx}
                      type="button"
                      className="prompt-pill-mini"
                      onClick={() => handleSendMessage(promptText)}
                      disabled={loadingChat}
                    >
                      {promptText}
                    </button>
                  ))}
                </div>
              )}

              <form
                className="chat-input-form"
                onSubmit={(e) => {
                  e.preventDefault();
                  handleSendMessage();
                }}
              >
                <input
                  type="text"
                  id="chat-user-input"
                  className="chat-input-field"
                  placeholder="Ask a question about your dataset, targets, or model performance..."
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  disabled={loadingChat}
                />
                <button
                  type="submit"
                  id="chat-send-button"
                  className="chat-send-btn"
                  disabled={loadingChat || !chatInput.trim()}
                >
                  {loadingChat ? (
                    <Loader2 size={16} className="spin" />
                  ) : (
                    <>
                      <span>Send</span>
                      <Send size={15} />
                    </>
                  )}
                </button>
              </form>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
