import { useMemo, useState } from "react";
import { BrowserRouter, Link, NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { useLocalStorage } from "./hooks/useLocalStorage";
import { useNetworkStatus } from "./hooks/useNetworkStatus";

const FEATURE_FIELDS = [
  { key: "common_ix_count", label: "Common IX Count", type: "number" },
  { key: "rank_diff", label: "AS Rank Difference", type: "number" },
  { key: "distance_km", label: "Approx Distance (km)", type: "number" },
  { key: "customer_cone_overlap", label: "Customer Cone Overlap", type: "number", step: "0.01" }
];

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function buildInitialFeatures() {
  return FEATURE_FIELDS.reduce((acc, field) => {
    acc[field.key] = "";
    return acc;
  }, {});
}

export default function App() {
  const isOnline = useNetworkStatus();
  const [history, setHistory] = useLocalStorage("prediction_history", []);
  const [result, setResult] = useState(() => history[0] || null);

  function handleSaveResult(nextResult) {
    setResult(nextResult);
    setHistory((prev) => [nextResult, ...prev].slice(0, 20));
  }

  return (
    <BrowserRouter>
      <div className="app-shell">
        <header className="hero">
          <p className="kicker">Peering Partner Prediction</p>
          <h1>Predict Link Potential Between Two ISPs</h1>
          <p className="subtext">
            Use your trained project model with custom or auto-filled features, then review outcomes on a dedicated results page.
          </p>
          <span className={`status ${isOnline ? "online" : "offline"}`}>
            {isOnline ? "Online" : "Offline"}
          </span>
          <nav className="top-nav" aria-label="Main navigation">
            <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              Home
            </NavLink>
            <NavLink to="/results" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              Results
            </NavLink>
            <NavLink to="/about" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              About
            </NavLink>
          </nav>
        </header>

        <BreadcrumbBar />

        <AnimatedRoutes isOnline={isOnline} onSaveResult={handleSaveResult} result={result} history={history} />
      </div>
    </BrowserRouter>
  );
}

function BreadcrumbBar() {
  const location = useLocation();

  return (
    <nav className="breadcrumbs" aria-label="Breadcrumb">
      <Link className={`crumb ${location.pathname === "/" ? "active" : ""}`} to="/">
        Home
      </Link>
      <span className="crumb-sep">/</span>
      <Link className={`crumb ${location.pathname === "/results" ? "active" : ""}`} to="/results">
        Results
      </Link>
      <span className="crumb-sep">/</span>
      <Link className={`crumb ${location.pathname === "/about" ? "active" : ""}`} to="/about">
        About
      </Link>
    </nav>
  );
}

function AnimatedRoutes({ isOnline, onSaveResult, result, history }) {
  const location = useLocation();

  return (
    <div className="route-stage" key={location.pathname}>
      <Routes location={location}>
        <Route path="/" element={<HomePage isOnline={isOnline} onSaveResult={onSaveResult} />} />
        <Route path="/results" element={<ResultsPage result={result} history={history} />} />
        <Route path="/about" element={<AboutPage />} />
        <Route
          path="*"
          element={
            <section className="card page-card">
              <h2>Page Not Found</h2>
              <p className="placeholder">The page you requested does not exist.</p>
              <Link to="/" className="text-link">
                Go back to Home
              </Link>
            </section>
          }
        />
      </Routes>
    </div>
  );
}

function HomePage({ isOnline, onSaveResult }) {
  const [ispA, setIspA] = useState("");
  const [ispB, setIspB] = useState("");
  const [modelName, setModelName] = useState("xgboost");
  const [features, setFeatures] = useState(buildInitialFeatures);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const payloadFeatures = useMemo(() => {
    return Object.fromEntries(
      Object.entries(features).map(([key, value]) => [key, value === "" ? null : Number(value)])
    );
  }, [features]);

  function handleFeatureChange(key, value) {
    setFeatures((prev) => ({ ...prev, [key]: value }));
  }

  async function handleAutoFill() {
    if (!ispA.trim() || !ispB.trim()) {
      setError("Enter both ISPs before fetching features.");
      return;
    }

    setError("");
    setLoading(true);

    try {
      const response = await fetch(
        `${API_BASE}/pair-features?ispA=${encodeURIComponent(ispA.trim())}&ispB=${encodeURIComponent(ispB.trim())}`
      );

      if (!response.ok) {
        throw new Error("Pair feature lookup failed.");
      }

      const data = await response.json();
      setFeatures((prev) => ({ ...prev, ...data.features }));
    } catch (err) {
      setError(err.message || "Could not auto-fill features.");
    } finally {
      setLoading(false);
    }
  }

  async function handlePredict(event) {
    event.preventDefault();

    if (!ispA.trim() || !ispB.trim()) {
      setError("Please provide both ISP values.");
      return;
    }

    if (ispA.trim() === ispB.trim()) {
      setError("ISP A and ISP B must be different.");
      return;
    }

    setError("");
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ispA: ispA.trim(),
          ispB: ispB.trim(),
          model: modelName,
          features: payloadFeatures
        })
      });

      if (!response.ok) {
        throw new Error("Prediction request failed.");
      }

      const data = await response.json();
      const nextResult = {
        ispA: ispA.trim(),
        ispB: ispB.trim(),
        model: modelName,
        label: data.label,
        probability: data.probability,
        createdAt: new Date().toISOString(),
        features: payloadFeatures
      };

      onSaveResult(nextResult);
      navigate("/results");
    } catch (err) {
      setError(err.message || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="grid">
      <section className="card wide">
        <h2>Input Pair + Features</h2>
        <form onSubmit={handlePredict}>
          <div className="row two-col">
            <label>
              ISP A / ASN
              <input
                value={ispA}
                onChange={(e) => setIspA(e.target.value)}
                placeholder="Example: AS3356"
              />
            </label>
            <label>
              ISP B / ASN
              <input
                value={ispB}
                onChange={(e) => setIspB(e.target.value)}
                placeholder="Example: AS1299"
              />
            </label>
          </div>

          <div className="row two-col">
            <label>
              Model
              <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
                <option value="xgboost">XGBoost</option>
                <option value="random_forest">Random Forest</option>
                <option value="decision_tree">Decision Tree</option>
              </select>
            </label>
            <div className="actions-inline">
              <button type="button" className="ghost" onClick={handleAutoFill} disabled={loading || !isOnline}>
                Auto-Fill Features
              </button>
            </div>
          </div>

          <div className="feature-grid">
            {FEATURE_FIELDS.map((field) => (
              <label key={field.key}>
                {field.label}
                <input
                  type={field.type}
                  step={field.step || "1"}
                  value={features[field.key]}
                  onChange={(e) => handleFeatureChange(field.key, e.target.value)}
                  placeholder="Enter value"
                />
              </label>
            ))}
          </div>

          {error ? <p className="error">{error}</p> : null}

          <div className="row">
            <button type="submit" className="primary" disabled={loading || !isOnline}>
              {loading ? "Processing..." : "Predict and Open Results"}
            </button>
            <Link className="text-link" to="/results">
              Go to Results Page
            </Link>
          </div>
        </form>
      </section>
    </main>
  );
}

function ResultsPage({ result, history }) {
  const navigate = useNavigate();
  const [error, setError] = useState("");
  const latestResult = result || history[0] || null;

  async function handleCopy() {
    if (!latestResult) return;
    const summary = `${latestResult.ispA} <-> ${latestResult.ispB} | ${latestResult.label} (${Math.round(latestResult.probability * 100)}%)`;

    try {
      await navigator.clipboard.writeText(summary);
    } catch {
      setError("Unable to copy result to clipboard.");
    }
  }

  async function handleShare() {
    if (!latestResult || !navigator.share) return;

    try {
      await navigator.share({
        title: "Peering Prediction",
        text: `${latestResult.ispA} and ${latestResult.ispB}: ${latestResult.label} (${Math.round(latestResult.probability * 100)}%)`
      });
    } catch {
      // User canceled or share unavailable.
    }
  }

  return (
    <main className="grid">
      <section className="card">
        <h2>Prediction Result</h2>
        {!latestResult ? (
          <p className="placeholder">No prediction yet. Go to Home and submit a pair first.</p>
        ) : (
          <>
            <div className="result-block">
              <p>
                <strong>Pair:</strong> {latestResult.ispA} and {latestResult.ispB}
              </p>
              <p>
                <strong>Model:</strong> {latestResult.model}
              </p>
              <p>
                <strong>Label:</strong> {latestResult.label}
              </p>
              <p>
                <strong>Probability:</strong> {Math.round(latestResult.probability * 100)}%
              </p>
              <div className="meter-wrap">
                <div
                  className="meter"
                  style={{ width: `${Math.max(0, Math.min(100, latestResult.probability * 100))}%` }}
                />
              </div>
            </div>

            <div className="row">
              <button type="button" onClick={handleCopy} className="ghost">
                Copy Summary
              </button>
              {navigator.share ? (
                <button type="button" onClick={handleShare} className="ghost">
                  Share
                </button>
              ) : null}
            </div>
          </>
        )}
        {error ? <p className="error">{error}</p> : null}
      </section>

      <aside className="card sticky-summary">
        <h2>Last Predicted Pair</h2>
        {!latestResult ? (
          <p className="placeholder">No recent pair yet.</p>
        ) : (
          <>
            <p>
              <strong>{latestResult.ispA}</strong> and <strong>{latestResult.ispB}</strong>
            </p>
            <p>{latestResult.label}</p>
            <p>{Math.round(latestResult.probability * 100)}% confidence</p>
          </>
        )}
        <div className="row">
          <button type="button" className="ghost" onClick={() => navigate(-1)}>
            Go Back
          </button>
          <Link className="text-link" to="/">
            Home
          </Link>
        </div>
      </aside>

      <section className="card wide">
        <h2>Recent Predictions ({history.length})</h2>
        {history.length === 0 ? (
          <p className="placeholder">No saved predictions in localStorage yet.</p>
        ) : (
          <div className="history-list">
            {history.map((item, index) => (
              <article key={`${item.createdAt}-${index}`} className="history-item">
                <p>
                  <strong>{item.ispA}</strong> and <strong>{item.ispB}</strong>
                </p>
                <p>
                  {item.label} | {Math.round(item.probability * 100)}% | {item.model}
                </p>
              </article>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}

function AboutPage() {
  const navigate = useNavigate();

  return (
    <main className="grid">
      <section className="card wide page-card">
        <h2>About This Project</h2>
        <p>
          This interface predicts peering likelihood between two ISPs using your trained ML models and feature inputs.
        </p>
        <p>
          Workflow: choose ISPs, auto-fill or edit features, run prediction on Home, and review outputs on Results.
        </p>
        <div className="row">
          <button type="button" className="ghost" onClick={() => navigate(-1)}>
            Go Back
          </button>
          <Link className="text-link" to="/">
            Home
          </Link>
          <Link className="text-link" to="/results">
            Results
          </Link>
        </div>
      </section>
    </main>
  );
}
