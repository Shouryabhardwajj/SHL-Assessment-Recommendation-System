import { useState } from "react";
import { recommend } from "./api";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    try {
      const data = await recommend(query);
      setResults(data);
    } catch (err) {
      alert("Failed to fetch recommendations");
    }
    setLoading(false);
  }

  return (
    <div className="page">
      <div className="container">
        <h1>SHL Assessment Recommendation System</h1>

        <form onSubmit={handleSubmit}>
          <textarea
            placeholder="Enter job description or hiring requirement..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
          />
          <button type="submit">
            {loading ? "Finding Assessments..." : "Recommend"}
          </button>
        </form>

        <div className="results">
          {results.map((r, i) => (
            <div className="card" key={i}>
              <a
                href={r.url.startsWith("http") ? r.url : `https://${r.url}`}
                target="_blank"
                rel="noopener noreferrer"
                className="title"
              >
                {r.name}
              </a>

              <p className="desc">{r.description}</p>

              <div className="meta">
                <span>
                  <b>Duration:</b> {r.duration ?? "N/A"} mins
                </span>
                <span>
                  <b>Adaptive:</b> {r.adaptive_support}
                </span>
                <span>
                  <b>Remote:</b> {r.remote_support}
                </span>
              </div>

              <div className="tags">
                {r.test_type.map((t, idx) => (
                  <span className="tag" key={idx}>
                    {t}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
