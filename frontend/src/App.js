import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const API_BASE_URL =
  process.env.REACT_APP_API_URL ||
  (window.location.hostname === "localhost"
    ? "http://localhost:8000"
    : "https://ai-resume-analyzer-roal.onrender.com");

const getErrorMessage = (error) => {
  if (error?.response?.data?.detail) {
    return error.response.data.detail;
  }

  if (error?.response?.data?.error) {
    return error.response.data.error;
  }

  if (error?.message) {
    return error.message;
  }

  return "Something went wrong while analyzing the resume.";
};

function App() {
  const [resumeFile, setResumeFile] = useState(null);
  const [jobDescriptionFile, setJobDescriptionFile] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");

  const hasJobDescription = jobDescription.trim() || jobDescriptionFile;

  const handleSubmit = async () => {
    if (!resumeFile || !hasJobDescription) {
      setResult(null);
      setErrorMessage("Please upload a resume and add the job description as text or PDF.");
      return;
    }

    const formData = new FormData();
    formData.append("file", resumeFile);
    formData.append("job_description", jobDescription);

    if (jobDescriptionFile) {
      formData.append("job_description_file", jobDescriptionFile);
    }

    try {
      setLoading(true);
      setResult(null);
      setErrorMessage("");
      setActiveTab("overview");

      const response = await axios.post(
        `${API_BASE_URL}/analyze-resume/`,
        formData,
        {
          timeout: 120000
        }
      );

      if (!response.data?.analysis_summary) {
        throw new Error("The analyzer returned an invalid report. Please try again.");
      }

      setResult(response.data);
    } catch (error) {
      console.error("API ERROR:", error);
      setResult(null);
      setErrorMessage(getErrorMessage(error));
    } finally {
      setLoading(false);
    }
  };

  const summary = result?.analysis_summary;
  const coach = result?.coach_report || {};
  const quality = coach.resume_quality || {};
  const requirements = result?.requirements_analysis?.required_items || [];
  const matchedRequirements = result?.requirements_analysis?.matched_requirements || [];
  const missingRequirements = result?.requirements_analysis?.missing_requirements || [];
  const hasValidReport = Boolean(summary);
  const score = summary?.ats_score_percentage || 0;
  const safeScore = Math.max(0, Math.min(Math.round(score), 100));
  const scoreColor = safeScore >= 70 ? "#15803d" : safeScore >= 45 ? "#b45309" : "#b91c1c";
  const scoreStyle = {
    background: `conic-gradient(${scoreColor} ${safeScore * 3.6}deg, #e2e8f0 0deg)`
  };

  const tabs = [
    { id: "overview", label: "Overview" },
    { id: "requirements", label: "Requirements" },
    { id: "rewrite", label: "Rewrite Plan" }
  ];

  const renderRequirement = (item, index) => (
    <li className={`requirement-row ${item.status}`} key={`${item.label}-${index}`}>
      <div className="requirement-copy">
        <div className="requirement-title-row">
          <span className={`status-marker ${item.status}`} />
          <strong>{item.label}</strong>
        </div>
        <span className="requirement-meta">
          {item.category} / {item.status === "matched" ? "Found" : "Missing"} / {item.confidence}% confidence
        </span>
      </div>

      {item.evidence ? (
        <p className="evidence-text">{item.evidence}</p>
      ) : (
        <p className="gap-text">{item.source}</p>
      )}
    </li>
  );

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <span className="eyebrow">Resume intelligence</span>
          <h1>AI Resume Analyzer</h1>
        </div>
        <div className="topbar-actions">
          <span className={`server-pill ${result ? "ready" : ""}`}>
            {loading ? "Analyzing" : result ? "Report ready" : "Ready"}
          </span>
        </div>
      </header>

      <div className="workspace">
        <aside className="input-panel">
          <div className="panel-title">
            <h2>Analyze Fit</h2>
            <span>PDF to report</span>
          </div>

          <section className="upload-card">
            <label htmlFor="resume-file">Resume PDF</label>
            <input
              id="resume-file"
              type="file"
              accept=".pdf"
              onChange={(event) => setResumeFile(event.target.files[0])}
            />
            <span className={resumeFile ? "file-name selected" : "file-name"}>
              {resumeFile ? resumeFile.name : "No resume selected"}
            </span>
          </section>

          <section className="upload-card">
            <label htmlFor="job-description-file">Job Description PDF</label>
            <input
              id="job-description-file"
              type="file"
              accept=".pdf"
              onChange={(event) => setJobDescriptionFile(event.target.files[0])}
            />
            <span className={jobDescriptionFile ? "file-name selected" : "file-name"}>
              {jobDescriptionFile ? jobDescriptionFile.name : "Optional PDF"}
            </span>
          </section>

          <section className="field-group">
            <div className="label-row">
              <label htmlFor="job-description">Job description text</label>
              <span>Optional</span>
            </div>
            <textarea
              id="job-description"
              rows="12"
              placeholder="Paste the job description here, or upload it as PDF above."
              value={jobDescription}
              onChange={(event) => setJobDescription(event.target.value)}
            />
          </section>

          <button className="primary-action" onClick={handleSubmit} disabled={loading}>
            {loading ? "Analyzing resume..." : "Analyze Resume"}
          </button>
        </aside>

        <section className="report-area">
          {errorMessage && (
            <section className="error-state">
              <span className="eyebrow">Analysis stopped</span>
              <h2>Could not create the report.</h2>
              <p>{errorMessage}</p>
            </section>
          )}

          {!hasValidReport && !errorMessage && (
            <section className="empty-state">
              <div className="empty-card">
                <span className="eyebrow">Waiting for files</span>
                <h2>Upload a resume and add a job description.</h2>
                <p>Use a job description PDF, pasted text, or both.</p>
              </div>
              <div className="empty-grid">
                <div>
                  <span>1</span>
                  <strong>Resume</strong>
                  <p>Upload the candidate PDF.</p>
                </div>
                <div>
                  <span>2</span>
                  <strong>Job description</strong>
                  <p>Upload JD PDF or paste text.</p>
                </div>
                <div>
                  <span>3</span>
                  <strong>Coach report</strong>
                  <p>Review matches, gaps, and rewrite actions.</p>
                </div>
              </div>
            </section>
          )}

          {hasValidReport && (
            <>
              <section className="hero-report">
                <div className="score-ring" style={scoreStyle}>
                  <span>{safeScore}%</span>
                </div>
                <div>
                  <span className="eyebrow">{summary.strength || "Report ready"}</span>
                  <h2>{coach.headline || "Resume analysis complete."}</h2>
                  <p>{coach.summary || "Review the requirement matches and rewrite suggestions below."}</p>
                  <div className="source-line">
                    <span>JD source</span>
                    <strong>{summary.job_description_source || "Text"}</strong>
                  </div>
                </div>
              </section>

              <section className="metric-strip">
                <div>
                  <span>Total needed</span>
                  <strong>{summary.total_requirements ?? 0}</strong>
                </div>
                <div>
                  <span>Found</span>
                  <strong>{summary.matched_requirements ?? 0}</strong>
                </div>
                <div>
                  <span>Missing</span>
                  <strong>{summary.missing_requirements ?? 0}</strong>
                </div>
                <div>
                  <span>Similarity</span>
                  <strong>{summary.overall_similarity_percentage ?? 0}%</strong>
                </div>
              </section>

              <nav className="tabs" aria-label="Report sections">
                {tabs.map((tab) => (
                  <button
                    className={activeTab === tab.id ? "active" : ""}
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    type="button"
                  >
                    {tab.label}
                  </button>
                ))}
              </nav>

              {activeTab === "overview" && (
                <div className="report-grid">
                  <section className="coach-panel">
                    <h3>AI Coach</h3>
                    <div className="chat-thread">
                      {(coach.messages || []).map((message, index) => (
                        <div className="chat-message" key={`${message.title}-${index}`}>
                          <span>{message.title}</span>
                          <p>{message.content}</p>
                        </div>
                      ))}
                    </div>
                  </section>

                  <section className="list-panel">
                    <h3>Strong Evidence</h3>
                    {(coach.strengths || []).length > 0 ? (
                      <ul className="insight-list">
                        {coach.strengths.map((item, index) => (
                          <li key={`${item.title}-strength-${index}`}>
                            <strong>{item.title}</strong>
                            <span>{item.detail}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="muted-text">No strong evidence found yet.</p>
                    )}
                  </section>

                  <section className="list-panel">
                    <h3>Priority Gaps</h3>
                    {(coach.priority_gaps || []).length > 0 ? (
                      <ul className="insight-list gaps">
                        {coach.priority_gaps.map((item, index) => (
                          <li key={`${item.title}-gap-${index}`}>
                            <strong>{item.title}</strong>
                            <span>{item.detail}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="muted-text">No major gaps found.</p>
                    )}
                  </section>

                  <section className="list-panel">
                    <h3>Resume Quality</h3>
                    <div className="quality-grid">
                      <div>
                        <span>Sections found</span>
                        <strong>{(quality.sections_found || []).join(", ") || "None"}</strong>
                      </div>
                      <div>
                        <span>Impact numbers</span>
                        <strong>{quality.has_measurable_impact ? "Present" : "Needs work"}</strong>
                      </div>
                    </div>
                  </section>
                </div>
              )}

              {activeTab === "requirements" && (
                <section className="requirements-panel">
                  <div className="section-heading">
                    <h3>Requirement Check</h3>
                    <span>{matchedRequirements.length} found / {missingRequirements.length} missing</span>
                  </div>
                  <ul className="requirement-list">
                    {requirements.map(renderRequirement)}
                  </ul>
                </section>
              )}

              {activeTab === "rewrite" && (
                <section className="rewrite-panel">
                  <h3>Rewrite Plan</h3>
                  <p className="recommendation">{result.recommendations}</p>
                  <ul className="rewrite-list">
                    {(coach.rewrite_suggestions || result.improvement_suggestions || []).map((suggestion, index) => (
                      <li key={`${suggestion}-${index}`}>{suggestion}</li>
                    ))}
                  </ul>
                </section>
              )}
            </>
          )}
        </section>
      </div>
    </main>
  );
}

export default App;
