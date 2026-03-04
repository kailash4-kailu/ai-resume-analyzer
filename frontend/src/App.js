import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {

  const [file, setFile] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {

    if (!file || !jobDescription) {
      alert("Please upload resume and enter job description");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("job_description", jobDescription);

    try {

      const response = await axios.post(
        "http://127.0.0.1:8000/analyze-resume/",
        formData
      );

      setResult(response.data);

    } catch (error) {
      console.error(error);
      alert("Error analyzing resume");
    }
  };

  const score = result?.analysis_summary?.ats_score_percentage || 0;

  let barClass = "progress-low";

  if (score >= 70) {
    barClass = "progress-high";
  }
  else if (score >= 40) {
    barClass = "progress-medium";
  }

  return (
    <div className="container">

      <h1 className="title">AI Resume Analyzer</h1>

      <div className="section">
        <h3>Upload Resume</h3>
        <input
          type="file"
          onChange={(e)=>setFile(e.target.files[0])}
        />
      </div>

      <div className="section">
        <h3>Paste Job Description</h3>

        <textarea
          rows="8"
          placeholder="Paste job description..."
          value={jobDescription}
          onChange={(e)=>setJobDescription(e.target.value)}
        />
      </div>

      <button onClick={handleSubmit}>
        Analyze Resume
      </button>

      {result && (

        <div className="result-box">

          <h2>ATS Score</h2>

          <div className="progress-container">

            <div
              className={`progress-bar ${barClass}`}
              style={{ width: `${score}%` }}
            >
              {score}%
            </div>

          </div>

          <h3>
            Match Strength: {result.analysis_summary.strength}
          </h3>

          <h3>Matched Skills</h3>

          <ul>
            {result.skills_analysis.matched_skills.map((skill,index)=>(
              <li key={index}>{skill}</li>
            ))}
          </ul>

          <h3>Missing Skills</h3>

          <ul>
            {result.skills_analysis.missing_skills.map((skill,index)=>(
              <li key={index}>{skill}</li>
            ))}
          </ul>

          <h3>Recommendation</h3>

          <p>{result.recommendations}</p>

        </div>

      )}

    </div>
  );
}

export default App;