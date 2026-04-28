# 🚀 AI Resume Analyzer

Ever wonder why great candidates get rejected before a human even reads their resume? It’s usually because of the **Applicant Tracking System (ATS)**. These automated systems filter out resumes that don't match the job description's keywords.

This project solves that problem. It's a smart tool that compares your resume against a job description, scores your chances, and gives you actionable advice on exactly what to fix to pass the robot filters.

## ✨ What It Does

- **Upload your Resume**: Drop your PDF resume into the app.
- **Provide the Job Description**: Paste the text or upload the job brochure PDF.
- **Get your ATS Score**: Instantly see your match percentage based on skills and core requirements.
- **Find Missing Skills**: Discover exactly which keywords the job asks for that your resume completely missed.
- **Smart Fallback Extraction**: Even if the job description is just general text without obvious "tech keywords," our backend will still extract the core expectations so the analysis never fails.
- **Actionable AI Coach**: Get a step-by-step rewrite plan telling you which bullet points to add or change.

## 🎨 Modern & Premium Design

The application features a gorgeous, state-of-the-art **Dark Glassmorphism UI** with:
- Deep gradient backgrounds
- Soft frosted glass panels
- Interactive hover effects and smooth animations
- Modern typography using the **Inter** font

## 🛠 What's Under the Hood?

- **Backend**: Python & FastAPI (for lightning-fast API responses).
- **Text Analysis (NLP)**: Scikit-learn (TF-IDF Vectorization to measure similarity) and NLTK (for natural language text processing).
- **Frontend**: ReactJS (with a custom, premium CSS architecture).
- **Deployment Ready**: Configured for Render (Backend) & Vercel (Frontend).

## 🚀 How to Run Locally

### Start the Backend (API)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
*(The backend will run on http://localhost:8000)*

### Start the Frontend (User Interface)
```bash
cd frontend
npm install
npm start
```
*(The frontend will automatically open on http://localhost:3000)*
