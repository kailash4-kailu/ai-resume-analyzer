from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io
import re
import nltk
from nltk.corpus import stopwords


# -----------------------------------
# Initialize FastAPI
# -----------------------------------
app = FastAPI(title="AI Resume Analyzer API")

# -----------------------------------
# CORS Configuration
# -----------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://ai-resume-analyzer-flax-tau.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# NLTK Setup (Render Safe)
# -----------------------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


# -----------------------------------
# Known Skills Database
# -----------------------------------
KNOWN_SKILLS = {
    "python", "java", "c", "c++", "mysql", "sql", "mongodb",
    "machine_learning", "deep_learning",
    "ai", "neural_networks",
    "fastapi", "flask", "django",
    "react", "javascript",
    "html", "css", "docker", "aws", "git",
    "pandas", "numpy", "tensorflow", "pytorch",
    "scikit_learn", "data_science"
}


# -----------------------------------
# Text Preprocessing
# -----------------------------------
def preprocess_text(text):

    text = text.lower()

    phrase_mappings = {
        "machine learning": "machine_learning",
        "deep learning": "deep_learning",
        "neural networks": "neural_networks",
        "scikit learn": "scikit_learn",
        "data science": "data_science"
    }

    for phrase, token in phrase_mappings.items():
        text = text.replace(phrase, token)

    text = re.sub(r'[^a-zA-Z_\s]', '', text)

    words = text.split()

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(filtered_words)


# -----------------------------------
# Health Check
# -----------------------------------
@app.get("/")
def root():
    return {"status": "AI Resume Analyzer API Running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# -----------------------------------
# Upload Resume Endpoint
# -----------------------------------
@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files allowed"}

    try:
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
    except Exception:
        return {"error": "Invalid or corrupted PDF"}

    raw_text = ""

    for page in pdf_reader.pages:
        raw_text += page.extract_text() or ""

    cleaned_text = preprocess_text(raw_text)

    return {
        "filename": file.filename,
        "cleaned_text_preview": cleaned_text[:1000]
    }


# -----------------------------------
# Resume Match Endpoint
# -----------------------------------
class ResumeRequest(BaseModel):
    resume_text: str
    job_description: str


@app.post("/match-resume/")
def match_resume(data: ResumeRequest):

    resume = preprocess_text(data.resume_text)
    job_desc = preprocess_text(data.job_description)

    resume_words = {skill for skill in KNOWN_SKILLS if skill in resume}
    job_words = {skill for skill in KNOWN_SKILLS if skill in job_desc}

    matched_skills = list(resume_words.intersection(job_words))
    missing_skills = list(job_words - resume_words)

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform([
        resume,
        job_desc
    ])

    similarity_score = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    return {
        "similarity_score": round(float(similarity_score) * 100, 2),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }


# -----------------------------------
# Full Resume Analysis
# -----------------------------------
@app.post("/analyze-resume/")
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):

    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files allowed"}

    try:
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
    except Exception:
        return {"error": "Invalid or corrupted PDF"}

    raw_text = ""

    for page in pdf_reader.pages:
        raw_text += page.extract_text() or ""

    resume = preprocess_text(raw_text)
    job_desc = preprocess_text(job_description)

    resume_words = {skill for skill in KNOWN_SKILLS if skill in resume}
    job_words = {skill for skill in KNOWN_SKILLS if skill in job_desc}

    matched_skills = list(resume_words.intersection(job_words))
    missing_skills = list(job_words - resume_words)

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform([
        resume,
        job_desc
    ])

    similarity_score = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    score = round(float(similarity_score) * 100, 2)

    if score >= 70:
        strength = "Strong Match"
    elif score >= 40:
        strength = "Moderate Match"
    else:
        strength = "Low Match"

    matched_display = [s.replace("_", " ") for s in matched_skills]
    missing_display = [s.replace("_", " ") for s in missing_skills]

    recommendation = (
        "Consider adding missing skills to improve ATS compatibility."
        if missing_display
        else "Your resume aligns well with this job description."
    )

    return {
        "analysis_summary": {
            "ats_score_percentage": score,
            "strength": strength
        },
        "skills_analysis": {
            "matched_skills": matched_display,
            "missing_skills": missing_display
        },
        "recommendations": recommendation
    }