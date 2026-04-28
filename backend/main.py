from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io
import re
import nltk
from nltk.corpus import stopwords


app = FastAPI(title="AI Resume Analyzer API")

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

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


SKILL_ALIASES = {
    "Python": ["python"],
    "Java": ["java"],
    "C": ["c", "c language"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "c sharp"],
    "Go": ["go", "golang"],
    "PHP": ["php"],
    "Ruby": ["ruby"],
    "R": ["r language"],
    "SQL": ["sql", "structured query language"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MySQL": ["mysql"],
    "MongoDB": ["mongodb", "mongo db"],
    "Redis": ["redis"],
    "Firebase": ["firebase"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "Data Science": ["data science"],
    "Artificial Intelligence": ["artificial intelligence", "ai"],
    "Neural Networks": ["neural networks", "neural network"],
    "NLP": ["nlp", "natural language processing"],
    "FastAPI": ["fastapi", "fast api"],
    "Flask": ["flask"],
    "Django": ["django"],
    "React": ["react", "reactjs", "react.js"],
    "Next.js": ["next.js", "nextjs", "next js"],
    "Angular": ["angular"],
    "Vue.js": ["vue.js", "vuejs", "vue js"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Express.js": ["express.js", "expressjs", "express js"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript", "ts"],
    "HTML": ["html"],
    "CSS": ["css"],
    "Tailwind CSS": ["tailwind", "tailwind css"],
    "Bootstrap": ["bootstrap"],
    "REST API": ["rest api", "restful api", "rest APIs", "api development"],
    "GraphQL": ["graphql"],
    "Microservices": ["microservices", "microservice"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud"],
    "Git": ["git"],
    "GitHub": ["github"],
    "CI/CD": ["ci/cd", "cicd", "continuous integration"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "Scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "OpenCV": ["opencv", "open cv"],
    "Hugging Face": ["hugging face", "transformers"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["excel", "microsoft excel"],
    "Linux": ["linux"],
    "Agile": ["agile", "scrum"],
    "Testing": ["testing", "unit testing", "automation testing"],
    "Jest": ["jest"],
    "Cypress": ["cypress"],
    "Selenium": ["selenium"],
    "Communication": ["communication", "stakeholder communication"],
    "Leadership": ["leadership", "team leadership"],
    "Problem Solving": ["problem solving", "problem-solving"],
}

SKILL_WEIGHTS = {
    "Python": 3,
    "Machine Learning": 3,
    "Deep Learning": 3,
    "Data Science": 3,
    "Artificial Intelligence": 3,
    "React": 2,
    "FastAPI": 2,
    "Django": 2,
    "Flask": 2,
    "Node.js": 2,
    "JavaScript": 2,
    "TypeScript": 2,
    "SQL": 2,
    "AWS": 2,
    "Docker": 2,
    "Kubernetes": 2,
    "Microservices": 2,
}

TECH_TERM_BLOCKLIST = {
    "ability",
    "applicant",
    "applications",
    "benefits",
    "candidate",
    "company",
    "degree",
    "description",
    "development",
    "duties",
    "environment",
    "excellent",
    "experience",
    "familiarity",
    "including",
    "knowledge",
    "preferred",
    "proficient",
    "qualification",
    "requirements",
    "responsibilities",
    "role",
    "skills",
    "team",
    "technologies",
    "tools",
    "understanding",
    "work",
    "years",
}

TECH_CONTEXT_PATTERN = re.compile(
    r"(?:experience with|knowledge of|proficient in|familiar with|skills?:|technologies?:|tools?:|tech stack:|using|work with)\s+([^.;\n]+)",
    re.IGNORECASE,
)

REQUIREMENT_MARKERS = (
    "required",
    "requirement",
    "qualification",
    "must",
    "should",
    "need",
    "experience",
    "years",
    "proficient",
    "knowledge",
    "familiar",
    "ability",
    "hands-on",
    "strong understanding",
    "degree",
    "bachelor",
    "master",
    "certification",
    "responsible for",
    "develop",
    "build",
    "design",
    "implement",
    "manage",
    "communication",
)


def normalize_spaces(text):
    return re.sub(r"\s+", " ", text or "").strip()


def phrase_in_text(text, phrase):
    if not text or not phrase:
        return False

    phrase = phrase.lower()

    if len(phrase) == 1 and phrase.isalnum():
        pattern = r"(?<![a-z0-9+#])" + re.escape(phrase) + r"(?![a-z0-9+#])"
    else:
        pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"

    return re.search(pattern, text.lower()) is not None


def find_phrase_snippet(text, phrase, window=85):
    lower_text = text.lower()
    index = lower_text.find(phrase.lower())

    if index == -1:
        return ""

    start = max(0, index - window)
    end = min(len(text), index + len(phrase) + window)
    return normalize_spaces(text[start:end])


def important_tokens(text):
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]*", (text or "").lower())
    return {
        token
        for token in tokens
        if len(token) > 2 and token not in stop_words
    }


def preprocess_text(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)
    words = text.split()
    return " ".join(word for word in words if word not in stop_words)


def clean_requirement(text):
    text = normalize_spaces(text)
    text = re.sub(r"^[\-\*\u2022\.\d\)\(]+", "", text).strip()
    text = re.sub(
        r"^(requirements?|qualifications?|skills?|responsibilities|must have|nice to have)\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = normalize_spaces(text)

    if len(text) > 180:
        text = text[:180].rsplit(" ", 1)[0] + "..."

    return text


def sentence_case(text):
    text = normalize_spaces(text)
    if not text:
        return text
    return text[0].upper() + text[1:]


def clean_skill_label(text):
    text = normalize_spaces(text)
    text = re.sub(r"^(and|or|with|using|plus)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(and|or|with|using|plus)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9+#./\-\s]", "", text)
    text = normalize_spaces(text)

    if not text:
        return ""

    normalized = text.replace(" js", ".js")
    if normalized.lower() in ("api", "apis", "rest", "software", "systems", "platforms"):
        return ""

    words = normalized.split()
    if len(words) > 4:
        return ""

    if normalized.lower() in TECH_TERM_BLOCKLIST:
        return ""

    return normalized.upper() if normalized.isupper() else normalized.title().replace(".Js", ".js")


def split_skill_terms(text):
    parts = re.split(r",|\s+and\s+|\s+or\s+|/", text or "", flags=re.IGNORECASE)
    return [clean_skill_label(part) for part in parts if clean_skill_label(part)]


def known_skill_labels_lower():
    labels = {skill.lower() for skill in SKILL_ALIASES}

    for aliases in SKILL_ALIASES.values():
        labels.update(alias.lower() for alias in aliases)

    return labels


def extract_dynamic_skills(job_description):
    known_labels = known_skill_labels_lower()
    candidates = []

    for match in TECH_CONTEXT_PATTERN.finditer(job_description or ""):
        candidates.extend(split_skill_terms(match.group(1)))

    candidates.extend(
        clean_skill_label(match.group(0))
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:\.js|JS)\b", job_description or "")
    )
    candidates.extend(
        clean_skill_label(match.group(0))
        for match in re.finditer(r"\b[A-Z][A-Z0-9+#]{1,8}\b", job_description or "")
    )

    dynamic = []
    seen = set()

    for candidate in candidates:
        if not candidate:
            continue

        key = candidate.lower()
        if key in seen or key in known_labels:
            continue

        if len(candidate) < 2:
            continue

        seen.add(key)
        dynamic.append({
            "label": candidate,
            "aliases": [candidate.lower(), candidate.replace(".", "").lower()],
        })

    return dynamic[:18]


def detect_skills_in_text(text):
    detected = []

    for skill, aliases in SKILL_ALIASES.items():
        if any(phrase_in_text(text, alias) for alias in aliases):
            detected.append(skill)

    return detected


def split_text_items(text):
    text = re.sub(r"[\u2022\u25cf\u25aa]", "\n", text or "")
    return [
        clean_requirement(part)
        for part in re.split(r"[\r\n]+|(?<=[.!?])\s+|;\s+", text)
        if clean_requirement(part)
    ]


def classify_requirement(text):
    lower = text.lower()

    if any(term in lower for term in ("degree", "bachelor", "master", "certification")):
        return "Education"

    if re.search(r"\b\d+\+?\s*(years?|yrs?)\b", lower) or "experience" in lower:
        return "Experience"

    if any(phrase_in_text(text, alias) for aliases in SKILL_ALIASES.values() for alias in aliases):
        return "Skill"

    return "Requirement"


def looks_like_requirement(text):
    lower = text.lower()
    word_count = len(text.split())

    if word_count < 3:
        return False

    if any(marker in lower for marker in REQUIREMENT_MARKERS):
        return True

    return text[:1].isupper() and word_count <= 14 and len(detect_skills_in_text(text)) > 0


def requirement_key(text):
    tokens = sorted(important_tokens(text))
    return " ".join(tokens) or text.lower()


def extract_requirement_statements(job_description):
    candidates = []

    for item in split_text_items(job_description):
        if not looks_like_requirement(item):
            continue

        skills_in_item = detect_skills_in_text(item)
        if len(skills_in_item) >= 2 and len(item.split()) <= 14:
            continue

        candidates.append(sentence_case(item))

    unique = []
    seen = set()

    for candidate in candidates:
        key = requirement_key(candidate)
        if key in seen:
            continue

        seen.add(key)
        unique.append(candidate)

    return unique[:12]


def extract_job_requirements(job_description):
    requirements = []
    seen = set()

    for skill in detect_skills_in_text(job_description):
        key = f"skill:{skill.lower()}"
        seen.add(key)
        requirements.append({
            "label": skill,
            "category": "Skill",
            "source": "Mentioned in job description",
            "aliases": SKILL_ALIASES[skill],
        })

    for skill in extract_dynamic_skills(job_description):
        key = f"skill:{skill['label'].lower()}"
        if key in seen:
            continue

        seen.add(key)
        requirements.append({
            "label": skill["label"],
            "category": "Skill",
            "source": "Detected from job description wording",
            "aliases": skill["aliases"],
        })

    for statement in extract_requirement_statements(job_description):
        key = requirement_key(statement)
        if key in seen:
            continue

        seen.add(key)
        requirements.append({
            "label": statement,
            "category": classify_requirement(statement),
            "source": statement,
            "aliases": [],
        })

    # Fallback if no specific requirements were detected
    if not requirements and job_description.strip():
        sentences = split_text_items(job_description)
        for s in sentences:
            if len(s.split()) >= 3:
                key = requirement_key(s)
                if key not in seen:
                    seen.add(key)
                    requirements.append({
                        "label": s,
                        "category": "Requirement",
                        "source": "Extracted from text",
                        "aliases": []
                    })
                if len(requirements) >= 5:
                    break

    # Ultimate fallback if text was too garbled or short
    if not requirements:
        requirements.append({
            "label": "General program fit and experience",
            "category": "Requirement",
            "source": "Default requirement",
            "aliases": []
        })

    return requirements


def build_resume_chunks(text):
    chunks = []

    for item in split_text_items(text):
        if len(item.split()) < 3:
            continue

        chunks.append(item)

    if not chunks and text:
        chunks.append(normalize_spaces(text[:600]))

    return chunks[:80]


def token_overlap_score(requirement, resume_chunk):
    required_tokens = important_tokens(requirement)

    if not required_tokens:
        return 0

    chunk_tokens = important_tokens(resume_chunk)
    return len(required_tokens.intersection(chunk_tokens)) / len(required_tokens)


def tfidf_similarity_matrix(requirements, chunks):
    if not requirements or not chunks:
        return []

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform(requirements + chunks)
        requirement_vectors = matrix[:len(requirements)]
        chunk_vectors = matrix[len(requirements):]
        return cosine_similarity(requirement_vectors, chunk_vectors)
    except ValueError:
        return []


def evaluate_requirements(requirements, resume_text):
    chunks = build_resume_chunks(resume_text)
    labels = [requirement["label"] for requirement in requirements]
    similarities = tfidf_similarity_matrix(labels, chunks)
    evaluated = []

    for index, requirement in enumerate(requirements):
        direct_alias = None
        evidence = ""
        best_similarity = 0
        best_overlap = 0
        best_chunk = ""

        for alias in requirement["aliases"]:
            if phrase_in_text(resume_text, alias):
                direct_alias = alias
                evidence = find_phrase_snippet(resume_text, alias)
                break

        if len(similarities) > index and chunks:
            row = similarities[index]
            best_index = row.argmax()
            best_similarity = float(row[best_index])
            best_chunk = chunks[best_index]
            best_overlap = token_overlap_score(requirement["label"], best_chunk)

        if not evidence:
            evidence = best_chunk

        if requirement["category"] == "Skill":
            matched = direct_alias is not None
        else:
            matched = (
                direct_alias is not None
                or best_overlap >= 0.55
                or best_similarity >= 0.32
                or (best_similarity >= 0.20 and best_overlap >= 0.35)
            )

        confidence = 100 if direct_alias else round(max(best_similarity, best_overlap) * 100, 1)

        evaluated.append({
            "label": requirement["label"],
            "category": requirement["category"],
            "status": "matched" if matched else "missing",
            "confidence": min(confidence, 100),
            "evidence": evidence if matched else "",
            "source": requirement["source"],
        })

    return evaluated


def calculate_overall_similarity(resume, job_desc):
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform([resume, job_desc])
        return cosine_similarity(matrix[0], matrix[1])[0][0]
    except ValueError:
        return 0


def score_requirements(evaluated):
    total_weight = 0
    matched_weight = 0

    for item in evaluated:
        weight = 1.5 if item["category"] == "Skill" else 1
        weight = SKILL_WEIGHTS.get(item["label"], weight)
        total_weight += weight

        if item["status"] == "matched":
            matched_weight += weight

    return (matched_weight / total_weight) * 100 if total_weight else 0


def strength_from_score(score):
    if score >= 75:
        return "Strong Match"
    if score >= 50:
        return "Moderate Match"
    return "Low Match"


def build_recommendations(missing_items):
    if not missing_items:
        return "Your resume covers the main requirements found in the job description."

    missing_labels = [item["label"] for item in missing_items[:6]]
    return (
        "Add clear resume evidence for: "
        + ", ".join(missing_labels)
        + ". Use exact job terms only where they are truthful, and include projects, tools, years, or measurable impact."
    )


def build_improvement_suggestions(missing_items):
    suggestions = []

    for item in missing_items[:5]:
        suggestions.append(
            f"Add a resume bullet that proves {item['label']} with a project, tool, result, or certification."
        )

    if missing_items:
        suggestions.append("Place the strongest matching skills near the top of the resume summary or skills section.")

    return suggestions


def extract_resume_quality_signals(resume_text):
    lower = resume_text.lower()
    sections = {
        "summary": any(term in lower for term in ("summary", "profile", "objective")),
        "skills": "skills" in lower or "technical skills" in lower,
        "experience": any(term in lower for term in ("experience", "work history", "employment")),
        "projects": "projects" in lower or "project" in lower,
        "education": "education" in lower or "degree" in lower,
    }
    metrics = re.findall(r"\b\d+(?:\.\d+)?%|\$[\d,]+|\b\d+\+?\s*(?:users|clients|projects|years|months|apis|models|reports)\b", resume_text, re.IGNORECASE)

    return {
        "sections_found": [section.title() for section, found in sections.items() if found],
        "sections_missing": [section.title() for section, found in sections.items() if not found],
        "has_measurable_impact": len(metrics) > 0,
        "measurable_examples": metrics[:5],
    }


def build_rewrite_suggestions(missing_items, matched_items):
    suggestions = []

    for item in missing_items[:3]:
        if item["category"] == "Skill":
            suggestions.append(
                f"Add one truthful bullet showing where you used {item['label']}, what you built, and the result."
            )
        elif item["category"] == "Experience":
            suggestions.append(
                f"Add a bullet that proves this experience requirement: {item['label']}"
            )
        else:
            suggestions.append(
                f"Make the resume explicitly cover this job requirement: {item['label']}"
            )

    if matched_items:
        suggestions.append(
            f"Move your strongest evidence for {matched_items[0]['label']} closer to the top of the resume."
        )

    return suggestions[:5]


def build_coach_report(score, matched_items, missing_items, matched_skills, missing_skills, resume_text):
    quality = extract_resume_quality_signals(resume_text)
    matched_count = len(matched_items)
    missing_count = len(missing_items)

    if score >= 75:
        headline = "Strong fit with a few targeted edits."
        summary = (
            f"The resume already proves {matched_count} important job requirements. "
            "Focus on making the missing items easier for a recruiter or ATS to find."
        )
    elif score >= 50:
        headline = "Good base, but the resume needs sharper targeting."
        summary = (
            f"The resume matches {matched_count} requirements, but {missing_count} important items are weak or absent. "
            "Add direct evidence for the top gaps before applying."
        )
    else:
        headline = "The resume needs stronger alignment for this role."
        summary = (
            f"The job description asks for several items that are not clearly visible in the resume. "
            "Rewrite the summary, skills, and project bullets around the missing requirements."
        )

    strengths = [
        {
            "title": item["label"],
            "detail": item["evidence"] or "This requirement appears to be covered in the resume.",
        }
        for item in matched_items[:5]
    ]
    gaps = [
        {
            "title": item["label"],
            "detail": item["source"],
        }
        for item in missing_items[:5]
    ]
    rewrite_suggestions = build_rewrite_suggestions(missing_items, matched_items)

    messages = [
        {
            "role": "assistant",
            "title": "AI Coach",
            "content": summary,
        },
        {
            "role": "assistant",
            "title": "Best next move",
            "content": rewrite_suggestions[0] if rewrite_suggestions else "Your resume is already well aligned. Keep the wording specific and measurable.",
        },
    ]

    if missing_skills:
        messages.append({
            "role": "assistant",
            "title": "Skill gap",
            "content": "Missing or unclear skills: " + ", ".join(missing_skills[:6]) + ".",
        })

    if not quality["has_measurable_impact"]:
        messages.append({
            "role": "assistant",
            "title": "Impact signal",
            "content": "Add numbers where possible: users, accuracy, time saved, revenue, projects delivered, or performance improvement.",
        })

    return {
        "headline": headline,
        "summary": summary,
        "strengths": strengths,
        "priority_gaps": gaps,
        "rewrite_suggestions": rewrite_suggestions,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "resume_quality": quality,
        "messages": messages,
    }


def extract_pdf_text(contents):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
    raw_text = ""

    for page in pdf_reader.pages:
        text = page.extract_text()

        if text:
            raw_text += text + "\n"

    return raw_text


@app.get("/")
def root():
    return {"status": "AI Resume Analyzer Running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze-resume/")
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: str = Form(""),
    job_description_file: Optional[UploadFile] = File(None)
):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Resume must be a PDF file")

    if job_description_file and (
        not job_description_file.filename
        or not job_description_file.filename.endswith(".pdf")
    ):
        raise HTTPException(status_code=400, detail="Job description file must be a PDF")

    contents = await file.read()
    raw_text = extract_pdf_text(contents)

    if not raw_text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    job_description_parts = []
    job_description_sources = []

    if job_description and job_description.strip():
        job_description_parts.append(job_description.strip())
        job_description_sources.append("pasted text")

    if job_description_file:
        job_description_contents = await job_description_file.read()
        job_description_pdf_text = extract_pdf_text(job_description_contents)

        if not job_description_pdf_text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from job description PDF"
            )

        job_description_parts.append(job_description_pdf_text)
        job_description_sources.append("PDF upload")

    job_description_text = "\n".join(job_description_parts).strip()

    if not job_description_text:
        raise HTTPException(
            status_code=400,
            detail="Paste a job description or upload a job description PDF"
        )

    requirements = extract_job_requirements(job_description_text)

    if not requirements:
        raise HTTPException(
            status_code=400,
            detail="Could not identify requirements from the job description"
        )

    evaluated_requirements = evaluate_requirements(requirements, raw_text)

    matched_requirements = [
        item for item in evaluated_requirements if item["status"] == "matched"
    ]
    missing_requirements = [
        item for item in evaluated_requirements if item["status"] == "missing"
    ]

    required_skills = [
        item for item in evaluated_requirements if item["category"] == "Skill"
    ]
    matched_skills = [
        item["label"] for item in required_skills if item["status"] == "matched"
    ]
    missing_skills = [
        item["label"] for item in required_skills if item["status"] == "missing"
    ]

    requirement_score = score_requirements(evaluated_requirements)
    similarity = calculate_overall_similarity(
        preprocess_text(raw_text),
        preprocess_text(job_description_text)
    )
    final_score = (requirement_score * 0.8) + (similarity * 100 * 0.2)
    score = round(final_score, 2)

    return {
        "analysis_summary": {
            "ats_score_percentage": score,
            "strength": strength_from_score(score),
            "total_requirements": len(evaluated_requirements),
            "matched_requirements": len(matched_requirements),
            "missing_requirements": len(missing_requirements),
            "overall_similarity_percentage": round(similarity * 100, 2),
            "job_description_source": " + ".join(job_description_sources),
        },
        "requirements_analysis": {
            "required_items": evaluated_requirements,
            "matched_requirements": matched_requirements,
            "missing_requirements": missing_requirements,
        },
        "skills_analysis": {
            "required_skills": [item["label"] for item in required_skills],
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
        },
        "coach_report": build_coach_report(
            score,
            matched_requirements,
            missing_requirements,
            matched_skills,
            missing_skills,
            raw_text,
        ),
        "recommendations": build_recommendations(missing_requirements),
        "improvement_suggestions": build_improvement_suggestions(missing_requirements),
    }
