import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from docx import Document
import os



load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("AI CV ↔ Job Matcher")
st.write("Compare your CV against a job description using NLP embeddings.")

skills_list = [
    # General workplace skills
    "communication", "leadership", "teamwork", "problem solving",
    "time management", "customer service", "project management",
    "stakeholder management", "report writing", "presentation skills",
    "attention to detail", "organisation", "training", "sales",
    "negotiation", "administration", "research", "planning",

    # Business and office skills
    "microsoft office", "excel", "word", "powerpoint", "outlook",
    "data entry", "crm", "reporting", "budgeting", "invoicing",
    "scheduling", "documentation", "compliance",

    # Data / tech skills
    "python", "sql", "data analysis", "data cleaning",
    "power bi", "tableau", "machine learning", "ai",
    "statistics", "dashboard", "database", "api",

    # Marketing
    "digital marketing", "seo", "social media", "content creation",
    "email marketing", "campaign management", "copywriting",

    # Finance
    "accounting", "bookkeeping", "payroll", "financial analysis",
    "reconciliation", "forecasting", "audit",

    # Healthcare / care
    "patient care", "safeguarding", "care planning",
    "risk assessment", "health and safety",

    # Education
    "teaching", "lesson planning", "curriculum", "mentoring",
    "tutoring", "classroom management",

    # Retail / hospitality
    "cash handling", "inventory management", "stock control",
    "food safety", "front of house", "hospitality"
]

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text

def extract_text_from_docx(uploaded_file):
    document = Document(uploaded_file)
    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text

def find_skills(text):
    text = text.lower()
    found_skills = []

    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)

    return found_skills

def generate_llm_suggestions(
    score,
    matched_skills,
    missing_skills,
    cv_text,
    job_text
):
    prompt = f"""
You are an expert UK career coach helping candidates across all industries.

Analyse this CV against the job description.

Match Score: {score}%

Matched Skills:
{matched_skills}

Missing Skills:
{missing_skills}

CV:
{cv_text[:3000]}

Job Description:
{job_text[:3000]}

Provide:
1. Overall assessment
2. Missing skills analysis
3. Practical CV improvements
4. Three rewritten stronger CV bullet points
5. Interview improvement advice

Keep advice concise and practical.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text

st.subheader("Upload CV")
uploaded_cv = st.file_uploader("Upload your CV as PDF or Word document", type=["pdf", "docx"])

cv_text_manual = st.text_area("Or paste your CV text here", height=200)

job_text = st.text_area("Paste the job description here", height=250)

if uploaded_cv is not None:

    if uploaded_cv.name.endswith(".pdf"):
        cv_text = extract_text_from_pdf(uploaded_cv)

    elif uploaded_cv.name.endswith(".docx"):
        cv_text = extract_text_from_docx(uploaded_cv)

    else:
        cv_text = ""

    st.success("CV text extracted successfully.")

else:
    cv_text = cv_text_manual

st.write("Extracted CV length:", len(cv_text))

if st.button("Analyse Match"):
    if cv_text.strip() == "" or job_text.strip() == "":
        st.warning("Please upload/paste your CV and paste the job description.")
    else:
        cv_embedding = model.encode([cv_text])
        job_embedding = model.encode([job_text])

        similarity = cosine_similarity(cv_embedding, job_embedding)[0][0]
        score = round(similarity * 100, 2)

        cv_skills = find_skills(cv_text)
        job_skills = find_skills(job_text)

        matched_skills = sorted(list(set(cv_skills).intersection(set(job_skills))))
        missing_skills = sorted(list(set(job_skills).difference(set(cv_skills))))

        st.subheader("Match Score")
        st.write(f"{score}%")

        if score >= 75:
            st.success("Strong match")
        elif score >= 50:
            st.info("Moderate match")
        else:
            st.warning("Weak match")

        st.subheader("Matched Skills")
        st.write(", ".join(matched_skills) if matched_skills else "No matched skills found.")

        st.subheader("Missing Skills")
        st.write(", ".join(missing_skills) if missing_skills else "No major missing skills found.")

        st.subheader("LLM CV Improvement Suggestions")
        suggestions = generate_llm_suggestions(
    score,
    matched_skills,
    missing_skills,
    cv_text,
    job_text
)

        
        st.markdown(suggestions)