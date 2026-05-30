import streamlit as st
import PyPDF2
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Screening System")
st.write("Upload a resume and compare it with a job description.")


SKILLS = [
    "python", "java", "c", "c++", "html", "css", "javascript",
    "sql", "mysql", "mongodb", "machine learning", "deep learning",
    "artificial intelligence", "data science", "pandas", "numpy",
    "scikit-learn", "tensorflow", "keras", "pytorch", "opencv",
    "nlp", "natural language processing", "streamlit", "flask",
    "django", "git", "github", "excel", "power bi", "tableau",
    "communication", "problem solving", "leadership", "teamwork",
    "data analysis", "matplotlib", "seaborn", "api", "rest api",
    "cloud", "aws", "azure", "docker"
]


def extract_text_from_pdf(uploaded_file):
    text = ""

    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

    except Exception as e:
        st.error("Error reading PDF file.")
        st.write(e)

    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s+#.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def calculate_match_score(resume_text, job_description):
    documents = [resume_text, job_description]

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    return round(similarity * 100, 2)


def find_skills(text):
    found_skills = []

    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text.lower()):
            found_skills.append(skill)

    return sorted(list(set(found_skills)))


def detect_sections(text):
    sections = {
        "Education": ["education", "degree", "b.tech", "bachelor", "university", "college"],
        "Skills": ["skills", "technical skills", "technologies"],
        "Projects": ["projects", "academic projects", "mini project", "major project"],
        "Experience": ["experience", "internship", "work experience"],
        "Certifications": ["certifications", "certificate", "courses"],
        "Achievements": ["achievements", "awards", "honors"]
    }

    detected = {}

    for section, keywords in sections.items():
        detected[section] = any(keyword in text.lower() for keyword in keywords)

    return detected


def generate_suggestions(score, missing_skills, sections):
    suggestions = []

    if score < 50:
        suggestions.append("Your resume match is low. Add more keywords from the job description.")
    elif score < 75:
        suggestions.append("Your resume is average for this role. Improve project descriptions and technical skills.")
    else:
        suggestions.append("Your resume is a strong match for this role.")

    if missing_skills:
        suggestions.append("Add missing skills if you know them: " + ", ".join(missing_skills))

    for section, present in sections.items():
        if not present:
            suggestions.append(f"Add a clear {section} section to your resume.")

    suggestions.append("Use action words like Built, Developed, Created, Improved, Designed, and Implemented.")
    suggestions.append("Add measurable results wherever possible, like accuracy, users, time saved, or performance improvement.")

    return suggestions


def calculate_ats_score(match_score, matched_skills, sections):
    score = 0

    score += match_score * 0.5
    score += min(len(matched_skills) * 5, 25)

    section_score = sum(1 for value in sections.values() if value)
    score += section_score * 4

    return min(round(score, 2), 100)


resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

job_description = st.text_area(
    "Paste Job Description",
    height=220,
    placeholder="Paste internship or job description here..."
)

analyze = st.button("Analyze Resume")

if analyze:
    if resume_file is None:
        st.error("Please upload your resume PDF.")
    elif job_description.strip() == "":
        st.error("Please paste the job description.")
    else:
        resume_text = extract_text_from_pdf(resume_file)

        if resume_text.strip() == "":
            st.error("Could not extract text from the PDF. Try another resume PDF.")
        else:
            cleaned_resume = clean_text(resume_text)
            cleaned_jd = clean_text(job_description)

            match_score = calculate_match_score(cleaned_resume, cleaned_jd)

            resume_skills = find_skills(cleaned_resume)
            jd_skills = find_skills(cleaned_jd)

            matched_skills = sorted(list(set(resume_skills) & set(jd_skills)))
            missing_skills = sorted(list(set(jd_skills) - set(resume_skills)))

            sections = detect_sections(cleaned_resume)
            ats_score = calculate_ats_score(match_score, matched_skills, sections)

            suggestions = generate_suggestions(match_score, missing_skills, sections)

            st.subheader("📊 Resume Analysis Result")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Match Score", f"{match_score}%")

            with col2:
                st.metric("ATS Score", f"{ats_score}%")

            with col3:
                st.metric("Matched Skills", len(matched_skills))

            with col4:
                st.metric("Missing Skills", len(missing_skills))

            st.progress(int(ats_score))

            st.subheader("✅ Matched Skills")
            if matched_skills:
                st.success(", ".join(matched_skills))
            else:
                st.warning("No matched skills found.")

            st.subheader("❌ Missing Skills")
            if missing_skills:
                st.error(", ".join(missing_skills))
            else:
                st.success("No major missing skills found.")

            st.subheader("📌 Resume Sections Detected")

            section_data = []
            for section, present in sections.items():
                section_data.append({
                    "Section": section,
                    "Status": "Present" if present else "Missing"
                })

            section_df = pd.DataFrame(section_data)
            st.table(section_df)

            st.subheader("💡 Suggestions to Improve Resume")
            for suggestion in suggestions:
                st.write("•", suggestion)

            st.subheader("📄 Resume Text Preview")
            with st.expander("View extracted resume text"):
                st.write(resume_text[:3000])

            report_data = {
                "Match Score": [match_score],
                "ATS Score": [ats_score],
                "Matched Skills": [", ".join(matched_skills)],
                "Missing Skills": [", ".join(missing_skills)],
                "Suggestions": [" | ".join(suggestions)]
            }

            report_df = pd.DataFrame(report_data)

            csv = report_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Report as CSV",
                data=csv,
                file_name="resume_screening_report.csv",
                mime="text/csv"
            )