import streamlit as st # type: ignore
import pdfplumber # type: ignore
import docx # type: ignore
import re
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from io import BytesIO
import tempfile
import os
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
from plotly.subplots import make_subplots # type: ignore

# Page configuration
st.set_page_config(
    page_title="JobFit Analyzer - Resume to Role Matching",
    page_icon="🎯",
    layout="wide"
)

# Pre-defined job role templates with detailed requirements
JOB_ROLE_TEMPLATES = {
    # AI/ML Roles
    "machine learning engineer": {
        "description": "Machine Learning Engineer with deep learning and model deployment experience",
        "required_skills": ["python", "tensorflow", "pytorch", "machine learning", "data preprocessing", "model deployment", "statistics"],
        "responsibilities": "Develop ML models, optimize algorithms, deploy to production, monitor performance, create data pipelines",
        "preferred": ["mlops", "cloud ai", "computer vision", "nlp", "kubernetes", "docker", "aws sage maker"]
    },
    
    "data scientist": {
        "description": "Data Scientist with strong analytical skills and machine learning expertise",
        "required_skills": ["python", "sql", "machine learning", "statistics", "pandas", "numpy", "data visualization", "r"],
        "responsibilities": "Analyze datasets, build predictive models, create data pipelines, generate insights, A/B testing",
        "preferred": ["tensorflow", "pytorch", "spark", "tableau", "power bi", "hadoop"]
    },
    
    "ai research scientist": {
        "description": "AI Research Scientist focused on cutting-edge AI algorithms",
        "required_skills": ["python", "machine learning", "deep learning", "research", "algorithm development", "mathematics"],
        "responsibilities": "Research new AI algorithms, publish papers, prototype solutions, collaborate with engineering teams",
        "preferred": ["phd", "research publications", "reinforcement learning", "computer vision", "nlp"]
    },
    
    "computer vision engineer": {
        "description": "Computer Vision Engineer specializing in image and video processing",
        "required_skills": ["python", "opencv", "tensorflow", "pytorch", "computer vision", "image processing", "deep learning"],
        "responsibilities": "Develop CV algorithms, implement object detection, work with video streams, optimize models",
        "preferred": ["c++", "yolo", "cnn", "gan", "azure cognitive services"]
    },
    
    "nlp engineer": {
        "description": "NLP Engineer with expertise in natural language processing",
        "required_skills": ["python", "nlp", "transformers", "spacy", "nltk", "machine learning", "text processing"],
        "responsibilities": "Develop NLP models, work with text data, implement chatbots, sentiment analysis, text generation",
        "preferred": ["bert", "gpt", "huggingface", "cloud nlp apis", "multilingual processing"]
    },

    # Software Development Roles
    "python developer": {
        "description": "Python Developer with expertise in web development and backend services",
        "required_skills": ["python", "django", "flask", "rest api", "sql", "git", "javascript", "html", "css"],
        "responsibilities": "Develop backend services, create APIs, implement database solutions, collaborate with frontend teams",
        "preferred": ["aws", "docker", "react", "unit testing", "postgresql", "fastapi"]
    },
    
    "java developer": {
        "description": "Java Developer with enterprise application experience",
        "required_skills": ["java", "spring boot", "hibernate", "rest api", "sql", "maven", "git", "microservices"],
        "responsibilities": "Develop Java applications, work with Spring framework, implement microservices, database integration",
        "preferred": ["aws", "docker", "kubernetes", "jenkins", "react", "angular"]
    },
    
    "frontend developer": {
        "description": "Frontend Developer specializing in modern web applications",
        "required_skills": ["javascript", "react", "html5", "css3", "typescript", "git", "responsive design"],
        "responsibilities": "Develop user interfaces, implement designs, optimize performance, ensure cross-browser compatibility",
        "preferred": ["vue.js", "angular", "next.js", "webpack", "sass", "redux"]
    },
    
    "backend developer": {
        "description": "Backend Developer focused on server-side logic and APIs",
        "required_skills": ["python", "java", "node.js", "rest api", "sql", "nosql", "git", "database design"],
        "responsibilities": "Develop server logic, create APIs, database management, performance optimization, security",
        "preferred": ["aws", "docker", "kubernetes", "graphql", "redis", "microservices"]
    },
    
    "fullstack developer": {
        "description": "Fullstack Developer proficient in both frontend and backend technologies",
        "required_skills": ["javascript", "react", "node.js", "python", "sql", "rest api", "git", "html", "css"],
        "responsibilities": "End-to-end web development, database design, API development, frontend implementation",
        "preferred": ["typescript", "aws", "docker", "mongodb", "express.js", "next.js"]
    },
    
    "mobile developer": {
        "description": "Mobile Developer specializing in iOS/Android applications",
        "required_skills": ["swift", "kotlin", "java", "mobile development", "rest api", "git", "ui/ux"],
        "responsibilities": "Develop mobile apps, implement designs, work with APIs, app store deployment",
        "preferred": ["react native", "flutter", "firebase", "ci/cd mobile", "push notifications"]
    },
    
    "game developer": {
        "description": "Game Developer with experience in game engines and graphics",
        "required_skills": ["c++", "c#", "unity", "unreal engine", "game development", "3d graphics", "physics"],
        "responsibilities": "Develop games, implement game mechanics, work with game engines, optimize performance",
        "preferred": ["opengl", "directx", "vr/ar", "multiplayer", "shader programming"]
    },
    
    "embedded systems engineer": {
        "description": "Embedded Systems Engineer with hardware-software integration experience",
        "required_skills": ["c", "c++", "embedded systems", "microcontrollers", "rtos", "hardware", "python"],
        "responsibilities": "Develop embedded software, work with hardware, firmware development, low-level programming",
        "preferred": ["arm", "raspberry pi", "arduino", "iot", "device drivers"]
    },

    # DevOps & Cloud Roles
    "devops engineer": {
        "description": "DevOps Engineer with cloud infrastructure and automation experience",
        "required_skills": ["aws", "azure", "docker", "kubernetes", "ci/cd", "linux", "bash", "terraform"],
        "responsibilities": "Manage cloud infrastructure, automate deployments, monitor systems, ensure security, optimize performance",
        "preferred": ["jenkins", "ansible", "prometheus", "grafana", "python", "serverless"]
    },
    
    "cloud engineer": {
        "description": "Cloud Engineer specializing in cloud infrastructure and services",
        "required_skills": ["aws", "azure", "gcp", "cloud infrastructure", "linux", "networking", "security"],
        "responsibilities": "Design cloud architecture, implement cloud solutions, optimize costs, ensure compliance",
        "preferred": ["terraform", "kubernetes", "docker", "python", "serverless", "cloud formation"]
    },
    
    "site reliability engineer": {
        "description": "SRE focused on system reliability and performance",
        "required_skills": ["linux", "monitoring", "automation", "python", "bash", "cloud", "incident response"],
        "responsibilities": "Ensure system reliability, automate operations, monitor performance, handle incidents",
        "preferred": ["prometheus", "grafana", "terraform", "kubernetes", "aws", "google cloud"]
    },
    
    "systems administrator": {
        "description": "Systems Administrator with server and network management experience",
        "required_skills": ["linux", "windows server", "networking", "bash", "powershell", "virtualization", "security"],
        "responsibilities": "Manage servers, maintain networks, ensure security, handle backups, user management",
        "preferred": ["aws", "azure", "docker", "ansible", "monitoring tools"]
    },

    # Data & Analytics Roles
    "data engineer": {
        "description": "Data Engineer with big data and pipeline experience",
        "required_skills": ["python", "sql", "big data", "etl", "spark", "hadoop", "data warehousing"],
        "responsibilities": "Build data pipelines, manage data infrastructure, ETL processes, data quality",
        "preferred": ["aws", "azure", "airflow", "kafka", "snowflake", "redshift"]
    },
    
    "data analyst": {
        "description": "Data Analyst with business intelligence and analytics skills",
        "required_skills": ["sql", "excel", "python", "data visualization", "statistics", "reporting", "tableau"],
        "responsibilities": "Analyze data, create reports, generate insights, business intelligence, dashboard creation",
        "preferred": ["power bi", "r", "bigquery", "looker", "data studio"]
    },
    
    "business intelligence developer": {
        "description": "BI Developer focused on data visualization and reporting",
        "required_skills": ["sql", "tableau", "power bi", "data warehousing", "etl", "dashboard development"],
        "responsibilities": "Develop BI solutions, create dashboards, data modeling, business reporting",
        "preferred": ["python", "aws", "azure", "looker", "qlik sense"]
    },
    
    "database administrator": {
        "description": "Database Administrator with database management expertise",
        "required_skills": ["sql", "database administration", "postgresql", "mysql", "oracle", "performance tuning", "backup"],
        "responsibilities": "Manage databases, optimize performance, ensure security, handle backups, data modeling",
        "preferred": ["aws rds", "azure sql", "mongodb", "redis", "cloud databases"]
    },

    # Security Roles
    "cybersecurity engineer": {
        "description": "Cybersecurity Engineer with security infrastructure experience",
        "required_skills": ["network security", "cybersecurity", "penetration testing", "firewalls", "linux", "python"],
        "responsibilities": "Implement security measures, conduct security audits, vulnerability assessment, incident response",
        "preferred": ["aws security", "azure security", "soc", "siem", "ceh", "cissp"]
    },
    
    "security analyst": {
        "description": "Security Analyst focused on threat detection and response",
        "required_skills": ["security monitoring", "siem", "incident response", "threat intelligence", "network security"],
        "responsibilities": "Monitor security events, investigate incidents, threat hunting, security reporting",
        "preferred": ["splunk", "wireshark", "python", "cloud security", "compliance"]
    },
    
    "penetration tester": {
        "description": "Penetration Tester with ethical hacking experience",
        "required_skills": ["penetration testing", "ethical hacking", "network security", "vulnerability assessment", "linux"],
        "responsibilities": "Conduct penetration tests, vulnerability assessments, security audits, report findings",
        "preferred": ["oscp", "ceh", "metasploit", "burp suite", "web application security"]
    },

    # QA & Testing Roles
    "qa engineer": {
        "description": "QA Engineer with software testing expertise",
        "required_skills": ["test automation", "selenium", "python", "java", "testing", "qa methodologies", "jira"],
        "responsibilities": "Create test plans, automate tests, bug tracking, regression testing, quality assurance",
        "preferred": ["cypress", "postman", "jenkins", "appium", "performance testing"]
    },
    
    "test automation engineer": {
        "description": "Test Automation Engineer focused on automated testing frameworks",
        "required_skills": ["test automation", "selenium", "python", "java", "ci/cd", "framework development"],
        "responsibilities": "Develop test automation frameworks, write automated tests, integrate with CI/CD",
        "preferred": ["cypress", "playwright", "docker", "aws", "performance testing"]
    },
    
    "performance engineer": {
        "description": "Performance Engineer with system performance optimization experience",
        "required_skills": ["performance testing", "load testing", "jmeter", "python", "monitoring", "optimization"],
        "responsibilities": "Conduct performance tests, identify bottlenecks, optimize performance, capacity planning",
        "preferred": ["gatling", "new relic", "datadog", "cloud monitoring", "apm tools"]
    },

    # Management & Leadership Roles
    "engineering manager": {
        "description": "Engineering Manager with technical leadership experience",
        "required_skills": ["leadership", "project management", "agile", "technical architecture", "team management"],
        "responsibilities": "Lead engineering teams, project management, technical decisions, hiring, mentoring",
        "preferred": ["aws", "cloud architecture", "python", "java", "budget management", "strategic planning"]
    },
    
    "technical lead": {
        "description": "Technical Lead with architecture and team guidance experience",
        "required_skills": ["technical leadership", "architecture", "code review", "mentoring", "agile", "system design"],
        "responsibilities": "Technical guidance, architecture decisions, code reviews, mentor developers, technical planning",
        "preferred": ["aws", "microservices", "python", "java", "devops", "cloud native"]
    },
    
    "product manager": {
        "description": "Product Manager with product development experience",
        "required_skills": ["product management", "agile", "user stories", "market research", "product strategy", "analytics"],
        "responsibilities": "Product planning, requirement gathering, stakeholder management, product strategy, roadmap",
        "preferred": ["sql", "python", "data analysis", "jira", "confluence", "customer development"]
    },
    
    "project manager": {
        "description": "Project Manager with software project delivery experience",
        "required_skills": ["project management", "agile", "scrum", "risk management", "budgeting", "stakeholder management"],
        "responsibilities": "Project planning, team coordination, risk management, delivery management, reporting",
        "preferred": ["jira", "confluence", "pmp", "prince2", "azure devops"]
    },

    # Specialized Roles
    "blockchain developer": {
        "description": "Blockchain Developer with distributed ledger technology experience",
        "required_skills": ["solidity", "blockchain", "ethereum", "smart contracts", "web3", "cryptography"],
        "responsibilities": "Develop smart contracts, blockchain applications, dApps, crypto protocols",
        "preferred": ["rust", "go", "hyperledger", "ipfs", "defi", "nft"]
    },
    
    "iot developer": {
        "description": "IoT Developer with Internet of Things experience",
        "required_skills": ["python", "c++", "iot", "embedded systems", "raspberry pi", "arduino", "mqtt"],
        "responsibilities": "Develop IoT solutions, work with sensors, device programming, cloud integration",
        "preferred": ["aws iot", "azure iot", "edge computing", "bluetooth", "zigbee"]
    },
    
    "ar/vr developer": {
        "description": "AR/VR Developer with augmented and virtual reality experience",
        "required_skills": ["unity", "unreal engine", "c#", "c++", "3d graphics", "ar/vr development"],
        "responsibilities": "Develop AR/VR applications, 3D modeling, interactive experiences, performance optimization",
        "preferred": ["blender", "maya", "opengl", "webgl", "mobile ar"]
    },
    
    "quantitative developer": {
        "description": "Quantitative Developer with financial modeling experience",
        "required_skills": ["python", "c++", "quantitative finance", "mathematics", "statistics", "algorithmic trading"],
        "responsibilities": "Develop trading algorithms, financial modeling, risk analysis, quantitative research",
        "preferred": ["r", "matlab", "financial derivatives", "machine learning", "high frequency trading"]
    },
    
    "solutions architect": {
        "description": "Solutions Architect with enterprise architecture experience",
        "required_skills": ["architecture", "cloud", "aws", "azure", "system design", "enterprise solutions"],
        "responsibilities": "Design solutions, technology selection, architecture planning, technical leadership",
        "preferred": ["terraform", "kubernetes", "microservices", "devops", "security architecture"]
    },
    
    "technical writer": {
        "description": "Technical Writer with software documentation experience",
        "required_skills": ["technical writing", "documentation", "api documentation", "markdown", "git", "content creation"],
        "responsibilities": "Create documentation, user guides, API documentation, technical content, knowledge base",
        "preferred": ["python", "javascript", "swagger", "readthedocs", "technical communication"]
    },
    
    "ux/ui designer": {
        "description": "UX/UI Designer with user experience and interface design expertise",
        "required_skills": ["ui design", "ux design", "figma", "sketch", "user research", "prototyping", "wireframing"],
        "responsibilities": "Design interfaces, user research, create prototypes, usability testing, design systems",
        "preferred": ["adobe xd", "invision", "html", "css", "user psychology", "accessibility"]
    },
    
    "software consultant": {
        "description": "Software Consultant with client solution expertise",
        "required_skills": ["consulting", "solution architecture", "project management", "client communication", "technical expertise"],
        "responsibilities": "Client consulting, solution design, requirements analysis, technical guidance, implementation",
        "preferred": ["aws", "azure", "agile", "python", "java", "cloud migration"]
    },
    
    "it support specialist": {
        "description": "IT Support Specialist with technical support experience",
        "required_skills": ["technical support", "troubleshooting", "windows", "linux", "networking", "helpdesk"],
        "responsibilities": "Technical support, troubleshooting, user assistance, system maintenance, IT helpdesk",
        "preferred": ["active directory", "azure ad", "office 365", "remote support", "itil"]
    }
}


def generate_job_description(job_role):
    """Generate comprehensive job description from template"""
    role_key = job_role.lower()
    for template_key in JOB_ROLE_TEMPLATES.keys():
        if template_key in role_key:
            template = JOB_ROLE_TEMPLATES[template_key]
            return f"""
            {template['description']}
            
            Required Skills: {', '.join(template['required_skills'])}
            Responsibilities: {template['responsibilities']}
            Preferred Qualifications: {', '.join(template['preferred'])}
            """
    
    # Default template for unknown roles
    return f"""
    {job_role.title()} Position
    Required Skills: Strong technical skills, problem-solving ability, team collaboration
    Responsibilities: Develop and maintain systems, collaborate with teams, deliver solutions
    Preferred: Industry experience, relevant certifications, portfolio of work
    """

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name
        
        with pdfplumber.open(tmp_file_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        os.unlink(tmp_file_path)
        return text.strip()
    except Exception as e:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        return f"Error reading PDF: {e}"

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(BytesIO(docx_file.read()))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error reading DOCX: {e}"

def clean_text(text):
    """Clean and normalize text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;:!?]', ' ', text)
    return text.lower().strip()

def extract_skills(text):
    """Extract potential skills from text"""
    common_skills = {
        'python', 'sql', 'java', 'javascript', 'html', 'css', 'react', 
        'angular', 'node', 'express', 'django', 'flask', 'mongodb', 
        'postgresql', 'mysql', 'aws', 'azure', 'docker', 'kubernetes',
        'git', 'jenkins', 'linux', 'machine learning', 'data analysis',
        'excel', 'power bi', 'tableau', 'communication', 'teamwork',
        'leadership', 'problem solving', 'project management', 'c++',
        'typescript', 'vue', 'spring', 'ruby', 'php', 'swift', 'kotlin',
        'nosql', 'redis', 'elasticsearch', 'graphql', 'rest api', 'agile',
        'scrum', 'ci/cd', 'devops', 'testing', 'debugging', 'api', 'microservices',
        'data science', 'deep learning', 'nlp', 'computer vision', 'tensorflow',
        'pytorch', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'statistics',
        'cloud', 'azure', 'gcp', 'serverless', 'lambda', 'ec2', 's3'
    }
    
    found_skills = []
    text_lower = text.lower()
    for skill in common_skills:
        if skill in text_lower:
            found_skills.append(skill)
    return found_skills

def extract_required_skills(jd_text):
    """Extract required skills from job description"""
    skills = extract_skills(jd_text)
    return skills

def calculate_relevance_score(resume_text, jd_text):
    """Calculate comprehensive relevance score"""
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(jd_text)
    
    if len(clean_resume.split()) < 10 or len(clean_jd.split()) < 5:
        return 0.0, set(), set(), [], 0, 0, 0
    
    # TF-IDF similarity
    tfidf_score = 0
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_jd])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        tfidf_score = max(0, similarity * 100)
    except:
        tfidf_score = 0
    
    # Keyword matching
    jd_keywords = set([word for word in clean_jd.split() if len(word) > 2])
    resume_keywords = set([word for word in clean_resume.split() if len(word) > 2])
    matched_keywords = jd_keywords.intersection(resume_keywords)
    keyword_score = (len(matched_keywords) / len(jd_keywords)) * 100 if jd_keywords else 0
    
    # Skills matching
    jd_skills = extract_required_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    matched_skills = set(jd_skills).intersection(set(resume_skills))
    skills_score = (len(matched_skills) / len(jd_skills)) * 100 if jd_skills else 0
    
    # Final score
    final_score = (tfidf_score * 0.4) + (keyword_score * 0.3) + (skills_score * 0.3)
    final_score = max(0, min(100, final_score))
    
    return round(final_score, 2), matched_keywords, matched_skills, jd_skills, tfidf_score, keyword_score, skills_score

def get_verdict(score):
    """Get suitability verdict based on score"""
    if score >= 75:
        return "Excellent Match! 🎯", "success"
    elif score >= 60:
        return "Good Match 👍", "warning"
    elif score >= 40:
        return "Moderate Match ⚠️", "warning"
    else:
        return "Needs Improvement 📈", "error"

def create_skills_radar_chart(matched_skills, jd_skills):
    """Create radar chart for skills comparison"""
    categories = ['Technical', 'Framework', 'Tools', 'Soft Skills', 'Domain']
    matched_values = [min(len(matched_skills) * 2, 100) for _ in categories]
    required_values = [80 for _ in categories]  # Ideal target
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=matched_values,
        theta=categories,
        fill='toself',
        name='Your Skills',
        line_color='blue'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=required_values,
        theta=categories,
        fill='toself',
        name='Required',
        line_color='red'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        height=300
    )
    
    return fig

def create_score_gauge(score):
    """Create gauge chart for overall score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 60], 'color': "orange"},
                {'range': [60, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

def create_skills_comparison(matched_skills, missing_skills):
    """Create skills comparison visualization"""
    df = pd.DataFrame({
        'Category': ['Matched Skills', 'Missing Skills'],
        'Count': [len(matched_skills), len(missing_skills)]
    })
    
    fig = px.bar(df, x='Category', y='Count', 
                 color='Category', color_discrete_map={
                     'Matched Skills': 'green', 
                     'Missing Skills': 'red'
                 })
    fig.update_layout(height=300, showlegend=False)
    return fig

def generate_personalized_recommendations(score, matched_skills, missing_skills, job_role):
    """Generate personalized recommendations"""
    recommendations = []
    
    if score >= 75:
        recommendations.append("🎯 Excellent match! Your skills align perfectly with this role")
        recommendations.append("💼 Focus on highlighting your relevant experience in interviews")
        recommendations.append("📈 Consider leadership or specialized roles in this field")
    elif score >= 60:
        recommendations.append("👍 Good foundation - focus on filling the skill gaps")
        recommendations.append("📚 Consider online courses for missing technical skills")
        recommendations.append("🔧 Work on projects that demonstrate the required skills")
    else:
        recommendations.append("📈 Significant skill gap - consider foundational courses")
        recommendations.append("🎯 Focus on the most critical missing skills first")
        recommendations.append("💡 Build portfolio projects to demonstrate capabilities")
    
    # Skill-specific recommendations
    if missing_skills:
        priority_skills = list(missing_skills)[:3]
        recommendations.append(f"🔍 Priority learning: {', '.join(priority_skills)}")
        
        # Learning resource suggestions
        tech_skills = [s for s in priority_skills if s in {'python', 'java', 'react', 'aws'}]
        if tech_skills:
            recommendations.append("💻 Check platforms like Coursera, Udemy for technical courses")
    
    return recommendations

def main():
    st.title("🎯 JobFit Analyzer")
    st.markdown("### Smart Resume to Job Role Matching System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        job_role = st.text_input(
            "Enter desired job role:",
            placeholder="e.g., Python Developer, Data Scientist...",
            help="Enter the job role you're targeting"
        )
        
        uploaded_file = st.file_uploader(
            "Upload your resume:",
            type=["pdf", "docx"],
            help="Supported formats: PDF and DOCX"
        )
        
        if not job_role:
            st.info("💡 Popular roles: Python Developer, Data Scientist, Frontend Developer")
    
    # Main content
    if job_role and uploaded_file:
        # Generate job description
        jd_text = generate_job_description(job_role)
        
        # Process resume
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = extract_text_from_docx(uploaded_file)
        
        if resume_text and not resume_text.startswith("Error"):
            # Calculate scores
            score, matched_keywords, matched_skills, jd_skills, tfidf_score, keyword_score, skills_score = calculate_relevance_score(resume_text, jd_text)
            verdict, verdict_color = get_verdict(score)
            missing_skills = set(jd_skills) - set(matched_skills)
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Skills Analysis", "💡 Recommendations"])
            
            with tab1:
                st.header(f"🎯 Match Analysis for {job_role.title()}")
                
                # Top metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Match Score", f"{score}%")
                with col2:
                    st.metric("Skills Matched", f"{len(matched_skills)}/{len(jd_skills)}")
                with col3:
                    st.metric("Match Verdict", verdict)
                
                # Visualizations
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_score_gauge(score), use_container_width=True)
                with col2:
                    st.plotly_chart(create_skills_comparison(matched_skills, missing_skills), use_container_width=True)
                
                # Quick stats
                st.subheader("📈 Quick Statistics")
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                with stats_col1:
                    st.info(f"**Required Skills:** {len(jd_skills)}")
                with stats_col2:
                    st.success(f"**Your Skills:** {len(matched_skills)}")
                with stats_col3:
                    st.error(f"**Skills to Learn:** {len(missing_skills)}")
            
            with tab2:
                st.header("🔧 Detailed Skills Analysis")
                
                # Skills comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("✅ Your Matching Skills")
                    if matched_skills:
                        for skill in sorted(matched_skills):
                            st.success(f"• {skill}")
                    else:
                        st.info("No skills match the requirements")
                
                with col2:
                    st.subheader("❌ Missing Skills")
                    if missing_skills:
                        for skill in sorted(missing_skills):
                            st.error(f"• {skill}")
                    else:
                        st.success("You have all required skills! 🎉")
                
                # Skills radar chart
                st.plotly_chart(create_skills_radar_chart(matched_skills, jd_skills), use_container_width=True)
            
            with tab3:
                st.header("💡 Personalized Recommendations")
                
                # Generate recommendations
                recommendations = generate_personalized_recommendations(score, matched_skills, missing_skills, job_role)
                
                for i, recommendation in enumerate(recommendations, 1):
                    st.info(f"{i}. {recommendation}")
                
                # Action plan
                st.subheader("🚀 Action Plan")
                if missing_skills:
                    st.write("**Priority Skills to Develop:**")
                    for skill in list(missing_skills)[:5]:
                        st.write(f"• 📚 **{skill}** - Online courses & practical projects")
                
                # Learning resources
                st.subheader("🎓 Learning Resources")
                resource_cols = st.columns(2)
                with resource_cols[0]:
                    st.write("**Platforms:**")
                    st.write("• Coursera")
                    st.write("• Udemy")
                    st.write("• LinkedIn Learning")
                    st.write("• FreeCodeCamp")
                
                with resource_cols[1]:
                    st.write("**Practice:**")
                    st.write("• Build portfolio projects")
                    st.write("• Contribute to open source")
                    st.write("• Practice on LeetCode")
                    st.write("• Join coding communities")
        
        else:
            st.error("❌ Error processing resume. Please try another file.")
    
    else:
        # Welcome screen
        st.info("👋 Welcome to JobFit Analyzer!")
        st.markdown("""
        ### 📋 How to use:
        1. **Enter your desired job role** in the sidebar
        2. **Upload your resume** (PDF or DOCX)
        3. **Get instant analysis** of your job fit
        4. **See detailed skills matching** and recommendations
        
        ### 🎯 Supported Job Roles:
        - Python Developer
        - Data Scientist  
        - Frontend Developer
        - DevOps Engineer
        - Fullstack Developer
        - Machine Learning Engineer
        - And many more!
        """)
        
        # Sample analysis preview
        st.subheader("📊 Sample Analysis Preview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Match Score", "82%", "Excellent")
        with col2:
            st.metric("Skills Matched", "18/20", "90%")
        with col3:
            st.metric("Recommendations", "3", "View details")

if __name__ == "__main__":
    main()
