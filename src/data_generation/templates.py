"""
Enhanced Resume NER Templates for the new 7-entity schema.
Includes Names, Emails, Locations, and Dates.
"""

# Realistic data for synthetic generation
NAMES = [
    "Karim Haddad", "Sarah Johnson", "Jean Dupont", "Ahmed Mansouri", 
    "Elena Rossi", "Liam Wilson", "Chen Wei", "Sofia Gomez",
    "Marco Silva", "Yuki Tanaka", "Fatima Al-Sayed", "David Miller",
    "Priya Sharma", "Lucas Meyer", "Anna Kowalska", "Omar Khalid",
    "Alice Brown", "Sébastien Morel", "Li Na", "Carlos Rodriguez",
    "Zainab Abbas", "Hiroshi Sato", "Emma Schmidt", "Alejandro Ruiz"
]

LOCATIONS = [
    "Sousse, Tunisie", "Paris, France", "New York, USA", "London, UK",
    "Berlin, Germany", "Tokyo, Japan", "Dubai, UAE", "Casablanca, Morocco",
    "San Francisco, CA", "Toronto, Canada", "Singapore", "Sydney, Australia",
    "Tunis, Tunisia", "Lyon, France", "Boston, MA", "Madrid, Spain",
    "Montreal, Canada", "Geneva, Switzerland", "Seoul, Korea", "Munich, Germany"
]

EMAILS = [
    "karim.haddad@example.com", "s.johnson@gmail.com", "j.dupont@orange.fr",
    "ahmed.m@outlook.com", "elena.rossi88@yahoo.it", "l.wilson@academic.edu",
    "contact@chenwei.cn", "sofia.gomez@empresa.es", "p.sharma@tech.in",
    "omar.khalid@univ.ae", "miller.david@work.com", "emma.s@startup.de",
    "carlos.r@global.mx", "alicia.b@consulting.co.uk"
]

SKILLS = [
    "Python", "Machine Learning", "TF-IDF", "SQLite", "Data Science", 
    "Deep Learning", "NLP", "Computer Vision", "React", "Node.js", 
    "Docker", "Kubernetes", "AWS", "Azure", "SQL", "NoSQL", "Java", 
    "C++", "JavaScript", "TypeScript", "TensorFlow", "PyTorch", 
    "Scikit-learn", "Pandas", "NumPy", "Git", "Jenkins", "Agile",
    "Project Management", "UI/UX Design", "Tableau", "Power BI",
    "Go", "Rust", "Swift", "Kotlin", "Spring Boot", "Django", "Flask",
    "CI/CD", "Terraform", "Ansible", "GraphQL", "Redis", "Elasticsearch"
]

DEGREES = [
    "Bachelor of Science in Computer Science", "Master of Science in Data Science",
    "PhD in Artificial Intelligence", "Bachelor of Engineering", "MBA",
    "Master in Software Engineering", "Bachelor of Technology",
    "PhD in Machine Learning", "Doctorate in Computer Science",
    "Master in Cybersecurity", "Bachelor of Business Administration",
    "Génie Logiciel", "DESS en Informatique", "Licence en Mathématiques",
    "Polytechnique", "Stanford", "MIT", "Oxford", "Harvard", "Sorbonne"
]

ROLES = [
    "Data Scientist", "Software Engineer", "Frontend Developer", "Backend Developer",
    "Full Stack Developer", "Machine Learning Engineer", "DevOps Engineer",
    "Project Manager", "Data Analyst", "Research Intern", "Lead Developer",
    "Technical Architect", "Product Owner", "Cybersecurity Analyst"
]

DATES = [
    "Août 2025", "Janvier 2023", "2020 - 2024", "Current", "June 2022",
    "2018 to 2021", "Present", "Septembre 2021", "May 2019", "2015-2017"
]

UNIVERSITIES = [
    "Ecole Polytechnique", "Stanford University", "MIT", "Oxford University", 
    "Harvard", "Sorbonne", "Tunis El Manar", "National University of Singapore",
    "ETH Zurich", "University of Toronto", "Cambridge University", "Berkeley",
    "Yale", "Columbia", "Princeton", "Imperial College London", "UCL",
    "Georgia Tech", "Caltech", "Carnegie Mellon", "NYU", "UCLA"
]

COMPANIES = [
    "Google", "Microsoft", "Amazon", "Meta", "Apple", "Netflix", "Tesla",
    "The Bridge", "IBM", "Intel", "Cisco", "Oracle", "Salesforce", "Adobe", 
    "Spotify", "Shopify", "Twitter", "LinkedIn", "Uber", "Airbnb", "Capgemini",
    "Accenture", "Deloitte", "Atos", "Sopra Steria", "Société Générale",
    "JPMorgan Chase", "Goldman Sachs", "Morgan Stanley"
]

# Words that should definitely be 'O' (Negative examples)
O_WORDS = [
    "passionné", "reconnu", "capacité", "solutions", "performantes",
    "experienced", "highly", "motivated", "senior", "junior", "team",
    "worked", "developed", "implemented", "responsible", "proven",
    "excellent", "skills", "proficient", "knowledge", "ability",
    "participated", "contributed", "managed", "led", "defined"
]

# Template structure for different resume sections
TEMPLATES = {
    "header": [
        "{name} | {email} | {location}",
        "{name} - {location}\nContact: {email}",
        "{name}\nEmail: {email}\nAddress: {location}",
        "Contact Details: {name}, {email}, {location}",
        "{location} | {name} | {email}"
    ],
    "education": [
        "Étudiant en {degree} à l' {university}",
        "Bachelor in {degree} from {university}, {date}",
        "Master en {degree} ({date})",
        "PhD in {degree} obtained at {university}",
        "Education: {degree}, {university}, {date}",
        "Diplôme: {degree} - {university} ({date})",
        "Formation: {degree} à {university}",
        "Studied {degree} at {university} during {date}"
    ],
    "experience": [
        "{role} chez {company} ({date})",
        "Working as {role} at {company} from {date}",
        "Previously {role} in {location} ({date})",
        "Current position: {role} at {company}",
        "{company}: {role}, {date}",
        "Professional Experience: {role} @ {company}, {date}",
        "Poste occupé: {role} au sein de {company} ({date})",
        "Software career at {company} as {role} ({date})"
    ],
    "skills_list": [
        "Skills: {skill1}, {skill2}, {skill3}, {skill4}",
        "Technical Stack: {skill1}, {skill2}, {skill3}",
        "Expertise in {skill1}, {skill2} and {skill3}",
        "Core competencies: {skill1}, {skill2}, {skill3}",
        "Main tools: {skill1}, {skill2}",
        "Compétences Clés: {skill1}, {skill2}, {skill3}",
        "Langages & Frameworks: {skill1}, {skill2}, {skill3}, {skill4}"
    ],
    "summary": [
        "Un {role} {passion} par les {solutions} {performantes}.",
        "Experienced {role} with a {proven} track record in {skill1}.",
        "Motivated student in {degree} looking for {role} opportunities.",
        "Expert in {skill1} and {skill2} with {excellent} {skills} in {team} work.",
        "Goal-oriented {role} specializing in {skill1}."
    ]
}

# Special mapping for variable filling
VARIABLE_MAP = {
    "name": NAMES,
    "email": EMAILS,
    "location": LOCATIONS,
    "skill1": SKILLS,
    "skill2": SKILLS,
    "skill3": SKILLS,
    "skill4": SKILLS,
    "degree": DEGREES,
    "university": UNIVERSITIES,
    "role": ROLES,
    "date": DATES,
    "company": COMPANIES,
    "passion": ["passionné", "motivated"],
    "solutions": ["solutions", "systems"],
    "performantes": ["performantes", "robust"],
    "proven": ["proven", "demonstrated"],
    "excellent": ["excellent", "strong"],
    "skills": ["skills", "abilities"],
    "team": ["team", "collaborative"]
}

# Entity labels for each variable
VARIABLE_TAGS = {
    "name": "NAME",
    "email": "EMAIL",
    "location": "LOCATION",
    "skill1": "SKILL",
    "skill2": "SKILL",
    "skill3": "SKILL",
    "skill4": "SKILL",
    "degree": "DEGREE",
    "university": "DEGREE",
    "role": "EXPERIENCE",
    "date": "DATE",
    "company": "EXPERIENCE",
    "passion": "O",
    "solutions": "O",
    "performantes": "O",
    "proven": "O",
    "excellent": "O",
    "skills": "O",
    "team": "O"
}
