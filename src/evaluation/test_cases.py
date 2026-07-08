"""Curated bilingual QA test cases for RAG evaluation."""

TEST_QA_PAIRS = [
    # ── Employee Handbook questions (EN) ──
    {
        "question": "How many days of annual leave do full-time employees get?",
        "ground_truth": "Full-time employees are entitled to 15 working days of paid annual leave per calendar year, accruing at 1.25 days per month.",
        "relevant_doc": "employee_handbook.docx",
        "language": "en",
    },
    {
        "question": "What is the remote work policy at AcmeTech?",
        "ground_truth": "Employees may work remotely up to 3 days per week with manager approval. Core working hours are 10:00 AM to 4:00 PM Beijing Time.",
        "relevant_doc": "employee_handbook.docx",
        "language": "en",
    },
    {
        "question": "What is the password policy for company systems?",
        "ground_truth": "Passwords require a minimum of 12 characters with at least one uppercase letter, one lowercase letter, one digit, and one special character. Passwords must be changed every 90 days. MFA is mandatory for systems with sensitive data.",
        "relevant_doc": "employee_handbook.docx",
        "language": "en",
    },
    {
        "question": "How much is the home office stipend?",
        "ground_truth": "The company provides a one-time home office stipend of 3,000 RMB for equipment purchases.",
        "relevant_doc": "employee_handbook.docx",
        "language": "en",
    },
    {
        "question": "What happens if I have unused annual leave at year end?",
        "ground_truth": "Unused leave up to 5 days may be carried over to the next year with manager approval.",
        "relevant_doc": "employee_handbook.docx",
        "language": "en",
    },

    # ── Employee Handbook questions (CN) ──
    {
        "question": "员工每年有多少天带薪年假？",
        "ground_truth": "全职员工每个日历年享有15个工作日的带薪年假，按每月1.25天累计。",
        "relevant_doc": "employee_handbook.docx",
        "language": "cn",
    },
    {
        "question": "公司的远程办公政策是什么？",
        "ground_truth": "员工经经理批准后每周最多可远程办公3天。核心工作时间为北京时间10:00至16:00。",
        "relevant_doc": "employee_handbook.docx",
        "language": "cn",
    },
    {
        "question": "薪酬结构包括哪些部分？",
        "ground_truth": "薪酬由基本工资（12个月）、绩效奖金（最高3个月基本工资，每年3月发放）和法定社会保险组成。",
        "relevant_doc": "employee_handbook.docx",
        "language": "cn",
    },

    # ── Compliance Guide ──
    {
        "question": "What data privacy law must AcmeTech comply with?",
        "ground_truth": "The company must comply with the Personal Information Protection Law (PIPL) of the People's Republic of China, effective November 1, 2021.",
        "relevant_doc": "compliance_guide.docx",
        "language": "en",
    },
    {
        "question": "What is the gift declaration threshold for anti-bribery compliance?",
        "ground_truth": "Gifts to or from business partners valued over 200 RMB must be declared to the compliance department.",
        "relevant_doc": "compliance_guide.docx",
        "language": "en",
    },
    {
        "question": "What are the requirements for cross-border data transfers?",
        "ground_truth": "Any transfer of personal data outside China requires a data export safety assessment and a signed standard contract with the overseas recipient.",
        "relevant_doc": "compliance_guide.docx",
        "language": "en",
    },
    {
        "question": "知识产权归谁所有？",
        "ground_truth": "员工在工作期间创造的所有发明、软件代码、设计和技术文档均为阿科美科技的知识产权。",
        "relevant_doc": "compliance_guide.docx",
        "language": "cn",
    },

    # ── Technical Specifications ──
    {
        "question": "What is the standard relational database at AcmeTech?",
        "ground_truth": "PostgreSQL 15+ is the standard relational database. Connection pooling must use PgBouncer with transaction-level pooling mode.",
        "relevant_doc": "tech_specifications.docx",
        "language": "en",
    },
    {
        "question": "What authentication method is required for APIs?",
        "ground_truth": "OAuth 2.0 with JWT access tokens using RS256 algorithm with 15-minute expiry and refresh tokens with 7-day expiry using one-time use with rotation.",
        "relevant_doc": "tech_specifications.docx",
        "language": "en",
    },
    {
        "question": "What are the API rate limiting rules?",
        "ground_truth": "Rate limiting: 100 requests per minute per user for general endpoints, 1000 requests per minute for authenticated internal services.",
        "relevant_doc": "tech_specifications.docx",
        "language": "en",
    },

    # ── Architecture Document ──
    {
        "question": "What is the daily data processing volume of the AcmeTech Data Platform?",
        "ground_truth": "The platform processes approximately 50TB of data daily with a peak throughput of 500,000 events per second.",
        "relevant_doc": "architecture_doc.docx",
        "language": "en",
    },
    {
        "question": "What is the disaster recovery RPO and RTO?",
        "ground_truth": "Disaster Recovery has an RPO of 15 minutes and RTO of 2 hours with active-passive setup across two availability zones.",
        "relevant_doc": "architecture_doc.docx",
        "language": "en",
    },
    {
        "question": "阿科美数据平台每天处理多少数据？",
        "ground_truth": "该平台每天处理约50TB数据，峰值吞吐量为每秒50万事件。",
        "relevant_doc": "architecture_doc.docx",
        "language": "cn",
    },

    # ── Edge cases (out of scope) ──
    {
        "question": "What is the company's policy on cryptocurrency investments?",
        "ground_truth": "There is no information about cryptocurrency investments in the available documents.",
        "relevant_doc": None,
        "language": "en",
    },
    {
        "question": "Who is the CEO of AcmeTech?",
        "ground_truth": "The name of the CEO is not mentioned in the available documents.",
        "relevant_doc": None,
        "language": "en",
    },
    {
        "question": "What is the dress code policy?",
        "ground_truth": "The dress code policy is not covered in the available documents.",
        "relevant_doc": None,
        "language": "en",
    },
]
