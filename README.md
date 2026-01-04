Metabolic Data Pipeline

An end-to-end pipeline for ingestion, preprocessing, analysis, and machine learning
on metabolic and diabetes-related data. The system supports both cohort-level analysis
and single-patient prediction with consent-aware data handling. The application includes
a Flask-based dashboard and is fully Dockerized for reproducible execution.

----------------------------------------------------------------------
1. High-level Workflows
----------------------------------------------------------------------

Cohort workflow:
User-provided cohort data
→ Upload via dashboard
→ Ingestion
→ Preprocessing & integration (SQLite)
→ Feature engineering & feature selection
→ Feature store (CSV)
→ Model training
→ Model artifacts & metadata

Single-patient workflow:
User-provided single-patient data
→ Upload via dashboard
→ Ingestion
→ Preprocessing
→ Feature transformation (no refitting)
→ Prediction
→ Optional storage / retraining (based on consent)

----------------------------------------------------------------------
2. Project Structure
----------------------------------------------------------------------

metabolic-data-pipeline/
├── Dockerfile
├── requirements.txt
├── README.md
├── config/
├── dashboard/
├── data/
│   ├── raw/          (internal storage, managed by the app)
│   ├── processed/
│   ├── uploads/
│   └── database.db
├── input_data/       (user-facing input folder)
├── src/
├── tests/
└── main.py

----------------------------------------------------------------------
3. User Input Data (input_data/)
----------------------------------------------------------------------

The input_data/ folder is intended for users to place raw datasets before uploading
them via the dashboard UI. Files are NOT processed automatically; users must select
them explicitly in the web interface.

Expected structure:

input_data/
├── cohort/
│   ├── cohort_pheno.csv
│   └── cohort_geno.csv
└── single_patient/
    ├── patient_0001_pheno.csv
    └── patient_0001_geno.csv

Purpose:
- Provides a clear input format for users
- Simplifies testing and onboarding
- Keeps internal storage separate from user inputs

----------------------------------------------------------------------
4. Internal Raw Data Storage (data/raw/)
----------------------------------------------------------------------

The data/raw/ directory is used internally by the application to store uploaded data
after ingestion. Users should not manually place files in this directory.

- Written only by the ingestion pipeline
- Subject to consent rules
- Ensures clean data provenance

----------------------------------------------------------------------
5. Example Phenotypic Dataset
----------------------------------------------------------------------

Cohort phenotypic data used for development is sampled (2,500 records) from:

Kaggle – 100,000 Diabetes Clinical Dataset
https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset

The dataset is used for research and demonstration purposes only.

----------------------------------------------------------------------
6. Running the Project with Docker (Recommended)
----------------------------------------------------------------------

Requirements:
- Docker Desktop

Build the Docker image (from project root):

docker build -t metabolic-pipeline .

Run the application:

docker run -p 5000:5000 metabolic-pipeline

Open in browser:
http://localhost:5000

----------------------------------------------------------------------
7. What the Dockerfile Does
----------------------------------------------------------------------

- Uses an official Python base image
- Sets a working directory inside the container
- Installs Python dependencies
- Copies project source code
- Exposes port 5000
- Starts the Flask dashboard

Docker removes the need for local Python, Conda, or virtual environments.

----------------------------------------------------------------------
8. Running Without Docker (Development Only)
----------------------------------------------------------------------

pip install -r requirements.txt
python dashboard/app.py

Offline training:
python main.py

----------------------------------------------------------------------
9. Notes
----------------------------------------------------------------------

- Place raw input files in input_data/
- Upload data only via the dashboard
- Do not manually modify data/raw/
- Docker is the recommended execution method
