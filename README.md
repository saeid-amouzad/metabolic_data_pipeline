# 🧬 Metabolic Data Pipeline

> A Dockerized end-to-end pipeline for ingestion, preprocessing, **analysis & visualization**,  
> and machine learning on metabolic and diabetes-related data.

---

## 📌 Overview

**Metabolic Data Pipeline** is a modular system designed for:

- Cohort-level analysis
- Single-patient prediction
- Consent-aware data handling
- Interactive **analysis & visualization dashboard**
- Reproducible execution using Docker

The project integrates **data engineering, analytics, visualization, and machine learning**
into a single, reproducible workflow.

---

## ✨ Key Features

- 📥 Raw data ingestion (cohort & single patient)
- 🧪 Preprocessing and feature engineering
- 📊 Rich analysis & visualization dashboard
- 🤖 Machine learning training and inference
- 🔒 Explicit consent-aware data storage
- 🐳 Fully Dockerized environment

---

## 🧠 High-level Workflows

### Cohort workflow
```
Raw cohort data
 → Upload via dashboard
 → Ingestion
 → Preprocessing & integration (SQLite)
 → Feature engineering & selection
 → Feature store (CSV)
 → Model training
 → Model artifacts & metadata
```

### Single-patient workflow
```
Raw single-patient data
 → Upload via dashboard
 → Ingestion
 → Preprocessing
 → Feature transformation (no refitting)
 → Prediction
 → Optional storage / retraining (based on consent)
```

---

## 🗂 Project Structure

```text
metabolic-data-pipeline/
├── Dockerfile
├── requirements.txt
├── README.md
├── config/
├── dashboard/                  # Flask dashboard (UI + backend)
├── data/
│   ├── raw/                    # Internal storage (managed by the app)
│   ├── processed/
│   ├── uploads/
│   └── database.db
├── input_data/                 # User-facing input examples
├── src/                        # Core pipeline implementation
├── tests/
└── main.py
```

---

## 📁 User Input Data (`input_data/`)

The `input_data/` directory is a **user-facing folder** intended to hold raw datasets
*before* uploading them through the dashboard UI.

> 📌 Files placed here are **not processed automatically**.  
> Users explicitly select them in the web interface.

### Expected structure

```text
input_data/
├── cohort/
│   ├── cohort_pheno.csv
│   └── cohort_geno.csv
└── single_patient/
    ├── patient_0001_pheno.csv
    └── patient_0001_geno.csv
```

---

## 🗄 Internal Data Storage (`data/raw/`)

- Managed **only by the application**
- Written after ingestion and consent checks
- Users should **not manually edit** this directory

This separation ensures:
- Clean data provenance
- Safer handling of sensitive data
- Clear distinction between input data and stored data

---

## 📊 Analysis & Visualization Dashboard

A core component of the project is the **analysis and visualization layer** exposed
via the Flask dashboard. These modules support exploratory data analysis, model
diagnostics, and feature understanding.

### Implemented analysis modules

| Module | Purpose |
|------|--------|
| `dataset_overview.py` | Dataset schema, size, missing values |
| `distribution_analysis.py` | Numeric & categorical distributions |
| `relationship_analysis.py` | Feature–target relationships |
| `temporal_analysis.py` | Time-based and registry-date trends |
| `snp_summary_analysis.py` | SNP-level summaries and distributions |
| `model_support_analysis.py` | Model diagnostics, metrics, statistical tests |

These analyses support both **cohort-level insights** and **model interpretation**.

---

## 🐳 Running with Docker (Recommended)

### Requirements
- Docker Desktop

### Build the image
```bash
docker build -t metabolic-pipeline .
```

### Run the application
```bash
docker run -p 5000:5000 metabolic-pipeline
```

Open in browser:
```
http://localhost:5000
```

---

## 🧩 What the Dockerfile Does

The `Dockerfile`:

- Uses an official Python base image
- Sets a consistent working directory
- Installs dependencies from `requirements.txt`
- Copies the project source code
- Exposes port `5000` for the dashboard
- Starts the Flask application

Docker removes the need for local Python, Conda, or virtual environments.

---

## 🧪 Running Without Docker (Development Only)

```bash
pip install -r requirements.txt
python dashboard/app.py
```

Offline training:
```bash
python main.py
```

---

## 📚 Example Dataset Source

Cohort phenotypic data used for development is **sampled (2,500 records)** from:

**Kaggle – 100,000 Diabetes Clinical Dataset**  
https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset

Used for research and demonstration purposes only.

---

## 🧱 Design Principles

- Separation of concerns
- Consent-aware data flow
- No feature leakage between training and inference
- Reproducible experiments
- Modular and extensible architecture

---

## 📝 Notes for Users & Reviewers

- Place raw files in `input_data/`
- Upload data only via the dashboard
- Do not manually modify `data/raw/`
- Docker is the recommended execution method
