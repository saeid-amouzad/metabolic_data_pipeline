# dashboard/app.py

"""
    Flask web application for genomic and phenotypic data analysis dashboard.
    Handles data ingestion, preprocessing, feature extraction, ML modeling,
    and various analyses with visualizations.

    | Route                       | Methods   | Purpose                                          |
    | --------------------------- | --------- | ------------------------------------------------ |
    | `/`                         | GET       | Home page                                        |
    | `/raw-data`                 | GET, POST | Cohort (raw) data ingestion + preprocessing + ML |
    | `/patient`                  | GET, POST | Single-patient prediction workflow               |
    | `/analysis`                 | GET, POST | Analysis & visualization dashboard               |
    | `/analysis/features`        | GET       | AJAX endpoint for feature lists                  |

"""

import os
import sys
import pandas as pd
import sqlite3
from flask import Flask, redirect, render_template, request, flash, jsonify, url_for, request
from flask import get_flashed_messages
from pathlib import Path

# =========================================================
# Fix Python path (important for src imports)
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================================================
# Internal imports
# =========================================================
from src.ingestion.ingest_data import IngestData
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.features.feature_pipeline import FeaturePipeline
from src.models.inference import InferenceService
from src.models.model_trainer import ModelTrainer
from src.utils.db import init_db
from src.utils.logger import setup_logger
from src.visualization.dataset_overview import DatasetOverviewAnalyzer
from src.visualization.distribution_analysis import DistributionAnalyzer
from src.visualization.relationship_analysis import RelationshipAnalyzer
from src.visualization.snp_summary_analysis import SNPSummaryAnalyzer
from src.visualization.temporal_analysis import TemporalAnalyzer
from src.visualization.model_support_analysis import ModelSupportAnalyzer

# =========================================================
# App configuration
# =========================================================
UPLOAD_FOLDER = "data/uploads"
RAW_FOLDER = "data/raw"
DB_PATH = "data/database.db"
STATIC_DIR = "dashboard/static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RAW_FOLDER, exist_ok=True)

logger = setup_logger("flask_app")

app = Flask(__name__)
app.secret_key = "dev-secret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

#========================================================
# Dataset configurations
#========================================================
DATASETS = {
    "raw_pheno": {
        "label": "Raw phenotypic",
        "type": "csv",
        "path": "data/raw/cohort/pheno_dataset_2500.csv",
    },
    "raw_geno": {
        "label": "Raw genotypic",
        "type": "csv",
        "path": "data/raw/cohort/geno_dataset_2500.csv",
    },
    "integrated": {
        "label": "Integrated cohort",
        "type": "db",
        "db_path": "data/database.db",
        "table": "analysis_master",
    },
    "features": {
        "label": "Feature-selected",
        "type": "csv",
        "path": "data/processed/features.csv",
    },
}

# -----------------------------
# Helper / factory functions
# -----------------------------
def get_distribution_analyzer(dataset_type):
    cfg = DATASETS.get(dataset_type)
    if not cfg:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    if cfg["type"] == "csv":
        return (
            DistributionAnalyzer(csv_path=cfg["path"]),
            cfg["label"],
        )

    if cfg["type"] == "db":
        return (
            DistributionAnalyzer(
                db_path=cfg["db_path"],
                table_name=cfg["table"],
            ),
            cfg["label"],
        )

    raise ValueError("Unsupported dataset configuration")

# =========================================================
# Initialize DB & pipelines
# =========================================================
init_db(DB_PATH)

ingestion_manager = IngestData(
    db_path=DB_PATH,
    raw_base_dir=RAW_FOLDER
)

preprocessing_pipeline = PreprocessingPipeline(
    db_path=DB_PATH
)

feature_pipeline = FeaturePipeline(
    db_path=DB_PATH,
    output_csv="data/processed/features.csv"
)

# =========================================================
# Routes
# =========================================================

@app.route("/")
def index():
    # Clear any stale flash messages
    get_flashed_messages()
    return render_template("index.html")

# ---------------------------------------------------------
# COHORT / RAW DATA INGESTION + PREPROCESSING + FEATURES + ML
# ---------------------------------------------------------
@app.route("/raw-data", methods=["GET", "POST"])
def raw_data():

    workflow_status = {
        "ingestion": False,
        "storage": False,
        "preprocessing": False,
        "feature": False,
        "ml": False,
    }

    # ✅ Check from the beginning
    cohort_raw_dir = Path("data/raw/cohort")
    cohort_raw_exists = cohort_raw_dir.exists()
    #cohort_raw_exists = False

    if request.method == "POST":
        consent = request.form.get("consent") == "yes"
        #confirm_override = request.form.get("confirm_override") == "yes"
        registry_date = request.form.get("registry_date")

        pheno_path = None
        geno_path = None

        # =========================
        # 1️⃣ Ingestion
        # =========================
        for i in range(1, 4):
            uploaded = request.files.get(f"file_{i}")
            data_type = request.form.get(f"type_{i}")
            source = request.form.get(f"source_{i}", "ui")

            if uploaded and uploaded.filename and data_type:
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded.filename)
                uploaded.save(save_path)

                ingestion_manager.ingest(
                    file_path=save_path,
                    data_type=data_type,
                    registry_date=registry_date,
                    consent_raw_storage=consent,
                    source=source
                )

                if data_type == "phenotypic":
                    pheno_path = save_path
                elif data_type == "genotypic":
                    geno_path = save_path

        if not pheno_path or not geno_path:
            flash("Both phenotypic and genotypic datasets are required.", "error")
            return render_template("raw_data.html", workflow_status=workflow_status)
        
        # ✅ Ingestion completed
        workflow_status["ingestion"] = True

        # =========================
        # Consent check (STOP EARLY)
        # =========================
        if not consent:
            flash(
                "You did not consent for data handling, so we cannot proceed with the rest of the workflow. "
                "Please update your consent choice to continue.",
                "error"
            )
            return render_template(
                "raw_data.html",
                workflow_status=workflow_status
            )

        # ✅ Raw storage completed
        workflow_status["storage"] = True

        # =========================
        # 2️⃣ Preprocessing
        # =========================
        try:
            preprocessing_pipeline.run(
                pheno_path=pheno_path,
                geno_path=geno_path,
                is_single_patient=False,
                consent_to_store=consent,
                registry_date=registry_date
            )
            workflow_status["preprocessing"] = True
        except Exception as e:
            flash(f"Preprocessing failed: {e}", "error")
            return render_template("raw_data.html", workflow_status=workflow_status)

        # =========================
        # 3️⃣ Feature pipeline
        # =========================
        try:
            df_processed = feature_pipeline.load_processed_data()
            X = feature_pipeline.fit_transform(df_processed)

            X_with_target = X.copy()
            X_with_target["diabetes"] = df_processed["diabetes"].values

            feature_pipeline.save_features(X_with_target)
            workflow_status["feature"] = True
        except Exception as e:
            flash(f"Feature pipeline failed: {e}", "error")
            return render_template("raw_data.html", workflow_status=workflow_status)

        # =========================
        # 4️⃣ ML modeling
        # =========================
        try:
            trainer = ModelTrainer("config/config.yaml")
            trainer.train_select_best()
            workflow_status["ml"] = True
        except Exception as e:
            flash(f"ML modeling failed: {e}", "error")
            return render_template("raw_data.html", workflow_status=workflow_status)

        flash("Cohort pipeline completed successfully.", "success")

    return render_template("raw_data.html", workflow_status=workflow_status,
                           cohort_raw_exists=cohort_raw_exists)
# ---------------------------------------------------------
# SINGLE PATIENT WORKFLOW
# ---------------------------------------------------------
@app.route("/patient", methods=["GET", "POST"])
def patient():

    prediction = None
    workflow_status = {
        "ingestion": False,
        "preprocessing": False,
        "prediction": False,
        "raw_storage": False,
        "integration": False,
        "retraining": False,
    }

    if request.method == "POST":
        patient_id = request.form["patient_id"]
        registry_date = request.form.get("registry_date")

        # ---- CONSENTS ----
        consent_raw = request.form.get("consent_raw") == "yes"
        consent_integrate = request.form.get("consent_integrate") == "yes"
        consent_retrain = request.form.get("consent_retrain") == "yes"
        
        if not registry_date:
            flash("Please enter a registry date before uploading.", "error")

        pheno_path = None
        geno_path = None

        # -------------------------------------------------
        # CONSENT VALIDATION (single patient)
        # -------------------------------------------------
        errors = []

        # ❌ Raw = NO, Integrate = YES; Retrain = NO/YES
        if not consent_raw and consent_integrate:
            errors.append(
                "Without raw storage consent, integration is not possible."
            )

        # ❌ Raw = NO, Retrain = YES; Integrate = NO/YES
        if not consent_raw and consent_retrain:
            errors.append(
                "Retrain consent requires raw storage and integration consent."
            )

        # ❌ Integrate = YES, Retrain = NO; Raw = YES/NO
        if consent_integrate and not consent_retrain:
            errors.append(
                "Integration consent without retrain consent does not make sense."
            )

        # ❌ Retrain = YES, Integrate = NO; Raw = YES/NO
        if consent_retrain and not consent_integrate:
            errors.append(
                "Retrain consent without integration consent does not make sense."
            )

        if errors:
            for msg in errors:
                flash(
                    msg + " Please change your consent choices.",
                    "error"
                )

            # ⛔ STOP processing, keep user inputs
            return render_template(
                "patient.html",
                workflow_status=workflow_status,
                prediction=None
            )


        # Prevent duplicate patient ID storage (raw data)
        single_patient_dir = Path("data/raw/single_patient")
        patient_folder = single_patient_dir / patient_id

        if consent_raw and patient_folder.exists():
            flash(
                f"Patient ID '{patient_id}' already exists in raw storage. "
                "Duplicate storage is not allowed.",
                "error"
            )
            return render_template(
                "patient.html",
                workflow_status=workflow_status,
                prediction=None
            )

        # =========================
        # 1️⃣ Ingestion
        # =========================
        for i in range(1, 4):
            uploaded = request.files.get(f"file_{i}")
            data_type = request.form.get(f"type_{i}")

            if uploaded and uploaded.filename and data_type:
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded.filename)
                uploaded.save(save_path)

                ingestion_manager.ingest(
                    file_path=save_path,
                    data_type=data_type,
                    patient_id=patient_id,
                    registry_date=registry_date,
                    consent_raw_storage=consent_raw,
                )

                if data_type == "phenotypic":
                    pheno_path = save_path
                elif data_type == "genotypic":
                    geno_path = save_path

        if not pheno_path or not geno_path:
            flash("Both phenotypic and genotypic datasets are required.", "error")
            return render_template("patient.html", workflow_status=workflow_status)

        workflow_status["ingestion"] = True
        workflow_status["raw_storage"] = consent_raw

        # =========================
        # 2️⃣ Preprocessing
        # =========================
        result = preprocessing_pipeline.run(
            pheno_path=pheno_path,
            geno_path=geno_path,
            is_single_patient=True,
            consent_to_store=consent_integrate,
            registry_date=registry_date
        )

        workflow_status["preprocessing"] = True
        workflow_status["integration"] = consent_integrate

        patient_df = result["final_df"]

        # =========================
        # 3️⃣ Feature pipeline + prediction
        # =========================

        X_patient = feature_pipeline.transform_single_patient(
            patient_df=patient_df,
            consent_to_store=False  # never append before prediction
        )

        inference = InferenceService()
        risk = inference.predict(X_patient)[0]
        prediction = f"{risk:.3f}" # (probability of Type 2 diabetes)
        workflow_status["prediction"] = True

        # =========================
        # 4️⃣ Optional retraining
        # =========================
        if consent_retrain and consent_integrate:
            feature_pipeline.transform_single_patient(
                patient_df=patient_df,
                consent_to_store=True   # append to features.csv
            )

            trainer = ModelTrainer("config/config.yaml")
            trainer.train_select_best()

            workflow_status["retraining"] = True

    return render_template(
        "patient.html",
        prediction=prediction,
        workflow_status=workflow_status,
        form_data=request.form   # ✅ pass form data back to template
    )

# -----------------------------
# Analysis route
# -----------------------------
@app.route("/analysis", methods=["GET", "POST"])
def analysis():

    summary = None
    dist_plots = []
    rel_plot = None
    plot_maf = None
    plot_var = None
    plot_temporal_size = None
    plot_temporal_prev = None
    plot_temporal_feat = None
    plot_corr = None
    plot_cm = None
    metrics = None
    plot_roc = None
    plot_pr = None
    plot_sig = None
    plot_freq = None
    plot_stack = None
    plot_ttest = None
    plot_chi2 = None

    active_step = None

    # --------------------------------------------------
    # Form state (A–F) for UI restoration
    # --------------------------------------------------
    overview_state = None
    dist_state = None
    rel_state = None
    snp_state = None
    temporal_state = None
    model_state = None

    try:
        if request.method == "POST":
            analysis_type = request.form.get("analysis_type")

            # ---------- A ----------
            if analysis_type == "overview":
                active_step = "A"
                dataset_type = request.form["dataset_type"]
                cfg = DATASETS.get(dataset_type)
                if not cfg:
                    flash("Invalid dataset selection.", "error")
                else:
                    analyzer = (
                        DatasetOverviewAnalyzer(cfg["label"], csv_path=cfg.get("path"))
                        if cfg["type"] == "csv"
                        else DatasetOverviewAnalyzer(
                            cfg["label"], db_path=cfg["db_path"], table_name=cfg["table"]
                        )
                    )
                    summary = analyzer.generate_summary()

                overview_state = {
                    "dataset_type": dataset_type,
                }

            # ---------- B ----------
            elif analysis_type == "distribution":
                active_step = "B"
                dataset_type = request.form.get("dataset_type")
                features = request.form.getlist("features")
                plot_type = request.form.get("plot_type")
                highlight_outliers = request.form.get("highlight_outliers") == "on"


                if not features:
                    flash("Please select at least one feature.", "error")
                    return render_template(
                        "analysis.html",
                        summary=summary,
                        dist_plots=[],
                        rel_plot=None,
                        rel_stats=None,
                        plot_maf=None,
                        plot_var=None,
                    )

                if highlight_outliers:
                    if plot_type != "box" or dataset_type not in {"raw_pheno", "raw_geno"}:
                        logger.warning(
                            "Outlier highlighting disabled due to incompatible settings"
                        )
                        highlight_outliers = False
                
                # ---------------------------------------------
                # Preserve Step B form state (for UI restore)
                # ---------------------------------------------
                dist_state = {
                    "dataset_type": dataset_type,
                    "plot_type": plot_type,
                    "features": features,
                    "highlight_outliers": highlight_outliers,
                }

                analyzer, label = get_distribution_analyzer(dataset_type)

                for f in features:
                    try:
                        fname = f"dist_{dataset_type}_{f}_{plot_type}.png"
                        fpath = os.path.join(STATIC_DIR, fname)

                        analyzer.plot_distribution(
                            feature=f,
                            plot_type=plot_type,
                            save_path=fpath,
                            dataset_label=label,
                            highlight_outliers=highlight_outliers,
                        )

                        dist_plots.append(fname)

                    except Exception as e:
                        app.logger.exception("Plot failed")
                        flash(f"Could not plot feature '{f}': {e}", "error")


            # ---------- C ----------
            elif analysis_type == "relationship":
                active_step = "C"
                feature = request.form.get("feature")
                plot_type = request.form.get("plot_type")

                assert feature, "Feature is required for relationship analysis"

                # ---------------------------------
                # Detect categorical features
                # ---------------------------------
                is_categorical = (
                    feature.startswith("rs")
                    or feature in {"gender", "hypertension", "heart_disease"}
                    or feature in {"race (encoded)", "smoking (encoded)"}
                )

                # Force grouped bar for categorical
                effective_plot = "bar" if is_categorical else plot_type

                rel_state = {
                    "feature": feature,
                    "plot_type": effective_plot,
                }

                analyzer = RelationshipAnalyzer(
                    db_path=DATASETS["integrated"]["db_path"],
                    table_name=DATASETS["integrated"]["table"],
                )

                fname = f"rel_{feature}_{effective_plot}.png"
                fpath = os.path.join(STATIC_DIR, fname)

                result = analyzer.analyze(
                    feature=feature,
                    plot_type=effective_plot,
                    save_path=fpath,
                )

                rel_plot = fname

            # ---------- F ----------
            elif analysis_type == "snp_summary":
                active_step = "F"

                analyzer = SNPSummaryAnalyzer(
                    db_path=DATASETS["integrated"]["db_path"],
                    table_name=DATASETS["integrated"]["table"],
                )

                snp_option = request.form.get("snp_option")

                # ---- SNP Overview ----
                if snp_option == "overview":
                    plot_maf = "snp_maf_hist.png"
                    plot_var = "snp_variance_hist.png"

                    analyzer.plot_maf_histogram(os.path.join(STATIC_DIR, plot_maf))
                    analyzer.plot_variance_histogram(os.path.join(STATIC_DIR, plot_var))

                # ---- SNP–Diabetes Association ----
                elif snp_option == "association":
                    analyzer.compute_snp_chi2()

                    plot_sig = "significant_snps_by_gene.png"
                    plot_freq = "significant_snp_genotype_frequencies.png"
                    plot_stack = "top10_snps_genotype_stacked_by_diabetes.png"

                    analyzer.plot_significant_snps(os.path.join(STATIC_DIR, plot_sig))
                    analyzer.plot_snp_genotype_frequencies(os.path.join(STATIC_DIR, plot_freq))
                    analyzer.plot_snp_genotype_stacked_by_diabetes(
                        os.path.join(STATIC_DIR, plot_stack)
                    )

                snp_state = {"option": snp_option}


            # ---------- D ----------
            elif analysis_type == "temporal":
                active_step = "D"
                analyzer = TemporalAnalyzer(
                    db_path=DATASETS["integrated"]["db_path"],
                    table_name=DATASETS["integrated"]["table"],
                )

                mode = request.form.get("temporal_mode", "year")
                agg = request.form.get("agg", "mean")
                plot_choice = request.form.get("temporal_plot")
                feature = request.form.get("feature")

                temporal_state = {
                    "mode": mode,
                    "plot_choice": plot_choice,
                    "feature": feature,
                    "agg": agg,
                }

                if plot_choice == "cohort":
                    plot_temporal_size = f"temporal_cohort_{mode}.png"
                    analyzer.plot_cohort_size(
                        os.path.join(STATIC_DIR, plot_temporal_size),
                        mode,
                    )

                elif plot_choice == "prevalence":
                    plot_temporal_prev = f"temporal_prev_{mode}.png"
                    analyzer.plot_prevalence(
                        os.path.join(STATIC_DIR, plot_temporal_prev),
                        mode,
                    )

                elif plot_choice == "feature":
                    if not feature:
                        raise ValueError("Feature must be selected for feature trend")

                    plot_temporal_feat = f"temporal_{feature}_{mode}.png"
                    analyzer.plot_feature_trend(
                        feature,
                        os.path.join(STATIC_DIR, plot_temporal_feat),
                        mode,
                        agg,
                    )
    
            # ---------- E ----------
            elif analysis_type == "model_support":
                active_step = "E"
                analyzer = ModelSupportAnalyzer(
                    features_csv="data/processed/features.csv",
                    artifacts_dir="src/models/artifacts",
                )

                option = request.form.get("support_option")

                model_state = {
                    "option": option,   
                }

                # --------------------------------------------------
                # GUARD: require trained model predictions
                # --------------------------------------------------
                if option in {"confusion", "roc", "pr"}:
                    if not analyzer.has_predictions():
                        flash(
                            "No model predictions found. Please train a model first.",
                            "error"
                        )
                        return render_template(
                            "analysis.html",
                            summary=summary,
                            dist_plots=dist_plots,
                            rel_plot=rel_plot,
                            plot_maf=plot_maf,
                            plot_var=plot_var,
                            plot_temporal_size=plot_temporal_size,
                            plot_temporal_prev=plot_temporal_prev,
                            plot_temporal_feat=plot_temporal_feat,
                            plot_corr=plot_corr,
                            plot_cm=plot_cm,
                            metrics=metrics,
                            plot_roc=plot_roc,
                            plot_pr=plot_pr,

                        )

                if option == "correlation":
                    plot_corr = "feature_correlation.png"
                    analyzer.plot_correlation_heatmap(
                        os.path.join(STATIC_DIR, plot_corr)
                    )

                elif option == "confusion":
                    plot_cm = "confusion_matrix.png"
                    analyzer.plot_confusion_matrix(
                        os.path.join(STATIC_DIR, plot_cm)
                    )

                elif option == "roc":
                    plot_roc = "roc_curve.png"
                    analyzer.plot_roc_curve(
                        os.path.join(STATIC_DIR, plot_roc)
                    )

                elif option == "pr":
                    plot_pr = "pr_curve.png"
                    analyzer.plot_pr_curve(
                        os.path.join(STATIC_DIR, plot_pr)
                    )

                elif option == "metrics":
                    metrics = analyzer.load_metrics()

                elif option == "stat_tests":
                    plot_ttest = "ttest_feature_significance.png"
                    plot_chi2 = "chi2_feature_significance.png"

                    analyzer.plot_ttest_pvalues(
                        db_path=DATASETS["integrated"]["db_path"],
                        table_name=DATASETS["integrated"]["table"],
                        save_path=os.path.join(STATIC_DIR, plot_ttest),
                    )

                    analyzer.plot_chi2_pvalues(
                        db_path=DATASETS["integrated"]["db_path"],
                        table_name=DATASETS["integrated"]["table"],
                        save_path=os.path.join(STATIC_DIR, plot_chi2),
                    )

                else:
                    raise ValueError("Invalid model support option")

            else:
                flash("Unknown analysis request.", "error")
                
    except Exception as e:
        app.logger.exception("Analysis failed")
        msg = str(e) if str(e) else e.__class__.__name__
        flash(f"Analysis failed: {msg}", "error")

    return render_template(
        "analysis.html",
        summary=summary,
        dist_plots=dist_plots,
        rel_plot=rel_plot,
        plot_maf=plot_maf,
        plot_var=plot_var,
        plot_temporal_size=plot_temporal_size,
        plot_temporal_prev=plot_temporal_prev,
        plot_temporal_feat=plot_temporal_feat,
        plot_corr=plot_corr,
        plot_cm=plot_cm,
        metrics=metrics,
        plot_roc=plot_roc,
        plot_pr=plot_pr,
        plot_sig=plot_sig,
        plot_freq=plot_freq,
        plot_stack=plot_stack,
        plot_ttest=plot_ttest,
        plot_chi2=plot_chi2,

        # state objects
        overview_state=overview_state,
        dist_state=dist_state,
        rel_state=rel_state,
        snp_state=snp_state,
        temporal_state=temporal_state,
        model_state=model_state,

        active_step=active_step,
    )


# returns dataset-specific feature names
@app.route("/analysis/features", methods=["GET"])

def get_analysis_features():
    dataset_type = request.args.get("dataset_type")
    mode = request.args.get("mode")  # distribution | relationship

    cfg = DATASETS.get(dataset_type)
    if not cfg:
        return jsonify({"error": "Invalid dataset type"}), 400

    # Load schema only
    if cfg["type"] == "csv":
        df = pd.read_csv(cfg["path"], nrows=1)
    else:
        with sqlite3.connect(cfg["db_path"]) as conn:
            df = pd.read_sql(
                f"SELECT * FROM {cfg['table']} LIMIT 1",
                conn
            )

    excluded = {"patient_id", "registry_date", "year", "location"}
    columns = [c for c in df.columns if c not in excluded]

    # ==================================================
    # STEP C: Relationship-safe feature list
    # ==================================================
    if mode == "relationship" or dataset_type == "integrated":
        features = []

        # ---- Race (encoded) ----
        race_cols = [
            "raceafricanamerican",
            "raceasian",
            "racecaucasian",
            "racehispanic",
            "raceother",
        ]
        if set(race_cols).issubset(set(columns)):
            features.append("race (encoded)")


        # ---- Smoking (encoded) ----
        smoking_cols = [
            "smoking_no_info",
            "smoking_current",
            "smoking_ever",
            "smoking_former",
            "smoking_never",
            "smoking_not_current",
        ]
        if set(smoking_cols).issubset(set(columns)):
            features.append("smoking (encoded)")


        # ---- Other features (exclude one-hot) ----
        for c in columns:
            if (
                not c.startswith("race")
                and not c.startswith("smoking_")
            ):
                features.append(c)

        return jsonify({"features": sorted(set(features))})

    # ==================================================
    # Default (Step B: distribution)
    # ==================================================
    return jsonify({"features": sorted(columns)})

# =========================================================
# Run app
# =========================================================
if __name__ == "__main__":
    logger.info("Starting Flask application")
    # app.run(debug=True, use_reloader=False) #--> run app.py directly
    app.run(debug=False, 
            host = "0.0.0.0",
            port = 5000) #--> run app.py via docker
