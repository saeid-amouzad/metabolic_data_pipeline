# src/visualization/snp_summary_analysis.py

"""
Step F – SNP Summary Analysis

- MAF
- Variance
- Chi-square SNP–diabetes association
- Significant SNP p-values
- Genotype frequency (overall)
- Genotype frequency stacked by diabetes

"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend (for Flask)
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

from src.utils.validators import validate_file_exists
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SNPSummaryAnalyzer:

    TARGET_COL = "diabetes"
    SNP_PREFIX = "rs"
    ALPHA = 0.05

    # assumed annotation (can be replaced later)
    SNP_GENE = "PDX1"
    SNP_CHROMOSOME = "13"
    SNP_TISSUE = "pancreas"

    def __init__(self, *, db_path: str, table_name: str):
        validate_file_exists(db_path)
        with sqlite3.connect(db_path) as conn:
            self.df = pd.read_sql(f"SELECT * FROM {table_name}", conn)

        self.snp_cols = [c for c in self.df.columns if c.startswith(self.SNP_PREFIX)]
        assert self.snp_cols, "No SNP columns found"

        self.results = None

    # --------------------------------------------------
    # Basic SNP statistics
    # --------------------------------------------------
    def compute_maf(self) -> pd.Series:
        """
        MAF = min(p, 1 - p), where p = mean(dosage) / 2
        """
        allele_freq = self.df[self.snp_cols].mean(skipna=True) / 2
        return allele_freq.apply(lambda p: min(p, 1 - p))

    def compute_variance(self) -> pd.Series:
        return self.df[self.snp_cols].var()

    # --------------------------------------------------
    # Chi-square SNP–diabetes association
    # --------------------------------------------------
    def compute_snp_chi2(self):
        rows = []

        for snp in self.snp_cols:
            sub = self.df[[snp, self.TARGET_COL]].dropna()
            geno = pd.to_numeric(sub[snp], errors="coerce").astype("Int64")
            geno = geno[geno.isin([0, 1, 2])]
            sub = sub.loc[geno.index]

            if sub.empty:
                continue

            ct = pd.crosstab(geno, sub[self.TARGET_COL])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue

            _, p, _, _ = chi2_contingency(ct)
            rows.append({"snp": snp, "p_value": p})

        res = pd.DataFrame(rows)
        rejected, p_fdr, _, _ = multipletests(
            res["p_value"], alpha=self.ALPHA, method="fdr_bh"
        )

        res["p_fdr"] = p_fdr
        res["significant"] = rejected

        self.results = res.sort_values("p_fdr")
        return self.results

    # --------------------------------------------------
    # Step F plots
    # --------------------------------------------------
    def plot_maf_histogram(self, save_path: str):
        maf = self.compute_maf()

        plt.figure()
        plt.hist(maf, bins=30)
        plt.xlabel("Minor Allele Frequency (MAF)")
        plt.ylabel("Number of SNPs")
        plt.title("SNP MAF distribution")

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        logger.info("Saved MAF histogram: %s", save_path)

    def plot_variance_histogram(self, save_path: str):
        var = self.compute_variance()

        plt.figure()
        plt.hist(var, bins=30)
        plt.xlabel("Genotype Variance")
        plt.ylabel("Number of SNPs")
        plt.title("SNP variance distribution")

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        logger.info("Saved SNP variance histogram: %s", save_path)

    def plot_significant_snps(self, save_path: str):
        sig = self.results[self.results["significant"]]
        if sig.empty:
            return

        plt.figure(figsize=(12, 5))
        plt.bar(sig["snp"], sig["p_fdr"], color="red")
        plt.yscale("log")
        plt.ylabel("FDR-corrected p-value")
        plt.xlabel("SNP")
        plt.title(
            "Significant SNPs (gene: PDX1)\nFDR-corrected Chi-square p-values"
        )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_snp_genotype_frequencies(self, save_path: str, top_k: int = 10):
        sig_snps = self.results[self.results["significant"]]["snp"].head(top_k)

        freq = []
        for snp in sig_snps:
            geno = pd.to_numeric(self.df[snp], errors="coerce").dropna().astype(int)
            geno = geno[geno.isin([0, 1, 2])]
            freq.append(geno.value_counts(normalize=True).reindex([0, 1, 2], fill_value=0))

        freq_df = pd.DataFrame(freq, index=sig_snps)

        x = np.arange(len(freq_df))
        w = 0.25

        plt.figure(figsize=(16, 6))
        plt.bar(x - w, freq_df[0], w, label="Genotype 0")
        plt.bar(x,     freq_df[1], w, label="Genotype 1")
        plt.bar(x + w, freq_df[2], w, label="Genotype 2")

        plt.xticks(x, freq_df.index, rotation=45, ha="right")
        plt.ylabel("Genotype frequency (proportion of samples)")
        plt.xlabel("SNP")
        plt.title(
            f"Genotype frequency distribution for top {len(freq_df)} significant SNPs\n"
            "Comparison of genotypes (0 / 1 / 2)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_snp_genotype_stacked_by_diabetes(self, save_path: str, top_k: int = 10):
        sig_snps = self.results[self.results["significant"]]["snp"].head(top_k)

        genotype_colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
        hatch_map = {
            0: ".",   # no diabetes
            1: "xx",   # diabetes
        }

        x = np.arange(len(sig_snps))
        w = 0.25

        plt.figure(figsize=(16, 6))

        for g_idx, g in enumerate([0, 1, 2]):
            bottoms = np.zeros(len(sig_snps))
            for outcome in [0, 1]:
                vals = []
                for snp in sig_snps:
                    sub = self.df[[snp, self.TARGET_COL]].dropna()
                    geno = pd.to_numeric(sub[snp], errors="coerce").astype("Int64")
                    geno = geno[geno.isin([0, 1, 2])]
                    sub = sub.loc[geno.index]

                    total = len(geno)
                    if total == 0:
                        vals.append(0)
                        continue

                    vals.append(((geno == g) & (sub[self.TARGET_COL] == outcome)).sum() / total)

                plt.bar(
                    x + (g_idx - 1) * w,
                    vals,
                    w,
                    bottom=bottoms,
                    color=genotype_colors[g],
                    hatch=hatch_map[outcome],
                    edgecolor="black",
                )
                bottoms += np.array(vals)

        plt.xticks(x, sig_snps, rotation=45, ha="right")
        plt.ylabel("Genotype frequency (proportion of samples)")
        plt.xlabel("SNP")
        plt.title(
            "Genotype frequency distribution for top significant SNPs\n"
            "Stacked by diabetes status, grouped by genotype"
        )
        # ---- legends ----
        genotype_legend = [
            plt.Rectangle((0, 0), 1, 1, color=genotype_colors[g])
            for g in [0, 1, 2]
        ]

        outcome_legend = [
            plt.Rectangle((0, 0), 1, 1, facecolor="white",
                        edgecolor="black", hatch=hatch_map[o])
            for o in [0, 1]
        ]

        plt.legend(
            genotype_legend + outcome_legend,
            ["Genotype 0", "Genotype 1", "Genotype 2",
            "No diabetes", "Diabetes"],
            title="Legend",
            loc="upper right",
        )
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# --------------------------------------------------
# Standalone example
# --------------------------------------------------
if __name__ == "__main__":
    analyzer = SNPSummaryAnalyzer(
        db_path="data/database.db",
        table_name="analysis_master",
    )

    analyzer.plot_maf_histogram("maf_hist.png")
    analyzer.plot_variance_histogram("snp_variance_hist.png")

    analyzer.compute_snp_chi2()

    analyzer.plot_significant_snps("significant_snps_by_gene.png")
    analyzer.plot_snp_genotype_frequencies("significant_snp_genotype_frequencies.png")
    analyzer.plot_snp_genotype_stacked_by_diabetes(
        "top10_snps_genotype_stacked_by_diabetes.png"
    )

    print("Step F – SNP summary analysis completed.")