# -*- coding: utf-8 -*-
"""
Feature selection / reduction: Remove redundancy features.
Reduces feature count

Examples:
    Remove correlated phenotypic features
    Remove low-variance SNPs
    
| Step                             | Explanation                                                                |
| 1. Variance filtering            | Removes useless SNPs quickly, reduces dimensionality dramatically.
| 2. Target correlation filtering  | Removes SNPs irrelevant to diabetes.
| 3. SNP–SNP correlation filtering | Removes redundancy only among meaningful SNPs.

"""
# src/features/feature_reduction.py

import pandas as pd
import numpy as np
import yaml
from typing import List

class FeatureReducer:
    """
    Performs:
      1. Remove correlated phenotypic features (> correlation_threshold)
      2. Remove phenotypic features weakly correlated with target (< target_corr_threshold)
      3. Remove low-variance SNPs (< min_snp_variance)
      4. Remove SNPs weakly correlated with target (< snp_target_corr_threshold)
      5. Remove highly correlated SNPs (> correlation_threshold_snp)
    """

    def __init__(
        self,
        correlation_threshold: float = 0.80,         # pheno-pheno correlation
        target_corr_threshold: float = 0.20,         # pheno-target corr cutoff
        min_snp_variance: float = 0.01,              # SNP variance filtering
        snp_target_corr_threshold: float = 0.05,     # SNP–target correlation
        correlation_threshold_snp: float = 0.98,     # SNP–SNP correlation removal
        log_path: str = "config/feature_selection_log.yaml"
    ):
        self.correlation_threshold = correlation_threshold
        self.target_corr_threshold = target_corr_threshold
        self.min_snp_variance = min_snp_variance
        self.snp_target_corr_threshold = snp_target_corr_threshold
        self.correlation_threshold_snp = correlation_threshold_snp
        self.log_path = log_path
        
        self.selected_phenotypic = None
        self.dropped_phenotypic = None
        self.selected_snps = None
        self.dropped_snps = None

    def load_selected_features(self, path: str = None):
        path = path or self.log_path

        with open(path, "r") as f:
            log = yaml.safe_load(f)

        self.selected_phenotypic = log["phenotypic"]["selected"]
        self.selected_snps = log["genotype"]["selected"]

        self.dropped_phenotypic = log["phenotypic"]["dropped"]
        self.dropped_snps = log["genotype"]["dropped"]


    def fit(self, df, phenotypic_cols, snp_cols, target_col):
    
        # --- Safety check: target_col must be a column name ---
        if not isinstance(target_col, str):
            raise ValueError(
                f"target_col must be a string (column name), but got {type(target_col)}.\n"
                f"Example correct usage: reducer.fit(df, pheno, snp, target_col='diabetes')"
            )
    
        if target_col not in df.columns:
            raise KeyError(
                f"target_col '{target_col}' does not exist in DataFrame columns.\n"
                f"Available columns: {list(df.columns)[:10]} ..."
            )
            
        # --- Phenotypic correlation filtering ---
        # ----------------------------------------------------------
        # 1. REMOVE HIGHLY CORRELATED PHENOTYPIC FEATURES
        # ----------------------------------------------------------
        corr = df[phenotypic_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        dropped_corr = [
            c for c in upper.columns
            if any(upper[c] > self.correlation_threshold)
        ]

        # ----------------------------------------------------------
        # 2. REMOVE PHENOTYPIC FEATURES WITH LOW CORRELATION TO TARGET
        # ----------------------------------------------------------
        target_corr = df[phenotypic_cols].corrwith(df[target_col]).abs()
        dropped_low_target = target_corr[target_corr < self.target_corr_threshold].index.tolist()

        self.dropped_phenotypic = sorted(set(dropped_corr + dropped_low_target))

        self.selected_phenotypic = [
            c for c in phenotypic_cols if c not in self.dropped_phenotypic
        ]

        # ----------------------------------------------------------
        # 3. LOW-VARIANCE SNP FILTERING
        # ----------------------------------------------------------
        variances = df[snp_cols].var()
        selected_by_variance = variances[variances >= self.min_snp_variance].index.tolist()

        # Keep only the SNPs that passed variance filtering
        snps_after_variance = selected_by_variance

        # ----------------------------------------------------------
        # 4. SNP-TARGET CORRELATION FILTERING
        # ----------------------------------------------------------
        snp_target_corr = df[snps_after_variance].corrwith(df[target_col]).abs()
        selected_by_target_corr = snp_target_corr[
            snp_target_corr >= self.snp_target_corr_threshold
        ].index.tolist()

        snps_after_target = selected_by_target_corr

        # ----------------------------------------------------------
        # 5. SNP–SNP CORRELATION REMOVAL (High redundancy)
        # ----------------------------------------------------------
        if len(snps_after_target) > 1:
            snp_corr = df[snps_after_target].corr().abs()
            upper_snp = snp_corr.where(np.triu(np.ones(snp_corr.shape), k=1).astype(bool))

            dropped_snp_corr = [
                c for c in upper_snp.columns
                if any(upper_snp[c] > self.correlation_threshold_snp)
            ]
        else:
            dropped_snp_corr = []

        # Final selected SNP list
        self.selected_snps = [
            c for c in snps_after_target if c not in dropped_snp_corr
        ]

        # Combine all dropped SNPs
        self.dropped_snps = [
            c for c in snp_cols if c not in self.selected_snps
        ]

        # Save logs
        self._log_feature_selection()

        return self

    def transform(self, df):
    
        assert self.selected_phenotypic is not None
        assert self.selected_snps is not None

        return df[self.selected_phenotypic + self.selected_snps]

    def _log_feature_selection(self):
        log = {
            "phenotypic": {
                "selected": self.selected_phenotypic,
                "dropped": self.dropped_phenotypic,
                "thresholds": {
                    "phenotype_correlation_threshold": self.correlation_threshold,
                    "phenotype_target_corr_threshold": self.target_corr_threshold,
                }
            },
            "genotype": {
                "selected": self.selected_snps,
                "dropped": self.dropped_snps,
                "thresholds": {
                    "snp_variance_threshold": self.min_snp_variance,
                    "snp_target_corr_threshold": self.snp_target_corr_threshold,
                    "snp_correlation_threshold": self.correlation_threshold_snp
                }
            }
        }
    
        with open(self.log_path, "w") as f:
            yaml.safe_dump(log, f, sort_keys=False)
