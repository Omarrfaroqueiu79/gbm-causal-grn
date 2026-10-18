#!/usr/bin/env Rscript
# ============================================================================
# PHASE I-B: SB28 INDEPENDENT MODEL PREPROCESSING (R - Seurat)
# Project: Perturbation-Aware Causal Machine Learning Framework for
#          Glioblastoma Regulatory Network Discovery
# Manuscript: A Perturbation-Aware Causal Machine Learning Framework Identifies
#             Candidate Regulatory Hubs and Radiotherapy-Associated Network
#             Rewiring in Glioblastoma
# ============================================================================
# Purpose: Preprocess independent SB28 murine glioblastoma single-cell dataset
#          to enable cross-model topological and biological validation in Phase VI.
# Dataset: Liu et al. (2024) - In Vivo SB28 Single-Cell Perturb-seq
# Accession: NCBI GEO GSE237062 / BioProject PRJNA1005229
# ============================================================================

cat("\n============================================\n")
cat("PHASE I-B: SB28 PREPROCESSING (R - Seurat)\n")
cat("============================================\n\n")

# ----------------------------------------------------------------------------
# 1B.1 LOAD REQUIRED LIBRARIES
# ----------------------------------------------------------------------------
cat("[1/6] Loading required libraries...\n")

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(tidyverse)
  library(Matrix)
  library(ggplot2)
})

set.seed(42)

# ----------------------------------------------------------------------------
# 1B.2 CONFIGURATION & PORTABLE PATH RESOLUTION
# ----------------------------------------------------------------------------
cat("[2/6] Setting up configuration...\n")

resolve_dir <- function(env_var, candidates, fallback) {
  val <- Sys.getenv(env_var)
  if (nzchar(val) && (dir.exists(val) || file.exists(val))) return(val)
  for (cand in candidates) {
    if (dir.exists(cand) || file.exists(cand)) return(normalizePath(cand, winslash = "/"))
  }
  return(fallback)
}

DATA_DIR    <- resolve_dir("GBM_DATA_DIR", c("Dataset", "../Dataset", "../../Dataset"), "L:/Thesis Work/Dataset")
SB28_FILE   <- "SB28_micro4_20230619.Rds"
OUTPUT_DIR  <- resolve_dir("GBM_PHASE1_SB28_DIR", c("outputs/phase1_sb28", "../outputs/phase1_sb28", "../../outputs/phase1_sb28"), "L:/Thesis Work/outputs/phase1_sb28")
FIGURE_DIR  <- file.path(OUTPUT_DIR, "figures")

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)

# Also load GL261 gene list so we export overlapping genes
GL261_GENE_FILE <- resolve_dir("GBM_GL261_GENE_FILE",
                               c(file.path(OUTPUT_DIR, "..", "phase1", "gene_names.txt"),
                                 "outputs/phase1/gene_names.txt",
                                 "../../outputs/phase1/gene_names.txt"),
                               "L:/Thesis Work/outputs/phase1/gene_names.txt")

# ----------------------------------------------------------------------------
# 1B.3 LOAD SB28 DATASET
# ----------------------------------------------------------------------------
cat("[3/6] Loading SB28 dataset...\n")

sb28_path <- file.path(DATA_DIR, SB28_FILE)

if (!file.exists(sb28_path)) {
  stop(sprintf("SB28 data file not found: %s", sb28_path))
}

sb28_data <- readRDS(sb28_path)
cat(sprintf("    Loaded SB28: %d cells x %d genes\n", ncol(sb28_data), nrow(sb28_data)))

# Inspect metadata columns
cat("    Available metadata columns:\n")
cat(sprintf("    %s\n", paste(colnames(sb28_data@meta.data), collapse = ", ")))

# Inspect available assays
cat(sprintf("    Available assays: %s\n", paste(names(sb28_data@assays), collapse = ", ")))
cat(sprintf("    Default assay: %s\n", DefaultAssay(sb28_data)))

# ----------------------------------------------------------------------------
# 1B.4 QUALITY CONTROL
# ----------------------------------------------------------------------------
cat("\n[4/6] Performing quality control...\n")

# Compute mitochondrial percentage if not already present
if (!("percent.mt" %in% colnames(sb28_data@meta.data))) {
  sb28_data[["percent.mt"]] <- PercentageFeatureSet(sb28_data, pattern = "^mt-")
}

# Compute nFeature_RNA if missing
if (!("nFeature_RNA" %in% colnames(sb28_data@meta.data))) {
  sb28_data[["nFeature_RNA"]] <- colSums(GetAssayData(sb28_data, assay = "RNA", layer = "counts") > 0)
}

cat(sprintf("    Cells before QC: %d\n", ncol(sb28_data)))

# Apply QC filters (same thresholds as GL261 Phase I)
sb28_filtered <- subset(sb28_data,
                        subset = nFeature_RNA > 200 &
                          percent.mt < 20)

cat(sprintf("    Cells after QC: %d\n", ncol(sb28_filtered)))

# ----------------------------------------------------------------------------
# 1B.5 NORMALIZATION AND HVG SELECTION
# ----------------------------------------------------------------------------
cat("\n[5/6] Normalization and HVG selection...\n")

# Use existing SCT assay if available, otherwise run SCTransform
if ("SCT" %in% names(sb28_filtered@assays)) {
  DefaultAssay(sb28_filtered) <- "SCT"
  cat("    Using existing SCT assay\n")
} else {
  cat("    Running SCTransform (this may take a few minutes)...\n")
  sb28_filtered <- SCTransform(sb28_filtered, verbose = FALSE)
}

# Find or reuse variable features
n_hvg <- 3000

if (DefaultAssay(sb28_filtered) == "SCT" && length(VariableFeatures(sb28_filtered[["SCT"]])) > 0) {
  cat("    Using existing variable features from SCT assay\n")
  VariableFeatures(sb28_filtered) <- head(VariableFeatures(sb28_filtered[["SCT"]]), n_hvg)
} else {
  target_assay <- ifelse(DefaultAssay(sb28_filtered) == "SCT", "RNA", DefaultAssay(sb28_filtered))
  sb28_filtered <- FindVariableFeatures(sb28_filtered, selection.method = "vst",
                                        assay = target_assay, nfeatures = n_hvg)
}

cat(sprintf("    Selected %d highly variable genes\n", length(VariableFeatures(sb28_filtered))))

# Find genes that overlap with GL261's HVG set (for cross-model comparison)
if (file.exists(GL261_GENE_FILE)) {
  gl261_genes <- readLines(GL261_GENE_FILE)
  sb28_hvg <- VariableFeatures(sb28_filtered)
  common_genes <- intersect(gl261_genes, sb28_hvg)
  cat(sprintf("    GL261 HVG genes: %d\n", length(gl261_genes)))
  cat(sprintf("    SB28 HVG genes: %d\n", length(sb28_hvg)))
  cat(sprintf("    Overlapping HVGs between GL261 and SB28: %d\n", length(common_genes)))
  writeLines(common_genes, file.path(OUTPUT_DIR, "common_genes_gl261_sb28.txt"))
} else {
  cat("    Warning: GL261 gene list not found. Run Phase I first.\n")
  common_genes <- VariableFeatures(sb28_filtered)
}

# ----------------------------------------------------------------------------
# 1B.6 EXPORT FOR PYTHON (Phase VI)
# ----------------------------------------------------------------------------
cat("\n[6/6] Exporting data for Phase VI Python validation...\n")

# Export HVG-subset expression matrix (memory-safe for Python)
hvg_genes <- VariableFeatures(sb28_filtered)
# Use RNA assay data layer for export (normalized log-counts)
available_assays <- names(sb28_filtered@assays)
if ("RNA" %in% available_assays) {
  expr_matrix <- GetAssayData(sb28_filtered, assay = "RNA", layer = "data")[hvg_genes, ]
} else {
  expr_matrix <- GetAssayData(sb28_filtered, assay = DefaultAssay(sb28_filtered), layer = "data")[hvg_genes, ]
}

writeMM(expr_matrix, file = file.path(OUTPUT_DIR, "expression_matrix.mtx"))
cat(sprintf("    Exported expression matrix: %d genes x %d cells\n", nrow(expr_matrix), ncol(expr_matrix)))

# Export gene and cell names
writeLines(rownames(expr_matrix), file.path(OUTPUT_DIR, "gene_names.txt"))
writeLines(colnames(expr_matrix), file.path(OUTPUT_DIR, "cell_names.txt"))

# Export metadata
write.csv(sb28_filtered@meta.data, file = file.path(OUTPUT_DIR, "metadata.csv"))

# Export summary
sb28_summary <- data.frame(
  Metric = c("Total Cells (raw)", "Total Cells (QC)", "Total Genes",
             "HVG Selected", "Common Genes with GL261"),
  Value = c(ncol(sb28_data), ncol(sb28_filtered), nrow(sb28_filtered),
            length(hvg_genes), length(common_genes))
)
write.csv(sb28_summary, file = file.path(OUTPUT_DIR, "sb28_preprocessing_summary.csv"), row.names = FALSE)
print(sb28_summary)

# Save processed Seurat object
saveRDS(sb28_filtered, file = file.path(OUTPUT_DIR, "sb28_preprocessed.Rds"))

cat("\n============================================\n")
cat("PHASE I-B: SB28 PREPROCESSING COMPLETED\n")
cat(sprintf("Output saved to: %s\n", OUTPUT_DIR))
cat("============================================\n")
