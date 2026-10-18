#!/usr/bin/env Rscript
# ============================================================================
# PHASE 1: DATA PREPROCESSING (R - Seurat)
# Project: Perturbation-Aware Causal Machine Learning Framework for
#          Glioblastoma Regulatory Network Discovery
# Manuscript: A Perturbation-Aware Causal Machine Learning Framework Identifies
#             Candidate Regulatory Hubs and Radiotherapy-Associated Network
#             Rewiring in Glioblastoma
# ============================================================================
# Dataset: Liu et al. (2024) - In Vivo Single-Cell CRISPR Perturb-seq (GL261)
# Accession: NCBI GEO GSE237062 / BioProject PRJNA1005229
# ============================================================================

# PHASE 1 START ===============================================================

cat("\n========================================\n")
cat("PHASE 1: DATA PREPROCESSING (R - Seurat)\n")
cat("========================================\n\n")

# ----------------------------------------------------------------------------
# 1.1 LOAD REQUIRED LIBRARIES
# ----------------------------------------------------------------------------
cat("[1/8] Loading required libraries...\n")

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(tidyverse)
  library(Matrix)
  library(future)
  library(future.apply)
  library(ggplot2)
  library(cowplot)
  library(patchwork)
  library(viridis)
  library(RColorBrewer)
  library(yaml)
})

# Set parallel processing
plan("multicore", workers = 4)
options(future.globals.maxSize = 16 * 1024^3)

# ----------------------------------------------------------------------------
# 1.2 LOAD CONFIGURATION & PORTABLE PATH RESOLUTION
# ----------------------------------------------------------------------------
cat("[2/8] Loading configuration...\n")

# Dynamic directory resolution for cross-platform portability
resolve_dir <- function(env_var, candidates, fallback) {
  val <- Sys.getenv(env_var)
  if (nzchar(val) && (dir.exists(val) || file.exists(val))) return(val)
  for (cand in candidates) {
    if (dir.exists(cand) || file.exists(cand)) return(normalizePath(cand, winslash = "/"))
  }
  return(fallback)
}

default_data_dir <- resolve_dir("GBM_DATA_DIR", c("Dataset", "../Dataset", "../../Dataset"), "L:/Thesis Work/Dataset")
default_phase1_dir <- resolve_dir("GBM_PHASE1_DIR", c("outputs/phase1", "../outputs/phase1", "../../outputs/phase1"), "L:/Thesis Work/outputs/phase1")

config_file <- "config.yaml"
if (file.exists(config_file)) {
  config <- yaml::read_yaml(config_file)
} else {
  cat("Notice: config.yaml not found. Using default paths with dynamic environment resolution.\n")
  config <- list(
    paths = list(
      data_dir = default_data_dir,
      gl261_file = "GL261_integrated_20230619.Rds",
      sb28_file = "SB28_micro4_20230619.Rds",
      phase1_dir = default_phase1_dir
    ),
    data = list(
      qc = list(min_genes = 200, max_mt_percent = 20),
      hvg = list(n_top_genes = 3000),
      test_mode = FALSE
    ),
    compute = list(random_seed = 42)
  )
}

# Set seed
set.seed(config$compute$random_seed)

# Create output directory
dir.create(config$paths$phase1_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------------------------------------------------------
# 1.3 LOAD PRIMARY DATASET: GL261
# ----------------------------------------------------------------------------
cat("[3/8] Loading GL261 dataset...\n")

gl261_path <- file.path(config$paths$data_dir, config$paths$gl261_file)

if (file.exists(gl261_path)) {
  gl261_data <- readRDS(gl261_path)
  cat(sprintf("    [DONE] Loaded GL261: %d cells x %d genes\n", ncol(gl261_data), nrow(gl261_data)))
} else {
  cat(sprintf("    ⚠ Data file not found: %s\n", gl261_path))
  cat("    Creating simulated data for testing...\n")
  
  set.seed(42)
  n_cells <- 10000
  n_genes <- 3000
  
  counts_matrix <- as(matrix(rpois(n_cells * n_genes, lambda = 2), nrow = n_genes), "dgCMatrix")
  rownames(counts_matrix) <- paste0("Gene", 1:n_genes)
  colnames(counts_matrix) <- paste0("Cell", 1:n_cells)
  
  gl261_data <- CreateSeuratObject(counts = counts_matrix, project = "GL261")
  
  gl261_data$orig.ident <- sample(c("GL261_noRT_preinf_1", "GL261_RT_CED", "GL261_48hit_noRT_1"), n_cells, replace = TRUE)
  gl261_data$nCount_RNA <- colSums(counts_matrix)
  gl261_data$nFeature_RNA <- apply(counts_matrix, 2, function(x) sum(x > 0))
  gl261_data$percent.mt <- runif(n_cells, 0, 20)
  gl261_data$sorted <- sample(c("MACSFACS", "unsorted", "invitro"), n_cells, replace = TRUE)
  gl261_data$source <- sample(c("preinf", "CED", "invitro"), n_cells, replace = TRUE)
  gl261_data$cond <- sample(c("noRT", "RT"), n_cells, replace = TRUE)
  gl261_data$seurat_clusters <- as.factor(sample(0:14, n_cells, replace = TRUE))
  gl261_data$scMRMA_manual <- sample(c("GL261 Tumor", "Macrophages", "Microglia", "T cells",
                                       "Oligodendrocytes", "NK cells", "Endothelial"),
                                     n_cells, replace = TRUE)
  gl261_data$sgRNA <- sample(c("NTC", "Bora", "Trp53", "Myc", "Pten", "None"), n_cells, replace = TRUE)
  gl261_data$sgRNA_binary <- ifelse(gl261_data$sgRNA == "None", 0, 1)
  gl261_data$sgRNA_UMI <- rpois(n_cells, lambda = 5)
  gl261_data$sgRNA_logUMI <- log1p(gl261_data$sgRNA_UMI)
  
  sct_assay <- CreateAssayObject(counts = counts_matrix)
  sct_assay <- SetAssayData(sct_assay, layer = "data", new.data = log1p(counts_matrix))
  gl261_data[["SCT"]] <- sct_assay
  
  gl261_data$nCount_SCT <- gl261_data$nCount_RNA
  gl261_data$nFeature_SCT <- gl261_data$nFeature_RNA
  
  pca_mat <- matrix(rnorm(n_cells * 50), nrow = n_cells)
  rownames(pca_mat) <- colnames(gl261_data)
  colnames(pca_mat) <- paste0("PC_", 1:50)
  gl261_data[["pca"]] <- CreateDimReducObject(embeddings = pca_mat, key = "PC_", assay = "SCT")
  
  umap_mat <- matrix(rnorm(n_cells * 2), nrow = n_cells)
  rownames(umap_mat) <- colnames(gl261_data)
  colnames(umap_mat) <- paste0("UMAP_", 1:2)
  gl261_data[["umap"]] <- CreateDimReducObject(embeddings = umap_mat, key = "UMAP_", assay = "SCT")
  
  DefaultAssay(gl261_data) <- "SCT"
}

# ----------------------------------------------------------------------------
# 1.4 QUALITY CONTROL
# ----------------------------------------------------------------------------
cat("[4/8] Performing quality control...\n")

if (!("percent.mt" %in% colnames(gl261_data@meta.data))) {
  gl261_data[["percent.mt"]] <- PercentageFeatureSet(gl261_data, pattern = "^mt-")
}

qc_params <- config$data$qc
gl261_filtered <- subset(gl261_data,
                         subset = nFeature_RNA > qc_params$min_genes &
                           percent.mt < qc_params$max_mt_percent)

cat(sprintf("    Cells before QC: %d\n", ncol(gl261_data)))
cat(sprintf("    Cells after QC: %d\n", ncol(gl261_filtered)))

# Group the identities by cell type class definitions
Idents(gl261_filtered) <- "scMRMA_manual"

# Publication-ready quality control violin plot (Nature Portfolio theme: sans-serif font, clear axes)
p1 <- VlnPlot(gl261_filtered, 
              features = c("nFeature_RNA", "nCount_RNA", "percent.mt"),
              ncol = 3, 
              pt.size = 0) & 
  theme_classic(base_size = 12, base_family = "sans") &
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, color = "black"),
    axis.text.y = element_text(color = "black"),
    axis.line = element_line(color = "black", size = 0.5),
    axis.ticks = element_line(color = "black", size = 0.5),
    axis.title.x = element_blank(),
    plot.title = element_blank(),
    legend.position = "none"
  )

p2 <- FeatureScatter(gl261_filtered, feature1 = "nCount_RNA", feature2 = "nFeature_RNA", group.by = "scMRMA_manual") +
  geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed") +
  theme_classic(base_size = 12, base_family = "sans") +
  theme(
    plot.title = element_blank(),
    axis.text = element_text(color = "black"),
    axis.line = element_line(color = "black", size = 0.5),
    axis.ticks = element_line(color = "black", size = 0.5),
    legend.text = element_text(size = 10, color = "black")
  )

# Widescreen outputs formatting setup to prevent layout overlap crashes
ggsave(filename = file.path(config$paths$phase1_dir, "figure_1a_qc_violin.png"), plot = p1, width = 16, height = 6, dpi = 300)
ggsave(filename = file.path(config$paths$phase1_dir, "figure_1b_qc_scatter.png"), plot = p2, width = 11, height = 6, dpi = 300)

# ----------------------------------------------------------------------------
# 1.5 NORMALIZATION AND HVG SELECTION
# ----------------------------------------------------------------------------
cat("[5/8] Normalization and HVG selection...\n")

if ("SCT" %in% names(gl261_filtered@assays)) {
  DefaultAssay(gl261_filtered) <- "SCT"
  cat("    Using existing SCT assay\n")
} else {
  cat("    Running SCTransform...\n")
  gl261_filtered <- SCTransform(gl261_filtered, verbose = FALSE)
}

hvg_params <- config$data$hvg

if (DefaultAssay(gl261_filtered) == "SCT" && length(VariableFeatures(gl261_filtered[["SCT"]])) > 0) {
  cat("    Using existing variable features from SCT assay\n")
  VariableFeatures(gl261_filtered) <- head(VariableFeatures(gl261_filtered[["SCT"]]), hvg_params$n_top_genes)
} else {
  target_assay <- ifelse(DefaultAssay(gl261_filtered) == "SCT", "RNA", DefaultAssay(gl261_filtered))
  if (ncol(gl261_filtered) > 50000) {
    cat("    Large dataset detected. Finding variable features on a random subset to prevent memory allocation errors...\n")
    subset_idx <- sample(1:ncol(gl261_filtered), 50000)
    gl261_sub <- gl261_filtered[, subset_idx]
    gl261_sub <- FindVariableFeatures(gl261_sub, selection.method = "vst", assay = target_assay, nfeatures = hvg_params$n_top_genes)
    VariableFeatures(gl261_filtered) <- VariableFeatures(gl261_sub)
    rm(gl261_sub)
    gc()
  } else {
    gl261_filtered <- FindVariableFeatures(gl261_filtered, selection.method = "vst", assay = target_assay, nfeatures = hvg_params$n_top_genes)
  }
}

cat(sprintf("    Selected %d highly variable genes\n", length(VariableFeatures(gl261_filtered))))

# ----------------------------------------------------------------------------
# 1.6 DIMENSIONALITY REDUCTION
# ----------------------------------------------------------------------------
cat("[6/8] Running dimensionality reduction...\n")

if (!("pca" %in% names(gl261_filtered@reductions))) {
  gl261_filtered <- RunPCA(gl261_filtered, features = VariableFeatures(object = gl261_filtered))
}

if (!("umap" %in% names(gl261_filtered@reductions))) {
  gl261_filtered <- FindNeighbors(gl261_filtered, dims = 1:30)
  gl261_filtered <- FindClusters(gl261_filtered, resolution = 0.8)
  gl261_filtered <- RunUMAP(gl261_filtered, dims = 1:30)
}

p3 <- DimPlot(gl261_filtered, reduction = "umap", group.by = "cond") +
  scale_color_manual(values = c("noRT" = "#2E86AB", "RT" = "#A23B72")) +
  theme_classic(base_size = 12, base_family = "sans") +
  theme(plot.title = element_blank(), axis.text = element_blank(), axis.ticks = element_blank(), axis.line = element_blank()) # Clean UMAPs often drop axes

p4 <- DimPlot(gl261_filtered, reduction = "umap", group.by = "scMRMA_manual") +
  theme_classic(base_size = 12, base_family = "sans") +
  theme(plot.title = element_blank(), axis.text = element_blank(), axis.ticks = element_blank(), axis.line = element_blank())

ggsave(filename = file.path(config$paths$phase1_dir, "figure_1c_umap_condition.png"), plot = p3, width = 10, height = 8, dpi = 300)
ggsave(filename = file.path(config$paths$phase1_dir, "figure_1d_umap_celltype.png"), plot = p4, width = 12, height = 8, dpi = 300)

# ----------------------------------------------------------------------------
# 1.7 CELL TYPE AND PERTURBATION SUMMARY
# ----------------------------------------------------------------------------
cat("[7/8] Generating cell type and perturbation summaries...\n")

celltype_summary <- gl261_filtered@meta.data %>%
  group_by(scMRMA_manual) %>%
  summarise(n_cells = n(), pct_total = n() / nrow(.) * 100, .groups = "drop") %>%
  arrange(desc(n_cells))
write.csv(celltype_summary, file = file.path(config$paths$phase1_dir, "celltype_summary.csv"), row.names = FALSE)

perturb_summary <- gl261_filtered@meta.data %>%
  group_by(sgRNA) %>%
  summarise(n_cells = n(), mean_umi = mean(sgRNA_UMI, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(n_cells))
write.csv(perturb_summary, file = file.path(config$paths$phase1_dir, "perturbation_summary.csv"), row.names = FALSE)

treatment_summary <- gl261_filtered@meta.data %>%
  group_by(cond) %>%
  summarise(n_cells = n(), .groups = "drop")

p5 <- ggplot(treatment_summary, aes(x = cond, y = n_cells, fill = cond)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("noRT" = "#2E86AB", "RT" = "#A23B72")) +
  labs(x = "Treatment", y = "Number of Cells") +
  theme_classic(base_size = 12, base_family = "sans") + 
  theme(
    legend.position = "none", 
    plot.title = element_blank(),
    axis.text = element_text(color = "black"),
    axis.line = element_line(color = "black", size = 0.5),
    axis.ticks = element_line(color = "black", size = 0.5)
  )

ggsave(filename = file.path(config$paths$phase1_dir, "figure_1e_treatment_counts.png"), plot = p5, width = 8, height = 6, dpi = 300)

# ----------------------------------------------------------------------------
# 1.8 EXPORT FOR PYTHON PHASES
# ----------------------------------------------------------------------------
cat("[8/8] Exporting data for Python phases...\n")

saveRDS(gl261_filtered, file = file.path(config$paths$phase1_dir, "gl261_preprocessed.Rds"))

# Subset to highly variable genes to prevent memory crashes in Python
hvg_genes <- VariableFeatures(gl261_filtered)
expr_matrix <- GetAssayData(gl261_filtered, assay = "RNA", layer = "data")[hvg_genes, ]
writeMM(expr_matrix, file = file.path(config$paths$phase1_dir, "expression_matrix.mtx"))

write.csv(gl261_filtered@meta.data, file = file.path(config$paths$phase1_dir, "metadata.csv"))
writeLines(rownames(expr_matrix), file.path(config$paths$phase1_dir, "gene_names.txt"))
writeLines(colnames(expr_matrix), file.path(config$paths$phase1_dir, "cell_names.txt"))

if ("pca" %in% names(gl261_filtered@reductions)) {
  write.csv(Embeddings(gl261_filtered, "pca"), file = file.path(config$paths$phase1_dir, "pca_embeddings.csv"))
}
if ("umap" %in% names(gl261_filtered@reductions)) {
  write.csv(Embeddings(gl261_filtered, "umap"), file = file.path(config$paths$phase1_dir, "umap_embeddings.csv"))
}

phase1_summary <- data.frame(
  Metric = c("Total Cells", "Total Genes", "HVG Selected", "noRT Cells", "RT Cells", "Cell Types"),
  Value = c(ncol(gl261_filtered), nrow(gl261_filtered), length(VariableFeatures(gl261_filtered)),
            sum(gl261_filtered$cond == "noRT", na.rm = TRUE), sum(gl261_filtered$cond == "RT", na.rm = TRUE),
            length(unique(gl261_filtered$scMRMA_manual)))
)
write.csv(phase1_summary, file = file.path(config$paths$phase1_dir, "phase1_summary.csv"), row.names = FALSE)

cat("\n========================================\n")
cat("PHASE 1 COMPLETED SUCCESSFULLY\n")
cat("========================================\n")
