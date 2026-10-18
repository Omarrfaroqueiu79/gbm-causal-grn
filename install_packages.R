#!/usr/bin/env Rscript
# ============================================================================
# R Package Dependencies Installer
# Causal Machine Learning Framework for Glioblastoma Gene Regulatory Networks
# ============================================================================

cat("Checking and installing required R packages...\n\n")

cran_packages <- c(
  "Seurat",
  "SeuratObject",
  "tidyverse",
  "Matrix",
  "future",
  "future.apply",
  "ggplot2",
  "cowplot",
  "patchwork",
  "viridis",
  "RColorBrewer",
  "yaml",
  "igraph"
)

# Function to check and install missing packages
install_if_missing <- function(packages) {
  missing_pkgs <- packages[!(packages %in% installed.packages()[, "Package"])]
  if (length(missing_pkgs) > 0) {
    cat(sprintf("Installing %d missing package(s): %s\n", length(missing_pkgs), paste(missing_pkgs, collapse = ", ")))
    install.packages(missing_pkgs, repos = "https://cloud.r-project.org")
  } else {
    cat("All CRAN packages are already installed.\n")
  }
}

install_if_missing(cran_packages)

cat("\nR package installation check complete.\n")
