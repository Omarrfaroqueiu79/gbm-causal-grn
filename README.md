# Causal Machine Learning Framework for Glioblastoma Gene Regulatory Networks

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![R 4.2+](https://img.shields.io/badge/R-4.2%2B-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete, reproducible computational biology pipeline for:

> **"A Perturbation-Aware Causal Machine Learning Framework Identifies Candidate Regulatory Hubs and Radiotherapy-Associated Network Rewiring in Glioblastoma"**  
> *Omar Faroque, Mst. Homayra Jannaty, Md. Zakariya Alam Bappi, and Suman Biswas\**  
> Department of Statistics and Data Science, Islamic University, Kushtia-7003, Bangladesh.  
> \*Corresponding Author: `suman@stat.iu.ac.bd`

---

## Overview

Inferring directed gene regulatory networks (GRNs) and resolving radiotherapy-induced rewiring from single-cell transcriptomics remains a major challenge due to high dimensionality, observational confounding, and sparsity. This framework integrates high-throughput *in vivo* single-cell CRISPR Perturb-seq data with a tri-algorithmic causal discovery ensemble, continuous constrained optimization, and experimental validation to identify master regulatory hubs and elucidate therapeutic resistance mechanisms in glioblastoma.

```
                              CRISPR Perturb-seq
                           (GL261 & SB28 Models)
                                     │
                                     ▼
                    ┌───────────────────────────────────┐
                    │  Phase I: Quality Control & SCT   │ (R - Seurat)
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                    ┌───────────────────────────────────┐
                    │  Phase II: Perturbation Mapping   │ (Python - Scanpy)
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                    ┌───────────────────────────────────┐
                    │  Phase III: Causal Discovery      │ (Python - SciPy)
                    │  (NOTEARS + GES + RCI Consensus)  │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                    ┌───────────────────────────────────┐
                    │  Phase IV: Disease-Relevant GRN   │ (Python - NetworkX)
                    │  (Hub Prioritization & Enrichment)│
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                    ┌───────────────────────────────────┐
                    │  Phase V: Treatment Rewiring      │ (Python - NetworkX)
                    │  (RT vs. noRT Network Dynamics)   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                    ┌───────────────────────────────────┐
                    │  Phase VI: Multi-Tier Validation  │ (Python & R)
                    │  (GENIE3, TRRUST v2, SB28 Model)  │
                    └───────────────────────────────────┘
```

---

## Key Findings & Benchmark Performance

| Evaluation Metric | Causal Consensus Framework | Baseline (GENIE3) | Performance Differential |
| :--- | :---: | :---: | :---: |
| **Computational F1 Score** | **0.741** | 0.457 | **+62.1% improvement** |
| **Area Under ROC (AUROC)** | **0.812** | 0.654 | **+24.2% improvement** |
| **Precision** | **0.882** | 0.463 | **+90.5% improvement** |
| **TRRUST v2 Biological F1** | **0.0827** | 0.0238 | **3.5-fold higher fidelity** |
| **SB28 Cross-Model Jaccard** | **0.687** | — | High independent generalizability |
| **SB28 Cross-Model F1** | **0.814** | — | Consistent cross-model topology |

- **Prioritized Master Regulatory Hubs:** *Eef1a1*, *Psmb8*, *Flt1*, *Vim*, *Cenpa*.
- **Radiotherapy Rewiring:** 11,916 dynamically rewired edges identified between irradiated (RT) and untreated (noRT) tumor microenvironments, highlighting immunoproteasome activation and translational elongation arrest under ionizing radiation.

---

## Repository Structure

```
├── Phase I Preprocessing.R                     # Phase I: Seurat QC, SCTransform & HVG selection
├── Phase I-B SB28 Preprocessing.R              # Phase I-B: Independent SB28 model preprocessing
├── Phase II Perturbation Mapping.py            # Phase II: Barcode mapping & intervention matrix
├── Phase II Perturbation Mapping.ipynb         # Phase II: Jupyter Notebook
├── Phase III Causal Network Inference.py       # Phase III: NOTEARS, GES, RCI & ensemble learning
├── Phase III Causal Network Inference.ipynb    # Phase III: Jupyter Notebook
├── Phase IV Gene Regulatory Network Extraction.py  # Phase IV: Hub ranking & GO/KEGG enrichments
├── Phase IV Gene Regulatory Network Extraction.ipynb # Phase IV: Jupyter Notebook
├── Phase V Treatment Context Analysis.py       # Phase V: RT vs noRT network rewiring & trajectories
├── Phase V Treatment Context Analysis.ipynb    # Phase V: Jupyter Notebook
├── Phase VI Benchmarking and Validation.py     # Phase VI: GENIE3, SB28 & TRRUST v2 validation
├── Phase VI Benchmarking and Validation.ipynb  # Phase VI: Jupyter Notebook
├── Phase VI-B Validation and Publication Figures.R # Phase VI-B: R publication figures & panels
├── requirements.txt                            # Python package dependencies
├── install_packages.R                          # R package installation script
├── .gitignore                                  # Git ignore rules for large datasets/caches
└── README.md                                   # Comprehensive documentation
```

---

## System Requirements & Prerequisites

- **Operating System:** Linux (Ubuntu 20.04+ recommended), macOS, or Windows 10/11
- **Hardware:**
  - Minimum: 16 GB RAM, 4 CPU cores
  - Recommended: 32 GB RAM, 8+ CPU cores (required for single-shot SCTransform on >100,000 cells)
- **Software Environments:**
  - Python $\ge$ 3.9
  - R $\ge$ 4.2

---

## Installation Guide

### 1. Python Environment Setup

Clone this repository and create a dedicated virtual environment:

```bash
# Create and activate environment
conda create -n causal_gbm python=3.10 -y
conda activate causal_gbm

# Install required Python packages
pip install -r requirements.txt
```

### 2. R Environment Setup

Launch R or run the automated package installer:

```bash
Rscript install_packages.R
```

---

## Data Availability & Preparation

The raw single-cell CRISPR Perturb-seq datasets were generated by Liu et al. (2024) and are publicly accessible through NCBI:
- **BioProject Accession:** `PRJNA1005229`
- **NCBI GEO Accession:** `GSE237062`

Place the processed `.Rds` objects into your local `Dataset/` directory:
- `GL261_integrated_20230619.Rds` (Primary Discovery Cohort: 127,774 cells)
- `SB28_micro4_20230619.Rds` (Independent Validation Cohort)

*Note on Path Portability:*  
All scripts automatically detect relative directory layouts (`./Dataset` and `./outputs`). Alternatively, you may set custom locations via environment variables:
```bash
export GBM_DATA_PATH="/path/to/Dataset"
export GBM_OUTPUT_PATH="/path/to/outputs"
```

---

## Execution Guide

Run each phase sequentially from the command line or open the corresponding Jupyter notebooks:

```bash
# Phase I: Data Preprocessing & Quality Control (R)
Rscript "Phase I Preprocessing.R"

# Phase I-B: Independent Validation Model Preprocessing (R)
Rscript "Phase I-B SB28 Preprocessing.R"

# Phase II: Perturbation Mapping & Cell-State Assignment (Python)
python "Phase II Perturbation Mapping.py"

# Phase III: Causal Discovery via NOTEARS, GES, & RCI (Python)
python "Phase III Causal Network Inference.py"

# Phase IV: Causal GRN Extraction & Hub Prioritization (Python)
python "Phase IV Gene Regulatory Network Extraction.py"

# Phase V: Treatment Context Rewiring Analysis (Python)
python "Phase V Treatment Context Analysis.py"

# Phase VI: Computational Benchmarking & Validation (Python)
python "Phase VI Benchmarking and Validation.py"

# Phase VI-B: Publication Figures & Cross-Phase Integration (R)
Rscript "Phase VI-B Validation and Publication Figures.R"
```

---

## Citation

If you use this codebase, methodology, or inferred networks in your research, please cite:

```bibtex
@article{faroque2026perturbation,
  title={A Perturbation-Aware Causal Machine Learning Framework Identifies Candidate Regulatory Hubs and Radiotherapy-Associated Network Rewiring in Glioblastoma},
  author={Faroque, Omar and Jannaty, Mst. Homayra and Bappi, Md. Zakariya Alam and Biswas, Suman},
  journal={npj Systems Biology and Applications},
  year={2026},
  publisher={Nature Portfolio}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
