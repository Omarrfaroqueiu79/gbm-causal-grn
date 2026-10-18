"""
============================================================================
PHASE II: PERTURBATION MAPPING AND CELL-STATE ANNOTATION (Python)
============================================================================
Project: Perturbation-Aware Causal Machine Learning Framework for 
         Glioblastoma Regulatory Network Discovery
Manuscript: A Perturbation-Aware Causal Machine Learning Framework Identifies
            Candidate Regulatory Hubs and Radiotherapy-Associated Network
            Rewiring in Glioblastoma
============================================================================
Description:
Maps CRISPR single-guide RNA (sgRNA) perturbation barcodes to single-cell
transcriptomes, identifies cell states, constructs binary intervention
matrices, and evaluates perturbation efficiency and downstream effects.
============================================================================
"""

# --- PHASE II START ---

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import os
import warnings
warnings.filterwarnings('ignore')

# Publication visualization standards (Nature Portfolio style)
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['text.color'] = 'black'
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
sns.set_style("ticks")

# Set random seed for reproducibility
np.random.seed(42)

# Dynamic Directory Resolution for Cross-Platform Portability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

OUTPUT_PATH = os.environ.get(
    "GBM_OUTPUT_PATH",
    os.path.join(REPO_ROOT, "outputs") if os.path.exists(os.path.join(REPO_ROOT, "outputs"))
    else (os.path.join(SCRIPT_DIR, "outputs") if os.path.exists(os.path.join(SCRIPT_DIR, "outputs"))
          else "L:/Thesis Work/outputs")
)
PHASE1_PATH = os.path.join(OUTPUT_PATH, "phase1")
PHASE2_PATH = os.path.join(OUTPUT_PATH, "phase2")
FIGURE_PATH = os.path.join(OUTPUT_PATH, "figures", "phase2")

# Create directories
os.makedirs(PHASE2_PATH, exist_ok=True)
os.makedirs(FIGURE_PATH, exist_ok=True)

print("=" * 80)
print("PHASE II: PERTURBATION MAPPING AND CELL-STATE ANNOTATION")
print("=" * 80)
print()

# 2.1 Load Data from Phase I
print("[1/5] Loading data from Phase I...")

# Load expression matrix from MTX format (saved by R)
mtx_path = os.path.join(PHASE1_PATH, "expression_matrix.mtx")
if not os.path.exists(mtx_path):
    raise FileNotFoundError(f"Expression matrix not found at {mtx_path}. Ensure Phase I was run correctly.")

print("    Loading MTX expression matrix (this may take a moment)...")
# Note: R's writeMM writes genes as rows and cells as columns. 
# sc.read_mtx reads it as cells as rows if we don't transpose, but we want cells as obs.
# We read and transpose to match AnnData convention (cells x genes).
adata = sc.read_mtx(mtx_path).T

# Load metadata
metadata = pd.read_csv(os.path.join(PHASE1_PATH, "metadata.csv"), index_col=0)

# Load gene names
with open(os.path.join(PHASE1_PATH, "gene_names.txt"), 'r') as f:
    gene_names = [line.strip() for line in f]

# Load cell names
with open(os.path.join(PHASE1_PATH, "cell_names.txt"), 'r') as f:
    cell_names = [line.strip() for line in f]

# Assign annotations to AnnData object
adata.obs_names = cell_names
adata.var_names = gene_names
# Filter metadata to match cell names and assign
adata.obs = metadata.loc[cell_names].copy()

print(f"    Loaded AnnData: {adata.n_obs} cells x {adata.n_vars} genes")

# 2.2 Perturbation Mapping Analysis
print("\n[2/5] Analyzing Perturbation Distribution...")

# Clean sgRNA column (handle NaNs and empty strings)
adata.obs['sgRNA'] = adata.obs['sgRNA'].fillna('Unassigned')
adata.obs['sgRNA'] = adata.obs['sgRNA'].replace({'NA': 'Unassigned', '': 'Unassigned'})

# Summary of perturbations (exclude Unassigned for clear counts if desired, but good to see all)
perturb_counts = adata.obs['sgRNA'].value_counts()
print(f"    Total unique perturbation labels (including Unassigned/Controls): {len(perturb_counts)}")
print(f"    Top 10 perturbations:")
print(perturb_counts.head(10))

# Create perturbation matrix
perturbations = [p for p in adata.obs['sgRNA'].unique() if p != 'Unassigned']
perturb_matrix = pd.DataFrame(0, index=adata.obs_names, columns=perturbations)
for sg in perturbations:
    perturb_matrix.loc[adata.obs['sgRNA'] == sg, sg] = 1

# Save perturbation matrix
perturb_matrix.to_csv(os.path.join(PHASE2_PATH, "perturbation_matrix.csv"))

# 2.3 Create Intervention Matrix for Causal Analysis
print("\n[3/5] Creating Intervention Matrix...")

# Identify control sgRNAs
control_guides = ['non-targeting', 'non-targeting_B', 'NTC', 'None', 'Unassigned']

def is_perturbed(sg):
    if pd.isna(sg) or sg in control_guides:
        return 0
    return 1

# Binary intervention matrix (1 = perturbed, 0 = control/NTC/Unassigned)
intervention_matrix = pd.DataFrame({
    'intervention': adata.obs['sgRNA'].apply(is_perturbed),
    'sgRNA': adata.obs['sgRNA'],
    'sgRNA_binary': adata.obs.get('sgRNA_binary', adata.obs['sgRNA'].apply(is_perturbed))
}, index=adata.obs_names)

intervention_matrix.to_csv(os.path.join(PHASE2_PATH, "intervention_matrix.csv"))
print(f"    Cells with defined perturbations: {intervention_matrix['intervention'].sum()}")
print(f"    Control/Unassigned cells: {(intervention_matrix['intervention'] == 0).sum()}")

# Create one-hot encoded intervention matrix for causal methods (exclude controls from features)
valid_sgrnas = adata.obs['sgRNA'].replace({c: np.nan for c in control_guides})
intervention_onehot = pd.get_dummies(valid_sgrnas, dummy_na=False)
intervention_onehot.to_csv(os.path.join(PHASE2_PATH, "intervention_onehot.csv"))

# 2.4 Cell-State Annotation Summary
print("\n[4/5] Cell-State Annotation Summary...")

# Handle potential missing column gracefully
if 'scMRMA_manual' not in adata.obs.columns:
    print("    Warning: 'scMRMA_manual' not found. Using 'seurat_clusters' as cell types.")
    adata.obs['scMRMA_manual'] = 'Cluster_' + adata.obs['seurat_clusters'].astype(str)

# Cell type distribution
ctype_dist = adata.obs['scMRMA_manual'].value_counts()
print("    Cell type distribution:")
print(ctype_dist.head(10))

# Treatment condition by cell type
if 'cond' in adata.obs.columns:
    cond_by_ctype = pd.crosstab(adata.obs['scMRMA_manual'], adata.obs['cond'])
    print("\n    Treatment condition by cell type:")
    print(cond_by_ctype.head(5))
else:
    cond_by_ctype = None
    print("\n    Warning: 'cond' column not found in metadata.")

# Perturbation by cell type
pert_by_ctype = pd.crosstab(adata.obs['scMRMA_manual'], adata.obs['sgRNA'])
print("\n    Top perturbations by cell type (first 5 columns):")
print(pert_by_ctype.iloc[:5, :5])

# 2.5 Generate Perturbation Analysis Figures
print("\n[5/5] Generating Figures...")

# Figure 2A: Perturbation Distribution (excluding Unassigned for clarity in plot)
fig, ax = plt.subplots(figsize=(12, 6))
plot_counts = perturb_counts.drop('Unassigned', errors='ignore')
plot_counts.head(20).plot(kind='bar', ax=ax, color='steelblue')
ax.set_xlabel('sgRNA Target', fontsize=12)
ax.set_ylabel('Number of Cells', fontsize=12)
# ax.set_title removed for publication style
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, 'figure_2a_perturbation_distribution.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'figure_2a_perturbation_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 2B: Cell Type Distribution
fig, ax = plt.subplots(figsize=(10, 6))
ctype_dist.head(15)[::-1].plot(kind='barh', ax=ax, color='forestgreen')
ax.set_xlabel('Number of Cells', fontsize=12)
ax.set_ylabel('Cell Type', fontsize=12)
# ax.set_title removed for publication style
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, 'figure_2b_celltype_distribution.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'figure_2b_celltype_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 2C: Treatment Condition Distribution
if cond_by_ctype is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Overall distribution
    cond_dist = adata.obs['cond'].value_counts()
    axes[0].pie(cond_dist.values, labels=cond_dist.index, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Treatment Condition Distribution', fontsize=12, fontweight='bold')
    
    # By cell type (stacked bar, top 10 cell types for clarity)
    cond_by_ctype_top = cond_by_ctype.loc[ctype_dist.head(10).index]
    cond_by_ctype_top[::-1].plot(kind='barh', stacked=True, ax=axes[1])
    axes[1].set_xlabel('Number of Cells', fontsize=12)
    axes[1].set_ylabel('Cell Type', fontsize=12)
    # axes[1].set_title removed for publication style
    axes[1].legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    
    # plt.suptitle removed for publication style
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_PATH, 'figure_2c_treatment_distribution.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURE_PATH, 'figure_2c_treatment_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Figure 2D: Perturbation Heatmap (Top genes x Top Cell types)
top_perturbs = plot_counts.head(15).index
top_ctypes = ctype_dist.head(10).index
pert_heatmap_data = pert_by_ctype.loc[top_ctypes, top_perturbs]

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(pert_heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
ax.set_xlabel('sgRNA Target', fontsize=12)
ax.set_ylabel('Cell Type', fontsize=12)
# ax.set_title removed for publication style
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, 'figure_2d_perturbation_heatmap.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'figure_2d_perturbation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 2E: Intervention Status by Cell Type
fig, ax = plt.subplots(figsize=(10, 6))
# Create a local binary column based on our robust 'is_perturbed' function
adata.obs['intervention_status'] = intervention_matrix['intervention'].map({1: 'Perturbed', 0: 'Control'})
int_by_ctype = pd.crosstab(adata.obs['scMRMA_manual'], adata.obs['intervention_status'])
int_by_ctype_top = int_by_ctype.loc[ctype_dist.head(10).index]
int_by_ctype_top.plot(kind='bar', stacked=True, ax=ax, color=['lightcoral', 'steelblue'])
ax.set_xlabel('Cell Type', fontsize=12)
ax.set_ylabel('Number of Cells', fontsize=12)
# ax.set_title removed for publication style
ax.legend(title='Status', frameon=False)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, 'figure_2e_intervention_by_celltype.pdf'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'figure_2e_intervention_by_celltype.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 2F: Perturbation Effect on Gene Expression (example)
# Find a valid control guide present in the data
present_controls = [c for c in control_guides if c in adata.obs['sgRNA'].values]
control_guide = present_controls[0] if present_controls else 'Unassigned'

perturbed_genes_in_data = [g for g in plot_counts.index[:10] if g in adata.var_names]
if len(perturbed_genes_in_data) > 0:
    selected_gene = perturbed_genes_in_data[0]
    expr_data = []
    
    for sgrna in [control_guide, selected_gene]:
        cells = adata.obs_names[adata.obs['sgRNA'] == sgrna]
        if len(cells) > 0:
            # Handle both sparse and dense representations safely
            expr = adata[cells, selected_gene].X
            if sparse.issparse(expr):
                expr = expr.toarray()
            expr = np.array(expr).flatten()
            expr_data.extend([(sgrna, e) for e in expr])

    if len(expr_data) > 0:
        expr_df = pd.DataFrame(expr_data, columns=['sgRNA', 'Expression'])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=expr_df, x='sgRNA', y='Expression', ax=ax, palette="Set2", linewidth=1.5)
        # ax.set_title removed for publication style
        plt.tight_layout()
        fig.savefig(os.path.join(FIGURE_PATH, f'figure_2f_perturbation_effect_{selected_gene}.pdf'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(FIGURE_PATH, f'figure_2f_perturbation_effect_{selected_gene}.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Figure 2H: Cell-State Programs Clustermap (Z-scored module score per sgRNA)
print("\n    Calculating Cell-State Program Module Scores...")
try:
    tumor_markers  = ["Nes","Sox2","Cd44","Egfr","Pdgfra","Olig2"]
    immune_markers = ["Ptprc","Cd68","Tmem119","P2ry12","Aif1","Csf1r"]
    prolif_markers = ["Mki67","Top2a","Pcna","Ccnb1"]
    rt_markers     = ["Cdkn1a","Gadd45a","Trp53","H2afx"]

    def valid_genes(genes):
        return [g for g in genes if g in adata.var_names]

    sc.tl.score_genes(adata, valid_genes(tumor_markers), score_name='Tumor_Stem')
    sc.tl.score_genes(adata, valid_genes(immune_markers), score_name='Immune_Micro')
    sc.tl.score_genes(adata, valid_genes(prolif_markers), score_name='Proliferating')
    sc.tl.score_genes(adata, valid_genes(rt_markers), score_name='RT_Response')

    programs = ['Tumor_Stem', 'Immune_Micro', 'Proliferating', 'RT_Response']
    scores = adata.obs[['sgRNA'] + programs].copy()
    scores = scores[~scores['sgRNA'].isin(['Unassigned', 'NTC', 'non-targeting', 'non-targeting_B', 'None', 'NA'])]
    
    mean_scores = scores.groupby('sgRNA').mean()
    mean_scores_z = (mean_scores - mean_scores.mean(axis=0)) / mean_scores.std(axis=0)

    g = sns.clustermap(mean_scores_z, cmap='RdBu_r', center=0, figsize=(10, 10), 
                       cbar_pos=(0.95, 0.4, 0.03, 0.2), 
                       dendrogram_ratio=(0.1, 0.1),
                       xticklabels=['Tumor Stem', 'Immune Micro', 'Proliferating', 'RT Response'])

    g.ax_heatmap.set_ylabel('')
    g.fig.suptitle('CRISPR Perturbation Effects on Cell-State Programs\n(Z-scored module score per sgRNA)', 
                   fontsize=14, fontweight='bold', y=1.05)

    g.savefig(os.path.join(FIGURE_PATH, 'figure_2h_cell_state_programs.pdf'), dpi=300, bbox_inches='tight')
    g.savefig(os.path.join(FIGURE_PATH, 'figure_2h_cell_state_programs.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    Generated Figure 2H: Cell-State Programs Heatmap")
except Exception as e:
    print(f"    Warning: Could not generate Figure 2H: {e}")


# 2.6 Calculate Diffusion Pseudotime (DPT)
print("\n[Saving] Calculating Diffusion Pseudotime (DPT)...")
try:
    # Need PCA and neighbors for diffmap
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.diffmap(adata)
    
    # Select root cell as the extreme of the first diffusion component
    root_id = np.argmin(adata.obsm['X_diffmap'][:, 1])
    adata.uns['iroot'] = root_id
    sc.tl.dpt(adata)
    
    # Create Early/Middle/Late bins
    dpt_max = adata.obs['dpt_pseudotime'].max()
    bins = pd.cut(adata.obs['dpt_pseudotime'], bins=[-np.inf, dpt_max/3, 2*dpt_max/3, np.inf], labels=['Early', 'Middle', 'Late'])
    adata.obs['pt_bin'] = bins
    
    # Save the pseudotime mapping
    pt_df = pd.DataFrame({
        'cell': adata.obs_names,
        'dpt_pseudotime': adata.obs['dpt_pseudotime'],
        'pt_bin': adata.obs['pt_bin']
    })
    pt_df.to_csv(os.path.join(PHASE2_PATH, "phase2_pseudotime.csv"), index=False)
    print(f"    Saved pseudotime to phase2_pseudotime.csv")
    
    # Generate Figure 2G: Pseudotime Distribution
    import seaborn as sns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: KDE plot comparing RT vs NoRT
    is_rt = adata.obs['orig.ident'].astype(str).str.contains('RT', case=False, na=False)
    adata.obs['Treatment'] = np.where(is_rt, 'RT', 'NoRT')
    
    sns.kdeplot(data=adata.obs, x='dpt_pseudotime', hue='Treatment', 
                fill=True, common_norm=False, palette={'NoRT': '#56B4E9', 'RT': '#D55E00'},
                linewidth=2.5, alpha=0.4, ax=axes[0])
    sns.rugplot(data=adata.obs, x='dpt_pseudotime', hue='Treatment', 
                palette={'NoRT': '#56B4E9', 'RT': '#D55E00'}, ax=axes[0], alpha=0.5, height=0.03)
    axes[0].set_xlabel('Pseudotime', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[0].set_title('Pseudotime Distribution by Treatment Condition', fontsize=14, fontweight='bold', pad=20)
    axes[0].text(0.5, 1.02, 'Comparison of cell-state progression in RT-treated vs. untreated tumours',
                 ha='center', va='bottom', transform=axes[0].transAxes, fontsize=11, color='dimgray')
    
    # Panel 2: Cell types across pseudotime bins
    pt_by_ctype = pd.crosstab(adata.obs['scMRMA_manual'], adata.obs['pt_bin'])
    top_ctypes_pt = pt_by_ctype.sum(axis=1).nlargest(8).index
    colors = {'Early': '#56B4E9', 'Middle': '#F0E442', 'Late': '#D55E00'}
    pt_by_ctype.loc[top_ctypes_pt].plot(kind='bar', stacked=True, ax=axes[1], color=[colors['Early'], colors['Middle'], colors['Late']])
    axes[1].set_xlabel('Cell Type', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Cells', fontsize=12, fontweight='bold')
    axes[1].set_title('Stage Composition by Cell Type', fontsize=14, fontweight='bold')
    axes[1].legend(title='Pseudotime Stage', frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_PATH, 'figure_2g_pseudotime_distribution.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURE_PATH, 'figure_2g_pseudotime_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    Generated Figure 2G: Pseudotime Distribution")
    
    # Figure 2I: Pseudotime UMAPs (Monocle3 Style)
    print("    Generating Pseudotime UMAPs...")
    sc.tl.umap(adata)
    
    # Plot 1: Continuous Pseudotime
    fig, ax = plt.subplots(figsize=(8, 8))
    sc.pl.umap(adata, color='dpt_pseudotime', cmap='plasma', ax=ax, show=False)
    ax.set_title('Pseudotime Trajectory', fontsize=16, fontweight='bold', pad=25)
    ax.text(0.5, 1.02, 'Cell-state progression ordered by transcriptional similarity',
            ha='center', va='bottom', transform=ax.transAxes, fontsize=12, color='dimgray')
    fig.savefig(os.path.join(FIGURE_PATH, 'figure_2i_continuous_umap.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURE_PATH, 'figure_2i_continuous_umap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Pseudotime Stages (Early, Middle, Late)
    fig, ax = plt.subplots(figsize=(8, 8))
    sc.pl.umap(adata, color='pt_bin', palette={'Early': '#56B4E9', 'Middle': '#F0E442', 'Late': '#D55E00'}, ax=ax, show=False)
    ax.set_title('Pseudotime Stage Classification', fontsize=16, fontweight='bold', pad=25)
    ax.text(0.5, 1.02, 'Cells binned into Early / Middle / Late progression stages',
            ha='center', va='bottom', transform=ax.transAxes, fontsize=12, color='dimgray')
    fig.savefig(os.path.join(FIGURE_PATH, 'figure_2i_stages_umap.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURE_PATH, 'figure_2i_stages_umap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("    Generated Figure 2I: Pseudotime UMAPs")
    
except Exception as e:
    print(f"    Warning: Could not calculate pseudotime: {e}")

# 2.7 Save Processed Data for Phase III
print("\n[Saving] Saving processed data for Phase III...")

# Save AnnData object
adata.write_h5ad(os.path.join(PHASE2_PATH, "phase2_processed.h5ad"))

# Save cell-type stratified indices (limit to top 20 to avoid clutter)
for ctype in ctype_dist.head(20).index:
    cells = adata.obs_names[adata.obs['scMRMA_manual'] == ctype]
    safe_name = str(ctype).replace(' ', '_').replace('/', '_')
    with open(os.path.join(PHASE2_PATH, f"cells_{safe_name}.txt"), 'w') as f:
        f.write('\n'.join(cells))

# Save condition-stratified indices
if 'cond' in adata.obs.columns:
    for cond in adata.obs['cond'].unique():
        if pd.isna(cond): continue
        cells = adata.obs_names[adata.obs['cond'] == cond]
        with open(os.path.join(PHASE2_PATH, f"cells_{cond}.txt"), 'w') as f:
            f.write('\n'.join(cells))

# Create summary report
summary_report = pd.DataFrame({
    'Metric': ['Total Cells', 'Total Perturbations (incl controls)', 'Cells with Defined Perturbations',
               'Control/Unassigned Cells', 'Cell Types', 'Treatment Conditions'],
    'Value': [adata.n_obs, len(perturb_counts),
              intervention_matrix['intervention'].sum(),
              (intervention_matrix['intervention'] == 0).sum(),
              len(ctype_dist), len(cond_dist) if 'cond' in adata.obs.columns else 0]
})
summary_report.to_csv(os.path.join(PHASE2_PATH, "phase2_summary.csv"), index=False)

# 2.8 Export Perturbation Coverage Table (S4 Table)
print("\n[S4 Table] Exporting Perturbation Coverage Table (S4_Table_Perturbation_Counts.csv)...")

def classify_sgrna(sg):
    if sg == 'Unassigned':
        return 'Unassigned'
    elif sg in control_guides:
        return 'Non-Targeting Control'
    return 'Gene-Targeting Perturbation'

s4_records = []
total_cells = len(adata.obs)
total_perturbed = int(intervention_matrix['intervention'].sum())

for sg, count in adata.obs['sgRNA'].value_counts().items():
    sub_obs = adata.obs[adata.obs['sgRNA'] == sg]
    category = classify_sgrna(sg)
    n_nort = int((sub_obs['cond'] == 'noRT').sum()) if 'cond' in sub_obs.columns else 0
    n_rt = int((sub_obs['cond'] == 'RT').sum()) if 'cond' in sub_obs.columns else 0
    n_invivo = int((sub_obs['scMRMA_manual'] == 'GL261 in vivo').sum()) if 'scMRMA_manual' in sub_obs.columns else 0
    n_invitro = int((sub_obs['scMRMA_manual'] == 'GL261 in vitro').sum()) if 'scMRMA_manual' in sub_obs.columns else 0
    n_tme = int(count - n_invivo - n_invitro)
    n_reps = int(sub_obs['orig.ident'].nunique()) if 'orig.ident' in sub_obs.columns else 0
    pct_pert = round((count / total_perturbed) * 100, 3) if category == 'Gene-Targeting Perturbation' and total_perturbed > 0 else 0.0

    s4_records.append({
        'sgRNA': sg,
        'Category': category,
        'Cell_Count': int(count),
        'Pct_Total_Cells': round((count / total_cells) * 100, 3),
        'Pct_Perturbed_Cells': pct_pert,
        'noRT_Cells': n_nort,
        'RT_Cells': n_rt,
        'GL261_in_vivo_Cells': n_invivo,
        'GL261_in_vitro_Cells': n_invitro,
        'TME_Immune_Stromal_Cells': n_tme,
        'N_Biological_Samples': n_reps
    })

s4_df = pd.DataFrame(s4_records)
s4_df.to_csv(os.path.join(OUTPUT_PATH, "S4_Table_Perturbation_Counts.csv"), index=False)
s4_df.to_csv(os.path.join(PHASE2_PATH, "S4_Table_Perturbation_Counts.csv"), index=False)
print(f"    Saved S4_Table_Perturbation_Counts.csv ({len(s4_df)} rows)")

print("\n" + "=" * 80)
print("[DONE] PHASE II COMPLETE")
print(f"Output saved to: {PHASE2_PATH}")
print(f"Figures saved to: {FIGURE_PATH}")
print("=" * 80)

# --- PHASE II END ---

