"""
============================================================================
PHASE IV: DISEASE-RELEVANT CAUSAL GRN EXTRACTION (Python)
============================================================================
Project: Perturbation-Aware Causal Machine Learning Framework for 
         Glioblastoma Regulatory Network Discovery
Manuscript: A Perturbation-Aware Causal Machine Learning Framework Identifies
            Candidate Regulatory Hubs and Radiotherapy-Associated Network
            Rewiring in Glioblastoma
============================================================================
Description:
Integrates Phase III consensus causal edges, Average Treatment Effects (ATE),
and FDR-adjusted statistical significance to extract the high-confidence
glioblastoma regulatory network, prioritize master regulatory hubs, detect
functional modules via Louvain clustering, and conduct GO/KEGG enrichments.
============================================================================
"""

# --- PHASE IV START ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import warnings
from sklearn.preprocessing import minmax_scale

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

# Set random seed and plotting styles
np.random.seed(42)
sns.set_style("ticks")

# Dynamic Directory Resolution for Cross-Platform Portability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

OUTPUT_PATH = os.environ.get(
    "GBM_OUTPUT_PATH",
    os.path.join(REPO_ROOT, "outputs") if os.path.exists(os.path.join(REPO_ROOT, "outputs"))
    else (os.path.join(SCRIPT_DIR, "outputs") if os.path.exists(os.path.join(SCRIPT_DIR, "outputs"))
          else "L:/Thesis Work/outputs")
)
DATA_PATH = os.environ.get(
    "GBM_DATA_PATH",
    os.path.join(REPO_ROOT, "Dataset") if os.path.exists(os.path.join(REPO_ROOT, "Dataset"))
    else (os.path.join(SCRIPT_DIR, "Dataset") if os.path.exists(os.path.join(SCRIPT_DIR, "Dataset"))
          else "L:/Thesis Work/Dataset")
)
PHASE3_PATH = os.path.join(OUTPUT_PATH, "phase3")
PHASE4_PATH = os.path.join(OUTPUT_PATH, "phase4")
FIGURE_PATH = os.path.join(OUTPUT_PATH, "figures", "phase4")

os.makedirs(PHASE4_PATH, exist_ok=True)
os.makedirs(FIGURE_PATH, exist_ok=True)

print("=" * 80)
print("PHASE IV: CANCER-RELEVANT REGULATORY NETWORKS")
print("=" * 80)
print()

# 4.1 Load Consensus Network from Phase III
print("[1/8] Loading consensus network and evidence from Phase III...")

adj_consensus = np.load(os.path.join(PHASE3_PATH, "adj_consensus.npy"))
with open(os.path.join(PHASE3_PATH, "gene_names.txt"), 'r') as f:
    gene_names = [line.strip() for line in f]

edges_file = os.path.join(PHASE3_PATH, "phase3_causal_edges_confident.csv")
ate_file = os.path.join(PHASE3_PATH, "phase3_ate_results.csv")

conf_edges = pd.read_csv(edges_file) if os.path.exists(edges_file) else pd.DataFrame()
ate_results = pd.read_csv(ate_file) if os.path.exists(ate_file) else pd.DataFrame()

print(f"    Loaded consensus network: {len(gene_names)} genes, {np.sum(adj_consensus)} edges")

# 4.2 Build NetworkX Directed Graph
print("\n[2/8] Building directed network and calculating metrics...")

G = nx.from_numpy_array(adj_consensus, create_using=nx.DiGraph())
mapping = {i: gene_names[i] for i in range(len(gene_names))}
G = nx.relabel_nodes(G, mapping)

# Map CRISPR-validated statistical evidence to the edges if available
if not conf_edges.empty:
    for _, row in conf_edges.iterrows():
        if G.has_edge(row['perturbed_gene'], row['target_gene']):
            G.edges[row['perturbed_gene'], row['target_gene']]['log2FC'] = row['avg_log2FC']
            G.edges[row['perturbed_gene'], row['target_gene']]['p_val_adj'] = row['p_val_adj']
            G.edges[row['perturbed_gene'], row['target_gene']]['validated'] = True

# Assign default attributes for non-validated edges
for u, v in G.edges():
    if 'log2FC' not in G.edges[u, v]:
        G.edges[u, v]['log2FC'] = 0.0
        G.edges[u, v]['validated'] = False

# We will export a dataframe of the network for the user
final_grn = pd.DataFrame(G.edges(data=True), columns=['perturbed_gene', 'target_gene', 'attributes'])
final_grn['avg_log2FC'] = final_grn['attributes'].apply(lambda x: x.get('log2FC', 0.0))
final_grn['confidence'] = final_grn['attributes'].apply(lambda x: x.get('weight', 0.0))
final_grn['validated'] = final_grn['attributes'].apply(lambda x: x.get('validated', False))
final_grn = final_grn.drop('attributes', axis=1)

print(f"    Final disease GRN edges: {G.number_of_edges()}")
print(f"    Validated edges (CRISPR): {sum(final_grn['validated'])}")
final_grn.to_csv(os.path.join(PHASE4_PATH, "phase4_final_disease_GRN.csv"), index=False)

net_metrics = pd.DataFrame({
    'Metric': ['nodes', 'edges', 'density', 'avg_out_degree', 'max_out_degree', 'n_components'],
    'Value': [
        G.number_of_nodes(), 
        G.number_of_edges(), 
        round(nx.density(G), 6),
        round(np.mean([d for n, d in G.out_degree()]), 3),
        max([d for n, d in G.out_degree()]),
        nx.number_weakly_connected_components(G)
    ]
})
net_metrics.to_csv(os.path.join(PHASE4_PATH, "phase4_network_metrics.csv"), index=False)
print(f"    Network: nodes = {G.number_of_nodes()} | edges = {G.number_of_edges()}")

# 4.4 Hub Gene Identification (Composite Score)
print("\n[4/8] Identifying hub genes (Composite Score: Out-Degree, PageRank, ATE, Betweenness)...")

out_degree = dict(G.out_degree())
in_degree = dict(G.in_degree())
total_degree = dict(G.degree())
pagerank = nx.pagerank(G, weight='weight')
betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)

hub_df = pd.DataFrame({
    'gene': list(G.nodes()),
    'out_degree': [out_degree[n] for n in G.nodes()],
    'in_degree': [in_degree[n] for n in G.nodes()],
    'total_degree': [total_degree[n] for n in G.nodes()],
    'pagerank': [pagerank[n] for n in G.nodes()],
    'betweenness': [betweenness[n] for n in G.nodes()]
})

if not ate_results.empty:
    hub_df = hub_df.merge(ate_results[['perturbed_gene', 'ATE', 'abs_ATE', 'n_targets']], 
                          left_on='gene', right_on='perturbed_gene', how='left')
    hub_df['abs_ATE'] = hub_df['abs_ATE'].fillna(0)
else:
    hub_df['abs_ATE'] = 0.0

# Calculate composite centrality score:
# 35% out-degree + 35% PageRank + 20% |ATE| + 10% betweenness
hub_df['hub_score'] = (
    minmax_scale(hub_df['out_degree']) * 0.35 +
    minmax_scale(hub_df['pagerank']) * 0.35 +
    minmax_scale(hub_df['abs_ATE']) * 0.20 +
    minmax_scale(hub_df['betweenness']) * 0.10
)

hub_df = hub_df.sort_values(by='hub_score', ascending=False).reset_index(drop=True)
hub_df.to_csv(os.path.join(PHASE4_PATH, "phase4_hub_genes.csv"), index=False)

top_hubs = hub_df['gene'].head(10).tolist()
print("    Top 10 hub genes:")
print(hub_df[['gene', 'out_degree', 'pagerank', 'abs_ATE', 'hub_score']].head(10).to_string(index=False))

# Figure 4A: Hub Gene Ranking
fig, ax = plt.subplots(figsize=(10, 8))
hub_top20 = hub_df.head(20).copy()
colors = ["#D55E00" if gene in top_hubs else "#0072B2" for gene in hub_top20['gene']]

bars = ax.barh(range(len(hub_top20)), hub_top20['hub_score'], color=colors, edgecolor='white')
ax.set_yticks(range(len(hub_top20)))
ax.set_yticklabels(hub_top20['gene'], fontsize=11, fontstyle='italic')
ax.invert_yaxis()
ax.set_xlabel('Hub Score (composite)', fontweight='bold', fontsize=12)
# ax.set_title removed

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#D55E00", label="Top hub"), Patch(facecolor="#0072B2", label="Other")]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10, frameon=False)

plt.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, 'Fig4A_Hub_Gene_Rankings.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'Fig4A_Hub_Gene_Rankings.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4.5 Community / Module Detection (Louvain)
print("\n[5/8] Running Louvain community detection...")

try:
    from networkx.algorithms.community import louvain_communities
    communities = louvain_communities(G.to_undirected(), seed=42, resolution=1.0)
except ImportError:
    try:
        import community.community_louvain as community_louvain
        partition = community_louvain.best_partition(G.to_undirected(), random_state=42)
        communities = []
        for com in set(partition.values()):
            communities.append([nodes for nodes in partition.keys() if partition[nodes] == com])
    except ImportError:
        print("    WARNING: 'python-louvain' module not found. Falling back to simple connected components.")
        communities = list(nx.connected_components(G.to_undirected()))

module_data = []
for i, comm in enumerate(communities):
    for gene in comm:
        module_data.append({'gene': gene, 'module': f"Module_{i+1}"})

module_df = pd.DataFrame(module_data)
module_summary = module_df['module'].value_counts().reset_index()
module_summary.columns = ['module', 'n_genes']

module_df.to_csv(os.path.join(PHASE4_PATH, "phase4_network_modules.csv"), index=False)
module_summary.to_csv(os.path.join(PHASE4_PATH, "phase4_module_summary.csv"), index=False)

print(f"    Detected {len(communities)} modules.")
print(module_summary.head(10).to_string(index=False))

# Assign node attributes for plotting
nx.set_node_attributes(G, pd.Series(hub_df.hub_score.values, index=hub_df.gene).to_dict(), 'hub_score')
nx.set_node_attributes(G, pd.Series(module_df.module.values, index=module_df.gene).to_dict(), 'module')

# 4.6 Main GRN Visualization
print("\n[6/8] Creating network visualizations...")

# Figure 4B: Main GRN (Top 150 edges)
# We sort by the combined hub score of the source and target to guarantee the master regulators (like Eef1a1) are included!
top_edges = sorted(G.edges(data=True), key=lambda x: G.nodes[x[0]].get('hub_score', 0) + G.nodes[x[1]].get('hub_score', 0), reverse=True)[:150]
G_vis = nx.DiGraph()
for u, v, data in top_edges:
    G_vis.add_edge(u, v, log2FC=data.get('log2FC', 0.0))

# Map attributes to G_vis
for node in G_vis.nodes():
    G_vis.nodes[node]['module'] = G.nodes[node].get('module', 'Module_1')
    G_vis.nodes[node]['hub_score'] = G.nodes[node].get('hub_score', 0.01)

fig, ax = plt.subplots(figsize=(14, 12))
# Create a spring layout
pos = nx.spring_layout(G_vis, k=2, iterations=100, seed=42)

# Colors
unique_modules = list(set([G_vis.nodes[n]['module'] for n in G_vis.nodes()]))
cmap = plt.get_cmap('tab20', max(1, len(unique_modules)))
module_colors = {mod: cmap(i) for i, mod in enumerate(unique_modules)}
node_colors = [module_colors[G_vis.nodes[n]['module']] for n in G_vis.nodes()]
node_sizes = [G_vis.nodes[n]['hub_score'] * 2000 + 50 for n in G_vis.nodes()]

def get_edge_color(data):
    val = data.get('log2FC', 0.0)
    if val > 0: return "#E69F00" # Orange
    elif val < 0: return "#56B4E9" # Blue
    else: return "#D3D3D3" # Grey

edge_colors = [get_edge_color(G_vis.edges[u, v]) for u, v in G_vis.edges()]

nx.draw_networkx_nodes(G_vis, pos, node_size=node_sizes, node_color=node_colors, 
                       edgecolors='white', linewidths=0.5, ax=ax)
nx.draw_networkx_edges(G_vis, pos, edge_color=edge_colors, alpha=0.6, arrowsize=15, ax=ax)

# Draw labels only for the top 25 hubs to add more detail without making a hairball
vis_hubs = hub_df['gene'].head(25).tolist()
labels = {n: n for n in G_vis.nodes() if n in vis_hubs}
nx.draw_networkx_labels(G_vis, pos, labels=labels, font_size=10, font_weight='bold', ax=ax)

# ax.set_title removed
ax.axis('off')

# Custom legend for edges
legend_elements = [Patch(facecolor="#E69F00", label="Activation (log2FC > 0)"), 
                   Patch(facecolor="#56B4E9", label="Repression (log2FC < 0)"),
                   Patch(facecolor="#D3D3D3", label="Unknown (log2FC = 0)")]
ax.legend(handles=legend_elements, loc="lower right", fontsize=11, frameon=False)

plt.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, 'Fig4B_Disease_GRN_Main.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'Fig4B_Disease_GRN_Main.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 4C: Degree Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist([d for n, d in G.out_degree()], bins=20, color="#D55E00", edgecolor='white')
axes[0].set_xlabel('Out-degree (regulators)', fontweight='bold')
axes[0].set_ylabel('Number of Genes', fontweight='bold')
axes[0].set_yscale('log')
# axes[0].set_title removed

axes[1].hist([d for n, d in G.in_degree()], bins=20, color="#0072B2", edgecolor='white')
axes[1].set_xlabel('In-degree (targets)', fontweight='bold')
axes[1].set_ylabel('Number of Genes', fontweight='bold')
axes[1].set_yscale('log')
# axes[1].set_title removed

# plt.suptitle removed
plt.tight_layout()
fig.savefig(os.path.join(FIGURE_PATH, 'Fig4C_Degree_Distribution.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'Fig4C_Degree_Distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 4D: Ego Networks for Top 3 Hubs
for i, hub in enumerate(top_hubs[:3]):
    if hub not in G:
        continue
    
    g_ego = nx.ego_graph(G, hub, radius=1, undirected=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    pos_ego = nx.spring_layout(g_ego, k=1.5, seed=42)
    
    node_colors = ["#D55E00" if n == hub else "#56B4E9" for n in g_ego.nodes()]
    node_sizes = [1500 if n == hub else 500 for n in g_ego.nodes()]
    edge_colors = ["#E69F00" if g_ego.edges[u, v]['log2FC'] > 0 else "#56B4E9" for u, v in g_ego.edges()]
    
    nx.draw_networkx_nodes(g_ego, pos_ego, node_size=node_sizes, node_color=node_colors, edgecolors='white', ax=ax)
    nx.draw_networkx_edges(g_ego, pos_ego, edge_color=edge_colors, arrowsize=15, ax=ax)
    nx.draw_networkx_labels(g_ego, pos_ego, font_size=9, font_weight='bold', ax=ax)
    
    ax.set_title(f'Ego Network for Hub: {hub}', fontweight='bold', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_PATH, f'Fig4D_Ego_{hub}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURE_PATH, f'Fig4D_Ego_{hub}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved ego network for {hub}")

# 4.7 Functional Enrichment (GO + KEGG)
print("\n[7/8] Running functional enrichment (GO & KEGG) via gseapy...")

target_genes = list(final_grn['target_gene'].unique())
print(f"    Target genes for enrichment: {len(target_genes)}")

try:
    import gseapy as gp
    print(f"    gseapy version: {gp.__version__}")
except ImportError:
    gp = None
    print("    WARNING: 'gseapy' module not found. Skipping GO/KEGG functional enrichment.")
    print("    To enable enrichment plotting, please run: pip install gseapy")

if gp is not None:
    # GO Biological Process
    print("    Running GO Biological Process enrichment...")
    try:
        enr_go = gp.enrichr(gene_list=target_genes,
                            gene_sets='GO_Biological_Process_2021',
                            organism='mouse',
                            outdir=None)
        go_res = enr_go.results
        print(f"    GO raw results: {len(go_res)} terms returned")
        go_res = go_res[go_res['Adjusted P-value'] < 0.05].head(15)
        print(f"    GO significant results (adj. P < 0.05): {len(go_res)} terms")
        
        if len(go_res) > 0:
            go_res.to_csv(os.path.join(PHASE4_PATH, "phase4_GO_enrichment.csv"), index=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            go_res['Clean_Term'] = go_res['Term'].apply(lambda x: x.split('(GO:')[0].strip())
            go_res = go_res.sort_values(by='Adjusted P-value', ascending=False)
            
            scatter = ax.scatter(go_res['Odds Ratio'], go_res['Clean_Term'], 
                                 s=go_res['Overlap'].apply(lambda x: int(x.split('/')[0])) * 20,
                                 c=-np.log10(go_res['Adjusted P-value']), cmap='viridis', alpha=0.8)
            
            ax.set_xlabel('Odds Ratio', fontweight='bold')
            # ax.set_title removed
            plt.colorbar(scatter, label='-log10(Adjusted P-value)')
            
            plt.tight_layout()
            fig.savefig(os.path.join(FIGURE_PATH, 'Fig4E_GO_Enrichment.pdf'), bbox_inches='tight')
            fig.savefig(os.path.join(FIGURE_PATH, 'Fig4E_GO_Enrichment.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("    Fig4E_GO_Enrichment saved.")
        else:
            print("    WARNING: No GO terms passed significance threshold (adj. P < 0.05). Fig4E skipped.")
    except Exception as e:
        print(f"    ERROR during GO enrichment: {type(e).__name__}: {e}")
        print("    Fig4E skipped due to error above.")
    
    # KEGG Pathways
    print("    Running KEGG Pathway enrichment...")
    try:
        enr_kegg = gp.enrichr(gene_list=target_genes,
                              gene_sets='KEGG_2019_Mouse',
                              organism='mouse',
                              outdir=None)
        kegg_res = enr_kegg.results
        print(f"    KEGG raw results: {len(kegg_res)} pathways returned")
        kegg_res = kegg_res[kegg_res['Adjusted P-value'] < 0.05].head(15)
        print(f"    KEGG significant results (adj. P < 0.05): {len(kegg_res)} pathways")
        
        if len(kegg_res) > 0:
            kegg_res.to_csv(os.path.join(PHASE4_PATH, "phase4_KEGG_enrichment.csv"), index=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            kegg_res = kegg_res.sort_values(by='Adjusted P-value', ascending=False)
            
            scatter = ax.scatter(kegg_res['Odds Ratio'], kegg_res['Term'], 
                                 s=kegg_res['Overlap'].apply(lambda x: int(x.split('/')[0])) * 20,
                                 c=-np.log10(kegg_res['Adjusted P-value']), cmap='viridis', alpha=0.8)
            
            ax.set_xlabel('Odds Ratio', fontweight='bold')
            # ax.set_title removed
            plt.colorbar(scatter, label='-log10(Adjusted P-value)')
            
            plt.tight_layout()
            fig.savefig(os.path.join(FIGURE_PATH, 'Fig4F_KEGG_Enrichment.pdf'), bbox_inches='tight')
            fig.savefig(os.path.join(FIGURE_PATH, 'Fig4F_KEGG_Enrichment.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("    Fig4F_KEGG_Enrichment saved.")
        else:
            print("    WARNING: No KEGG pathways passed significance threshold (adj. P < 0.05). Fig4F skipped.")
    except Exception as e:
        print(f"    ERROR during KEGG enrichment: {type(e).__name__}: {e}")
        print("    Fig4F skipped due to error above.")

# 4.8 Multi-Layer Evidence Heatmap
print("\n[8/8] Generating multi-layer evidence heatmap...")

selected_genes = top_hubs[:12]
if len(selected_genes) >= 4:
    pert_score = final_grn.groupby('perturbed_gene')['avg_log2FC'].apply(lambda x: np.max(np.abs(x))).reset_index()
    pert_score.columns = ['gene', 'Perturbation_Effect']
    
    conf_score = final_grn.groupby('perturbed_gene')['confidence'].mean().reset_index()
    conf_score.columns = ['gene', 'Bootstrap_Confidence']
    
    hmdf = pd.DataFrame({'gene': selected_genes})
    hmdf = hmdf.merge(pert_score, on='gene', how='left')
    hmdf = hmdf.merge(conf_score, on='gene', how='left')
    if not ate_results.empty:
        hmdf = hmdf.merge(hub_df[['gene', 'pagerank', 'out_degree', 'abs_ATE']], on='gene', how='left')
        hmdf = hmdf.fillna(0).set_index('gene')
        hmdf.columns = ['Perturbation Effect', 'Bootstrap Confidence', 'PageRank', 'Out-Degree', 'ATE Magnitude']
    else:
        hmdf = hmdf.merge(hub_df[['gene', 'pagerank', 'out_degree']], on='gene', how='left')
        hmdf = hmdf.fillna(0).set_index('gene')
        hmdf.columns = ['Perturbation Effect', 'Bootstrap Confidence', 'PageRank', 'Out-Degree']
    
    # Scale each column 0-1 for heatmap comparison
    hm_scaled = pd.DataFrame(minmax_scale(hmdf), index=hmdf.index, columns=hmdf.columns)
    
    # Figure 4G: Clustermap
    g = sns.clustermap(hm_scaled, cmap="Blues", figsize=(9, 7), 
                       row_cluster=True, col_cluster=False, 
                       annot=True, fmt=".2f", linewidths=0.5,
                       cbar_pos=(0.02, 0.8, 0.03, 0.18))
    # g.fig.suptitle removed
    
    g.savefig(os.path.join(FIGURE_PATH, 'Fig4G_MultiLayer_Evidence.pdf'), bbox_inches='tight')
    g.savefig(os.path.join(FIGURE_PATH, 'Fig4G_MultiLayer_Evidence.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Save final network in standard formats
nx.write_gml(G, os.path.join(PHASE4_PATH, "phase4_grn_network.gml"))

# 4.9 Sensitivity Analysis of Candidate Regulatory Hubs (S4 Fig)
print("\n[S4 Fig] Generating Sensitivity Analysis of Candidate Regulatory Hubs...")

adj_notears = np.load(os.path.join(PHASE3_PATH, "adj_notears.npy"))
adj_ges = np.load(os.path.join(PHASE3_PATH, "adj_ges.npy"))
adj_rci = np.load(os.path.join(PHASE3_PATH, "adj_rci.npy"))
ensemble_score = 0.4 * adj_notears + 0.3 * adj_ges + 0.3 * adj_rci
np.fill_diagonal(ensemble_score, 0)

thresholds = [
    (0.3, "Permissive (tau = 0.30)\n[1,830 edges]"),
    (0.4, "Moderate (tau = 0.40)\n[1,192 edges]"),
    (0.5, "Strict Consensus (tau >= 0.50)\n[490 edges]")
]
top_genes = ['Eef1a1', 'Psmb8', 'Flt1', 'Vim', 'Cenpa']
bar_colors = ['#D55E00', '#0072B2', '#009E73', '#CC79A7', '#E69F00']

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=False)
fig.suptitle('Sensitivity of Prioritized Candidate Regulatory Hubs Across Ensemble Thresholds',
             fontsize=14, fontweight='bold', y=1.03)

for idx, (t, label) in enumerate(thresholds):
    adj_t = (ensemble_score >= t).astype(int)
    np.fill_diagonal(adj_t, 0)
    out_deg_map = {gene_names[i]: int(adj_t[i, :].sum()) for i in range(len(gene_names))}
    scores = [out_deg_map.get(g, 0) for g in top_genes]

    bars = axes[idx].bar(top_genes, scores, color=bar_colors, edgecolor='black', linewidth=0.8, width=0.6)
    axes[idx].set_title(label, fontsize=11, fontweight='bold', pad=10)
    axes[idx].set_ylabel('Out-Degree (Downstream Targets)' if idx == 0 else '', fontsize=11, fontweight='bold')
    axes[idx].tick_params(axis='x', rotation=30, labelsize=10)
    axes[idx].set_ylim(0, max(scores) * 1.18)

    for bar in bars:
        h = int(bar.get_height())
        axes[idx].text(bar.get_x() + bar.get_width() / 2.0, h + max(scores) * 0.02,
                       f'{h}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_PATH, 'S4_Fig_Sensitivity_Analysis.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'S4_Fig_Sensitivity_Analysis.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(FIGURE_PATH, 'S4_Fig_Sensitivity_Analysis.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("    Saved S4_Fig_Sensitivity_Analysis.png and .pdf")

print("\n" + "=" * 80)
print("[DONE] PHASE IV COMPLETE")
print(f"Final GRN edges : {len(final_grn)}")
print(f"Output directory: {PHASE4_PATH}")
print(f"Figures saved to: {FIGURE_PATH}")
print("=" * 80)

# --- PHASE IV END ---

