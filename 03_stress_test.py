import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# --- 1. Load Data ---
df_banks = pd.read_csv("banks_risk_predicted.csv")
df_edges = pd.read_csv("interbank_exposures.csv")

# Build NetworkX graph
G = nx.DiGraph()

for _, row in df_banks.iterrows():
    G.add_node(
        int(row["bank_id"]),
        name=row["bank_name"],
        reserves=row["capital_reserves_m"],
        risk=row["gnn_predicted_risk_score"]
    )

for _, row in df_edges.iterrows():
    G.add_edge(int(row["borrower_id"]), int(row["lender_id"]), exposure=row["exposure_amount_m"])

# --- 2. Causal Shock Simulation Function ---
def simulate_causal_shock(graph, shock_node_id, shock_magnitude=0.8):
    current_reserves = {node: data["reserves"] for node, data in graph.nodes(data=True)}
    current_reserves[shock_node_id] *= (1.0 - shock_magnitude)
    failed_nodes = {shock_node_id} if current_reserves[shock_node_id] < 20.0 else set()
    
    propagation_history = [len(failed_nodes)]
    
    for round_num in range(1, 5):
        new_failures = set()
        for node in graph.nodes():
            if node in failed_nodes:
                continue
            
            incoming_loss = 0.0
            for borrower in graph.predecessors(node):
                if borrower in failed_nodes:
                    incoming_loss += graph[borrower][node]["exposure"] * 0.5
            
            current_reserves[node] -= incoming_loss
            if current_reserves[node] < 20.0:
                new_failures.add(node)
                
        failed_nodes.update(new_failures)
        propagation_history.append(len(failed_nodes))
        
    return failed_nodes, propagation_history

# --- 3. Run Simulation & Plot ---
SHOCK_NODE = 0
failed_banks, history = simulate_causal_shock(G, shock_node_id=SHOCK_NODE, shock_magnitude=0.9)

plt.figure(figsize=(10, 7))
node_colors = ['#ff4d4d' if node in failed_banks else '#66b3ff' for node in G.nodes()]
node_sizes = [v["reserves"] * 3 for k, v in G.nodes(data=True)]

pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=12, edge_color="gray", alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

plt.title(f"Financial Network Stress Propagation (Shocked Node: Bank_{SHOCK_NODE})\nRed = Failed Banks, Size = Capital Reserves")

# Save PNG image file
plt.savefig("network_stress_results.png", bbox_inches='tight')
print("IMAGE SAVED SUCCESSFULLY AS network_stress_results.png!")

# POPUP WINDOW FOR DISPLAY
plt.show()