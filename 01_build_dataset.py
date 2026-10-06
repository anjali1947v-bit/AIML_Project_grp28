import pandas as pd
import numpy as np
import networkx as nx

# Fix seed for reproducible results
np.random.seed(42)

NUM_BANKS = 25

# 1. Generate Bank Profiles (Nodes)
bank_names = [f"Bank_{chr(65 + i) if i < 26 else i}" for i in range(NUM_BANKS)]

data_nodes = {
    "bank_id": list(range(NUM_BANKS)),
    "bank_name": bank_names,
    "capital_reserves_m": np.round(np.random.uniform(50, 500, NUM_BANKS), 2),  # in $ Millions
    "total_assets_m": np.round(np.random.uniform(500, 5000, NUM_BANKS), 2),
    "leverage_ratio": np.round(np.random.uniform(8.0, 25.0, NUM_BANKS), 2),
    "liquidity_coverage_ratio": np.round(np.random.uniform(0.8, 2.0, NUM_BANKS), 2)
}

df_banks = pd.DataFrame(data_nodes)

# 2. Generate Interbank Exposures / Loans (Edges)
# Create directed graph representing financial liabilities
G = nx.erdos_renyi_graph(n=NUM_BANKS, p=0.2, directed=True)

edges_list = []
for source, target in G.edges():
    if source != target:
        # Loan exposure amount between banks
        loan_amount_m = round(np.random.uniform(5, 50), 2)
        edges_list.append({
            "borrower_id": source,
            "lender_id": target,
            "exposure_amount_m": loan_amount_m
        })

df_edges = pd.DataFrame(edges_list)

# 3. Save to CSV files
df_banks.to_csv("banks_data.csv", index=False)
df_edges.to_csv("interbank_exposures.csv", index=False)

print("Dataset Created Successfully!")
print(f"- Banks profile saved to 'banks_data.csv' ({len(df_banks)} rows)")
print(f"- Financial links saved to 'interbank_exposures.csv' ({len(df_edges)} rows)\n")

# Preview first few rows
print("--- Sample Bank Profiles ---")
print(df_banks.head(3))
print("\n--- Sample Interbank Loans ---")
print(df_edges.head(3))