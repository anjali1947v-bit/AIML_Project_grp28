import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# --- 1. Load Data from CSV Files ---
df_banks = pd.read_csv("banks_data.csv")
df_edges = pd.read_csv("interbank_exposures.csv")

# Extract bank features: Capital Reserves, Total Assets, Leverage, Liquidity
features = df_banks[["capital_reserves_m", "total_assets_m", "leverage_ratio", "liquidity_coverage_ratio"]].values
x = torch.tensor(features, dtype=torch.float)

# Normalize features so AI learns effectively
x = (x - x.mean(dim=0)) / x.std(dim=0)

# Convert network links to PyTorch tensor
edge_index = torch.tensor([df_edges["borrower_id"].values, df_edges["lender_id"].values], dtype=torch.long)

# Create PyTorch Geometric Graph Data Object
graph_data = Data(x=x, edge_index=edge_index)

# --- 2. Define Graph Neural Network Architecture ---
class FinancialRiskGNN(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(FinancialRiskGNN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, x, edge_index):
        # Aggregate financial risk from neighboring banks
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)  # Outputs risk score between 0.0 and 1.0

# --- 3. Run GNN Model ---
model = FinancialRiskGNN(in_features=4, hidden_dim=16)
model.eval()

with torch.no_grad():
    risk_scores = model(graph_data.x, graph_data.edge_index).squeeze().numpy()

# Save predicted risk scores back to bank data
df_banks["gnn_predicted_risk_score"] = list(map(lambda r: round(float(r), 4), risk_scores))
df_banks.to_csv("banks_risk_predicted.csv", index=False)

print("--- Top 5 Riskiest Banks Predicted by GNN ---")
print(df_banks[["bank_id", "bank_name", "leverage_ratio", "gnn_predicted_risk_score"]].sort_values(by="gnn_predicted_risk_score", ascending=False).head(5))
print("\nPredictions saved to 'banks_risk_predicted.csv'!")