# Graph Neural Network-Based Financial Contagion and Interbank Risk Analysis

##  Overview

Financial institutions are highly interconnected through lending and borrowing relationships. Because of these connections, the failure of one bank can potentially affect other banks and create a cascading effect known as **financial contagion**.

This project uses a **Graph Neural Network (GNN)** to model interbank relationships and analyze financial risk at the network level.

Banks are represented as **nodes**, while interbank lending relationships are represented as **edges**. A **Graph Convolutional Network (GCN)** learns from both the financial characteristics of individual banks and their connections with neighboring banks.

The project also includes a **stress-testing simulation** in which one bank is artificially failed to observe how the resulting financial shock propagates through the network.

---

##  Objectives

- Represent an interconnected banking system as a graph.
- Model interbank lending relationships using nodes and edges.
- Use a Graph Convolutional Network to estimate bank-level risk.
- Generate baseline risk scores for individual banks.
- Simulate the failure of a major bank.
- Analyze the propagation of financial shocks.
- Visualize surviving and failed banks after the stress test.

---

##  Key Idea

The project can be understood using a **domino effect** analogy:

> If one bank fails, the banks connected to it may also experience financial losses. If those banks become unstable, the shock can continue spreading through the network.

Traditional machine learning models generally analyze institutions individually, while a GNN can consider both **individual bank features and network relationships**.

---

##  Project Architecture

```text
                 Financial & Network Data
                           │
                           ▼
                 ┌─────────────────────┐
                 │  Graph Construction │
                 │ 01_build_dataset.py │
                 └──────────┬──────────┘
                            │
                            ▼
                     Interbank Graph
                            │
                            ▼
                 ┌─────────────────────┐
                 │     GCN Training    │
                 │  02_train_gnn.py   │
                 └──────────┬──────────┘
                            │
                            ▼
                      Bank Risk Scores
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Stress Testing    │
                 │ 03_stress_test.py  │
                 └──────────┬──────────┘
                            │
                            ▼
                    Shock Propagation
                            │
                            ▼
                  Contagion Visualization
