import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print("==================================================")
print("  CAUSAL AI FOR FINANCIAL SYSTEMIC RISK PIPELINE  ")
print("==================================================\n")

print("[1/3] Generating Financial Dataset...")
os.system("python 01_build_dataset.py")

print("\n[2/3] Training Graph Neural Network...")
os.system("python 02_train_gnn.py")

print("\n[3/3] Running Causal Stress Test Simulation...")
os.system("python 03_stress_test.py")

print("\n==================================================")
print("          PROJECT EXECUTION COMPLETE!             ")
print("==================================================")

# Display the output network diagram
if os.path.exists("network_stress_results.png"):
    img = mpimg.imread("network_stress_results.png")
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Final Output: Financial Shock Contagion Network")
    plt.show()
