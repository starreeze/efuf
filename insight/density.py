import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Generate sample data: two normal distributions with different means
np.random.seed(0)  # For reproducibility
data1 = np.random.normal(loc=0, scale=1, size=1000)  # Mean=0, StdDev=1
data2 = np.random.normal(loc=5, scale=1, size=1000)  # Mean=5, StdDev=1

# Create the seaborn density plot
# sns.set(style="white")  # Set the style for the plots
plt.figure(figsize=(10, 6))  # Set the figure size for better readability

# Plot the density plots on the same figure
sns.kdeplot(data1, bw_adjust=1.5, label="Distribution 1: Mean=0", fill=True)
sns.kdeplot(data2, bw_adjust=1.5, label="Distribution 2: Mean=5", fill=True)

# Beautify the plot
plt.title("Density Plots of Two Normal Distributions", fontsize=16)
plt.xlabel("Value", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=12)

# Show the plot
plt.show()
