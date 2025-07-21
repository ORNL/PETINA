import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('mnist_accuracy_with_dp.csv')

# Define larger font size
font_size = 16

# Plot 1: Accuracy vs Step
plt.figure(figsize=(10,6))
plt.plot(df['Step'], df['Accuracy_no_DP'], label='Accuracy - No DP')
plt.plot(df['Step'], df['accuracy_sketch_DP'], label='Accuracy - Sketch DP')
plt.plot(df['Step'], df['accuracy_laplace_DP'], label='Accuracy - Laplace DP')
plt.xlabel('Step (Epoch)', fontsize=font_size)
plt.ylabel('Accuracy', fontsize=font_size)
plt.title('Accuracy vs Step for 3 Experiments', fontsize=font_size + 2)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.legend(fontsize=font_size)
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_vs_step.png')
plt.show()

# Plot 2: Epsilon vs Step
plt.figure(figsize=(10,6))
plt.plot(df['Step'], df['Used_Epsilon_No_DP'], label='Total Epsilon Used - No DP')
plt.plot(df['Step'], df['Epsilon_Sketch_DP'], label='Total Epsilon Used - Sketch DP')
plt.plot(df['Step'], df['epsilon_laplace_DP'], label='Total Epsilon Used - Laplace DP')
plt.xlabel('Step (Epoch)', fontsize=font_size)
plt.ylabel('Epsilon', fontsize=font_size)
plt.title('Epsilon vs Step for 3 Experiments', fontsize=font_size + 2)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.legend(fontsize=font_size)
plt.grid(True)
plt.tight_layout()
plt.savefig('epsilon_vs_step.png')
plt.show()
