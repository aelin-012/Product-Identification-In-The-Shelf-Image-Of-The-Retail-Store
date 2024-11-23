import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
results_path = 'runs/classify/train4/results.csv'
results = pd.read_csv(results_path)

# Display the actual column names to identify the issue
print(results.columns)

# Clean up the column names by stripping leading/trailing whitespace
results.columns = results.columns.str.strip()

# Plot train and validation loss vs epochs
plt.figure()
plt.plot(results['epoch'], results['train/loss'], label='train loss')
plt.plot(results['epoch'], results['val/loss'], label='val loss', c='red')
plt.grid()
plt.title('Loss vs epochs')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
# Plot validation accuracy vs epochs
plt.figure()
plt.plot(results['epoch'], results['metrics/accuracy_top1'] * 100)
plt.grid()
plt.title('Validation accuracy vs epochs')
plt.ylabel('accuracy (%)')
plt.xlabel('epochs')
plt.show()
