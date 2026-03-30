import matplotlib.pyplot as plt
import csv

# File path (adjust if necessary)
csv_file = "out_train_val_loss.csv"

# Lists to store loss values
train_loss = []
val_loss = []

# Read the CSV and populate the lists
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) == 2:  # Ensure proper format
            try:
                train_loss.append(float(row[0]))
                val_loss.append(float(row[1]))
            except ValueError:
                continue  # Skip malformed rows

# Epoch numbers (0-indexed)
epochs = list(range(1, len(train_loss) + 1))

# Plotting
plt.figure(figsize=(5, 4))
plt.plot(epochs, train_loss, 'r-', label='Train Loss')  # red line
plt.plot(epochs, val_loss, 'b-', label='Validation Loss')  # blue line

# Labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('mpp_lsc_train_val_loss.png', dpi=300)
plt.close()

