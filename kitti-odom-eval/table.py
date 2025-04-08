import re
import matplotlib.pyplot as plt

# Read and parse the result.txt file
filename = "result/test_02/result.txt"
with open(filename, 'r') as f:
    lines = f.readlines()

data = []
current = {}
for line in lines:
    line = line.strip()
    if not line:
        if current:
            data.append(current)
            current = {}
        continue
    # Match lines like "Key:    Value"
    match = re.match(r"([^:]+):\s+(.+)", line)
    if match:
        key = match.group(1).strip()
        value = match.group(2).strip()
        current[key] = value
if current:
    data.append(current)

# Define table headers and build rows
headers = ["Sequence", "Trans. err. (%)", "Rot. err. (deg/100m)", "ATE (m)", "RPE (m)", "RPE (deg)"]
table_data = []
for entry in data:
    row = [
        entry.get("Sequence", ""),
        entry.get("Trans. err. (%)", ""),
        entry.get("Rot. err. (deg/100m)", ""),
        entry.get("ATE (m)", ""),
        entry.get("RPE (m)", ""),
        entry.get("RPE (deg)", "")
    ]
    table_data.append(row)

# Create a figure and add the table
# Adjust the figure height based on the number of rows
fig, ax = plt.subplots(figsize=(8, len(table_data)*0.6 + 1))
ax.axis('off')  # Hide axes

# Create table at the center of the axis
table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)  # Adjust scaling if needed

plt.tight_layout()

# Save the table as an image (PNG) with high resolution
plt.savefig("result_table.png", dpi=300)
plt.show()

