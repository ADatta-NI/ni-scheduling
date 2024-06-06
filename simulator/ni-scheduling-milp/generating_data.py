import numpy as np

table = np.random.uniform(low=1, high=5, size=(30, 100))

# Print to console
for row in table:
    print("\t".join("{:.4f}".format(value) for value in row))

# Save to file
with open("dummy.data", "w") as file:
    for row in table:
        line = "\t".join("{:.4f}".format(value) for value in row)
        file.write(line + "\n")
