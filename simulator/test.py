data = [
    [1.2, 2.3, 3.4],
    [4.5, 5.6, 6.7],
    [7.8, 8.9, 9.0]
]

# Transpose the data to work with columns instead of rows
columns = list(map(list, zip(*data)))

# Calculate mean, variance, min, and max for each column
column_stats = []

for column in columns:
    mean = sum(column) / len(column)
    variance = sum((x - mean) ** 2 for x in column) / len(column)
    min_value = min(column)
    max_value = max(column)
    
    column_stats.append({
        'mean': mean,
        'variance': variance,
        'min': min_value,
        'max': max_value
    })

# Printing the statistics for each column
for i, stats in enumerate(column_stats, start=1):
    print(f"Column {i}:")
    print(f"Mean: {stats['mean']}")
    print(f"Variance: {stats['variance']}")
    print(f"Min: {stats['min']}")
    print(f"Max: {stats['max']}")
    print()
