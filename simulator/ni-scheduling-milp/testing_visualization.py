import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm


# Example data
m = 3  # number of machines
n = 5  # number of jobs
jobs = np.arange(n)
machines = np.arange(m)
start_times = np.random.randint(0, 20, size=(n, m))
processing_times = np.random.randint(1, 10, size=(n, m))
setup_times = np.random.randint(0, 5, size=(n, m))

# Create figure and axes
fig, ax = plt.subplots()

# Set y-ticks and y-tick labels
ax.set_yticks(jobs)
ax.set_yticklabels([f'Job {i}' for i in jobs])

# Set x-ticks and x-tick labels
ax.set_xticks(np.arange(0, np.max(start_times + processing_times + setup_times), step=5))
ax.set_xlabel('Time')

# Loop over machines and jobs to create horizontal bars
# Map integer values to a color from a colormap
cmap = cm.get_cmap('viridis')
norm = colors.Normalize(vmin=0, vmax=5)
for i in machines:
    machine_label = f'Machine {i}'
    for j in jobs:
        start_time = start_times[j, i]
        setup_time = setup_times[j, i]
        processing_time = processing_times[j, i]
        height = 0.3
        ax.barh(j, setup_time, left=start_time, height=height, align='center', color='blue')
        ax.barh(j, processing_time, left=start_time+setup_time, height=height, align='center', color=cmap(j))
        bar_width = setup_time + processing_time
        ax.text(start_time + bar_width/2, j + height/2, f'Op {j+1}', ha='center', va='center')

    # Add machine label to the left of the plot
    ax.text(-2, n-0.5, machine_label, ha='center', va='center', rotation=90)

# Set the limits of the x-axis and y-axis
ax.set_xlim(0, np.max(start_times + processing_times + setup_times))
ax.set_ylim(-0.5, n-0.5)

# Show the plot
plt.show()
