import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Define tasks and timelines
tasks = [
    ("Finalize dataset and ontology", "2025-01-15", "2025-01-21"),
    ("Data collection and preprocessing", "2025-01-22", "2025-02-04"),
    ("NER PrimeKGIntegration implementation", "2025-01-29", "2025-02-11"),  # Runs in parallel with data preprocessing
    ("Relation extraction models", "2025-02-12", "2025-02-25"),
    ("Post-process extracted triplets", "2025-02-20", "2025-03-07"),
    ("Knowledge graph integration", "2025-03-07", "2025-03-15"),
    ("Graph reduction and exploratory analysis", "2025-03-18", "2025-03-30"),
    ("GNN tasks and report finalization", "2025-03-31", "2025-04-15")
]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define date format and convert string dates to datetime
start_dates = [datetime.strptime(task[1], "%Y-%m-%d") for task in tasks]
end_dates = [datetime.strptime(task[2], "%Y-%m-%d") for task in tasks]
task_names = [task[0] for task in tasks]
durations = [(end - start).days for start, end in zip(start_dates, end_dates)]

# Plot bars for each task with some overlapping
for i, (start, duration, name) in enumerate(zip(start_dates, durations, task_names)):
    overlap_offset = -0.2 if i % 2 == 0 else 0.2  # Alternate overlapping offset
    ax.barh(i + overlap_offset, duration, left=start, height=0.4, align='center', label=name if i == 0 else "")

# Format the timeline
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45)
ax.set_yticks(range(len(task_names)))
ax.set_yticklabels(task_names)

# Add labels and grid
ax.set_xlabel("Timeline")
ax.set_ylabel("Tasks")
ax.set_title("Project Timeline Gantt Chart")
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
