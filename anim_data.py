import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

df = pd.read_csv('test_data/noisy/output_data.csv')

# Prepare the plot
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)  # lw is line width
ax.set_xlim(-50, len(df)+50)  # Set the x-axis to match the number of rows

# Initialize an empty data structure for the plot
def init():
    line.set_data([], [])
    return line,

# Update function for the animation
def update(frame):
    x = list(range(frame + 1))  # Row numbers up to the current frame
    y = df['rewards'].iloc[:frame + 1]  # Reward values up to the current frame
    line.set_data(x, y)

    ax.relim()  # Recalculate limits
    ax.autoscale_view()  # Automatically adjust axis

    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True, repeat=False)

# Save the animation
ani.save('rewards_animation.mp4', writer='ffmpeg', fps=60)