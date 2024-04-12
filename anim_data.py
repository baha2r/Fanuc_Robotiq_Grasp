import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set the directory where your data is stored
dir = 'test_data/test10/'

# Load the data from CSV
df = pd.read_csv(dir + 'output_data.csv')

# Function to create and save an animation for a specified column
def create_animation(df, column_name, filename):
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)  # lw is line width
    ax.set_xlim(-50, len(df) + 50)  # Adjust x-axis to match the number of rows
    # ax.set_ylim(df[column_name].min() - 10, df[column_name].max() + 10)  # Adjust y-axis based on min and max values of the column

    # Initialize an empty data structure for the plot
    def init():
        line.set_data([], [])
        return line,

    # Update function for the animation
    def update(frame):
        x = list(range(frame + 1))  # Row numbers up to the current frame
        y = df[column_name].iloc[:frame + 1]  # Column values up to the current frame
        line.set_data(x, y)

        ax.relim()  # Recalculate limits
        ax.autoscale_view()  # Automatically adjust axis

        return line,

    # Create the animation
    animation = FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True, repeat=False)

    # Save the animation
    animation.save(dir + filename, writer='ffmpeg', fps=60)

# Create and save animations for both 'closest_distance' and 'reward' columns
create_animation(df, 'closest_distance', 'closest_distance_vid.mp4')
create_animation(df, 'rewards', 'reward_vid.mp4')
