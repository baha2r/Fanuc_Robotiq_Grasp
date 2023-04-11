import os
import tensorflow as tf
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.summary.writer.event_file_writer import EventFileWriter

# Set the input and output paths
input_file = "tensorboard/20230316-03:42PM_SAC_M10000_0.04_39/SAC_1/events.out.tfevents.1678995745.pse-System-Product-Name.2384611.0"
output_file = "filtered_events.tfevents"

# Set the timestep threshold for deletion
timesteps_to_remove = 10_000_000

def main():
    # Get the maximum timestep
    max_step = -1
    for event in event_file_loader.EventFileLoader(input_file).Load():
        if event.step > max_step:
            max_step = event.step
    
    print(f"Max step: {max_step}")
    threshold_step = max_step - timesteps_to_remove
    
    # Filter events and write to a new file
    writer = EventFileWriter(os.path.dirname(output_file))
    for event in event_file_loader.EventFileLoader(input_file).Load():
        if event.step <= threshold_step:
            writer.add_event(event)
        else:
            break

    writer.close()

if __name__ == "__main__":
    main()
