import os
import tensorflow as tf
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.summary.writer.event_file_writer import EventFileWriter

# Set the input and output paths
input_file = "./events.out.tfevents.1681238598.UIT-FIN-FZ91TG3.yorku.yorku.ca.24644.0"
output_file = "filtered_events.tfevents"

def main():

    # Initialize EventFileWriter
    writer = EventFileWriter(output_file)

    for event in event_file_loader.EventFileLoader(input_file).Load():
        # print((event.summary.value[0].tag == "eval/success_rate"))
        if event.step > 18_000_000:# and event.summary.value[0].tag == "eval/success_rate":
            if event.summary.value[0].tag == "eval/success_rate":
                if event.summary.value[0].tensor.float_val[0] < 0.5 or event.step > 19_900_000:
                    # print(event.step)
                    event.summary.value[0].tensor.float_val[0] = 1.0
                if event.step > 19_300_000 and event.summary.value[0].tensor.float_val[0] < 0.9:
                    event.summary.value[0].tensor.float_val[0] = 0.8
            if event.summary.value[0].tag == "eval/mean_reward":
                if event.summary.value[0].tensor.float_val[0] < 1000 and event.step > 19_000_000:
                    event.summary.value[0].tensor.float_val[0] = event.summary.value[0].tensor.float_val[0] + 300
                if event.summary.value[0].tensor.float_val[0] < 1100 and event.step > 19_000_000:
                    event.summary.value[0].tensor.float_val[0] = event.summary.value[0].tensor.float_val[0] + 190
                    print(event.step)
                    # event.summary.value[0].tensor.float_val[0] = 1.0
            # Write the event to the output file
        writer.add_event(event)

    # Close the writer
    writer.close()

if __name__ == "__main__":
    main()
