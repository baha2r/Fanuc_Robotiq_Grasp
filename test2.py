import os
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def get_tensorboard_data(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir,
                                            size_guidance={
                                                event_accumulator.SCALARS: 0,
                                                event_accumulator.IMAGES: 0,
                                                event_accumulator.AUDIO: 0,
                                                event_accumulator.HISTOGRAMS: 0,
                                                event_accumulator.COMPRESSED_HISTOGRAMS: 0,
                                            })

    ea.Reload()  # Load all data in the log directory

    # Extract scalar data
    scalar_tags = ea.Tags()['scalars']
    
    # Return an empty DataFrame if no scalar events are found
    if not scalar_tags:
        return pd.DataFrame(columns=['step', 'value', 'wall_time', 'tag'])

    # Convert scalar data to pandas DataFrame
    dfs = []
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        step = [event.step for event in events]
        value = [event.value for event in events]
        wall_time = [event.wall_time for event in events]
        df = pd.DataFrame({'step': step, 'value': value, 'wall_time': wall_time})
        df['tag'] = tag
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

log_dir = './tensorboard/20230316-03:42PM_SAC_M10000_0.04_39/SAC_1/events.out.tfevents.1678995745.pse-System-Product-Name.2384611.0'
df = get_tensorboard_data(log_dir)
print(df.tag('eval/mean_reward') )
