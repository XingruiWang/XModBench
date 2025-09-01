
import pandas as pd
from collections import defaultdict

# Load CSVs
audio_csv = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/Urbansas/annotations/audio_annotations.csv"
video_csv = "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/Urbansas/annotations/video_annotations.csv"

audio_df = pd.read_csv(audio_csv)
video_df = pd.read_csv(video_csv)

# Preprocess video annotations by clip
video_df['frame_time'] = video_df['frame_id'] * 0.5
video_grouped = video_df.groupby('filename')

# Define helper to load and process
def get_directional_info(video_subset, label, start_time, end_time):
    matched = video_subset[(video_subset['label'] == label) &
                           (video_subset['frame_time'] >= start_time) &
                           (video_subset['frame_time'] <= end_time) &
                           (video_subset['visibility'] > 0.5)]

    if matched.empty:
        return None, None, None, None

    object_tracks = matched.groupby('track_id')
    max_area = 0
    selected_obj = None

    for track_id, group in object_tracks:
        start_frame = group.nsmallest(1, 'frame_time')
        end_frame = group.nlargest(1, 'frame_time')
        x0, x1 = start_frame['x'].values[0], end_frame['x'].values[0]
        y0, y1 = start_frame['y'].values[0], end_frame['y'].values[0]
        w0 = start_frame['w'].values[0]
        h0 = start_frame['h'].values[0]
        area = w0 * h0

        if area > max_area:
            max_area = area
            direction_x = x1 - x0
            direction_y = y1 - y0
            selected_obj = (x0, y0, direction_x, direction_y)

    if selected_obj:
        return selected_obj
    return None, None, None, None

# Process each audio annotation
records = []

for _, row in audio_df.iterrows():
    filename = row['filename']
    label = row['label']
    start = float(row['start'])
    end = float(row['end'])
    # import ipdb; ipdb.set_trace()
    if row['class_id'] == -1:
        # no car scene
        records.append({
            "filename": filename,
            "class_id": row['class_id'],
            "label": label,
            "non_identifiable_vehicle_sound": row['non_identifiable_vehicle_sound'],
            "start": 0,
            "end": 4,
            "start_position": -1,
            "direction_x": 0,
            "direction_y": 0
        })
    if row['non_identifiable_vehicle_sound'] == 1:
        continue  # skip non-identifiable vehicle sounds

    if filename not in video_grouped.groups:
        continue

    video_subset = video_grouped.get_group(filename)
    x0, y0, direction_x, direction_y = get_directional_info(video_subset, label, start, end)

    records.append({
        "filename": filename,
        "class_id": row['class_id'],
        "label": label,
        "non_identifiable_vehicle_sound": row['non_identifiable_vehicle_sound'],
        "start": start,
        "end": end,
        "start_position": x0 if x0 is not None else -1,
        "direction_x": direction_x,
        "direction_y": direction_y
    })

result_df = pd.DataFrame(records)
output_csv_path = "audio_with_motion_and_visibility.csv"
result_df.to_csv(output_csv_path, index=False)
