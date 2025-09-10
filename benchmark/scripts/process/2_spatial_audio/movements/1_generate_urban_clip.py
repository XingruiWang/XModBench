
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
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
        return None, None, None, None, None

    object_tracks = matched.groupby('track_id')
    max_area = 0
    selected_obj = None

    for track_id, group in object_tracks:
        bounding_box = []
        
        for _, frame in group.iterrows():
            bounding_box.append((frame['x'], frame['y'], frame['w'], frame['h'], frame['frame_time']))
        
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
            selected_obj = (x0, y0, direction_x, direction_y, bounding_box)
    if selected_obj:
        return selected_obj
    return None, None, None, None, None

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
    try:
        x0, y0, direction_x, direction_y, bounding_box = get_directional_info(video_subset, label, start, end)
    except Exception as e:
        import ipdb; ipdb.set_trace()
        continue

    records.append({
        "filename": filename,
        "class_id": row['class_id'],
        "label": label,
        "non_identifiable_vehicle_sound": row['non_identifiable_vehicle_sound'],
        "start": start,
        "end": end,
        "start_position": x0 if x0 is not None else -1,
        "direction_x": direction_x,
        "direction_y": direction_y,
        "bounding_box": bounding_box
    })

result_df = pd.DataFrame(records)

# check if any record contains an overlap with another record and with different class_id or direction_x or direction_y

print(f"Total records: {len(result_df)}")
invalid_records = []
for i, row in tqdm(result_df.iterrows()):
    for j, row2 in result_df.iterrows():
        if i != j and j not in invalid_records and row['filename'] == row2['filename']:
            if row['start'] < row2['end'] and row['end'] > row2['start'] :
                if row['class_id'] != row2['class_id'] or row['direction_x'] * row2['direction_x'] < 0 or row['direction_y'] * row2['direction_y'] < 0:
                    invalid_records.append(i)
                    print(f"Invalid record: {row['filename']} {row['start']} {row['end']} {row2['start']} {row2['end']}")

result_df = result_df.drop(invalid_records)

# filter valid records
result_df = result_df[~result_df.index.isin(invalid_records)]

output_csv_path = "audio_with_motion_and_visibility_bounding_box.csv"
result_df.to_csv(output_csv_path, index=False)
