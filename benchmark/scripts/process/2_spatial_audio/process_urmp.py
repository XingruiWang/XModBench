import os
import cv2
import ipdb
import random
import numpy as np
import soundfile as sf
import librosa
import itertools
from tqdm import tqdm
import shutil



URMP_id2class = {
    'vn': 'Violin',
    'vc': 'Cello',
    'va': 'Viola',
    'fl': 'Flute',
    'cl': 'Clarinet',
    'tpt': 'Trumpet',
    'sax': 'Saxophone',
    'tbn': 'Trombone',
    'tba': 'Tuba',
    'ob': 'Oboe',
    'hn': 'French Horn',
    'db': 'Double Bass',
    'bn': 'Bassoon'}
URMP_class2id = {v: k for k, v in URMP_id2class.items()}

data_root = '/home/xwang378/scratch/2025/AudioBench/benchmark/Data/URMP'
processed_data_root = '/home/xwang378/scratch/2025/AudioBench/benchmark/Data/URMP_processed'


def iterate_all_orders(lst):
    """
    Yields all permutations (orders) of the input list.

    Args:
        lst (list): Input list.

    Yields:
        list: A permutation of the input list.
    """
    for perm in itertools.permutations(lst):
        yield list(perm)
        

def process_image(image_path, number_of_objects, full_object_names):
    """
    Process the image to create a description based on the number of objects.
    """
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    all_images = []
    for i in range(number_of_objects):
        object_image = image.copy()
        object_image = object_image[:h, i * (w // number_of_objects):(i + 1) * (w // number_of_objects)]
        all_images.append({
            'name': full_object_names[i],
            'image': object_image,
            # 'description': f"A {full_object_names[i].capitalize()}."
        })
    return all_images


def gen_spatial_audio(audios, number_of_objects):
    """
    Simulate stereo audio with sources positioned from -45° (left front) to +45° (right front).

    Args:
        audios (List[np.ndarray]): List of mono audio waveforms (1D numpy arrays).
        number_of_objects (int): Number of objects/sources to spatialize.

    Returns:
        np.ndarray: Stereo waveform as shape (samples, 2)
    """
    assert number_of_objects <= len(audios), "Not enough audio clips for the number of objects."

    # Normalize length of all audios
    max_len = max(len(a) for a in audios[:number_of_objects])
    padded_audios = [np.pad(a, (0, max_len - len(a))) for a in audios[:number_of_objects]]

    stereo_output = np.zeros((max_len, 2))

    for i, audio in enumerate(padded_audios):
        # Distribute angles between -45° to +45°, convert to pan value [-1, 1]
        angle_deg = -45 + 90 * i / (number_of_objects - 1) if number_of_objects > 1 else 0
        pan = np.sin(np.radians(angle_deg))  # -1 (left) to 1 (right)

        # Equal power panning
        left_gain = np.cos((pan + 1) * np.pi / 4)
        right_gain = np.sin((pan + 1) * np.pi / 4)

        stereo_output[:, 0] += audio * left_gain
        stereo_output[:, 1] += audio * right_gain

    # Normalize to prevent clipping
    max_val = np.max(np.abs(stereo_output))
    if max_val > 1.0:
        stereo_output /= max_val

    return stereo_output

def extract_peak_audio_segment(audio_data, sr = 16000, window=2.0):

    mono_audio = audio_data if audio_data.ndim == 1 else audio_data.mean(axis=1)

    win_len = int(sr * window)
    max_energy = -np.inf
    best_start = 0
    for start in range(0, len(mono_audio) - win_len, len(mono_audio) // 32):
        segment = mono_audio[start:start + win_len]
        energy = np.mean(segment ** 2)
        if energy > max_energy:
            max_energy = energy
            best_start = start
    peak_audio = audio_data[best_start:best_start + win_len, :]
    return peak_audio
    
for instance_name in tqdm(os.listdir(data_root)):
    if instance_name in ['Supplementary_Files']:
        continue
    instance_path = os.path.join(data_root, instance_name)
    if os.path.isdir(instance_path):
        objects_name = instance_name.split('_')[2:]
        full_object_names = [URMP_id2class[obj] for obj in objects_name]
        number_of_objects = len(full_object_names)
        
        # Load audios
        all_separate_audios = [file_name for file_name in os.listdir(instance_path) if (file_name.startswith('AuSep') and file_name.endswith('.wav'))]
        all_separate_audios = sorted(all_separate_audios)
        all_separate_audios = [os.path.join(instance_path, file_name) for file_name in all_separate_audios]
        for i in range(len(all_separate_audios)):
            audio, sr = sf.read(all_separate_audios[i])
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            all_separate_audios[i] = audio

        # Crop images
        images = os.path.join(processed_data_root, instance_name, 'Vid_'+instance_name+'.jpg')
        all_seperate_images = process_image(images, number_of_objects, full_object_names)
        
        # Random compose
        rm_instance_path = os.path.join(processed_data_root, instance_name, 'reordered')

        shutil.rmtree(rm_instance_path, ignore_errors=True)
        
        for i, rand_order in enumerate(iterate_all_orders(list(range(number_of_objects)))):
            
            rand_seperate_images = [all_seperate_images[i] for i in rand_order]
            rand_separate_audios = [all_separate_audios[i] for i in rand_order]
            
            # Generate spatial audio
            spatial_audio = gen_spatial_audio(rand_separate_audios, number_of_objects)
            spatial_audio = extract_peak_audio_segment(spatial_audio, sr=16000, window=2.0)
            
            # generate new scene image by combining all images
            scene_image = cv2.hconcat([img['image'] for img in rand_seperate_images])
            
            object_names = [img['name'] for img in rand_seperate_images]
            
            # save processed data
            order_name = '_'.join([URMP_class2id[img['name']] for img in rand_seperate_images])
            processed_instance_path = os.path.join(processed_data_root, instance_name, 'reordered', f'{order_name}')
            
            if os.path.exists(processed_instance_path):
                continue
            
            os.makedirs(processed_instance_path, exist_ok=True)
            cv2.imwrite(os.path.join(processed_instance_path, 'image.jpg'), scene_image)
            
            sf.write(os.path.join(processed_instance_path, 'audio.wav'), spatial_audio, 16000)
            with open(os.path.join(processed_instance_path, 'objects.txt'), 'w') as f:
                for name in object_names:
                    f.write(name + '\n')
            
            
            
        
                    
        
        
        
