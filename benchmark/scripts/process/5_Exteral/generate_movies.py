import os
import re
import random
import json
import pandas as pd
import ast
from pathlib import Path


# ==================== Movie Multimodal Matching Question-Answer Generator ====================


def load_movie_dataset(csv_path, poster_dir, trailer_dir):
    """
    Load movie dataset from CSV and organize with available media files.
    
    Args:
        csv_path (str): Path to the movie dataset CSV file
        poster_dir (str): Directory containing movie poster images
        trailer_dir (str): Directory containing movie trailer audio clips
        
    Returns:
        dict: Dictionary with movie_id as keys and movie data as values
    """
    # Load movie data from CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} movies from CSV")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return {}
    
    movie_data = {}
    
    # Get available poster files
    available_posters = {}
    if os.path.exists(poster_dir):
        for poster_file in os.listdir(poster_dir):
            if poster_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Extract movie ID from filename (e.g., "7451.jpg" -> "7451")
                movie_id = os.path.splitext(poster_file)[0]
                available_posters[movie_id] = os.path.join(poster_dir, poster_file)
    
    # Get available trailer files
    available_trailers = {}
    if os.path.exists(trailer_dir):
        for trailer_file in os.listdir(trailer_dir):
            if trailer_file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                # Extract movie ID from filename (e.g., "movie_7451_trailer_0.wav" -> "7451")
                match = re.search(r'movie_(\d+)_trailer', trailer_file)
                if match:
                    movie_id = match.group(1)
                    available_trailers[movie_id] = os.path.join(trailer_dir, trailer_file)
    
    print(f"Found {len(available_posters)} posters and {len(available_trailers)} trailers")
    
    # Process each movie and match with available media
    for _, row in df.iterrows():
        movie_id = str(row['id'])
        
        # Check if this movie has both poster and trailer
        has_poster = movie_id in available_posters
        has_trailer = movie_id in available_trailers
        
        if has_poster and has_trailer and pd.notna(row['overview']) and row['overview'].strip():
            # Parse genres if they're in string format
            genres = []
            try:
                if pd.notna(row['genres']):
                    if isinstance(row['genres'], str):
                        # Handle string representation of list
                        genres = ast.literal_eval(row['genres'])
                    elif isinstance(row['genres'], list):
                        genres = row['genres']
            except:
                genres = []
            
            # Parse cast if available
            cast = []
            try:
                if pd.notna(row['cast']):
                    if isinstance(row['cast'], str):
                        cast = ast.literal_eval(row['cast'])
                    elif isinstance(row['cast'], list):
                        cast = row['cast']
            except:
                cast = []
            
            movie_data[movie_id] = {
                'id': movie_id,
                'title': row['title'],
                'original_title': row.get('original_title', row['title']),
                'overview': row['overview'].strip(),
                'release_date': row.get('release_date', ''),
                'genres': genres,
                'cast': cast,
                'popularity': row.get('popularity', 0),
                'vote_average': row.get('vote_average', 0),
                'vote_count': row.get('vote_count', 0),
                'poster_path': available_posters[movie_id],
                'trailer_path': available_trailers[movie_id],
                'runtime': row.get('runtime', 0),
                'budget': row.get('budget', 0),
                'revenue': row.get('revenue', 0)
            }
    
    print(f"Successfully processed {len(movie_data)} movies with complete data")
    return movie_data


def sample_mixed_movie_instances(movie_data, target_movie_id, n_samples=4):
    """
    Sample movie instances with one correct target and distractors.
    
    Args:
        movie_data (dict): Dictionary of all movie data
        target_movie_id (str): The correct movie ID for this question
        n_samples (int): Total number of options (default: 4)
        
    Returns:
        tuple: (sampled_movies, correct_answer) or (None, None) if sampling fails
    """
    if target_movie_id not in movie_data or len(movie_data) < n_samples:
        return None, None
    
    # Get target movie
    target_movie = movie_data[target_movie_id]
    
    # Sample distractor movies
    other_movie_ids = [mid for mid in movie_data.keys() if mid != target_movie_id]
    if len(other_movie_ids) < n_samples - 1:
        return None, None
    
    distractor_ids = random.sample(other_movie_ids, n_samples - 1)
    selected_ids = [target_movie_id] + distractor_ids
    random.shuffle(selected_ids)
    
    # Find the correct answer position
    correct_idx = selected_ids.index(target_movie_id)
    correct_answer = chr(ord('A') + correct_idx)
    
    # Get sampled movie data
    sampled_movies = [movie_data[mid] for mid in selected_ids]
    
    return sampled_movies, correct_answer


def generate_question_audio_vision(movies, correct_answer, target_movie):
    """Generate audio -> vision question: listen to trailer and choose matching poster."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_movie = movies[correct_idx]
    
    question = {
        "question": "Listen to this movie trailer. Which movie poster belongs to the same film? Answer with A, B, C, or D",
        "target_movie_id": target_movie['id'],
        "target_movie_title": target_movie['title'],
        "conditions": {
            "modality": "Audio",
            "input": correct_movie['trailer_path'],
            "movie_id": correct_movie['id'],
            "title": correct_movie['title']
        },
        "options": {
            "A": {"modality": "Image", "input": movies[0]['poster_path'], "movie_id": movies[0]['id'], "title": movies[0]['title']},
            "B": {"modality": "Image", "input": movies[1]['poster_path'], "movie_id": movies[1]['id'], "title": movies[1]['title']},
            "C": {"modality": "Image", "input": movies[2]['poster_path'], "movie_id": movies[2]['id'], "title": movies[2]['title']},
            "D": {"modality": "Image", "input": movies[3]['poster_path'], "movie_id": movies[3]['id'], "title": movies[3]['title']}
        },
        "correct_answer": correct_answer,
        "correct_movie_id": target_movie['id']
    }
    return question


def generate_question_vision_audio(movies, correct_answer, target_movie):
    """Generate vision -> audio question: look at poster and choose matching trailer."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_movie = movies[correct_idx]
    
    question = {
        "question": "Look at this movie poster. Which trailer audio belongs to the same film? Answer with A, B, C, or D",
        "target_movie_id": target_movie['id'],
        "target_movie_title": target_movie['title'],
        "conditions": {
            "modality": "Image",
            "input": correct_movie['poster_path'],
            "movie_id": correct_movie['id'],
            "title": correct_movie['title']
        },
        "options": {
            "A": {"modality": "Audio", "input": movies[0]['trailer_path'], "movie_id": movies[0]['id'], "title": movies[0]['title']},
            "B": {"modality": "Audio", "input": movies[1]['trailer_path'], "movie_id": movies[1]['id'], "title": movies[1]['title']},
            "C": {"modality": "Audio", "input": movies[2]['trailer_path'], "movie_id": movies[2]['id'], "title": movies[2]['title']},
            "D": {"modality": "Audio", "input": movies[3]['trailer_path'], "movie_id": movies[3]['id'], "title": movies[3]['title']}
        },
        "correct_answer": correct_answer,
        "correct_movie_id": target_movie['id']
    }
    return question


def generate_question_vision_text(movies, correct_answer, target_movie):
    """Generate vision -> text question: look at poster and choose matching overview."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_movie = movies[correct_idx]
    
    question = {
        "question": "Look at this movie poster. Which plot overview describes the same film? Answer with A, B, C, or D",
        "target_movie_id": target_movie['id'],
        "target_movie_title": target_movie['title'],
        "conditions": {
            "modality": "Image",
            "input": correct_movie['poster_path'],
            "movie_id": correct_movie['id'],
            "title": correct_movie['title']
        },
        "options": {
            "A": {"modality": "Text", "input": movies[0]['overview'], "movie_id": movies[0]['id'], "title": movies[0]['title']},
            "B": {"modality": "Text", "input": movies[1]['overview'], "movie_id": movies[1]['id'], "title": movies[1]['title']},
            "C": {"modality": "Text", "input": movies[2]['overview'], "movie_id": movies[2]['id'], "title": movies[2]['title']},
            "D": {"modality": "Text", "input": movies[3]['overview'], "movie_id": movies[3]['id'], "title": movies[3]['title']}
        },
        "correct_answer": correct_answer,
        "correct_movie_id": target_movie['id']
    }
    return question


def generate_question_text_vision(movies, correct_answer, target_movie):
    """Generate text -> vision question: read overview and choose matching poster."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_movie = movies[correct_idx]
    
    question = {
        "question": "Read this plot overview. Which movie poster belongs to the film described? Answer with A, B, C, or D",
        "target_movie_id": target_movie['id'],
        "target_movie_title": target_movie['title'],
        "conditions": {
            "modality": "Text",
            "input": correct_movie['overview'],
            "movie_id": correct_movie['id'],
            "title": correct_movie['title']
        },
        "options": {
            "A": {"modality": "Image", "input": movies[0]['poster_path'], "movie_id": movies[0]['id'], "title": movies[0]['title']},
            "B": {"modality": "Image", "input": movies[1]['poster_path'], "movie_id": movies[1]['id'], "title": movies[1]['title']},
            "C": {"modality": "Image", "input": movies[2]['poster_path'], "movie_id": movies[2]['id'], "title": movies[2]['title']},
            "D": {"modality": "Image", "input": movies[3]['poster_path'], "movie_id": movies[3]['id'], "title": movies[3]['title']}
        },
        "correct_answer": correct_answer,
        "correct_movie_id": target_movie['id']
    }
    return question


def generate_question_text_audio(movies, correct_answer, target_movie):
    """Generate text -> audio question: read overview and choose matching trailer."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_movie = movies[correct_idx]
    
    question = {
        "question": "Read this plot overview. Which trailer audio belongs to the film described? Answer with A, B, C, or D",
        "target_movie_id": target_movie['id'],
        "target_movie_title": target_movie['title'],
        "conditions": {
            "modality": "Text",
            "input": correct_movie['overview'],
            "movie_id": correct_movie['id'],
            "title": correct_movie['title']
        },
        "options": {
            "A": {"modality": "Audio", "input": movies[0]['trailer_path'], "movie_id": movies[0]['id'], "title": movies[0]['title']},
            "B": {"modality": "Audio", "input": movies[1]['trailer_path'], "movie_id": movies[1]['id'], "title": movies[1]['title']},
            "C": {"modality": "Audio", "input": movies[2]['trailer_path'], "movie_id": movies[2]['id'], "title": movies[2]['title']},
            "D": {"modality": "Audio", "input": movies[3]['trailer_path'], "movie_id": movies[3]['id'], "title": movies[3]['title']}
        },
        "correct_answer": correct_answer,
        "correct_movie_id": target_movie['id']
    }
    return question


def generate_question_audio_text(movies, correct_answer, target_movie):
    """Generate audio -> text question: listen to trailer and choose matching overview."""
    correct_idx = ord(correct_answer) - ord('A')
    correct_movie = movies[correct_idx]
    
    question = {
        "question": "Listen to this movie trailer. Which plot overview describes the same film? Answer with A, B, C, or D",
        "target_movie_id": target_movie['id'],
        "target_movie_title": target_movie['title'],
        "conditions": {
            "modality": "Audio",
            "input": correct_movie['trailer_path'],
            "movie_id": correct_movie['id'],
            "title": correct_movie['title']
        },
        "options": {
            "A": {"modality": "Text", "input": movies[0]['overview'], "movie_id": movies[0]['id'], "title": movies[0]['title']},
            "B": {"modality": "Text", "input": movies[1]['overview'], "movie_id": movies[1]['id'], "title": movies[1]['title']},
            "C": {"modality": "Text", "input": movies[2]['overview'], "movie_id": movies[2]['id'], "title": movies[2]['title']},
            "D": {"modality": "Text", "input": movies[3]['overview'], "movie_id": movies[3]['id'], "title": movies[3]['title']}
        },
        "correct_answer": correct_answer,
        "correct_movie_id": target_movie['id']
    }
    return question


def generate_all_modality_combinations(movies, correct_answer, target_movie):
    """Generate all possible modality combinations for movie matching."""
    questions = {}
    
    questions['audio_vision'] = generate_question_audio_vision(movies, correct_answer, target_movie)
    questions['vision_audio'] = generate_question_vision_audio(movies, correct_answer, target_movie)
    questions['vision_text'] = generate_question_vision_text(movies, correct_answer, target_movie)
    questions['text_vision'] = generate_question_text_vision(movies, correct_answer, target_movie)
    questions['text_audio'] = generate_question_text_audio(movies, correct_answer, target_movie)
    questions['audio_text'] = generate_question_audio_text(movies, correct_answer, target_movie)
    
    return questions


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    
    # Configuration parameters
    DATASET_NAME = 'movie_matching'
    csv_path = "/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/process/5_Exteral/movie_academic_dataset.csv"  # Update to your CSV path
    poster_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/process/5_Exteral/posters"  # Update to your posters directory
    trailer_dir = "/home/xwang378/scratch/2025/AudioBench/benchmark/scripts/process/5_Exteral/trailer_clips"  # Update to your trailer clips directory
    
    N = 200  # Number of questions to generate
    
    # Select modality combinations to generate
    GENERATE_COMBINATIONS = [
        'audio_vision',  # Trailer -> Poster (Listen to trailer, choose poster)
        'vision_audio',  # Poster -> Trailer (Look at poster, choose trailer)
        'vision_text',   # Poster -> Overview (Look at poster, choose overview)
        'text_vision',   # Overview -> Poster (Read overview, choose poster)
        'text_audio',    # Overview -> Trailer (Read overview, choose trailer)
        'audio_text'     # Trailer -> Overview (Listen to trailer, choose overview)
    ]
    
    export_dir = f'/home/xwang378/scratch/2025/AudioBench/benchmark/tasks/05_Exteral/movie_matching'
    
    # Load the movie dataset
    print("Loading movie dataset...")
    movie_data = load_movie_dataset(csv_path, poster_dir, trailer_dir)
    
    if len(movie_data) < 4:
        print("Error: Need at least 4 movies with complete data to generate questions!")
        exit(1)
    
    print(f"Dataset loaded successfully:")
    print(f"  Total movies with complete data: {len(movie_data)}")
    
    # Sample some movie titles for preview
    sample_titles = list(movie_data.values())[:5]
    for movie in sample_titles:
        print(f"  - {movie['title']} ({movie['id']})")
    
    # Initialize question lists for each combination
    all_questions = {combo: [] for combo in GENERATE_COMBINATIONS}
    movie_stats = {}
    
    # Generate questions using different target movies
    movie_ids = list(movie_data.keys())
    successful_generations = 0
    
    for i in range(N):
        # Randomly select a target movie
        target_movie_id = random.choice(movie_ids)
        target_movie = movie_data[target_movie_id]
        
        # Sample mixed instances (1 correct + 3 distractors)
        sampled_movies, correct_answer = sample_mixed_movie_instances(
            movie_data, target_movie_id, 4
        )
        
        if sampled_movies is None:
            print(f"  Skipping iteration {i+1}: sampling failed")
            continue
        
        # Generate questions for all modality combinations
        questions = generate_all_modality_combinations(
            sampled_movies, correct_answer, target_movie
        )
        
        # Add to corresponding lists
        for combo in GENERATE_COMBINATIONS:
            if combo in questions:
                all_questions[combo].append(questions[combo])
        
        successful_generations += 1
        
        # Track which movies were used as targets
        if target_movie_id not in movie_stats:
            movie_stats[target_movie_id] = 0
        movie_stats[target_movie_id] += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {successful_generations} successful questions out of {i + 1} attempts")
    
    # Save output files
    os.makedirs(export_dir, exist_ok=True)
    for combo in GENERATE_COMBINATIONS:
        filename = f"{export_dir}/{DATASET_NAME}_questions_{combo}.json"
        with open(filename, "w") as f:
            json.dump(all_questions[combo], f, indent=4)
        print(f"Saved {len(all_questions[combo])} questions to {filename}")
    
    # Save generation statistics
    stats = {
        "dataset": DATASET_NAME,
        "total_movies": len(movie_data),
        "questions_generated": successful_generations,
        "total_questions_per_combination": successful_generations,
        "movie_usage_stats": movie_stats,
        "combinations": GENERATE_COMBINATIONS,
        "task_type": "movie_instance_matching_across_modalities",
        "data_sources": {
            "csv": csv_path,
            "posters": poster_dir,
            "trailers": trailer_dir
        }
    }
    
    with open(f"{export_dir}/{DATASET_NAME}_generation_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    
    print(f"\n=== Generation Summary ===")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Total movies in dataset: {len(movie_data)}")
    print(f"Task: Movie instance matching across modalities")
    print(f"Successful questions generated: {successful_generations}")
    print(f"Questions per combination: {successful_generations}")
    print(f"Generated combinations: {', '.join(GENERATE_COMBINATIONS)}")
    print(f"Export directory: {export_dir}")
    
    print(f"\nTop 10 most frequently used target movies:")
    sorted_movies = sorted(movie_stats.items(), key=lambda x: x[1], reverse=True)[:10]
    for movie_id, count in sorted_movies:
        movie_title = movie_data[movie_id]['title']
        print(f"  {movie_title} (ID: {movie_id}): {count} times")