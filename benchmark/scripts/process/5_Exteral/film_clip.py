import pandas as pd
import requests
import json
import time
from pathlib import Path
import yt_dlp
import subprocess
import os
import warnings
from googleapiclient.discovery import build
import numpy as np
import librosa
import scipy.signal
from collections import defaultdict

# Try to import browser cookie support
try:
    import browser_cookie3
    BROWSER_COOKIES_AVAILABLE = True
except ImportError:
    BROWSER_COOKIES_AVAILABLE = False
    print("⚠️ browser_cookie3 not installed. Install with: pip install browser-cookie3")

class AudioClipGenerator:
    def __init__(self, tmdb_api_key, youtube_api_key=None, cookies_file=None):
        self.tmdb_api_key = tmdb_api_key
        self.youtube_api_key = youtube_api_key
        self.tmdb_base_url = "https://api.themoviedb.org/3"
        self.cookies_file = cookies_file
        
        # Create directories
        self.clips_dir = Path("trailer_clips")
        self.clips_dir.mkdir(exist_ok=True)
        
        # Create directory for full videos
        self.videos_dir = Path("trailer_videos")
        self.videos_dir.mkdir(exist_ok=True)
        
        # Create directory for processed audio (30s clips with dialogue)
        self.dialogue_clips_dir = Path("dialogue_clips")
        self.dialogue_clips_dir.mkdir(exist_ok=True)
        
        # Check if ffmpeg is available
        self.ffmpeg_available = self._check_ffmpeg()
        if not self.ffmpeg_available:
            print("ERROR: FFmpeg not found. Please install FFmpeg first.")
            print("Visit: https://ffmpeg.org/download.html")
            exit(1)
        
        print("✓ FFmpeg found - ready to extract audio clips and videos")
        
        # Check if librosa is available for advanced audio processing
        try:
            import librosa
            self.librosa_available = True
            print("✓ librosa found - advanced dialogue detection enabled")
        except ImportError:
            assert False, "librosa not found"
            self.librosa_available = False
            print("⚠️ librosa not found. Install with: pip install librosa")
            print("   (Will use basic audio processing without dialogue detection)")
        
        # Setup cookies for YouTube
        self._setup_cookies()
    
    def _check_ffmpeg(self):
        """Check if FFmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _setup_cookies(self):
        """Setup cookies for YouTube authentication"""
        self.cookie_method = None
        
        # Method 1: User-provided cookies file
        if self.cookies_file and Path(self.cookies_file).exists():
            print(f"✓ Using cookies file: {self.cookies_file}")
            self.cookie_method = 'file'
            return
        
        # Method 2: Try to extract from browser
        if BROWSER_COOKIES_AVAILABLE:
            try:
                # Try different browsers
                for browser_name, browser_func in [
                    ('Chrome', browser_cookie3.chrome),
                    ('Firefox', browser_cookie3.firefox),
                    ('Edge', browser_cookie3.edge),
                    ('Safari', browser_cookie3.safari),
                ]:
                    try:
                        cookies = browser_func(domain_name='youtube.com')
                        if cookies:
                            # Create temporary cookies file
                            self.cookies_file = 'temp_youtube_cookies.txt'
                            self._save_cookies_to_file(cookies, self.cookies_file)
                            print(f"✓ Extracted cookies from {browser_name}")
                            self.cookie_method = 'browser'
                            return
                    except Exception as e:
                        continue
            except Exception as e:
                print(f"⚠️ Could not extract browser cookies: {e}")
        
        # Method 3: Instructions for manual cookie extraction
        print("⚠️ No cookies available - YouTube may block requests")
        print("\nTo fix this:")
        print("1. Install: pip install browser-cookie3")
        print("2. Or manually export cookies:")
        print("   - Visit YouTube in your browser")
        print("   - Install 'Get cookies.txt' browser extension")
        print("   - Export cookies to 'youtube_cookies.txt'")
        print("   - Pass the file path when creating the generator")
    
    def _save_cookies_to_file(self, cookies, filename):
        """Save browser cookies to Netscape format file"""
        try:
            with open(filename, 'w') as f:
                f.write("# Netscape HTTP Cookie File\n")
                for cookie in cookies:
                    f.write(f"{cookie.domain}\tTRUE\t{cookie.path}\t{'TRUE' if cookie.secure else 'FALSE'}\t{int(cookie.expires) if cookie.expires else 0}\t{cookie.name}\t{cookie.value}\n")
        except Exception as e:
            print(f"⚠️ Could not save cookies: {e}")
    
    def detect_dialogue_segments(self, audio_path, min_duration=30):
        """
        Detect segments with dialogue using advanced audio analysis
        Returns list of (start_time, end_time, confidence_score) tuples
        """
        if not self.librosa_available:
            # Fallback: return middle segment
            return [(60, 90, 0.5)]  # Simple fallback: 60-90 seconds
        
        try:
            print("    🎵 Analyzing audio for dialogue segments...")
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050)
            duration = len(y) / sr
            
            if duration < min_duration:
                print(f"    ⚠️ Audio too short ({duration:.1f}s), using full duration")
                return [(0, duration, 0.3)]
            
            # Parameters for analysis
            frame_length = 2048
            hop_length = 512
            
            # 1. Voice Activity Detection using spectral features
            # Extract features that indicate speech
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
            
            # 2. RMS energy (volume)
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            
            # 3. Spectral features for voice detection
            # Voice typically has energy concentrated in 85-255 Hz (fundamental freq)
            # and 1000-4000 Hz (formants)
            stft = librosa.stft(y, hop_length=hop_length)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
            
            # Voice frequency bands
            voice_low_idx = np.where((freqs >= 85) & (freqs <= 255))[0]
            voice_mid_idx = np.where((freqs >= 1000) & (freqs <= 4000))[0]
            
            voice_energy = np.sum(np.abs(stft[voice_low_idx, :]), axis=0) + \
                          np.sum(np.abs(stft[voice_mid_idx, :]), axis=0)
            
            # 4. Combine features for dialogue confidence score
            # Normalize features
            rms_norm = (rms - np.mean(rms)) / (np.std(rms) + 1e-8)
            voice_energy_norm = (voice_energy - np.mean(voice_energy)) / (np.std(voice_energy) + 1e-8)
            spectral_centroid_norm = (spectral_centroids - np.mean(spectral_centroids)) / (np.std(spectral_centroids) + 1e-8)
            
            # Dialogue confidence: combination of energy, voice frequencies, and spectral characteristics
            dialogue_confidence = (
                0.3 * np.clip(rms_norm, 0, 2) +  # Energy (clipped to avoid outliers)
                0.4 * np.clip(voice_energy_norm, 0, 2) +  # Voice frequency energy
                0.2 * (1 - np.abs(spectral_centroid_norm)) +  # Stable spectral centroid
                0.1 * (1 - zcr)  # Lower zero crossing rate (less noise)
            )
            
            # Smooth the confidence score
            dialogue_confidence = scipy.signal.savgol_filter(dialogue_confidence, 
                                                           window_length=min(21, len(dialogue_confidence)//2*2+1), 
                                                           polyorder=2)
            
            # 5. Find segments with high dialogue confidence
            time_frames = librosa.frames_to_time(np.arange(len(dialogue_confidence)), 
                                               sr=sr, hop_length=hop_length)
            
            # Find peaks in dialogue confidence
            peaks, _ = scipy.signal.find_peaks(dialogue_confidence, 
                                             height=np.percentile(dialogue_confidence, 60),
                                             distance=int(sr/hop_length * 10))  # At least 10s apart
            
            segments = []
            
            # Create segments around peaks
            for peak_idx in peaks:
                peak_time = time_frames[peak_idx]
                confidence = dialogue_confidence[peak_idx]
                
                # Find segment boundaries
                start_time = max(0, peak_time - min_duration/2)
                end_time = min(duration, start_time + min_duration)
                
                # Adjust if we're too close to the end
                if end_time - start_time < min_duration:
                    start_time = max(0, end_time - min_duration)
                
                segments.append((start_time, end_time, confidence))
            
            # If no good segments found, use segments with highest average confidence
            if not segments:
                print("    📊 No clear dialogue peaks found, analyzing by segments...")
                segment_size = min_duration
                num_segments = int(duration // segment_size)
                
                for i in range(num_segments):
                    start_time = i * segment_size
                    end_time = min(start_time + segment_size, duration)
                    
                    # Calculate average confidence for this segment
                    start_frame = int(start_time * sr / hop_length)
                    end_frame = int(end_time * sr / hop_length)
                    
                    if end_frame > start_frame:
                        avg_confidence = np.mean(dialogue_confidence[start_frame:end_frame])
                        segments.append((start_time, end_time, avg_confidence))
            
            # Sort by confidence (highest first)
            segments.sort(key=lambda x: x[2], reverse=True)
            
            print(f"    ✓ Found {len(segments)} potential dialogue segments")
            if segments:
                print(f"    📈 Best segment: {segments[0][0]:.1f}s-{segments[0][1]:.1f}s (confidence: {segments[0][2]:.2f})")
            
            return segments[:5]  # Return top 5 segments
            
        except Exception as e:
            print(f"    ⚠️ Dialogue detection failed: {e}")
            # Fallback to simple heuristic
            return [(max(0, duration/2 - 15), min(duration, duration/2 + 15), 0.3)]
    
    def extract_dialogue_clip_from_video(self, video_path, movie_id, movie_title, duration=30):
        """
        Extract 30-second audio clip with dialogue from existing video
        """
        try:
            output_path = self.dialogue_clips_dir / f"movie_{movie_id}_trailer_0.wav"
            
            # Skip if already exists
            if output_path.exists():
                print(f"    ✓ Dialogue clip already exists: {output_path}")
                return str(output_path)
            
            print(f"    🎬 Processing video: {Path(video_path).name}")
            
            # First, extract full audio for analysis
            temp_audio_path = self.dialogue_clips_dir / f"temp_full_audio_{movie_id}.wav"
            
            # Extract full audio from video
            extract_cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', '22050',
                '-ac', '1',  # Mono
                str(temp_audio_path),
                '-y', '-loglevel', 'quiet'
            ]
            
            result = subprocess.run(extract_cmd, capture_output=True)
            if result.returncode != 0:
                print(f"    ✗ Failed to extract audio from video: {video_path}")
                return None
            
            # Detect dialogue segments
            dialogue_segments = self.detect_dialogue_segments(str(temp_audio_path), duration)
            
            if not dialogue_segments:
                print(f"    ⚠️ No dialogue segments detected, using middle section")
                # Get video duration first
                probe_cmd = [
                    'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
                ]
                duration_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                try:
                    video_duration = float(duration_result.stdout.strip())
                    start_time = max(0, video_duration/2 - duration/2)
                except:
                    start_time = 30  # Default fallback
                
                dialogue_segments = [(start_time, start_time + duration, 0.3)]
            
            # Use the best dialogue segment
            best_segment = dialogue_segments[0]
            start_time, end_time, confidence = best_segment
            
            print(f"    🎯 Extracting dialogue: {start_time:.1f}s-{end_time:.1f}s (confidence: {confidence:.2f})")
            
            # Extract the specific dialogue segment
            clip_cmd = [
                'ffmpeg', '-i', str(video_path),
                '-ss', str(start_time),
                '-t', str(min(duration, end_time - start_time)),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', '22050',
                '-ac', '1',  # Mono
                '-af', 'highpass=f=80,lowpass=f=8000',  # Filter for speech frequency range
                str(output_path),
                '-y', '-loglevel', 'quiet'
            ]
            
            result = subprocess.run(clip_cmd, capture_output=True)
            
            # Clean up temporary file
            if temp_audio_path.exists():
                temp_audio_path.unlink()
            
            if result.returncode == 0:
                print(f"    ✅ Dialogue clip saved: {output_path}")
                return str(output_path)
            else:
                print(f"    ✗ Failed to extract dialogue clip")
                return None
                
        except Exception as e:
            print(f"    ✗ Error extracting dialogue from {movie_title}: {e}")
            
            # Clean up temporary files
            if 'temp_audio_path' in locals() and temp_audio_path.exists():
                try:
                    temp_audio_path.unlink()
                except:
                    pass
            
            return None
    
    def process_existing_videos(self, csv_file_path=None, max_movies=None):
        """
        Process existing videos to extract 30-second dialogue clips
        """
        print("🎬 Processing existing videos for dialogue extraction...")
        print("=" * 60)
        
        # Find all existing video files
        video_files = list(self.videos_dir.glob("movie_*_trailer_*.mp4"))
        
        if not video_files:
            print("❌ No video files found in trailer_videos directory!")
            return
        
        print(f"📁 Found {len(video_files)} video files")
        
        # Load dataset for movie titles if provided
        movie_info = {}
        if csv_file_path and Path(csv_file_path).exists():
            try:
                df = pd.read_csv(csv_file_path)
                movie_info = {row['id']: row['title'] for _, row in df.iterrows()}
                print(f"📊 Loaded movie information for {len(movie_info)} movies")
            except Exception as e:
                print(f"⚠️ Could not load movie dataset: {e}")
        
        # Process videos
        results = []
        successful_extractions = 0
        
        # Sort and optionally limit
        video_files.sort()
        if max_movies:
            video_files = video_files[:max_movies]
            print(f"🎯 Processing first {max_movies} videos")
        
        for i, video_file in enumerate(video_files):
            # Extract movie ID from filename
            try:
                filename = video_file.stem  # movie_123_trailer_0
                parts = filename.split('_')
                movie_id = int(parts[1])
                movie_title = movie_info.get(movie_id, f"Movie {movie_id}")
            except Exception as e:
                print(f"⚠️ Could not parse filename {video_file.name}: {e}")
                continue
            
            print(f"\n[{i+1}/{len(video_files)}] Processing: {movie_title}")
            print(f"    📹 Video: {video_file.name}")
            
            try:
                # Extract dialogue clip
                dialogue_path = self.extract_dialogue_clip_from_video(
                    str(video_file), movie_id, movie_title, duration=30
                )
                
                if dialogue_path:
                    successful_extractions += 1
                    status = 'success'
                else:
                    status = 'failed'
                
                results.append({
                    'movie_id': movie_id,
                    'movie_title': movie_title,
                    'video_path': str(video_file),
                    'dialogue_clip_path': dialogue_path,
                    'status': status,
                    'clip_duration': '30s',
                    'audio_features': 'dialogue_optimized'
                })
                
            except Exception as e:
                print(f"    ✗ Error processing {movie_title}: {e}")
                results.append({
                    'movie_id': movie_id,
                    'movie_title': movie_title,
                    'video_path': str(video_file),
                    'dialogue_clip_path': None,
                    'status': 'error',
                    'clip_duration': '30s',
                    'audio_features': 'dialogue_optimized'
                })
        
        # Save results
        self.save_dialogue_results(results)
        
        print(f"\n" + "=" * 60)
        print(f"🎯 DIALOGUE EXTRACTION SUMMARY:")
        print(f"Total videos processed: {len(video_files)}")
        print(f"Successful extractions: {successful_extractions}")
        print(f"Failed extractions: {len(video_files) - successful_extractions}")
        print(f"Success rate: {successful_extractions/len(video_files)*100:.1f}%")
        print(f"📁 Dialogue clips saved in: {self.dialogue_clips_dir}")
        
        return results
    
    def save_dialogue_results(self, results):
        """Save the dialogue extraction results"""
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_path = Path("datasets") / "dialogue_clip_results.csv"
        Path("datasets").mkdir(exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"\n✅ Results saved to: {results_path}")
        
        # Create manifest for successful extractions
        successful_clips = [r for r in results if r['status'] == 'success']
        
        manifest = {
            'extraction_info': {
                'clip_duration': '30 seconds',
                'audio_format': 'WAV',
                'sample_rate': '22050 Hz',
                'channels': 'mono',
                'features': 'dialogue_optimized',
                'processing': 'speech_frequency_filtered',
                'dialogue_detection': 'advanced' if self.librosa_available else 'basic'
            },
            'clips': []
        }
        
        for clip_info in successful_clips:
            manifest['clips'].append({
                'movie_id': clip_info['movie_id'],
                'movie_title': clip_info['movie_title'],
                'source_video': clip_info['video_path'],
                'dialogue_clip_path': clip_info['dialogue_clip_path'],
                'duration': clip_info['clip_duration'],
                'optimization': 'dialogue_focused'
            })
        
        manifest_path = Path("datasets") / "dialogue_clips_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Dialogue manifest saved to: {manifest_path}")
        print(f"🎵 {len(successful_clips)} dialogue clips ready for use")
        
        return results_df
    
    # Keep all existing methods for backward compatibility
    def load_existing_dataset(self, csv_file_path):
        """Load the existing movie dataset"""
        try:
            df = pd.read_csv(csv_file_path)
            print(f"✓ Loaded {len(df)} movies from {csv_file_path}")
            return df
        except FileNotFoundError:
            print(f"ERROR: File {csv_file_path} not found!")
            return None
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
            return None
    
    def get_movie_trailers(self, movie_id, movie_title):
        """Get trailer information for a specific movie"""
        trailers = []
        
        # 1. Get trailers from TMDb
        try:
            url = f"{self.tmdb_base_url}/movie/{movie_id}"
            params = {
                'api_key': self.tmdb_api_key,
                'append_to_response': 'videos',
                'language': 'en-US'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'videos' in data and 'results' in data['videos']:
                for video in data['videos']['results']:
                    if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                        trailers.append({
                            'name': video['name'],
                            'youtube_url': f"https://www.youtube.com/watch?v={video['key']}",
                            'source': 'tmdb'
                        })
            
            time.sleep(0.25)  # Rate limiting
            
        except Exception as e:
            print(f"  Warning: Could not get TMDb trailers for {movie_title}: {e}")
        
        # 2. If no TMDb trailers and YouTube API available, search YouTube
        if not trailers and self.youtube_api_key:
            try:
                youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
                
                search_response = youtube.search().list(
                    q=f"{movie_title} official trailer",
                    part='id,snippet',
                    maxResults=2,
                    type='video',
                    order='relevance'
                ).execute()
                
                for item in search_response['items']:
                    trailers.append({
                        'name': item['snippet']['title'],
                        'youtube_url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                        'source': 'youtube_search'
                    })
                
                time.sleep(0.5)  # Rate limiting for YouTube API
                
            except Exception as e:
                print(f"  Warning: Could not search YouTube for {movie_title}: {e}")
        
        return trailers
    
    def download_trailer_and_clip(self, youtube_url, movie_id, movie_title, trailer_name, trailer_index=0):
        """Download both the full video trailer and 5-second audio clip"""
        try:
            audio_filename = self.clips_dir / f"movie_{movie_id}_trailer_{trailer_index}.wav"
            video_filename = self.videos_dir / f"movie_{movie_id}_trailer_{trailer_index}.mp4"
            
            # Skip if both already exist
            if audio_filename.exists() and video_filename.exists():
                print(f"  ✓ Both audio and video already exist for {movie_title}")
                return str(audio_filename), str(video_filename)
            
            print(f"  Downloading from: {trailer_name}")
            
            # Base yt-dlp options
            base_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                }
            }
            
            # Add cookies if available
            if self.cookies_file and Path(self.cookies_file).exists():
                base_opts['cookiefile'] = self.cookies_file
            
            # Try downloading both audio and video
            audio_path = None
            video_path = None
            
            # Download full video if not exists
            if not video_filename.exists():
                video_opts = base_opts.copy()
                video_opts.update({
                    'format': 'best[height<=720]/best',  # Max 720p to save space
                    'outtmpl': str(video_filename),
                })
                
                try:
                    print(f"    Downloading video...")
                    with yt_dlp.YoutubeDL(video_opts) as ydl:
                        ydl.download([youtube_url])
                    
                    if video_filename.exists():
                        video_path = str(video_filename)
                        print(f"    ✓ Video saved: {video_filename}")
                    
                except Exception as e:
                    print(f"    ⚠️ Video download failed: {str(e)[:80]}...")
            else:
                video_path = str(video_filename)
                print(f"    ✓ Video already exists")
            
            # Download and process audio clip if not exists
            if not audio_filename.exists():
                audio_opts = base_opts.copy()
                audio_opts.update({
                    'format': 'bestaudio/best',
                    'outtmpl': str(self.clips_dir / f'temp_full_audio_{movie_id}_{trailer_index}.%(ext)s'),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                })
                
                try:
                    print(f"    Downloading full audio for clipping...")
                    with yt_dlp.YoutubeDL(audio_opts) as ydl:
                        ydl.download([youtube_url])
                    
                    # Find the downloaded full audio file
                    temp_files = list(self.clips_dir.glob(f'temp_full_audio_{movie_id}_{trailer_index}.*'))
                    if temp_files:
                        temp_audio_file = temp_files[0]
                        
                        # Extract 5-second clip using FFmpeg
                        cmd = [
                            'ffmpeg', '-i', str(temp_audio_file),
                            '-ss', '10',  # Start at 10 seconds
                            '-t', '5',    # Duration: 5 seconds
                            '-acodec', 'pcm_s16le',  # Standard WAV format
                            '-ar', '22050',  # Sample rate
                            '-ac', '1',      # Mono
                            str(audio_filename),
                            '-y',  # Overwrite if exists
                            '-loglevel', 'quiet'  # Suppress FFmpeg output
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True)
                        if result.returncode == 0:
                            audio_path = str(audio_filename)
                            print(f"    ✓ Audio clip saved: {audio_filename}")
                        else:
                            print(f"    ✗ FFmpeg failed for {movie_title}")
                        
                        # Clean up temporary full audio file
                        temp_audio_file.unlink()
                    
                except Exception as e:
                    print(f"    ⚠️ Audio download failed: {str(e)[:80]}...")
                    
                    # Clean up any temporary files
                    for temp_file in self.clips_dir.glob(f'temp_full_audio_{movie_id}_{trailer_index}.*'):
                        try:
                            temp_file.unlink()
                        except:
                            pass
            else:
                audio_path = str(audio_filename)
                print(f"    ✓ Audio clip already exists")
            
            return audio_path, video_path
            
        except Exception as e:
            print(f"  ✗ Error downloading for {movie_title}: {e}")
            
            # Clean up any temporary files
            for temp_file in self.clips_dir.glob(f'temp_full_audio_{movie_id}_{trailer_index}.*'):
                try:
                    temp_file.unlink()
                except:
                    pass
            
            return None, None
    
    def generate_clips_for_dataset(self, csv_file_path, max_movies=None, download_videos=True):
        """Generate audio clips and optionally full videos for all movies in the dataset"""
        # Load existing dataset
        df = self.load_existing_dataset(csv_file_path)
        if df is None:
            return
        
        # Limit number of movies if specified
        if max_movies:
            df = df.head(max_movies)
            print(f"Processing first {max_movies} movies")
        
        print(f"\nStarting {'audio + video' if download_videos else 'audio only'} generation for {len(df)} movies...")
        print("=" * 50)
        
        results = []
        successful_downloads = 0
        
        for i, row in df.iterrows():
            movie_id = row['id']
            movie_title = row['title']
            
            print(f"\n[{i+1}/{len(df)}] Processing: {movie_title}")
            
            try:
                # Get trailers for this movie
                trailers = self.get_movie_trailers(movie_id, movie_title)
                
                if not trailers:
                    print(f"  ✗ No trailers found for {movie_title}")
                    results.append({
                        'movie_id': movie_id,
                        'movie_title': movie_title,
                        'status': 'no_trailers',
                        'audio_clip_path': None,
                        'video_path': None
                    })
                    continue
                
                # Download from first available trailer
                audio_path = None
                video_path = None
                
                for idx, trailer in enumerate(trailers[:1]):  # Just first trailer
                    if download_videos:
                        audio_path, video_path = self.download_trailer_and_clip(
                            trailer['youtube_url'], 
                            movie_id, 
                            movie_title, 
                            trailer['name'],
                            idx
                        )
                    else:
                        # Audio only version (keeping backward compatibility)
                        audio_path = self.download_5s_audio_clip(
                            trailer['youtube_url'], 
                            movie_id, 
                            movie_title, 
                            trailer['name'],
                            idx
                        )
                    
                    if audio_path:
                        successful_downloads += 1
                        break
                    time.sleep(2)  # Rate limiting between attempts
                
                results.append({
                    'movie_id': movie_id,
                    'movie_title': movie_title,
                    'status': 'success' if audio_path else 'failed',
                    'audio_clip_path': audio_path,
                    'video_path': video_path,
                    'trailer_count': len(trailers)
                })
                
                # Rate limiting between movies
                time.sleep(1)
                
            except Exception as e:
                print(f"  ✗ Error processing {movie_title}: {e}")
                results.append({
                    'movie_id': movie_id,
                    'movie_title': movie_title,
                    'status': 'error',
                    'audio_clip_path': None,
                    'video_path': None
                })
                continue
        
        # Save results
        self.save_clip_results(results, download_videos)
        
        print(f"\n" + "=" * 50)
        print(f"SUMMARY:")
        print(f"Total movies processed: {len(df)}")
        print(f"Successful downloads: {successful_downloads}")
        print(f"Failed downloads: {len(df) - successful_downloads}")
        print(f"Success rate: {successful_downloads/len(df)*100:.1f}%")
        
        return results
    
    def download_5s_audio_clip(self, youtube_url, movie_id, movie_title, trailer_name, trailer_index=0):
        """Download only a 5-second audio clip (backward compatibility)"""
        audio_path, _ = self.download_trailer_and_clip(youtube_url, movie_id, movie_title, trailer_name, trailer_index)
        return audio_path
    
    def save_clip_results(self, results, includes_videos=True):
        """Save the clip generation results"""
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_path = Path("datasets") / "audio_clip_results.csv"
        Path("datasets").mkdir(exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Results saved to: {results_path}")
        
        # Create manifest for successful downloads
        successful_clips = [r for r in results if r['status'] == 'success']
        
        manifest = []
        for clip_info in successful_clips:
            entry = {
                'movie_id': clip_info['movie_id'],
                'movie_title': clip_info['movie_title'],
                'audio_clip_path': clip_info['audio_clip_path'],
                'audio_duration': '5s',
                'audio_format': 'WAV',
                'audio_sample_rate': '22050Hz',
                'audio_channels': 'mono'
            }
            
            if includes_videos and clip_info.get('video_path'):
                entry['video_path'] = clip_info['video_path']
                entry['video_format'] = 'MP4'
                entry['video_max_quality'] = '720p'
            
            manifest.append(entry)
        
        manifest_path = Path("datasets") / "media_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Media manifest saved to: {manifest_path}")
        
        return results_df


# Enhanced usage example
if __name__ == "__main__":
    import shutil
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Movie Trailer Audio Processor')
    parser.add_argument('--mode', choices=['download', 'extract', 'both'], default='extract',
                      help='Mode: download new videos, extract from existing, or both')
    parser.add_argument('--max-movies', type=int, help='Maximum number of movies to process')
    parser.add_argument('--csv-file', default='movie_academic_dataset.csv', 
                      help='Path to movie dataset CSV file')
    parser.add_argument('--cookies-file', help='Path to YouTube cookies file')
    
    args = parser.parse_args()
    
    print("=== Enhanced Movie Trailer Audio Processor ===\n")
    
    # Check requirements
    print("Requirements:")
    print("- yt-dlp: pip install yt-dlp")
    print("- librosa: pip install librosa (for advanced dialogue detection)")
    print("- scipy: pip install scipy (for signal processing)")
    print("- browser-cookie3: pip install browser-cookie3 (for auto cookie extraction)")
    print("- FFmpeg: https://ffmpeg.org/download.html")
    print()
    
    # API keys
    TMDB_API_KEY = "3675b6ea75dc426e1c958eec9932fa91"
    YOUTUBE_API_KEY = "AIzaSyC9dgKqYXIs28yFKoCoLbQGy7-lgXb0k2M"  # Optional
    
    if not TMDB_API_KEY:
        print("TMDb API key is required!")
        exit(1)
    
    # Cookie file setup
    cookies_file = args.cookies_file
    if not cookies_file:
        # Default locations
        default_cookies = [
            "/home/xwang378/scratch/2025/AudioBench/benchmark/Data/solos/cookies.txt",
            "/scratch/xwang378/2025/AudioBench/benchmark/Data/solos/cookies.txt",
            "cookies.txt",
            "youtube_cookies.txt"
        ]
        
        for cookie_path in default_cookies:
            if Path(cookie_path).exists():
                cookies_file = cookie_path
                break
    
    # Copy cookies file if needed
    if cookies_file and Path(cookies_file).exists():
        cookies_copy = "/scratch/xwang378/2025/AudioBench/benchmark/Data/solos/cookies_copy.txt"
        try:
            shutil.copy(cookies_file, cookies_copy)
            cookies_file = cookies_copy
            print(f"✓ Using cookies file: {cookies_file}")
        except Exception as e:
            print(f"⚠️ Could not copy cookies file: {e}")
    
    # Initialize generator
    generator = AudioClipGenerator(
        tmdb_api_key=TMDB_API_KEY,
        youtube_api_key=YOUTUBE_API_KEY if YOUTUBE_API_KEY else None,
        cookies_file=cookies_file 
    )
    
    print("\n" + "="*60)
    print("🎯 PROCESSING OPTIONS:")
    print("1. 'download' - Download new videos and create 5s audio clips")
    print("2. 'extract' - Extract 30s dialogue clips from existing videos")  
    print("3. 'both' - Do both operations")
    print(f"📊 Current mode: {args.mode}")
    print("="*60)
    
    # Execute based on mode
    if args.mode in ['download', 'both']:
        print("\n🔄 PHASE 1: Downloading videos and creating 5s clips...")
        download_results = generator.generate_clips_for_dataset(
            args.csv_file, 
            max_movies=args.max_movies,
            download_videos=True
        )
        
        print(f"\n🎵 5-second clips saved in: trailer_clips/")
        print(f"🎬 Videos saved in: trailer_videos/")
    
    if args.mode in ['extract', 'both']:
        print("\n🔄 PHASE 2: Extracting 30s dialogue clips from existing videos...")
        dialogue_results = generator.process_existing_videos(
            csv_file_path=args.csv_file,
            max_movies=args.max_movies
        )
        
        print(f"\n🎤 30-second dialogue clips saved in: dialogue_clips/")
    
    print("\n" + "="*60)
    print("📁 FINAL OUTPUT DIRECTORIES:")
    print(f"• trailer_clips/     - 5-second audio clips")
    print(f"• trailer_videos/    - Full video trailers (MP4)")  
    print(f"• dialogue_clips/    - 30-second dialogue clips (optimized)")
    print(f"• datasets/          - Results and manifests")
    print("="*60)
    
    print("\n💡 USAGE TIPS:")
    print("• For quick dialogue extraction from existing videos: --mode extract")
    print("• For downloading new content: --mode download")
    print("• For complete processing pipeline: --mode both")
    print("• Limit processing: --max-movies 100")
    print("• Custom dataset: --csv-file your_movies.csv")
    
    # Clean up temporary cookies
    if cookies_file and "cookies_copy.txt" in cookies_file:
        try:
            Path(cookies_file).unlink()
        except:
            pass
    
    if Path("temp_youtube_cookies.txt").exists():
        try:
            Path("temp_youtube_cookies.txt").unlink()
        except:
            pass