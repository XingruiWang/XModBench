import pandas as pd
import requests
import json
import time
from pathlib import Path
from googleapiclient.discovery import build

class MovieDatasetBuilder:
    def __init__(self, tmdb_api_key, youtube_api_key=None):
        self.tmdb_api_key = tmdb_api_key
        self.youtube_api_key = youtube_api_key
        self.tmdb_base_url = "https://api.themoviedb.org/3"
        self.dataset = []
        
    def get_popular_movies(self, pages=25):
        """Get list of popular movies"""
        movies = []
        for page in range(1, pages + 1):
            try:
                url = f"{self.tmdb_base_url}/movie/popular"
                params = {
                    'api_key': self.tmdb_api_key,
                    'page': page,
                    'language': 'en-US'  # Changed to English
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                movies.extend(data['results'])
                print(f"Retrieved page {page}, {len(data['results'])} movies")
                time.sleep(0.25)  # Rate limiting
                
            except requests.exceptions.RequestException as e:
                print(f"Error retrieving page {page}: {e}")
                continue
                
        return movies
    
    def get_movie_details(self, movie_id):
        """Get detailed movie information"""
        try:
            url = f"{self.tmdb_base_url}/movie/{movie_id}"
            params = {
                'api_key': self.tmdb_api_key,
                'append_to_response': 'videos,images,credits',
                'language': 'en-US'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error getting details for movie {movie_id}: {e}")
            return {}
    
    def get_youtube_trailers(self, movie_title):
        """Search for trailers on YouTube"""
        if not self.youtube_api_key:
            return []
            
        try:
            youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
            
            search_response = youtube.search().list(
                q=f"{movie_title} official trailer",
                part='id,snippet',
                maxResults=3,
                type='video',
                order='relevance'
            ).execute()
            
            trailers = []
            for item in search_response['items']:
                trailer_info = {
                    'title': item['snippet']['title'],
                    'video_id': item['id']['videoId'],
                    'description': item['snippet']['description'][:200],  # Limit description length
                    'thumbnail': item['snippet']['thumbnails'].get('medium', {}).get('url'),
                    'youtube_url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                }
                trailers.append(trailer_info)
                
            return trailers
            
        except Exception as e:
            print(f"Error searching trailers for {movie_title}: {e}")
            return []
    
    def download_poster(self, poster_path, movie_id):
        """Download movie poster"""
        if not poster_path:
            return None
            
        try:
            url = f"https://image.tmdb.org/t/p/w500{poster_path}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Create poster storage directory
            poster_dir = Path("posters")
            poster_dir.mkdir(exist_ok=True)
            
            file_path = poster_dir / f"{movie_id}.jpg"
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            return str(file_path)
            
        except Exception as e:
            print(f"Error downloading poster for {movie_id}: {e}")
            return None
    
    def collect_movie_data(self, num_movies=500):
        """Collect movie data"""
        print("Starting movie data collection...")
        
        # 1. Get list of popular movies
        pages_needed = (num_movies // 20) + 1  # TMDb returns 20 movies per page
        movies = self.get_popular_movies(pages=pages_needed)
        
        print(f"Retrieved {len(movies)} movies total, processing first {num_movies}...")
        
        for i, movie in enumerate(movies[:num_movies]):
            try:
                print(f"Processing ({i+1}/{num_movies}): {movie['title']}")
                
                # Get detailed information
                details = self.get_movie_details(movie['id'])
                
                # Get trailer links (if YouTube API key is available)
                trailers = self.get_youtube_trailers(movie['title']) if self.youtube_api_key else []
                
                # Download poster image
                poster_path = self.download_poster(movie.get('poster_path'), movie['id'])
                
                # Process genre information
                genres = []
                if 'genres' in details:
                    genres = [genre['name'] for genre in details['genres']]
                
                # Process cast information
                cast = []
                if 'credits' in details and 'cast' in details['credits']:
                    cast = [actor['name'] for actor in details['credits']['cast'][:5]]  # Get top 5 main actors
                
                movie_data = {
                    'id': movie['id'],
                    'title': movie['title'],
                    'original_title': movie.get('original_title'),
                    'overview': movie.get('overview', ''),
                    'release_date': movie.get('release_date'),
                    'genres': genres,
                    'popularity': movie.get('popularity'),
                    'vote_average': movie.get('vote_average'),
                    'vote_count': movie.get('vote_count'),
                    'poster_path': poster_path,
                    'backdrop_path': movie.get('backdrop_path'),
                    'runtime': details.get('runtime'),
                    'budget': details.get('budget'),
                    'revenue': details.get('revenue'),
                    'cast': cast,
                    'trailers': trailers,
                    'tmdb_url': f"https://www.themoviedb.org/movie/{movie['id']}"
                }
                
                self.dataset.append(movie_data)
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error processing {movie.get('title', 'Unknown')}: {e}")
                continue
        
        print(f"Data collection complete! Collected {len(self.dataset)} movies")
    
    def save_dataset(self, filename="movie_dataset"):
        """Save dataset"""
        if not self.dataset:
            print("No data to save")
            return
        
        # Save as JSON format
        json_filename = f"{filename}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=2)
        print(f"JSON data saved to: {json_filename}")
        
        # Save as CSV format
        try:
            df = pd.json_normalize(self.dataset)
            csv_filename = f"{filename}.csv"
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"CSV data saved to: {csv_filename}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
    
    def get_dataset_stats(self):
        """Get dataset statistics"""
        if not self.dataset:
            return "Dataset is empty"
        
        stats = {
            "Total movies": len(self.dataset),
            "Movies with posters": len([m for m in self.dataset if m['poster_path']]),
            "Movies with trailers": len([m for m in self.dataset if m['trailers']]),
            "Average rating": sum(m['vote_average'] for m in self.dataset if m['vote_average']) / len(self.dataset),
            "Earliest year": min(m['release_date'][:4] for m in self.dataset if m['release_date']),
            "Latest year": max(m['release_date'][:4] for m in self.dataset if m['release_date'])
        }
        
        return stats

# 使用示例
if __name__ == "__main__":
    # 设置API密钥
    TMDB_API_KEY = "3675b6ea75dc426e1c958eec9932fa91"  # 在 https://www.themoviedb.org/settings/api 申请
    YOUTUBE_API_KEY = "AIzaSyC9dgKqYXIs28yFKoCoLbQGy7-lgXb0k2M"  # 在 Google Cloud Console 申请（可选）

    # 创建数据集构建器
    builder = MovieDatasetBuilder(
        tmdb_api_key=TMDB_API_KEY,
        youtube_api_key=YOUTUBE_API_KEY  # 如果没有YouTube API可以设为None
    )
    
    # 收集数据
    builder.collect_movie_data(num_movies=100)  # 先试试100部电影
    
    # 保存数据
    builder.save_dataset("movie_academic_dataset")
    
    # 查看统计信息
    stats = builder.get_dataset_stats()
    print("\n数据集统计:")
    for key, value in stats.items():
        print(f"{key}: {value}")