import gradio as gr
import json
import os
import random
from pathlib import Path
import base64
from typing import Dict, List, Tuple, Any
import datetime
import uuid
import threading
import time

class MultiUserManager:
    """Manage multiple users and real-time updates"""
    def __init__(self):
        self.active_users = {}  # user_id -> user_info
        self.current_questions = []
        self.current_answers = []
        self.user_responses = {}  # user_id -> responses
        self.leaderboard = []
        self.question_set_id = None
        self.lock = threading.Lock()
        
    def register_user(self, username: str = None):
        """Register a new user"""
        user_id = str(uuid.uuid4())[:8]
        if not username:
            username = f"User_{user_id}"
        
        with self.lock:
            self.active_users[user_id] = {
                "id": user_id,
                "username": username,
                "joined_at": datetime.datetime.now(),
                "answers": [],
                "score": 0,
                "submitted": False
            }
            
            # Initialize responses for current questions
            if self.current_questions:
                self.user_responses[user_id] = [""] * len(self.current_questions)
        
        return user_id, username
    
    def get_user_stats(self):
        """Get current user statistics"""
        with self.lock:
            total_users = len(self.active_users)
            submitted_users = sum(1 for user in self.active_users.values() if user["submitted"])
            
            # Update leaderboard
            self.leaderboard = sorted(
                [user for user in self.active_users.values() if user["submitted"]], 
                key=lambda x: x["score"], 
                reverse=True
            )
            
            return {
                "total_users": total_users,
                "submitted_users": submitted_users,
                "active_users": [user["username"] for user in self.active_users.values()],
                "leaderboard": self.leaderboard[:10]  # Top 10
            }
    
    def submit_user_answer(self, user_id: str, question_idx: int, answer: str):
        """Submit answer for a specific user"""
        with self.lock:
            if user_id in self.user_responses:
                if question_idx < len(self.user_responses[user_id]):
                    self.user_responses[user_id][question_idx] = answer
                    return True
        return False
    
    def submit_user_final(self, user_id: str):
        """Submit final answers and calculate score"""
        with self.lock:
            if user_id in self.active_users and user_id in self.user_responses:
                user_answers = self.user_responses[user_id]
                score = sum(1 for ua, ca in zip(user_answers, self.current_answers) if ua == ca)
                
                self.active_users[user_id]["answers"] = user_answers
                self.active_users[user_id]["score"] = score
                self.active_users[user_id]["submitted"] = True
                self.active_users[user_id]["submitted_at"] = datetime.datetime.now()
                
                return score, len(self.current_answers)
        return 0, 0
    
    def set_questions(self, questions, answers):
        """Set new questions for users"""
        with self.lock:
            self.current_questions = questions
            self.current_answers = answers
            self.question_set_id = str(uuid.uuid4())[:8]
            
            # Reset user responses
            self.user_responses = {}
            for user_id in self.active_users:
                self.user_responses[user_id] = [""] * len(questions)
                self.active_users[user_id]["submitted"] = False
                self.active_users[user_id]["score"] = 0
                self.active_users[user_id]["answers"] = []

# Global multi-user manager
multi_user_manager = MultiUserManager()

class AudioBenchDemo:
    def __init__(self, base_path="/home/xwang378/scratch/2025/AudioBench/benchmark/tasks", max_display_questions=15):
        self.base_path = Path(base_path)
        self.current_questions = []
        self.current_answers = []
        self.user_answers = []
        self.cached_questions = {}  # Cache loaded questions
        self.session_id = str(uuid.uuid4())[:8]  # Generate unique session ID
        self.results_dir = Path("user_results")
        self.results_dir.mkdir(exist_ok=True)  # Create results directory
        
        # Add user attributes for multi-user mode
        self.user_id = None
        self.username = None
        
        # Add question type tracking and display control
        self.questions_per_type = 5  # Default N questions per type
        self.max_display_questions = max_display_questions  # Maximum questions to display in UI
        
        print(f"🆔 Session ID: {self.session_id}")
        print(f"📺 Max display questions: {self.max_display_questions}")
        
    def save_user_results(self, task: str, subtask: str):
        """Save user results to local file"""
        if not self.current_questions or not self.user_answers:
            return
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.username}_session_{self.session_id}_{task}_{subtask}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Prepare results data
        results = {
            "session_id": self.session_id,
            "timestamp": timestamp,
            "task": task,
            "subtask": subtask,
            "total_questions": len(self.current_questions),
            "user_answers": self.user_answers,
            "correct_answers": self.current_answers,
            "questions_per_type": self.questions_per_type,
            "questions": []
        }
        
        # Add question details
        for i, (question, user_answer, correct_answer) in enumerate(zip(
            self.current_questions, self.user_answers, self.current_answers
        )):
            question_data = {
                "question_number": i + 1,
                "question_text": question.get('question', ''),
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "is_correct": user_answer == correct_answer,
                "source_file": question.get('source_file', ''),
                "question_id": question.get('question_id', ''),
                "question_type": question.get('question_type', 'unknown')
            }
            results["questions"].append(question_data)
        
        # Calculate score
        correct_count = sum(1 for ua, ca in zip(self.user_answers, self.current_answers) if ua == ca)
        results["score"] = {
            "correct": correct_count,
            "total": len(self.current_answers),
            "percentage": (correct_count / len(self.current_answers)) * 100 if self.current_answers else 0
        }
        
        # Save to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return None
    
    def manual_save_results(self, task: str, subtask: str):
        """Manual save user results"""
        if not self.current_questions or not self.user_answers:
            return "No data to save. Please load questions and provide answers first."
        
        # Check if user has answered any questions
        answered_count = sum(1 for answer in self.user_answers if answer.strip())
        if answered_count == 0:
            return "No answers provided yet. Please answer at least one question before saving."
        
        filepath = self.save_user_results(task, subtask)
        if filepath:
            correct_count = sum(1 for ua, ca in zip(self.user_answers, self.current_answers) if ua == ca)
            total_count = len(self.current_answers)
            percentage = (correct_count / total_count) * 100 if total_count > 0 else 0
            
            return f"✅ **Results saved successfully!**\n\n📊 **Score:** {correct_count}/{total_count} ({percentage:.1f}%)\n📁 **File:** {filepath.name}\n🆔 **Session:** {self.session_id}"
        else:
            return "❌ Error saving results. Please try again."
    
    def record_answer(self, question_num: int, selected_answer: str):
        """Record answer for current user"""
        if self.user_id:
            success = multi_user_manager.submit_user_answer(self.user_id, question_num, selected_answer)
            if success:
                return f"Selected: {selected_answer}"
        
        # Fallback to local storage
        if question_num < len(self.user_answers):
            self.user_answers[question_num] = selected_answer
            return f"Selected: {selected_answer}"
        return "Error recording answer"
    
    def get_user_stats_display(self):
        """Get formatted user statistics"""
        stats = multi_user_manager.get_user_stats()
        
        stats_text = f"""## 🌐 Multi-User Statistics
        
👥 **Active Users:** {stats['total_users']}
✅ **Submitted:** {stats['submitted_users']}

🏆 **Top Performers:**
"""
        
        for i, user in enumerate(stats['leaderboard'][:5], 1):
            percentage = (user['score'] / len(self.current_answers)) * 100 if self.current_answers else 0
            stats_text += f"{i}. **{user['username']}** - {user['score']}/{len(self.current_answers)} ({percentage:.1f}%)\n"
        
        if stats['active_users']:
            stats_text += f"\n👤 **Online:** {', '.join(stats['active_users'][:10])}"
            if len(stats['active_users']) > 10:
                stats_text += f" +{len(stats['active_users']) - 10} more"
        
        return stats_text
        
    def get_available_tasks(self) -> List[str]:
        """Get list of available main tasks"""
        tasks = []
        for item in self.base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.') and 'archived' not in item.name:
                tasks.append(item.name)
        return sorted(tasks)
    
    def get_available_subtasks(self, task: str) -> List[str]:
        """Get list of available subtasks for a given task"""
        if not task:
            return []
        
        print(f"🔍 Scanning subtasks for task: {task}")
        task_path = self.base_path / task
        subtasks = []
        
        try:
            for item in task_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    subtasks.append(item.name)
            print(f"✅ Found {len(subtasks)} subtasks: {subtasks}")
        except Exception as e:
            print(f"❌ Error scanning subtasks for {task}: {e}")
        
        return sorted(subtasks)
    
    def get_json_files(self, task: str, subtask: str) -> List[Path]:
        """Get all JSON files in the subtask directory"""
        if not task or not subtask:
            return []
        
        subtask_path = self.base_path / task / subtask
        json_files = []
        
        if subtask_path.exists():
            json_files = list(subtask_path.glob("*.json"))
        
        return json_files
    
    def load_questions_from_files(self, json_files: List[Path]) -> Dict[str, List[Dict]]:
        """Load questions from JSON files grouped by file (question type)"""
        cache_key = str(sorted(json_files))
        
        # Check cache first
        if cache_key in self.cached_questions:
            return self.cached_questions[cache_key]
        
        questions_by_type = {}
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    questions = json.load(f)
                    
                    # Use filename as question type
                    question_type = json_file.stem
                    
                    # Add metadata to each question
                    for i, q in enumerate(questions):
                        q['source_file'] = str(json_file)
                        q['question_id'] = f"{question_type}_{i}"
                        q['question_type'] = question_type
                    
                    questions_by_type[question_type] = questions
                    print(f"📁 Loaded {len(questions)} questions from {question_type}")
                    
            except Exception as e:
                print(f"❌ Error loading {json_file}: {e}")
                continue
        
        # Cache the results
        self.cached_questions[cache_key] = questions_by_type
        return questions_by_type
    
    def format_media_content(self, modality: str, input_path: str) -> Tuple[str, Any]:
        """Format media content based on modality - simplified without file checking"""
        print(f"🎬 Processing {modality} content: {input_path}")
        
        if modality.lower() == "text":
            return "text", input_path
        elif modality.lower() == "audio":
            if os.path.exists(input_path):
                print(f"✅ Audio file exists: {input_path}")
                return "audio", input_path
            else:
                print(f"❌ Audio file not found: {input_path}")
                return "text", f"[Audio file not found: {input_path}]"
        elif modality.lower() in ["visual", "image"]:
            if os.path.exists(input_path):
                print(f"✅ Image file exists: {input_path}")
                return "image", input_path
            else:
                print(f"❌ Image file not found: {input_path}")
                return "text", f"[Image file not found: {input_path}]"
        elif modality.lower() == "video":
            if os.path.exists(input_path):
                print(f"✅ Video file exists: {input_path}")
                print(f"📁 Video file size: {os.path.getsize(input_path)} bytes")
                print(f"🎞️ Video extension: {os.path.splitext(input_path)[1]}")
                return "video", input_path
            else:
                print(f"❌ Video file not found: {input_path}")
                return "text", f"[Video file not found: {input_path}]"
        else:
            return "text", input_path
    
    def generate_n_questions_per_type(self, task: str, subtask: str, questions_per_type: int = 5, max_display: int = None):
        """Generate N questions from each question type (JSON file)"""
        print(f"🎲 Generating {questions_per_type} questions per type from {task}/{subtask}")
        
        # Use instance max_display_questions if max_display not provided
        if max_display is None:
            max_display = self.max_display_questions
        
        if not task or not subtask:
            return "Please select both task and subtask first.", [], []
        
        # Store the questions_per_type setting
        self.questions_per_type = questions_per_type
        
        # Get JSON files
        print(f"🔍 Looking for JSON files in {task}/{subtask}")
        json_files = self.get_json_files(task, subtask)
        if not json_files:
            print(f"❌ No JSON files found in {task}/{subtask}")
            return f"No JSON files found in {task}/{subtask}", [], []
        
        print(f"📁 Found {len(json_files)} JSON files (question types)")
        
        # Load questions grouped by type
        questions_by_type = self.load_questions_from_files(json_files)
        if not questions_by_type:
            print("❌ No questions found in the selected files")
            return "No questions found in the selected files.", [], []
        
        # Select N questions from each type
        selected_questions = []
        type_summary = []
        
        for question_type, questions in questions_by_type.items():
            available_count = len(questions)
            select_count = min(questions_per_type, available_count)
            
            # Randomly sample questions from this type
            if select_count > 0:
                sampled_questions = random.sample(questions, select_count)
                selected_questions.extend(sampled_questions)
                type_summary.append(f"{question_type}: {select_count}/{available_count}")
                # print(f"📋 Selected {select_count} questions from {question_type} (available: {available_count})")
        
        print(f"🎯 Total selected questions: {len(selected_questions)}")
        
        # Shuffle all questions together
        random.shuffle(selected_questions)
        
        # Don't limit display here - show all selected questions up to max_display
        display_questions = selected_questions[:min(len(selected_questions), max_display)]
        
        # Store questions locally and globally
        self.current_questions = selected_questions
        self.current_answers = [q.get('correct_answer', 'Unknown') for q in selected_questions]
        self.user_answers = [""] * len(selected_questions)  # Reset user answers
        
        # Set questions for all users in multi-user mode
        multi_user_manager.set_questions(selected_questions, self.current_answers)
        
        # Format questions for display (all selected questions up to max_display)
        print("🎨 Formatting questions for display...")
        question_displays = []
        for i, q in enumerate(display_questions):
            question_display = self.format_question_display(q, i)
            question_displays.append(question_display)
        
        # Create summary with type breakdown
        summary = f"🌐 Questions loaded! {len(selected_questions)} total questions ({questions_per_type} per type)\n"
        summary += f"📊 Types: {', '.join(type_summary)}\n"
        summary += f"🎮 Displaying {len(display_questions)} questions in UI"
        
        print(f"✅ {summary}")
        return summary, question_displays, self.current_answers
    
    def generate_random_questions(self, task: str, subtask: str, num_questions: int = 5):
        """Generate random questions from selected task and subtask (legacy method)"""
        print(f"🎲 Generating {num_questions} random questions from {task}/{subtask}")
        
        if not task or not subtask:
            return "Please select both task and subtask first.", [], []
        
        # Get JSON files
        print(f"🔍 Looking for JSON files in {task}/{subtask}")
        json_files = self.get_json_files(task, subtask)
        if not json_files:
            print(f"❌ No JSON files found in {task}/{subtask}")
            return f"No JSON files found in {task}/{subtask}", [], []
        
        print(f"📁 Found {len(json_files)} JSON files")
        
        # Load all questions (flatten from all types)
        questions_by_type = self.load_questions_from_files(json_files)
        all_questions = []
        for questions in questions_by_type.values():
            all_questions.extend(questions)
            
        if not all_questions:
            print("❌ No questions found in the selected files")
            return "No questions found in the selected files.", [], []
        
        print(f"📚 Total available questions: {len(all_questions)}")
        
        # Randomly select questions
        selected_questions = random.sample(
            all_questions, 
            min(num_questions, len(all_questions))
        )
        
        print(f"🎯 Selected {len(selected_questions)} questions")
        
        # Store questions locally and globally
        self.current_questions = selected_questions
        self.current_answers = [q.get('correct_answer', 'Unknown') for q in selected_questions]
        self.user_answers = [""] * len(selected_questions)  # Reset user answers
        
        # Set questions for all users in multi-user mode
        multi_user_manager.set_questions(selected_questions, self.current_answers)
        
        # Format questions for display
        print("🎨 Formatting questions for display...")
        question_displays = []
        for i, q in enumerate(selected_questions):
            question_display = self.format_question_display(q, i)
            question_displays.append(question_display)
        
        summary = f"🌐 Questions loaded for users! {len(selected_questions)} questions from {task}/{subtask}"
        print(f"✅ {summary}")
        return summary, question_displays, self.current_answers
    
    def format_question_display(self, question: Dict, question_num: int) -> Dict:
        """Format a single question for display in Gradio"""
        display = {
            "question_text": f"**Question {question_num + 1}:** {question['question']}",
            "condition_type": None,
            "condition_content": None,
            "options": {}
        }
        
        # Format condition/input
        if 'conditions' in question:
            condition = question['conditions']
            modality = condition.get('modality', 'Text')
            input_data = condition.get('input', '')
            
            content_type, content = self.format_media_content(modality, input_data)
            display["condition_type"] = content_type
            display["condition_content"] = content
        
        # Format options
        if 'options' in question:
            for option_key, option_data in question['options'].items():
                modality = option_data.get('modality', 'Text')
                input_data = option_data.get('input', '')
                
                content_type, content = self.format_media_content(modality, input_data)
                display["options"][option_key] = {
                    "type": content_type,
                    "content": content
                }
        
        return display

# Initialize the demo with configurable max display questions
MAX_DISPLAY_QUESTIONS = 50  # Hyperparameter: Set this to control max displayable questions

print("🚀 Initializing AudioBench Demo...")
demo_instance = AudioBenchDemo(max_display_questions=MAX_DISPLAY_QUESTIONS)
print("✅ Demo instance created")

def update_subtasks(task):
    """Update subtask dropdown based on selected task"""
    subtasks = demo_instance.get_available_subtasks(task)
    return gr.update(choices=subtasks, value=None)

def load_questions(task, subtask, questions_per_type, use_per_type_mode):
    """Load and display questions - either N per type or random selection"""
    print(f"🎮 User requested: {questions_per_type} {'questions per type' if use_per_type_mode else 'random questions'} from {task}/{subtask}")
    
    if use_per_type_mode:
        # Use N questions per type mode - use instance max_display_questions as absolute maximum
        summary, questions, answers = demo_instance.generate_n_questions_per_type(
            task, subtask, questions_per_type, max_display=demo_instance.max_display_questions
        )
    else:
        # Use legacy random mode
        summary, questions, answers = demo_instance.generate_random_questions(
            task, subtask, min(questions_per_type, demo_instance.max_display_questions)
        )
    
    print("🎨 Building UI outputs...")
    print(f"📊 Total questions to display: {len(questions)}")
    question_outputs = []
    
    # For each of the question slots (controlled by max_display_questions hyperparameter)
    # But only iterate through the actual questions we want to display
    for i in range(demo_instance.max_display_questions):
        if i < len(questions):
            print(f"📝 Processing question {i+1}/{len(questions)}")
            q = questions[i]
            
            # Question text with proper formatting
            question_text = f"**Question {i+1}:** {q['question_text'].split(':', 1)[1].strip() if ':' in q['question_text'] else q['question_text']}"
            
            # Components for this question (23 total)
            outputs = [question_text]  # 0: question_display
            
            # Input media components (1-4)
            if q["condition_type"] == "text":
                outputs.extend([
                    gr.update(visible=False),                               # 1: input_audio
                    gr.update(visible=False),                               # 2: input_image
                    gr.update(visible=False),                               # 3: input_video
                    gr.update(visible=True, value=f"**Input:** {q['condition_content']}")  # 4: input_text
                ])
            elif q["condition_type"] == "audio":
                outputs.extend([
                    gr.update(visible=True, value=q["condition_content"]),  # 1: input_audio
                    gr.update(visible=False),                               # 2: input_image
                    gr.update(visible=False),                               # 3: input_video
                    gr.update(visible=False)                                # 4: input_text
                ])
            elif q["condition_type"] == "image":
                outputs.extend([
                    gr.update(visible=False),                               # 1: input_audio
                    gr.update(visible=True, value=q["condition_content"]),  # 2: input_image
                    gr.update(visible=False),                               # 3: input_video
                    gr.update(visible=False)                                # 4: input_text
                ])
            elif q["condition_type"] == "video":
                print(f"🎬 Loading video: {q['condition_content']}")
                outputs.extend([
                    gr.update(visible=False),                               # 1: input_audio
                    gr.update(visible=False),                               # 2: input_image
                    gr.update(visible=True, value=q["condition_content"]),  # 3: input_video
                    gr.update(visible=False)                                # 4: input_text
                ])
            else:
                # No input - hide all
                outputs.extend([
                    gr.update(visible=False),  # 1: input_audio
                    gr.update(visible=False),  # 2: input_image
                    gr.update(visible=False),  # 3: input_video
                    gr.update(visible=False)   # 4: input_text
                ])
            
            # Option components (5-20): A,B,C,D each has audio,image,video,text
            for opt_key in ['A', 'B', 'C', 'D']:
                if opt_key in q["options"]:
                    opt_data = q["options"][opt_key]
                    if opt_data["type"] == "text":
                        outputs.extend([
                            gr.update(visible=False),                            # audio
                            gr.update(visible=False),                            # image
                            gr.update(visible=False),                            # video
                            gr.update(visible=True, value=opt_data["content"])   # text
                        ])
                    elif opt_data["type"] == "audio":
                        outputs.extend([
                            gr.update(visible=True, value=opt_data["content"]),  # audio
                            gr.update(visible=False),                            # image
                            gr.update(visible=False),                            # video
                            gr.update(visible=False)                             # text
                        ])
                    elif opt_data["type"] == "image":
                        outputs.extend([
                            gr.update(visible=False),                            # audio
                            gr.update(visible=True, value=opt_data["content"]),  # image
                            gr.update(visible=False),                            # video
                            gr.update(visible=False)                             # text
                        ])
                    elif opt_data["type"] == "video":
                        print(f"🎬 Loading option {opt_key} video: {opt_data['content']}")
                        outputs.extend([
                            gr.update(visible=False),                            # audio
                            gr.update(visible=False),                            # image
                            gr.update(visible=True, value=opt_data["content"]),  # video
                            gr.update(visible=False)                             # text
                        ])
                else:
                    # No option - hide all
                    outputs.extend([
                        gr.update(visible=False),  # audio
                        gr.update(visible=False),  # image
                        gr.update(visible=False),  # video
                        gr.update(visible=False)   # text
                    ])
            
            # Answer status and group visibility (21-22)
            outputs.extend([
                "",                          # 21: answer_status
                gr.update(visible=True)      # 22: question_group
            ])
            
            question_outputs.extend(outputs)
            
        else:
            # Hidden question slot - 23 components all hidden/empty
            question_outputs.extend([
                "",                          # 0: question_display
                gr.update(visible=False),    # 1: input_audio
                gr.update(visible=False),    # 2: input_image
                gr.update(visible=False),    # 3: input_video
                gr.update(visible=False),    # 4: input_text
                # Options A,B,C,D (5-20)
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),  # A
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),  # B
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),  # C
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),  # D
                "",                          # 21: answer_status
                gr.update(visible=False)     # 22: question_group
            ])
    
    # Update answers display
    total_questions = len(demo_instance.current_answers)
    displayed_questions = len(questions)
    answers_text = f"**Questions Loaded:** {total_questions} total ({displayed_questions} displayed)\n"
    answers_text += "**Sample Answers:**\n" + "\n".join([f"Q{i+1}: {ans}" for i, ans in enumerate(answers[:displayed_questions])])
    
    print("✅ UI outputs ready")
    return [summary, answers_text] + question_outputs

def show_answers():
    """Show the correct answers and compare with user answers"""
    if not demo_instance.current_answers:
        return "No questions loaded yet."
    
    # Get current user's answers from multi-user system or fallback
    if demo_instance.user_id and demo_instance.user_id in multi_user_manager.user_responses:
        current_user_answers = multi_user_manager.user_responses[demo_instance.user_id]
    else:
        current_user_answers = demo_instance.user_answers
    
    answers_text = "**Results:**\n\n"
    correct_count = 0
    
    for i, correct_answer in enumerate(demo_instance.current_answers):
        user_answer = current_user_answers[i] if i < len(current_user_answers) else ""
        
        if user_answer == correct_answer:
            status = "✅ Correct"
            correct_count += 1
        elif user_answer == "":
            status = "⚪ Not answered"
        else:
            status = "❌ Wrong"
        
        answers_text += f"**Question {i+1}:** Your answer: **{user_answer}** | Correct: **{correct_answer}** | {status}\n\n"
    
    score = f"**Score: {correct_count}/{len(demo_instance.current_answers)} ({correct_count/len(demo_instance.current_answers)*100:.1f}%)**\n\n"
    
    # Show session info without auto-saving
    session_info = f"\n📋 **Session ID:** {demo_instance.session_id}\n💡 **Tip:** Click 'Submit Results' to save your answers permanently"
    
    return score + answers_text + session_info

def submit_results(task, subtask):
    """Submit and save user results manually"""
    if demo_instance.user_id:
        # Submit to multi-user system
        score, total = multi_user_manager.submit_user_final(demo_instance.user_id)
        percentage = (score / total) * 100 if total > 0 else 0
        
        # Update local answers for saving
        if demo_instance.user_id in multi_user_manager.user_responses:
            demo_instance.user_answers = multi_user_manager.user_responses[demo_instance.user_id]
        
        # Also save individual file
        result = demo_instance.manual_save_results(task, subtask)
        
        return f"✅ **Submitted to Leaderboard!**\n\n📊 **Your Score:** {score}/{total} ({percentage:.1f}%)\n👤 **Player:** {demo_instance.username}\n\n{result}"
    else:
        return demo_instance.manual_save_results(task, subtask)

def join_session(username):
    """Join the multi-user session"""
    if not username.strip():
        username = None
    
    # Call register_user on the multi_user_manager, not demo_instance
    user_id, assigned_username = multi_user_manager.register_user(username)
    
    # Store user info in demo_instance for later use
    demo_instance.user_id = user_id
    demo_instance.username = assigned_username
    
    return f"✅ Joined as: {assigned_username} (ID: {user_id})"

def get_live_stats():
    """Get live user statistics"""
    return demo_instance.get_user_stats_display()

def clear_user_answers():
    """Clear current user's answers"""
    if demo_instance.user_id and demo_instance.user_id in multi_user_manager.user_responses:
        # Clear in multi-user system
        with multi_user_manager.lock:
            if demo_instance.user_id in multi_user_manager.user_responses:
                multi_user_manager.user_responses[demo_instance.user_id] = [""] * len(demo_instance.current_answers)
    
    # Also clear local answers
    demo_instance.user_answers = [""] * len(demo_instance.current_answers)
    
    username = demo_instance.username or 'Anonymous'
    return f"All answers cleared! User: {username}"

# Create Gradio interface with custom CSS
css = """
.large-text {
    font-size: 18px !important;
    line-height: 1.4 !important;
}

.section-header {
    font-size: 20px !important;
    font-weight: bold !important;
    margin: 15px 0 10px 0 !important;
}

.option-label {
    font-size: 16px !important;
    font-weight: bold !important;
    text-align: center !important;
    margin-bottom: 8px !important;
}

.option-text {
    font-size: 14px !important;
    text-align: center !important;
}

.input-text {
    font-size: 16px !important;
}

.answer-status {
    font-size: 16px !important;
    font-weight: bold !important;
}

/* Make media components more compact */
.gradio-container .gradio-audio {
    max-width: 300px !important;
    height: 60px !important;
}

.gradio-container .gradio-image {
    max-height: 150px !important;
    max-width: 150px !important;
}

.gradio-container .gradio-video {
    max-height: 150px !important;
    max-width: 150px !important;
}

/* Button styling */
button {
    font-size: 16px !important;
    font-weight: bold !important;
    min-height: 45px !important;
    min-width: 60px !important;
}

/* Overall spacing */
.gradio-group {
    padding: 20px !important;
    margin: 15px 0 !important;
}

/* Column width for options */
.gradio-column {
    flex: 1 !important;
    max-width: 200px !important;
}
"""

with gr.Blocks(title="AudioBench VQA Demo - Multi-User Enhanced", theme=gr.themes.Soft(), css=css) as app:
    gr.Markdown("# 🌐 AudioBench VQA - Multi-User Demo (Enhanced)")
    gr.Markdown("**Real-time multiplayer VQA testing with N questions per type!** Join with your username and compete with others.")
    
    # User registration section
    with gr.Row():
        with gr.Column(scale=2):
            username_input = gr.Textbox(label="Your Username", placeholder="Enter your name", value="")
            join_btn = gr.Button("Join Session", variant="primary")
            user_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Column(scale=1):
            stats_display = gr.Markdown("## 📊 Waiting for users...")
            refresh_stats_btn = gr.Button("🔄 Refresh Stats")
    
    # Question control section
    with gr.Row():
        task_dropdown = gr.Dropdown(
            label="Select Task",
            choices=demo_instance.get_available_tasks(),
            interactive=True,
            value="01_perception"  # Default value
        )
        subtask_dropdown = gr.Dropdown(
            label="Select Subtask",
            choices=demo_instance.get_available_subtasks("01_perception"),  # Load subtasks for default task
            interactive=True,
            value="vggss"  # Default value
        )
    
    # Enhanced question loading options
    with gr.Row():
        questions_per_type_slider = gr.Slider(
            label="Questions per Type (N)",
            minimum=1,
            maximum=8,
            value=3,
            step=1,
            info="Number of questions to select from each JSON file (question type)"
        )
        use_per_type_mode = gr.Checkbox(
            label="Use N-per-type mode", 
            value=True,
            info="Check to use N questions per type, uncheck for random selection"
        )
    
    with gr.Row():
        load_btn = gr.Button("🚀 Load Questions for Users", variant="primary", size="lg")
        show_answers_btn = gr.Button("Show My Results", variant="secondary")
        submit_btn = gr.Button("📤 Submit to Leaderboard", variant="stop")
        clear_answers_btn = gr.Button("Clear My Answers")
    
    summary_text = gr.Textbox(label="Status", interactive=False)
    answers_text = gr.Textbox(label="Results", interactive=False, visible=False, lines=10)
    submit_status = gr.Textbox(label="Submission Status", interactive=False, visible=False, lines=5)
    
    # Simplified question display - now supports configurable number of questions
    question_components = []
    
    for i in range(demo_instance.max_display_questions):  # Use hyperparameter for question slots
        with gr.Group(visible=False) as question_group:
            # Question text - larger font
            question_display = gr.Markdown(elem_classes="large-text")
            
            # Input Media section
            gr.Markdown("### Input Media:", elem_classes="section-header")
            with gr.Row():
                input_audio = gr.Audio(label="Audio", visible=False)
                input_image = gr.Image(label="Image", visible=False) 
                input_video = gr.Video(label="Video", visible=False, format="mp4")
                input_text = gr.Markdown(visible=False, elem_classes="input-text")
            
            # Options section  
            gr.Markdown("### Options:", elem_classes="section-header")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**A**", elem_classes="option-label")
                    option_a_audio = gr.Audio(visible=False, show_label=False)
                    option_a_image = gr.Image(visible=False, show_label=False)
                    option_a_video = gr.Video(visible=False, show_label=False, format="mp4")
                    option_a_text = gr.Markdown(visible=False, elem_classes="option-text")
                
                with gr.Column():
                    gr.Markdown("**B**", elem_classes="option-label")
                    option_b_audio = gr.Audio(visible=False, show_label=False)
                    option_b_image = gr.Image(visible=False, show_label=False)
                    option_b_video = gr.Video(visible=False, show_label=False, format="mp4")
                    option_b_text = gr.Markdown(visible=False, elem_classes="option-text")
                
                with gr.Column():
                    gr.Markdown("**C**", elem_classes="option-label")
                    option_c_audio = gr.Audio(visible=False, show_label=False)
                    option_c_image = gr.Image(visible=False, show_label=False)
                    option_c_video = gr.Video(visible=False, show_label=False, format="mp4")
                    option_c_text = gr.Markdown(visible=False, elem_classes="option-text")
                
                with gr.Column():
                    gr.Markdown("**D**", elem_classes="option-label")
                    option_d_audio = gr.Audio(visible=False, show_label=False)
                    option_d_image = gr.Image(visible=False, show_label=False)
                    option_d_video = gr.Video(visible=False, show_label=False, format="mp4")
                    option_d_text = gr.Markdown(visible=False, elem_classes="option-text")
            
            # Your Answer section
            gr.Markdown("### Your Answer:", elem_classes="section-header")
            with gr.Row():
                answer_a = gr.Button("A", variant="secondary")
                answer_b = gr.Button("B", variant="secondary")
                answer_c = gr.Button("C", variant="secondary")
                answer_d = gr.Button("D", variant="secondary")
            
            answer_status = gr.Textbox(label="Selected", interactive=False, show_label=False, elem_classes="answer-status")
        
        # Store components: 23 components per question
        question_components.extend([
            question_display,           # 0
            input_audio,               # 1
            input_image,               # 2
            input_video,               # 3
            input_text,                # 4
            option_a_audio,            # 5
            option_a_image,            # 6
            option_a_video,            # 7
            option_a_text,             # 8
            option_b_audio,            # 9
            option_b_image,            # 10
            option_b_video,            # 11
            option_b_text,             # 12
            option_c_audio,            # 13
            option_c_image,            # 14
            option_c_video,            # 15
            option_c_text,             # 16
            option_d_audio,            # 17
            option_d_image,            # 18
            option_d_video,            # 19
            option_d_text,             # 20
            answer_status,             # 21
            question_group             # 22
        ])
        
        # Add button click handlers
        def make_answer_handler(q_idx, choice):
            def handler():
                return demo_instance.record_answer(q_idx, choice)
            return handler
        
        # Answer status is at index 21 for each question
        status_component = question_components[i * 23 + 21]
        
        answer_a.click(make_answer_handler(i, "A"), outputs=[status_component])
        answer_b.click(make_answer_handler(i, "B"), outputs=[status_component])
        answer_c.click(make_answer_handler(i, "C"), outputs=[status_component])
        answer_d.click(make_answer_handler(i, "D"), outputs=[status_component])
    
    # Event handlers
    print("🔗 Setting up event handlers...")
    
    # User registration
    join_btn.click(
        join_session,
        inputs=[username_input],
        outputs=[user_status]
    )
    
    # Stats refresh
    refresh_stats_btn.click(
        get_live_stats,
        outputs=[stats_display]
    )
    
    task_dropdown.change(
        update_subtasks,
        inputs=[task_dropdown],
        outputs=[subtask_dropdown]
    )
    
    load_btn.click(
        lambda task, subtask, n_per_type, use_per_type: [
            *load_questions(task, subtask, n_per_type, use_per_type),
            get_live_stats()
        ],
        inputs=[task_dropdown, subtask_dropdown, questions_per_type_slider, use_per_type_mode],
        outputs=[summary_text, answers_text] + question_components + [stats_display]
    )
    
    show_answers_btn.click(
        lambda: gr.update(visible=True, value=show_answers()),
        outputs=[answers_text]
    )
    
    submit_btn.click(
        lambda task, subtask: [
            gr.update(visible=True, value=submit_results(task, subtask)),
            gr.update(visible=True, value=show_answers()),
            get_live_stats()
        ],
        inputs=[task_dropdown, subtask_dropdown],
        outputs=[submit_status, answers_text, stats_display]
    )
    
    clear_answers_btn.click(
        lambda: [
            clear_user_answers(),
            gr.update(visible=False),
            gr.update(visible=False)
        ],
        outputs=[summary_text, answers_text, submit_status]
    )

print("🎉 Gradio interface ready!")

if __name__ == "__main__":
    print("🌐 Launching Gradio app...")
    app.launch(share=True, debug=True)