import gradio as gr
import json
import os
import random
from pathlib import Path
import base64
from typing import Dict, List, Tuple, Any
import datetime
import uuid

class AudioBenchDemo:
    def __init__(self, base_path="/home/xwang378/scratch/2025/AudioBench/benchmark/tasks"):
        self.base_path = Path(base_path)
        self.current_questions = []
        self.current_answers = []
        self.user_answers = []
        self.cached_questions = {}  # Cache loaded questions
        self.session_id = str(uuid.uuid4())[:8]  # Generate unique session ID
        self.results_dir = Path("user_results")
        self.results_dir.mkdir(exist_ok=True)  # Create results directory
        print(f"🆔 Session ID: {self.session_id}")
        
    def save_user_results(self, task: str, subtask: str):
        """Save user results to local file"""
        if not self.current_questions or not self.user_answers:
            return
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{self.session_id}_{task}_{subtask}_{timestamp}.json"
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
                "question_id": question.get('question_id', '')
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
        """Record answer without auto-saving"""
        if question_num < len(self.user_answers):
            self.user_answers[question_num] = selected_answer
            return f"Selected: {selected_answer}"
        return "Error recording answer"
        
    def get_available_tasks(self) -> List[str]:
        """Get list of available main tasks"""
        return ['01_perception']
        tasks = []
        for item in self.base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                tasks.append(item.name)
        return sorted(tasks)
    
    def get_available_subtasks(self, task: str) -> List[str]:
        """Get list of available subtasks for a given task"""
        return ['vggss']
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
    
    def load_questions_from_files(self, json_files: List[Path]) -> List[Dict]:
        """Load all questions from JSON files with caching"""
        cache_key = str(sorted(json_files))
        
        # Check cache first
        if cache_key in self.cached_questions:
            return self.cached_questions[cache_key]
        
        all_questions = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    questions = json.load(f)
                    for i, q in enumerate(questions):
                        q['source_file'] = str(json_file)
                        q['question_id'] = i
                        all_questions.append(q)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        # Cache the results
        self.cached_questions[cache_key] = all_questions
        return all_questions
    
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
    
    def generate_random_questions(self, task: str, subtask: str, num_questions: int = 5):
        """Generate random questions from selected task and subtask"""
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
        
        # Load all questions
        all_questions = self.load_questions_from_files(json_files)
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
        
        self.current_questions = selected_questions
        self.current_answers = [q.get('correct_answer', 'Unknown') for q in selected_questions]
        self.user_answers = [""] * len(selected_questions)  # Reset user answers
        
        # Format questions for display
        print("🎨 Formatting questions for display...")
        question_displays = []
        for i, q in enumerate(selected_questions):
            question_display = self.format_question_display(q, i)
            question_displays.append(question_display)
        
        summary = f"Loaded {len(selected_questions)} questions from {task}/{subtask}"
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

# Initialize the demo
print("🚀 Initializing AudioBench Demo...")
demo_instance = AudioBenchDemo()
print("✅ Demo instance created")

def update_subtasks(task):
    """Update subtask dropdown based on selected task"""
    subtasks = demo_instance.get_available_subtasks(task)
    return gr.update(choices=subtasks, value=None)

def load_questions(task, subtask, num_questions):
    """Load and display random questions"""
    print(f"🎮 User requested: {num_questions} questions from {task}/{subtask}")
    
    summary, questions, answers = demo_instance.generate_random_questions(task, subtask, num_questions)
    
    print("🎨 Building UI outputs...")
    question_outputs = []
    
    # For each of the 3 question slots
    for i in range(3):
        if i < len(questions):
            print(f"📝 Processing question {i+1}")
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
    answers_text = "**Correct Answers:**\n" + "\n".join([f"Q{i+1}: {ans}" for i, ans in enumerate(answers)])
    
    print("✅ UI outputs ready")
    return [summary, answers_text] + question_outputs

def show_answers():
    """Show the correct answers and compare with user answers"""
    if not demo_instance.current_answers:
        return "No questions loaded yet."
    
    answers_text = "**Results:**\n\n"
    correct_count = 0
    
    for i, correct_answer in enumerate(demo_instance.current_answers):
        user_answer = demo_instance.user_answers[i] if i < len(demo_instance.user_answers) else ""
        
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
    return demo_instance.manual_save_results(task, subtask)

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

with gr.Blocks(title="AudioBench VQA Demo", theme=gr.themes.Soft(), css=css) as app:
    gr.Markdown("# AudioBench VQA Task Demo")
    gr.Markdown(f"**Session ID: {demo_instance.session_id}** | Answer questions and manually submit your results when ready.")
    
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
        num_questions_slider = gr.Slider(
            label="Number of Questions",
            minimum=1,
            maximum=3,
            value=2,  # Reduce default to 2 for faster loading
            step=1
        )
    
    with gr.Row():
        load_btn = gr.Button("Load Random Questions", variant="primary")
        show_answers_btn = gr.Button("Show Results", variant="secondary")
        submit_btn = gr.Button("Submit Results", variant="stop")
        clear_answers_btn = gr.Button("Clear My Answers")
    
    summary_text = gr.Textbox(label="Status", interactive=False)
    answers_text = gr.Textbox(label="Results", interactive=False, visible=False, lines=10)
    submit_status = gr.Textbox(label="Submission Status", interactive=False, visible=False, lines=5)
    
    # Simplified question display - 4 components per question
    question_components = []
    
    for i in range(3):  # 3 questions maximum
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
        
        # Store components: 22 components per question
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
    
    task_dropdown.change(
        update_subtasks,
        inputs=[task_dropdown],
        outputs=[subtask_dropdown]
    )
    
    load_btn.click(
        load_questions,
        inputs=[task_dropdown, subtask_dropdown, num_questions_slider],
        outputs=[summary_text, answers_text] + question_components
    )
    
    show_answers_btn.click(
        lambda: gr.update(visible=True, value=show_answers()),
        outputs=[answers_text]
    )
    
    submit_btn.click(
        lambda task, subtask: [
            gr.update(visible=True, value=submit_results(task, subtask)),
            gr.update(visible=True, value=show_answers())
        ],
        inputs=[task_dropdown, subtask_dropdown],
        outputs=[submit_status, answers_text]
    )
    
    clear_answers_btn.click(
        lambda: [
            (setattr(demo_instance, 'user_answers', [""] * len(demo_instance.user_answers)), 
             f"All answers cleared! Session: {demo_instance.session_id}")[1],
            gr.update(visible=False),
            gr.update(visible=False)
        ],
        outputs=[summary_text, answers_text, submit_status]
    )

print("🎉 Gradio interface ready!")

if __name__ == "__main__":
    print("🌐 Launching Gradio app...")
    app.launch(share=True, debug=True)