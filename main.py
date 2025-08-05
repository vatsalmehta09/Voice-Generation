import streamlit as st
import os
import uuid
import json
from datetime import datetime
import tempfile
import io
from openai import OpenAI
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import PyPDF2
import docx
import pandas as pd
import mimetypes
import base64
import asyncio
import edge_tts  


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration (same as before - no changes needed)
class Config:
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "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")
    GITHUB_BASE_URL = os.environ.get("GITHUB_BASE_URL", "https://models.github.ai/inference")
    AUDIO_OUTPUT_DIR = os.getenv('AUDIO_OUTPUT_DIR', './audio_files')
    UPLOAD_DIR = os.getenv('UPLOAD_DIR', './uploads')
    DATABASE_PATH = os.getenv('DATABASE_PATH', './voiceover_db.sqlite')
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB max file size
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'es': 'Spanish', 
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh': 'Chinese',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'pl': 'Polish',
        'nl': 'Dutch'
    }
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'csv', 'xlsx', 'xls', 'json', 'md'}
    MAX_TEXT_LENGTH = 50000  # Maximum characters for text input
    
    # Updated Voices for edge-tts - only the specified models
    DEFAULT_VOICES = {
        'Neerja (en-IN, Female)': 'en-IN-NeerjaNeural',
        'Prabhat (en-IN, Male)': 'en-IN-PrabhatNeural',
        'Jenny (en-US, Female)': 'en-US-JennyNeural',
        'Davis (en-US, Male)': 'en-US-DavisNeural',
        'Mia (en-GB, Female)': 'en-GB-MiaNeural',
        'Ryan (en-GB, Male)': 'en-GB-RyanNeural',
        'Clara (en-CA, Female)': 'en-CA-ClaraNeural',
        'Liam (en-CA, Male)': 'en-CA-LiamNeural',
        'Duncan (en-AU, Male)': 'en-AU-DuncanNeural',
        'Tina (en-AU, Female)': 'en-AU-TinaNeural',
    }


# Ensure directories exist
os.makedirs(Config.AUDIO_OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.UPLOAD_DIR, exist_ok=True)


# Initialize OpenAI client
client = OpenAI(
    base_url=Config.GITHUB_BASE_URL,
    api_key=Config.GITHUB_TOKEN
)


# Database setup
def init_db():
    """Initialize SQLite database"""
    with sqlite3.connect(Config.DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('PRAGMA table_info(voiceover_projects)')
        columns = [row[1] for row in cursor.fetchall()]
        
        if not columns or 'voice_id' not in columns:
            conn.execute('DROP TABLE IF EXISTS voiceover_projects')
            conn.execute('''
                CREATE TABLE voiceover_projects (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    original_content TEXT,
                    source_type TEXT,
                    source_filename TEXT,
                    generated_script TEXT,
                    language TEXT,
                    voice_id TEXT,
                    voice_name TEXT,
                    status TEXT,
                    audio_file_path TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            ''')
            logger.info("Database table 'voiceover_projects' created/updated.")
        conn.commit()


@contextmanager
def get_db():
    """Database context manager"""
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


class VoiceoverGenerator:
    def __init__(self):
        self.projects = {}
        self.available_voices = []
        self._load_available_voices()
    
    def _load_available_voices(self):
        """Load available voices for edge-tts"""
        self.available_voices = [
            {
                'voice_id': voice_id,
                'name': name,
                'category': 'edge-tts'
            } for name, voice_id in Config.DEFAULT_VOICES.items()
        ]
        logger.info(f"Loaded {len(self.available_voices)} voices for edge-tts")


    def allowed_file(self, filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    
    def extract_text_from_file(self, file_path: str, filename: str) -> str:
        """Extract text from various file formats"""
        try:
            file_extension = filename.rsplit('.', 1)[1].lower()
            
            if file_extension == 'txt' or file_extension == 'md':
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            
            elif file_extension == 'pdf':
                return self._extract_from_pdf(file_path)
            
            elif file_extension in ['docx', 'doc']:
                return self._extract_from_docx(file_path)
            
            elif file_extension == 'csv':
                return self._extract_from_csv(file_path)
            
            elif file_extension in ['xlsx', 'xls']:
                return self._extract_from_excel(file_path)
            
            elif file_extension == 'json':
                return self._extract_from_json(file_path)
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {str(e)}")
            raise Exception(f"Failed to extract text from {filename}: {str(e)}")
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    
    def _extract_from_csv(self, file_path: str) -> str:
        """Extract text from CSV file"""
        df = pd.read_csv(file_path)
        text = f"Data Summary:\n"
        text += f"Number of rows: {len(df)}\n"
        text += f"Columns: {', '.join(df.columns)}\n\n"
        
        text += "Sample Data:\n"
        text += df.head(10).to_string(index=False)
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            text += f"\n\nNumeric Data Summary:\n"
            text += df[numeric_cols].describe().to_string()
        
        return text
    
    def _extract_from_excel(self, file_path: str) -> str:
        """Extract text from Excel file"""
        df = pd.read_excel(file_path)
        # Reuse CSV extractor to get summary for Excel data
        return self._extract_from_csv(file_path)
    
    def _extract_from_json(self, file_path: str) -> str:
        """Extract text from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if isinstance(data, dict):
            text = "JSON Data Content:\n"
            text += json.dumps(data, indent=2, ensure_ascii=False)
        elif isinstance(data, list):
            text = f"JSON Array with {len(data)} items:\n"
            text += json.dumps(data[:5], indent=2, ensure_ascii=False)
            if len(data) > 5:
                text += f"\n... and {len(data) - 5} more items"
        else:
            text = str(data)
        
        return text
    
    def generate_script(self, content: str, instructions: str = "", target_length: str = "medium", 
                       custom_duration: int = None) -> str:
        """Generate script from content using GitHub Models API"""
        try:
            duration_guidance = self._get_duration_guidance(target_length, custom_duration)
            
            system_prompt = f"""
            You are a professional script writer for video voiceovers using Microsoft edge-tts voices. 
            Create engaging, clear, and well-structured scripts suitable for high-quality AI voice synthesis.
            
            Guidelines:
            - Write in a conversational, engaging tone optimized for edge-tts voices
            - Use clear, simple language with natural speech patterns
            - Include natural pauses using ellipses (...) where appropriate
            - {duration_guidance}
            - Average speaking rate: 150-160 words per minute
            - For very short durations (15-30 seconds), focus on key points and maintain impact
            - Use punctuation to control pacing and emphasis
            - Only provide the script in output (NO EXTRA MARKERS)
            """
            
            user_prompt = f"""
            Content to convert to script:
            {content}
            
            Additional instructions: {instructions}
            
            Please create a professional voiceover script optimized for edge-tts based on this content.
            """
            
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="openai/gpt-4o",
                temperature=0.8,
                max_tokens=4096,
                top_p=1
            )
            
            script = response.choices[0].message.content.strip()
            return script
            
        except Exception as e:
            logger.error(f"Error generating script: {str(e)}")
            raise Exception(f"Failed to generate script: {str(e)}")
    
    def _get_duration_guidance(self, target_length: str, custom_duration: int = None) -> str:
        """Get duration guidance for script generation with support for short durations"""
        if custom_duration:
            if custom_duration < 15:
                custom_duration = 15
            elif custom_duration > 300:
                custom_duration = 300  # 5 minutes max
                
            minutes = custom_duration // 60
            seconds = custom_duration % 60
            target_words = int(custom_duration * 2.5)
            
            duration_str = f"{minutes}:{seconds:02d}" if minutes > 0 else f"{seconds} seconds"
            return f"Target duration: exactly {duration_str} ({target_words} words approximately)"
        else:
            duration_mapping = {
                "short": "Target duration: 15-30 seconds (38-75 words approximately)",
                "medium": "Target duration: 1-2 minutes (150-320 words approximately)", 
                "long": "Target duration: 3-5 minutes (450-750 words approximately)"
            }
            return duration_mapping.get(target_length, duration_mapping["medium"])
    
    def estimate_duration(self, script: str) -> dict:
        """Estimate duration of script based on word count"""
        clean_script = script.replace('...', ' ')
        words = len(clean_script.split())
        
        duration_seconds = int((words / 150) * 60)
        if duration_seconds < 15:
            duration_seconds = 15
        minutes = duration_seconds // 60
        seconds = duration_seconds % 60
        
        return {
            "word_count": words,
            "estimated_duration_seconds": duration_seconds,
            "estimated_duration_formatted": f"{minutes}:{seconds:02d}",
            "speaking_rate_wpm": 150
        }


    async def _generate_voice_async(self, script: str, voice_id: str, output_path: str):
        """Generate speech audio asynchronously using edge-tts"""
        try:
            communicate = edge_tts.Communicate(script, voice_id)
            await communicate.save(output_path)
        except Exception as e:
            logger.error(f"Error in _generate_voice_async: {str(e)}")
            raise

    def generate_voice(self, script: str, voice_id: str) -> str:
        """Generate voice from script using edge-tts with default settings"""
        try:
            # Clean script for better TTS processing
            clean_script = script.replace('[PAUSE]', '... ')
            
            audio_filename = f"{uuid.uuid4()}.wav"  # Changed to .wav for better compatibility
            audio_path = os.path.join(Config.AUDIO_OUTPUT_DIR, audio_filename)
            
            # Run edge-tts asynchronously
            asyncio.run(self._generate_voice_async(clean_script, voice_id, audio_path))
            
            return audio_path
            
        except Exception as e:
            logger.error(f"Error generating voice with edge-tts: {str(e)}")
            raise Exception(f"Failed to generate voice: {str(e)}")
    
    # **FIXED: Voice preview method**
    async def _get_voice_preview_async(self, voice_id: str) -> Optional[bytes]:
        """Get a voice preview asynchronously using edge-tts"""
        try:
            text = "Hello! I am a AI-powered video voiceover generation system that takes input from documents or videos, uses large language model to generate a script, and converts it into natural-sounding speech using TTS engines."
            
            # Create a temporary file for the preview
            temp_preview_path = os.path.join(Config.AUDIO_OUTPUT_DIR, f"preview_{uuid.uuid4()}.wav")
            
            communicate = edge_tts.Communicate(text, voice_id)
            await communicate.save(temp_preview_path)
            
            # Read the generated audio file
            with open(temp_preview_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temporary file
            if os.path.exists(temp_preview_path):
                os.remove(temp_preview_path)
            
            return audio_data

        except Exception as e:
            logger.error(f"Error generating voice preview: {str(e)}")
            return None

    def get_voice_preview(self, voice_id: str) -> Optional[bytes]:
        """Synchronous wrapper for voice preview"""
        try:
            return asyncio.run(self._get_voice_preview_async(voice_id))
        except Exception as e:
            logger.error(f"Error in get_voice_preview: {str(e)}")
            return None


# Initialize generator and database
generator = VoiceoverGenerator()
init_db()



# Streamlit App
def main():
    st.set_page_config(
        page_title="AI Voiceover Generator",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("🎙️ AI Voiceover Generator")
    st.markdown("Transform your text and documents into professional voiceovers with AI-powered script generation and Microsoft edge-tts voices.")
    
    
    # Initialize session state
    if 'current_project' not in st.session_state:
        st.session_state.current_project = None
    if 'generated_script' not in st.session_state:
        st.session_state.generated_script = None
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Create New Project", "My Projects", "Voice Gallery", "Supported Formats", "About"]
    )
    
    if page == "Create New Project":
        create_project_page()
    elif page == "My Projects":
        projects_page()
    elif page == "Voice Gallery":
        voice_gallery_page()
    elif page == "Supported Formats":
        formats_page()
    elif page == "About":
        about_page()



def create_project_page():
    st.header("Create New Voiceover Project")
    
    # Project title
    project_title = st.text_input("Project Title", placeholder="Enter a title for your project")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "File Upload"]
    )
    
    content = ""
    source_type = "text"
    source_filename = None
    
    if input_method == "Text Input":
        content = st.text_area(
            "Enter your content:",
            height=200,
            max_chars=Config.MAX_TEXT_LENGTH,
            placeholder="Paste your text here..."
        )
        
        if content:
            st.info(f"Character count: {len(content)}/{Config.MAX_TEXT_LENGTH}")
    
    elif input_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=list(Config.ALLOWED_EXTENSIONS),
            help=f"Supported formats: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            try:
                # Extract text from file
                content = generator.extract_text_from_file(tmp_file_path, uploaded_file.name)
                source_type = "file"
                source_filename = uploaded_file.name
                
                if not project_title:
                    project_title = uploaded_file.name.rsplit('.', 1)[0]
                
                st.success(f"✅ File processed successfully!")
                st.info(f"Extracted {len(content)} characters from {uploaded_file.name}")
                
                # Show preview
                with st.expander("Preview extracted content"):
                    st.text_area("Content preview", content[:1000] + "..." if len(content) > 1000 else content, height=200, disabled=True)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                content = ""
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
    
    # Script generation options
    if content:
        st.subheader("Script Generation Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_length = st.selectbox(
                "Target Length",
                ["short", "medium", "long"],
                index=1,
                help="Short: 15-30 sec, Medium: 1-2 min, Long: 3-5 min"
            )
        
        with col2:
            use_custom_duration = st.checkbox("Use custom duration")
            custom_duration = None
            if use_custom_duration:
                custom_duration = st.number_input(
                    "Duration (seconds)",
                    min_value=15,
                    max_value=300,
                    value=60,
                    step=5,
                    help="Enter duration between 15 seconds and 5 minutes"
                )
        
        instructions = st.text_area(
            "Additional Instructions (optional)",
            placeholder="e.g., Make it more casual, include humor, emphasize key points...",
            height=100
        )
        
        # Generate script button
        if st.button("🎯 Generate Script", type="primary"):
            if not project_title:
                st.error("Please enter a project title")
            else:
                with st.spinner("Generating script..."):
                    try:
                        script = generator.generate_script(
                            content, 
                            instructions, 
                            target_length, 
                            custom_duration
                        )
                        
                        duration_info = generator.estimate_duration(script)
                        
                        # Save to database
                        project_id = str(uuid.uuid4())
                        with get_db() as conn:
                            conn.execute('''
                                INSERT INTO voiceover_projects 
                                (id, title, original_content, source_type, source_filename, 
                                 generated_script, status, created_at, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (project_id, project_title, content, source_type, source_filename, 
                                  script, 'script_generated', datetime.now(), datetime.now()))
                            conn.commit()
                        
                        st.session_state.current_project = project_id
                        st.session_state.generated_script = script
                        
                        st.success("✅ Script generated successfully!")
                        
                        # Display duration info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Word Count", duration_info["word_count"])
                        with col2:
                            st.metric("Estimated Duration", duration_info["estimated_duration_formatted"])
                        with col3:
                            st.metric("Speaking Rate", f"{duration_info['speaking_rate_wpm']} WPM")
                        
                    except Exception as e:
                        st.error(f"Error generating script: {str(e)}")
    
    # Show generated script and voice options
    if st.session_state.generated_script:
        st.subheader("Generated Script")
        
        # Script editor
        edited_script = st.text_area(
            "Review and edit your script:",
            value=st.session_state.generated_script,
            height=300,
            help="You can modify the script before generating the voice"
        )
        
        # Update duration estimate if script is edited
        if edited_script != st.session_state.generated_script:
            duration_info = generator.estimate_duration(edited_script)
            st.info(f"Updated duration estimate: {duration_info['estimated_duration_formatted']} ({duration_info['word_count']} words)")
        
        # Voice generation options
        st.subheader("Voice Generation Options")
        
        # Voice selection
        voice_names = [voice['name'] for voice in generator.available_voices]
        
        if not voice_names:
            st.error("No voices available.")
            return
        
        selected_voice_name = st.selectbox(
            "Select Voice",
            options=voice_names,
            help="Choose from available Microsoft edge-tts voices"
        )
        
        selected_voice = next((voice for voice in generator.available_voices if voice['name'] == selected_voice_name), None)
        
        if not selected_voice:
            st.error("Selected voice not found")
            return
        
        # Voice preview
        if st.button("🔊 Preview Voice"):
            with st.spinner("Generating voice preview..."):
                try:
                    preview_audio = generator.get_voice_preview(selected_voice['voice_id'])
                    if preview_audio:
                        st.audio(preview_audio, format='audio/mp3')
                    else:
                        st.error("Could not generate voice preview")
                except Exception as e:
                    st.error(f"Error generating preview: {str(e)}")
        
        # Generate voice button
        if st.button("🎵 Generate Voice", type="primary"):
            with st.spinner("Generating voice... This may take a moment."):
                try:
                    # Update script if edited
                    if edited_script != st.session_state.generated_script:
                        with get_db() as conn:
                            conn.execute('''
                                UPDATE voiceover_projects 
                                SET generated_script = ?, updated_at = ?
                                WHERE id = ?
                            ''', (edited_script, datetime.now(), st.session_state.current_project))
                            conn.commit()
                    
                    # Generate voice with default settings
                    audio_path = generator.generate_voice(
                        edited_script, 
                        selected_voice['voice_id']
                    )
                    
                    # Update database
                    with get_db() as conn:
                        conn.execute('''
                            UPDATE voiceover_projects 
                            SET voice_id = ?, voice_name = ?, 
                                audio_file_path = ?, status = ?, updated_at = ?
                            WHERE id = ?
                        ''', (selected_voice['voice_id'], selected_voice['name'], 
                              audio_path, 'voice_generated', datetime.now(), st.session_state.current_project))
                        conn.commit()
                    
                    st.session_state.audio_file = audio_path
                    st.success("✅ Voice generated successfully!")
                    
                    # Display audio player
                    with open(audio_path, 'rb') as audio_file:
                        st.audio(audio_file.read(), format='audio/mp3')
                    
                    # Download button
                    with open(audio_path, 'rb') as audio_file:
                        st.download_button(
                            label="📥 Download Audio",
                            data=audio_file.read(),
                            file_name=f"{project_title}.mp3",
                            mime="audio/mpeg"
                        )
                    
                except Exception as e:
                    st.error(f"Error generating voice: {str(e)}")


def projects_page():
    st.header("My Projects")
    
    try:
        with get_db() as conn:
            projects = conn.execute('''
                SELECT * FROM voiceover_projects 
                ORDER BY created_at DESC
            ''').fetchall()
            
            if not projects:
                st.info("No projects found. Create your first project!")
                return
            
            for project in projects:
                with st.expander(f"🎙️ {project['title']} - {project['status']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Created:** {project['created_at']}")
                        st.write(f"**Status:** {project['status']}")
                        if project['voice_name']:
                            st.write(f"**Voice:** {project['voice_name']}")
                    
                    with col2:
                        if project['source_type'] == 'file':
                            st.write(f"**Source:** {project['source_filename']}")
                        else:
                            st.write(f"**Source:** Text input")
                        
                        if project['generated_script']:
                            duration_info = generator.estimate_duration(project['generated_script'])
                            st.write(f"**Duration:** {duration_info['estimated_duration_formatted']}")
                    
                    # Show script
                    if project['generated_script']:
                        st.text_area(
                            "Generated Script",
                            value=project['generated_script'],
                            height=150,
                            disabled=True,
                            key=f"script_{project['id']}"
                        )
                    
                    # Show audio player if available
                    if project['audio_file_path'] and os.path.exists(project['audio_file_path']):
                        st.subheader("Generated Audio")
                        with open(project['audio_file_path'], 'rb') as audio_file:
                            st.audio(audio_file.read(), format='audio/mp3')
                        
                        # Download button
                        with open(project['audio_file_path'], 'rb') as audio_file:
                            st.download_button(
                                label="📥 Download Audio",
                                data=audio_file.read(),
                                file_name=f"{project['title']}.mp3",
                                mime="audio/mpeg",
                                key=f"download_{project['id']}"
                            )
                    
                    # Action buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🔄 Edit Project", key=f"edit_{project['id']}"):
                            st.session_state.current_project = project['id']
                            st.session_state.generated_script = project['generated_script']
                            st.info("Go to 'Create New Project' to continue editing")
                    
                    with col2:
                        if st.button(f"🗑️ Delete Project", key=f"delete_{project['id']}", type="secondary"):
                            # Delete project
                            conn.execute('DELETE FROM voiceover_projects WHERE id = ?', (project['id'],))
                            conn.commit()
                            
                            # Delete audio file if exists
                            if project['audio_file_path'] and os.path.exists(project['audio_file_path']):
                                os.remove(project['audio_file_path'])
                            
                            st.success("Project deleted successfully!")
                            st.experimental_rerun()
    
    except Exception as e:
        st.error(f"Error loading projects: {str(e)}")


def voice_gallery_page():
    st.header("🎭 Voice Gallery")
    st.markdown("Explore available Microsoft edge-tts voices and listen to previews")
    
    if not generator.available_voices:
        st.error("No voices available.")
        return
    
    # Show updated voices with details
    st.subheader("Edge-TTS Voices")
    st.markdown("**Available Voices (Indian, US, British, Canadian, and Australian accents):**")
    
    cols = st.columns(2)
    for idx, voice in enumerate(generator.available_voices):
        with cols[idx % 2]:
            st.write(f"**{voice['name']}**")
            
            # Add accent information
            if 'en-IN' in voice['voice_id']:
                st.write("🇮🇳 Indian English accent")
            elif 'en-US' in voice['voice_id']:
                st.write("🇺🇸 American English accent")
            elif 'en-GB' in voice['voice_id']:
                st.write("🇬🇧 British English accent")
            elif 'en-CA' in voice['voice_id']:
                st.write("🇨🇦 Canadian English accent")
            elif 'en-AU' in voice['voice_id']:
                st.write("🇦🇺 Australian English accent")
            
            if st.button(f"🔊 Preview", key=f"preview_{voice['voice_id']}"):
                with st.spinner(f"Generating preview for {voice['name']}..."):
                    try:
                        preview_audio = generator.get_voice_preview(voice['voice_id'])
                        if preview_audio:
                            st.audio(preview_audio, format='audio/mp3')
                        else:
                            st.error("Could not generate preview")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            st.write("---")


def formats_page():
    st.header("Supported Formats")
    
    st.subheader("📄 File Formats")
    
    format_descriptions = {
        'txt': ('Plain Text Files', 'Simple text documents'),
        'md': ('Markdown Files', 'Formatted text with markdown syntax'),
        'pdf': ('PDF Documents', 'Portable Document Format files'),
        'docx': ('Word Documents', 'Microsoft Word (newer format)'),
        'doc': ('Word Documents', 'Microsoft Word (older format)'),
        'csv': ('CSV Files', 'Comma-separated values data'),
        'xlsx': ('Excel Spreadsheets', 'Microsoft Excel (newer format)'),
        'xls': ('Excel Spreadsheets', 'Microsoft Excel (older format)'),
        'json': ('JSON Files', 'JavaScript Object Notation data')
    }
    
    for ext, (name, desc) in format_descriptions.items():
        st.write(f"**{name} (.{ext})** - {desc}")
    
    st.subheader("🌍 Supported Languages")
    
    for code, name in Config.SUPPORTED_LANGUAGES.items():
        st.write(f"**{name}** ({code})")
    
    st.subheader("📊 Limitations")
    
    st.write(f"- **Maximum file size:** {Config.MAX_CONTENT_LENGTH // (1024 * 1024)} MB")
    st.write(f"- **Maximum text length:** {Config.MAX_TEXT_LENGTH:,} characters")
    st.write(f"- **Duration range:** 15 seconds to 5 minutes")
    st.write(f"- **Speaking rate:** ~150 words per minute")
    st.write(f"- **Voice quality:** Premium Microsoft edge-tts voices with default settings")


def about_page():
    st.header("About AI Voiceover Generator with edge-tts")
    
    st.markdown("""
    ### 🎯 What is this?
    
    The AI Voiceover Generator with edge-tts is a streamlined tool that transforms your text and documents into professional-quality voiceovers using Microsoft's edge-tts technology with optimized default settings.
    
    ### 🚀 Features
    
    - **Diverse Voice Selection**: Indian, American, British, Canadian, and Australian English voices
    - **Premium Voice Quality**: Powered by Microsoft edge-tts technology
    - **Multiple Input Methods**: Support for text input and various file formats
    - **AI-Powered Script Generation**: Automatically creates engaging scripts from your content
    - **Voice Gallery**: Browse and preview available voices with accent information
    - **Flexible Duration Control**: Set custom durations from 15 seconds to 5 minutes
    - **Project Management**: Save and manage multiple projects
    - **Easy Export**: Download your voiceovers as high-quality MP3 files
    - **Simplified Workflow**: Uses optimized default voice settings for quick generation
    
    ### 🎭 Voice Selection
    
    **Available Voices:**
    - **Indian English**: Neerja (Female), Prabhat (Male)
    - **American English**: Jenny (Female), Davis (Male)
    - **British English**: Mia (Female), Ryan (Male)
    - **Canadian English**: Clara (Female), Liam (Male)
    - **Australian English**: Tina (Female), Duncan (Male)
    
    ### 🔧 How it works
    
    1. **Input**: Provide your content via text or file upload
    2. **Script Generation**: AI creates an optimized script for voiceover
    3. **Voice Selection**: Choose from premium edge-tts voices with different accents
    4. **Voice Generation**: Convert the script to high-quality audio with default settings
    5. **Export**: Download your professional voiceover
    
    ### 💡 Use Cases
    
    - **Educational Content**: Create voiceovers for online courses
    - **Marketing Materials**: Generate audio for presentations and ads
    - **Podcast Intros**: Create professional introductions
    - **Accessibility**: Convert text content to audio format
    - **Content Creation**: Enhance videos and multimedia projects
    - **Audiobooks**: Convert written content to spoken format
    - **Global Reach**: Use different English accents for diverse audiences
    
    ### 🛠️ Technical Details
    
    - **AI Model**: GitHub Models (openai/gpt-4o) for script generation
    - **Voice Synthesis**: Microsoft edge-tts with optimized default settings
    - **Database**: SQLite for project management
    - **Framework**: Streamlit for user interface
    - **Audio Quality**: High-fidelity MP3 output
    - **Accent Support**: Multiple English variants for global appeal
    
    ### ⚡ Simplified Experience
    
    This version uses optimized default voice settings to provide:
    - **Faster Generation**: No need to adjust complex settings
    - **Consistent Quality**: Professionally tuned defaults
    - **Streamlined Workflow**: Focus on content, not configuration
    - **Reliable Results**: Tested settings for best performance
    - **Accent Variety**: Choose the right English accent for your audience
    
    ---
    
    Built with ❤️ for content creators who want professional results with minimal configuration and global appeal.
    """)



if __name__ == "__main__":
    main()
