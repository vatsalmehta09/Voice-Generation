# 🎙️ AI Voiceover Generator with Microsoft edge-tts

Transform text and documents into professional-quality voiceovers using AI-powered script generation and Microsoft Edge TTS voices. This app is built with Streamlit and is perfect for educators, content creators, marketers, and anyone looking to quickly create realistic, engaging audio from written content.

---

## ✨ Features

- **Diverse Voice Selection:** Choose from Indian, US, UK, Canadian, and Australian English voices (both male and female).
- **AI-Powered Script Generation:** Generate engaging, natural-sounding scripts from any text or supported document file using an OpenAI language model (via GitHub Models API).
- **Multiple Input Methods:** Paste your text or upload files (TXT, PDF, DOCX/DOC, CSV, XLSX/XLS, JSON, Markdown).
- **Flexible Duration Control:** Select default short/medium/long durations or specify a custom duration (15s to 5min).
- **Project Management:** Save, view, edit, and delete voiceover projects. All projects are stored and managed using SQLite.
- **Voice Preview:** Quickly preview what each voice sounds like.
- **High-Quality Audio Output:** Audio is generated using Microsoft edge-tts with optimized settings and can be downloaded as a high-fidelity MP3 file.
- **Streamlit UI:** Clean, intuitive, and responsive web interface.
- **Secure:** API keys and tokens are managed through environment variables; sensitive files ignored via `.gitignore`.

---

## 🚀 Installation

### 1. Clone the Repository

git clone https://github.com/vatsalmehta09/AI-Voice-Generation-edge-tts.git

### 2. Create and Activate a Virtual Environment (Recommended)

On Unix/macOS:
python3 -m venv venv
source venv/bin/activate

On Windows:
python -m venv venv
venv\Scripts\activate


### 3. Install Dependencies

pip install -r requirements.txt

OR

pip install streamlit openai PyPDF2 python-docx pandas edge-tts openpyxl xlrd asyncio-throttle pathlib2 python-dotenv loguru typing-extensions requests


---

## 🔑 Environment Variables

Create a `.env` file in your root directory:

GITHUB_TOKEN=#your_github_models_api_token
GITHUB_BASE_URL=https://models.github.ai/inference # or your custom base url if different


- `AUDIO_OUTPUT_DIR`, `UPLOAD_DIR`, `DATABASE_PATH` can be changed in the code if you want a different location.

---
##🎤 Available Voice Models
This application uses the following Microsoft Edge-TTS voices for professional-quality audio synthesis. All voices are pre-configured in the app for quick selection and consistent results.

Voice Name	Gender	Accent/Description	Model ID
Neerja	Female	Indian English accent	en-IN-NeerjaNeural
Prabhat	Male	Indian English accent	en-IN-PrabhatNeural
Jenny	Female	US accent, popular, natural	en-US-JennyNeural
Davis	Male	US accent, clear, authoritative	en-US-DavisNeural
Mia	Female	British accent, clear, prof.	en-GB-MiaNeural
Ryan	Male	British accent, classic	en-GB-RyanNeural
Clara	Female	Canadian accent	en-CA-ClaraNeural
Liam	Male	Canadian accent	en-CA-LiamNeural
Duncan	Male	Australian, professional	en-AU-DuncanNeural
Tina	Female	Australian, professional	en-AU-TinaNeural


## 🏃 Usage

1. **Start the app:**

streamlit run app.py (to run the app in terminal)

2. **Navigate to the web interface:**  
Open your browser and go to: [http://localhost:8501](http://localhost:8501)
3. **Create a project:**  
- Enter a project title.
- Choose input method (text or file upload).
- Set script and voiceover options.
- Generate the script, review/edit it.
- Select a voice and generate/download audio.
4. **Browse previous projects and manage or preview them from the sidebar.**

---

## 📂 Supported File Formats

- **Text:** `.txt`, `.md`
- **PDF:** `.pdf`
- **Word:** `.docx`, `.doc`
- **Spreadsheets:** `.csv`, `.xlsx`, `.xls`
- **Structured data:** `.json`

---

## 🌍 Supported Languages & Voices

- **Inputs supported in:** English, Spanish, French, German, Italian, Portuguese, Russian, Japanese, Korean, Chinese, Arabic, Hindi, Polish, Dutch.
- **Voice synthesis:** Indian, American, British, Canadian, and Australian English voices. (See the "Voice Gallery" in the app for options.)

---

## ⚙️ Project Structure

app.py
requirements.txt
README.md
.env # (not committed) API keys and confidential configs
audio_files/ # Generated audio files (auto-created)
uploads/ # Uploaded input documents (auto-created)
voiceover_db.sqlite # SQLite database for projects (auto-created)


---

## 🛑 .gitignore Example

Python
pycache/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

Streamlit
.streamlit/

Database
*.sqlite
*.db

Audio files
audio_files/
*.mp3
*.wav

Uploads
uploads/

Environment variables
.env
.env.local

IDE
.vscode/
.idea/
*.swp
*.swo

OS
.DS_Store
Thumbs.db

---

## 📝 Customization Tips

- **Add voices:** Open `app.py` and modify `Config.DEFAULT_VOICES` to include new edge-tts voices.
- **Add language support:** Update `Config.SUPPORTED_LANGUAGES` in code.
- **Tweak AI prompts or durations:** Edit the prompt templates in `VoiceoverGenerator` within `app.py`.

---

## 🛠️ Troubleshooting

- **`sqlite3` errors:** Make sure you're running Python 3.7+ with the full standard library.
- **Dependency errors:** Check your Python and pip versions; upgrade pip if needed (`pip install --upgrade pip`).
- **GitHub API/Model errors:** Confirm your API token and URL are correct and that you have access to the model.

---

## 📄 License

# Put your license or remove if none.
MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Credits

- [Streamlit](https://streamlit.io/) – UI framework
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Microsoft edge-tts](https://github.com/ranyelhousieny/edge-tts)
- [PyPDF2](https://github.com/py-pdf/PyPDF2), [python-docx](https://github.com/python-openxml/python-docx), and all other libraries listed in `requirements.txt`.

---

For questions, support, or collaboration, please open an issue on GitHub or contact vatsal9mehta@gmail.com


