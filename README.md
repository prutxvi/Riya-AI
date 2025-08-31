
# Veda: Telugu AI Assistant for Students & Learners

Veda is a professional, Telugu-first AI assistant designed to help students, teachers, and users understand any topic in a friendly, clear, and engaging way. Veda uses advanced AI models and natural Telugu conversation to make learning easy and enjoyable.

---

## Features

- **Telugu-first experience:** All explanations, greetings, and interactions are in clear, simple Telugu.
- **Topic explanation:** Ask about any subject—science, math, history, technology, or general knowledge—and get step-by-step, easy-to-understand answers.
- **Analogies & examples:** Veda uses real-world analogies and examples to make complex topics simple.
- **Student-friendly:** Designed for all ages, especially students preparing for exams or seeking homework help.
- **Voice & text modes:** Switch between text and audio modes for a personalized experience.
- **Noise cancellation:** Enhanced audio clarity using advanced noise cancellation.
- **Powered by Google Gemini & LiveKit:** Fast, reliable, and secure AI backend.

---

## Getting Started

### 1. Clone the repository
```sh
git clone https://github.com/yourusername/veda-telugu-ai-assistant.git
cd veda-telugu-ai-assistant
```

### 2. Install dependencies
You can use pip or uv:
```sh
pip install -r requirements.txt
# or
uv pip install -r requirements.txt
```

### 3. Set up environment variables
Copy the example file and fill in your credentials:
```sh
cp .env.local.example .env.local
# Edit .env.local and add your LiveKit and Google API keys
```

### 4. Run the assistant
```sh
uv run agent.py console
```

---

## Usage

- When you start Veda, you’ll see a greeting in Telugu.
- Type or speak your question or topic (e.g., "జ్యామితి గురించి వివరించండి" / "Explain geometry").
- Veda will respond in Telugu, breaking down the topic with analogies and examples.
- You can ask follow-up questions, request more details, or switch between text/audio modes.

---

## Project Structure

- `agent.py` — Main Telugu AI assistant code
- `.env.local.example` — Example environment file
- `.gitignore` — Ignore secrets, cache, and build files
- `requirements.txt` — Python dependencies
- `pyproject.toml` — Project metadata
- `README.md` — Project documentation

---

## Troubleshooting

- **API errors:** Double-check your API keys in `.env.local`.
- **Audio issues:** Make sure your microphone and speakers are working and permissions are granted.
- **Dependency issues:** Run `pip install -r requirements.txt` again if you see missing package errors.
- **Duplicate tool errors:** Ensure you are not running multiple agent instances or duplicate tool names in your code.

---

## Contributing

Pull requests, feature suggestions, and bug reports are welcome! Please open an issue or PR on GitHub.

---

## License

MIT — Free for personal and commercial use. See LICENSE file for details.
