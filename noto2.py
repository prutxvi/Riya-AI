import asyncio
import logging
import os
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import google, noise_cancellation
from datetime import datetime
import PyPDF2

# Lightweight emotion detection (just keyword example for demo)
class EmotionalState:
    CONFUSED = "confused"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"
    SATISFIED = "satisfied"
    NEUTRAL = "neutral"
    keywords = {
        CONFUSED: ["confused", "unclear", "don't get", "lost"],
        EXCITED: ["awesome", "wow", "amazing"],
        FRUSTRATED: ["annoy", "frustrated", "difficult", "stuck"],
        SATISFIED: ["thanks", "great", "perfect"],
    }
    @classmethod
    def detect(cls, text):
        t = text.lower()
        for state, words in cls.keywords.items():
            if any(w in t for w in words):
                return state
        return cls.NEUTRAL

class NotoContext:
    def __init__(self):
        self.last_user = None
        self.last_topic = None
        self.emotion = EmotionalState.NEUTRAL
        self.turn = 0
        self.last_intro = False
        self.start_time = datetime.now()

load_dotenv(".env.local")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("noto")

PDF_PATH = "document.pdf"
VOICE = "Orus"
MODEL = "gemini-2.0-flash-live-001"
noto_context = NotoContext()

def extract_pdf_text(pdf_path: str) -> str:
    text = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
    return "\n".join(text)

def build_prompt(document_text, user_state, emotion):
    intro = (
        "You are Noto, an advanced emotional-intelligent PDF AI assistant for Pruthvi."
        f"\nYou have the content of document.pdf as context below.\n"
        "You are clear, concise, friendly, and adapt your style to the user's emotion."
        f"\nUser emotion: {emotion}\n"
        "ONLY answer with information from this PDF unless asked about agent features or the workflow."
        "\nHere is the PDF text:\n"
        "-------------------------\n"
        f"{document_text[:12000]}{'... [truncated]' if len(document_text)>12000 else ''}\n"
        "-------------------------\n"
    )
    return intro

class NotoAgent(Agent):
    def __init__(self, prompt):
        super().__init__(instructions=prompt)

async def start_session(ctx: agents.JobContext) -> AgentSession:
    document_text = extract_pdf_text(PDF_PATH)
    prompt = build_prompt(document_text, noto_context, noto_context.emotion)
    agent = NotoAgent(prompt)
    rt = google.beta.realtime.RealtimeModel(
        model=MODEL,
        voice=VOICE,
        language="en-US",
        temperature=0.5,
        vertexai=False,
    )
    session = AgentSession(llm=rt)
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    # Personalized intro, offers choice, only once per session
    intro = (
        "Hello! I am Noto, your advanced AI PDF assistant powered by Gemini and emotional intelligence. "
        "I've successfully loaded your file 'document.pdf'. Would you like a summary of the document, "
        "or do you have specific questions or doubts you'd like clarified? "
        "Just say 'summary' for an overview or ask anything, and I'll do my best to help!"
    )
    noto_context.last_intro = True
    await session.generate_reply(instructions=intro)
    return session

async def send_reply(session: AgentSession, text: str):
    """Smart, adaptive reply based on PDF and user emotional state."""
    noto_context.turn += 1
    emot = EmotionalState.detect(text)
    noto_context.emotion = emot
    instructions = ""
    # If the user says something like 'summary', auto-answer with summary
    if "summary" in text.lower():
        instructions = "Summarize the content of the document above in clear bullet points."
    elif text.strip() != "":
        instructions = (
            f"User question: '{text}'\n"
            "Answer using the PDF content. If the question is unclear, ask for clarification and offer to summarize the relevant section."
            f"\nAdjust tone to match if the user is {emot}."
        )
    else:
        instructions = (
            "The user didn't provide input or was silent. Ask if they want to continue discussing the PDF, "
            "offer to summarize, or if they have a question."
        )
    await session.generate_reply(instructions=instructions)

async def entrypoint(ctx: agents.JobContext):
    try:
        session = await start_session(ctx)
        logger.info("Noto session started successfully.")
        # LiveKit handles input loop, but you may add hooks here if you want to log/monitor further
    except Exception as e:
        logger.error(f"Failed to start Noto session: {e}")
        raise

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
