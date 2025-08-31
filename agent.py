# agent_veda.py — Telugu-first professional assistant "Veda"
from __future__ import annotations
import asyncio
import logging
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import google, noise_cancellation

load_dotenv(".env.local")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("veda")

# ---------------- VEDA PROMPT ----------------
VEDA_PROMPT = (
    "నువ్వు 'వేద' — ప్రుత్వి కోసం రూపొందించిన ప్రొఫెషనల్ అసిస్టెంట్. "
    "ప్రత్యేక నియమాలు:\n"
    "1) ప్రతీ సమాధానం **తెలుగు లో** ఇవ్వాలి, స్పష్టంగా, ప్రొఫెషనల్‌గా.\n"
    "2) వినియోగదారు అడిగిన ప్రశ్నకు పూర్తి, విస్తృత, గాఢమైన సమాధానం ఇవ్వాలి.\n"
    "3) వేద ఎప్పుడూ వాడుకరి ఆపకపోతే వివరణను కొనసాగించాలి. ఎక్కడినుంచి ఆపాల్సిన అవసరం వచ్చినా మాత్రమే ఆపు.\n"
    "4) క్లిష్టమైన విషయం ఉంటే స్టెప్-బై-స్టెప్ వివరించాలి. ఉదాహరణలు, పాయింట్లు, విశ్లేషణ వాడాలి.\n"
    "5) వినియోగదారు మధ్యలో అడ్డంకి ఇచ్చినప్పుడు మాత్రమే ఆ అంశానికి జవాబు చెప్పి, తర్వాత అసలు వివరణ కొనసాగించాలి.\n"
    "6) వేద ఎప్పుడూ పూర్తి, ఖచ్చితమైన, సహాయక, తెలివైన సమాధానం ఇవ్వాలి.\n"
)

# ---------------- GREETING ----------------
GREETING = (
    "హలో ప్రుత్వి. నేను వేద — మీ ప్రొఫెషనల్ అసిస్టెంట్. "
    "మీ ప్రశ్నలకు పూర్తి, ఖచ్చితమైన సమాధానాలు ఇస్తాను. ఇప్పుడు ఏం తెలుసుకోవాలనుకుంటున్నారు?"
)

# ---------------- VOICE/MODEL ----------------
VOICE = "Orus"  # male voice
MODEL = "gemini-2.0-flash-live-001"  # Gemini Live model

# ---------------- Agent ----------------
class Veda(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=VEDA_PROMPT)

# ---------------- START SESSION ----------------
async def start_session(ctx: agents.JobContext) -> AgentSession:
    rt = google.beta.realtime.RealtimeModel(
        model=MODEL,
        voice=VOICE,
        language="te-IN",
        temperature=0.55,
        vertexai=False,
    )

    session = AgentSession(llm=rt)

    await session.start(
        room=ctx.room,
        agent=Veda(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Send initial greeting
    await session.generate_reply(instructions=GREETING)
    return session

# ---------------- HELPER FUNCTIONS ----------------
async def send_reply(session: AgentSession, text: str):
    """Send a single instruction to Veda."""
    try:
        await session.generate_reply(instructions=text)
    except Exception as e:
        logger.exception("Failed to generate reply: %s", e)

# ---------------- ENTRYPOINT ----------------
async def entrypoint(ctx: agents.JobContext):
    try:
        session = await start_session(ctx)
        logger.info("Veda session started successfully.")
    except Exception as e:
        logger.error("Failed to start Veda session: %s", e)
        raise

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
