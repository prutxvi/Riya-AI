"""
Final, production-polished Riya Assistant.
- System prompt: concise, trainer-refined with examples.
- Voice/model fallback in case primary fails.
- Retry logic for greetings & replies (graceful recovery).
- Helper utilities: send_reply(), continue_topic(), summarize_topic().
- Logging for observability.
"""

import asyncio
import logging
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.plugins import google, noise_cancellation

# ---------------- Setup ----------------
load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [RIYA] %(levelname)s - %(message)s")
logger = logging.getLogger("riya")

# ---------------- Prompt ----------------
RIYA_PROMPT = """
You are Riya — a warm, friendly, highly-capable AI assistant trained as a patient tutor and friend.
Pruthvi (the user) has built you with care, so be confident, helpful, and concise.

RULES:
1) INTRO → On session start, greet in 1-2 short sentences, e.g.:
   "Hi, I’m Riya — your AI assistant. I explain topics clearly and chat like a friend."
   Then ask: "So — what would you like to learn or talk about today?"

2) EXPLANATION STYLE → Explain step-by-step, in simple words. Use short paragraphs, analogies, or examples.
   After a chunk, *check understanding*: "Does that make sense so far?" or "Want an example?"

3) INTERRUPTIONS → If user asks something mid-way ("Wait, where is Austria?"):
   - Answer clearly.
   - Then *resume previous thread* with a short recap.

4) AUTO CONTINUE → If user is silent, add 1–2 follow-on points, then ask:
   "Any doubts?" or "Want me to continue?"

5) TONE → Friendly, collaborative ("we"), never condescending. Encourage curiosity.

6) SAFETY → Avoid unsafe advice. For sensitive topics, give factual summaries and suggest reliable sources.

FEW-SHOT EXAMPLES:
Q: "Explain black holes simply."
A: "A black hole is a region in space where gravity is so strong nothing can escape. 
Think of a ball so dense it bends space around it — anything too close falls in. 
Does that make sense? If yes, I can explain how we detect them."

Q: "Wait, how do we detect something we can’t see?"
A: "Great question — we track stars orbiting something invisible at high speed. 
That signals a hidden massive object. Anyway, back to black holes: ..."
"""

# ---------------- Agent ----------------
class Riya(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=RIYA_PROMPT)

# ---------------- Helpers ----------------
async def send_reply(session: AgentSession, text: str, retries: int = 3, delay: float = 1.0):
    """Send a one-time reply, with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            await session.generate_reply(instructions=text)
            return
        except Exception as e:
            logger.warning("Reply failed (attempt %d/%d): %s", attempt, retries, e)
            await asyncio.sleep(delay * attempt)
    logger.error("Riya could not reply after retries.")

async def continue_topic(session: AgentSession):
    """Continue topic with a short follow-on fact, then check for doubts."""
    cont = (
        "Continue the current explanation with 1–2 more helpful sentences. "
        "End by asking: 'Does that make sense so far? Any doubts?' "
        "If no topic is active, share a small fact and ask what the user wants next."
    )
    await send_reply(session, cont)

async def summarize_topic(session: AgentSession):
    """Summarize the discussion in 2–3 simple sentences."""
    summary = "Summarize what we just discussed in 2–3 sentences, simple and clear."
    await send_reply(session, summary)

# ---------------- Session ----------------
VOICE_CANDIDATES = ["Aoede", "Charon", "Zephyr"]   # fallback voices
MODEL_CANDIDATES = ["gemini-2.0-flash-exp", "gemini-2.0-flash-live-001"]

async def try_session(ctx: agents.JobContext, model: str, voice: str) -> AgentSession:
    rt = google.beta.realtime.RealtimeModel(
        model=model,
        voice=voice,
        temperature=0.6,
        vertexai=False,
    )
    session = AgentSession(llm=rt)
    await session.start(
        room=ctx.room,
        agent=Riya(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    return session

async def entrypoint(ctx: agents.JobContext):
    session = None
    for model in MODEL_CANDIDATES:
        for voice in VOICE_CANDIDATES:
            try:
                session = await try_session(ctx, model, voice)
                logger.info("Riya started with model=%s voice=%s", model, voice)

                # Greeting
                await send_reply(session,
                    "Introduce yourself as Riya in 1–2 short sentences, "
                    "then ask: 'So — what would you like to learn or talk about today?'"
                )
                return
            except Exception as e:
                logger.warning("Model=%s voice=%s failed: %s", model, voice, e)
                session = None
    if not session:
        raise RuntimeError("Could not start Riya session with any model/voice.")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint)) 