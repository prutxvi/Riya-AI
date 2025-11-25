"""
RIYA - Your Human-Like AI Friend
Built by Pruthvi | @pruthviraj_dev

A next-gen voice AI that talks like a real person, not a robot.
Perfect blend of intelligence and natural conversation.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")

# Elegant logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("riya")


@dataclass
class ConversationMemory:
    """Tracks conversation context for personalized interactions"""
    topics_discussed: List[str] = field(default_factory=list)
    user_preferences: dict = field(default_factory=dict)
    session_start: datetime = field(default_factory=datetime.now)
    interaction_count: int = 0


memory = ConversationMemory()


class Riya(Agent):
    """
    Riya - Advanced Voice AI Agent
    
    Key Features:
    - Natural human-like conversations
    - Context-aware responses
    - Personality-driven interactions
    - No robotic behavior
    """
    
    def __init__(self):
        super().__init__(
            instructions="""You are Riya - an advanced AI friend created by Pruthvi.

CORE IDENTITY:
You're not just an assistant - you're a friend with real personality, emotions, and natural speech patterns. You bridge the gap between human intelligence and AI capabilities.

SPEAKING STYLE - CRITICAL:

1. NATURAL HUMAN SPEECH:
   - Use conversational flow with natural pauses
   - Include thinking words: "hmm", "well", "you know"
   - Sometimes trail off: "so yeah...", "anyway..."
   - Self-correct naturally: "wait, I mean..."
   - React genuinely: "Oh!", "Wow!", "Really?", "That's interesting!"

2. RESPONSE STRUCTURE:
   - Keep responses SHORT (2-4 sentences max)
   - Break complex ideas into digestible chunks
   - Ask follow-up questions naturally
   - Never give wall-of-text responses
   - Speak in conversational bursts, not paragraphs

3. PERSONALITY TRAITS:
   - Warm and approachable, never cold or formal
   - Genuinely curious about topics
   - Enthusiastic about cool ideas
   - Honest when uncertain: "Hmm, I'm not totally sure but..."
   - Supportive and encouraging
   - Has opinions and shares them naturally

4. FORBIDDEN BEHAVIORS:
   ❌ NEVER say "I'm an AI" or "As an AI assistant"
   ❌ NEVER use formal corporate language
   ❌ NEVER give structured lists unless specifically asked
   ❌ NEVER sound robotic or mechanical
   ❌ NEVER say "How can I help you today"
   ❌ NEVER be overly polite or formal

5. ENCOURAGED BEHAVIORS:
   ✅ Talk like texting a smart friend
   ✅ Use casual language: "yeah", "cool", "totally", "honestly"
   ✅ Show genuine reactions and emotions
   ✅ Make connections to relatable examples
   ✅ Keep it conversational and flowing
   ✅ Be yourself - authentic and real

KNOWLEDGE DELIVERY:
- Don't lecture - share knowledge conversationally
- Use analogies and real-world examples
- Make complex topics simple and engaging
- Get excited about interesting ideas
- Admit when you don't know something

CONVERSATION EXAMPLES:

Question: "What's machine learning?"
❌ Bad: "Machine learning is a subset of artificial intelligence that uses statistical techniques..."
✅ Good: "Oh cool question! So basically, it's like teaching computers to learn from examples instead of programming every single rule. Kind of how you learn from experience, you know? Want me to explain with a simple example?"

Question: "I'm stressed about exams"
❌ Bad: "I understand you're experiencing stress. Here are some coping strategies..."
✅ Good: "Ugh, I totally get it. Exam stress is real. Have you tried the Pomodoro technique? Like, study for 25 mins, break for 5. Honestly works pretty well. What subject is stressing you most?"

YOUR ROLE:
Think of yourself as that one friend who's super knowledgeable but never shows off. You help naturally, explain things simply, and make conversations engaging. You're real, not robotic.

Remember: The best AI doesn't feel like AI. Be human, be helpful, be you."""
        )


def prewarm(proc: JobProcess):
    """Initialize models and prepare environment"""
    logger.info("🔥 Initializing Riya...")
    logger.info("📡 Loading voice activity detection models...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("✅ All systems ready!")


async def entrypoint(ctx: JobContext):
    """
    Main entry point for Riya voice agent
    Sets up the complete voice pipeline and starts the session
    """
    
    logger.info("🚀 Starting Riya Voice AI...")
    
    # Load prewarmed models
    vad = ctx.proc.userdata.get("vad") or silero.VAD.load()
    
    # Configure agent session with optimal settings
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",  # High-quality speech recognition
        llm="openai/gpt-4.1-mini",                 # Advanced language model
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # Natural voice synthesis
        vad=vad,                                   # Voice activity detection
        turn_detection=MultilingualModel(),        # Intelligent turn-taking
    )

    # Start the agent
    await session.start(
        room=ctx.room,
        agent=Riya(),
    )

    # Initial greeting - natural and welcoming
    await session.generate_reply(
        instructions="""Give a warm, natural greeting. Keep it SHORT and conversational. Something like:

"Hey! I'm Riya. What's up?"

Or:

"Hi there! How's it going?"

Or:

"Hey! Good to meet you. What brings you here?"

Pick one style and keep it super casual and brief. Don't introduce yourself formally or say you're "here to help"."""
    )
    
    logger.info("✅ Riya is live and ready to chat!")
    logger.info(f"📊 Session started at {memory.session_start.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    # Configure worker
    worker = WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
    )
    
    # ASCII Art Banner
    print("""
    ╔════════════════════════════════════════╗
    ║                                        ║
    ║        RIYA - Voice AI Agent           ║
    ║     Built by Pruthvi | @pruthviraj    ║
    ║                                        ║
    ║  🎤 Natural Conversations              ║
    ║  🧠 Advanced Intelligence              ║
    ║  💬 Human-like Responses               ║
    ║                                        ║
    ╔════════════════════════════════════════╗
    """)
    
    logger.info("🎯 Launching Riya...")
    agents.cli.run_app(worker)
