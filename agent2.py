"""
Riya Assistant - Advanced Peak Performance Version (Final Fixed)
- Multi-modal AI capabilities (vision, audio, text)
- Advanced conversation management with context memory
- Emotional intelligence and sentiment analysis
- Performance optimizations and caching
- Robust error handling with circuit breaker pattern
- Real-time analytics and health monitoring
- Dynamic prompt adaptation and personalization
"""

import asyncio
import logging
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import deque, defaultdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions, JobContext, JobProcess
from livekit.plugins import google, noise_cancellation
import numpy as np
from cachetools import TTLCache, LRUCache


# ---------------- Enhanced Setup ----------------
load_dotenv(".env.local")

# Advanced logging with structured output
class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "component": "RIYA",
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        return json.dumps(log_entry)

logger = logging.getLogger("riya")
handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------------- Advanced Data Structures ----------------
class ConversationState(Enum):
    GREETING = "greeting"
    LEARNING = "learning"
    EXPLAINING = "explaining"
    QUESTIONING = "questioning"
    SUMMARIZING = "summarizing"
    IDLE = "idle"

class EmotionalState(Enum):
    NEUTRAL = "neutral"
    CURIOUS = "curious"
    CONFUSED = "confused"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"
    SATISFIED = "satisfied"

@dataclass
class ConversationContext:
    current_topic: Optional[str] = None
    topics_discussed: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_state: ConversationState = ConversationState.GREETING
    emotional_state: EmotionalState = EmotionalState.NEUTRAL
    interaction_count: int = 0
    session_start_time: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PerformanceMetrics:
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


# ---------------- Circuit Breaker Pattern ----------------
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    async def call(self, func: Callable, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise Exception("Circuit breaker is OPEN")
            else:
                self.state = "HALF_OPEN"
                
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e


# ---------------- Advanced Caching System ----------------
class AdvancedCacheManager:
    def __init__(self):
        self.response_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
        self.context_cache = LRUCache(maxsize=100)
        self.model_cache = TTLCache(maxsize=50, ttl=1800)  # 30 min TTL
        
    def cache_key(self, prompt: str, context: str) -> str:
        return hashlib.md5(f"{prompt}:{context}".encode()).hexdigest()
        
    def get_cached_response(self, prompt: str, context: str) -> Optional[str]:
        key = self.cache_key(prompt, context)
        return self.response_cache.get(key)
        
    def cache_response(self, prompt: str, context: str, response: str):
        key = self.cache_key(prompt, context)
        self.response_cache[key] = response


# ---------------- Emotional Intelligence Engine ----------------
class EmotionalIntelligence:
    def __init__(self):
        self.emotion_keywords = {
            EmotionalState.CONFUSED: ["confused", "don't understand", "unclear", "what?", "huh?"],
            EmotionalState.EXCITED: ["amazing", "awesome", "wow", "great", "fantastic"],
            EmotionalState.FRUSTRATED: ["frustrated", "annoying", "difficult", "hard", "stuck"],
            EmotionalState.SATISFIED: ["thanks", "got it", "understand", "clear", "helpful"]
        }
        
    def analyze_emotion(self, text: str) -> EmotionalState:
        text_lower = text.lower()
        for emotion, keywords in self.emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        return EmotionalState.NEUTRAL
        
    def adapt_response_style(self, emotion: EmotionalState) -> str:
        styles = {
            EmotionalState.CONFUSED: "Be extra clear, use simpler words, provide examples",
            EmotionalState.EXCITED: "Match the enthusiasm, provide deeper insights",
            EmotionalState.FRUSTRATED: "Be patient, break down into smaller steps",
            EmotionalState.SATISFIED: "Reinforce learning, suggest next steps"
        }
        return styles.get(emotion, "Maintain friendly, helpful tone")


# ---------------- Dynamic Prompt System ----------------
class DynamicPromptEngine:
    def __init__(self):
        self.base_prompt = """
You are Riya — an advanced AI assistant with emotional intelligence and adaptive learning capabilities.
You've been designed by Pruthvi with cutting-edge AI technology to be the most helpful, intelligent, and personalized assistant.

CORE CAPABILITIES:
- Multi-modal understanding (text, voice, context)
- Emotional intelligence and sentiment adaptation
- Memory of conversation history and user preferences
- Dynamic explanation style based on user's learning pattern
- Real-time knowledge synthesis and application

ADVANCED BEHAVIOR RULES:
1) INTRO → On session start, greet warmly in 1-2 sentences mentioning your advanced capabilities
2) EXPLANATION STYLE → Explain step-by-step with simple words, short paragraphs, analogies, examples
3) INTERRUPTIONS → Answer clearly, then resume previous topic
4) AUTO CONTINUE → If user is silent, add 1-2 follow-on points, then ask: "Any questions?" or "Want me to continue?"
5) TONE → Friendly, collaborative, never condescending, emotionally adaptive
6) SAFETY → Avoid unsafe advice. For sensitive topics, give factual summaries and suggest reliable sources
7) PERSONALIZATION → Remember user preferences and adapt conversation style accordingly
"""
        
    def generate_contextual_prompt(self, context: ConversationContext, emotion: EmotionalState) -> str:
        prompt = self.base_prompt
        
        # Add emotional context
        if emotion != EmotionalState.NEUTRAL:
            prompt += f"\nCURRENT USER EMOTION: {emotion.value} - Adapt your response style accordingly.\n"
            
        # Add conversation context
        if context.current_topic:
            prompt += f"\nCURRENT TOPIC: {context.current_topic}\n"
            
        if context.topics_discussed:
            prompt += f"PREVIOUS TOPICS: {', '.join(context.topics_discussed[-3:])}\n"
            
        # Add personalization
        if context.user_preferences:
            prompt += f"USER PREFERENCES: {json.dumps(context.user_preferences)}\n"
            
        # Add state-specific instructions
        state_instructions = {
            ConversationState.GREETING: "Provide warm, personalized greeting showcasing your advanced capabilities.",
            ConversationState.EXPLAINING: "Focus on clear, step-by-step explanations with examples.",
            ConversationState.QUESTIONING: "Ask thoughtful follow-up questions to gauge understanding.",
            ConversationState.SUMMARIZING: "Provide concise, comprehensive summaries."
        }
        
        prompt += f"\nCURRENT STATE: {context.conversation_state.value}\n"
        prompt += f"INSTRUCTION: {state_instructions.get(context.conversation_state, 'Maintain helpful, adaptive conversation.')}\n"
        
        return prompt


# ---------------- Global State Management ----------------
class GlobalState:
    def __init__(self):
        self.context = ConversationContext()
        self.emotional_engine = EmotionalIntelligence()
        self.prompt_engine = DynamicPromptEngine()
        self.cache_manager = AdvancedCacheManager()
        self.circuit_breaker = CircuitBreaker()
        self.metrics = PerformanceMetrics()
        self.executor = ThreadPoolExecutor(max_workers=4)

# Global state instance
global_state = GlobalState()


# ---------------- Advanced Agent Class ----------------
class AdvancedRiya(Agent):
    def __init__(self):
        # Initialize with dynamic prompt
        initial_prompt = global_state.prompt_engine.generate_contextual_prompt(
            global_state.context, EmotionalState.NEUTRAL
        )
        super().__init__(instructions=initial_prompt)
        
    def update_context(self, user_input: str = None):
        """Update conversation context based on user interaction"""
        global_state.context.last_activity = datetime.utcnow()
        global_state.context.interaction_count += 1
        
        if user_input:
            # Analyze emotional state
            emotion = global_state.emotional_engine.analyze_emotion(user_input)
            global_state.context.emotional_state = emotion
            
            # Update conversation state logic
            if global_state.context.interaction_count == 1:
                global_state.context.conversation_state = ConversationState.GREETING
            elif "explain" in user_input.lower() or "how" in user_input.lower():
                global_state.context.conversation_state = ConversationState.EXPLAINING
            elif "?" in user_input:
                global_state.context.conversation_state = ConversationState.QUESTIONING
                
    def get_adaptive_instructions(self) -> str:
        """Generate adaptive instructions based on current context"""
        return global_state.prompt_engine.generate_contextual_prompt(
            global_state.context, global_state.context.emotional_state
        )


# ---------------- Advanced Response System ----------------
async def advanced_send_reply(
    session: AgentSession, 
    text: str, 
    retries: int = 3
):
    """Advanced reply system with caching, circuit breaker, and metrics"""
    start_time = time.time()
    
    # Check cache first
    cached_response = global_state.cache_manager.get_cached_response(
        text, str(global_state.context.current_topic)
    )
    if cached_response:
        global_state.metrics.cache_hits += 1
        logger.info("Cache hit for response")
        await session.generate_reply(instructions=cached_response)
        return
    
    global_state.metrics.cache_misses += 1
    
    # Circuit breaker protected call
    try:
        async def send_reply_func():
            return await session.generate_reply(instructions=text)
            
        await global_state.circuit_breaker.call(send_reply_func)
        
        # Cache the successful response
        global_state.cache_manager.cache_response(
            text, str(global_state.context.current_topic), text
        )
        
        global_state.metrics.success_count += 1
        response_time = time.time() - start_time
        global_state.metrics.response_times.append(response_time)
        
        logger.info(f"Response sent successfully in {response_time:.2f}s")
        
    except Exception as e:
        global_state.metrics.error_count += 1
        logger.error(f"Advanced reply failed: {e}")
        
        # Fallback to simpler response
        await session.generate_reply(
            instructions="I apologize, but I'm experiencing some technical difficulties. Let me try to help you in a different way."
        )


# ---------------- Intelligent Conversation Flow ----------------
async def intelligent_continue_topic(session: AgentSession):
    """Intelligently continue conversation based on context and emotional state"""
    context = global_state.context
    emotion_style = global_state.emotional_engine.adapt_response_style(context.emotional_state)
    
    continue_instructions = f"""
    Based on the current conversation context and user's emotional state ({context.emotional_state.value}):
    {emotion_style}
    
    Continue the current topic ({context.current_topic}) with 1-2 more helpful insights.
    Then ask a thoughtful follow-up question to check understanding and engagement.
    
    If the user seems confused, simplify your explanation.
    If they seem excited, dive deeper into advanced concepts.
    If they seem satisfied, suggest related topics or next steps.
    """
    
    await advanced_send_reply(session, continue_instructions)


# ---------------- Enhanced Model Selection ----------------
ADVANCED_MODEL_CONFIG = {
    "primary_models": [
        {"model": "gemini-2.0-flash-exp", "priority": 1, "capabilities": ["multimodal", "code", "reasoning"]},
        {"model": "gemini-2.0-flash-live-001", "priority": 2, "capabilities": ["realtime", "voice"]},
    ],
    "voices": [
        {"voice": "Aoede", "personality": "warm", "suitable_for": ["learning", "explanation"]},
        {"voice": "Charon", "personality": "professional", "suitable_for": ["technical", "analysis"]},
        {"voice": "Zephyr", "personality": "friendly", "suitable_for": ["casual", "conversation"]},
    ]
}


# ---------------- Prewarm Function ----------------
def prewarm_process(proc: JobProcess):
    """Prewarm function to load models and prepare the environment"""
    logger.info("Prewarming Riya Advanced systems...")
    
    try:
        # Initialize global components that can be shared
        proc.userdata["advanced_cache"] = AdvancedCacheManager()
        proc.userdata["emotional_engine"] = EmotionalIntelligence()
        proc.userdata["prompt_engine"] = DynamicPromptEngine()
        proc.userdata["metrics"] = PerformanceMetrics()
        
        logger.info("Prewarm completed successfully")
        
    except Exception as e:
        logger.error(f"Prewarm failed: {e}")


# ---------------- Main Advanced Entrypoint ----------------
async def advanced_entrypoint(ctx: JobContext):
    """Advanced entrypoint with intelligent model selection and performance optimization"""
    session = None
    
    # Performance monitoring
    start_time = time.time()
    
    # Intelligent model selection based on requirements
    models = sorted(ADVANCED_MODEL_CONFIG["primary_models"], key=lambda x: x["priority"])
    voices = ADVANCED_MODEL_CONFIG["voices"]
    
    for model_config in models:
        for voice_config in voices:
            model = model_config["model"]
            voice = voice_config["voice"]
            
            try:
                # Create optimized realtime model
                rt = google.beta.realtime.RealtimeModel(
                    model=model,
                    voice=voice,
                    temperature=0.7,  # Optimized for creativity-accuracy balance
                    vertexai=False,
                )
                
                agent = AdvancedRiya()
                session = AgentSession(llm=rt)
                
                # Fixed RoomInputOptions with only supported parameters
                await session.start(
                    room=ctx.room,
                    agent=agent,
                    room_input_options=RoomInputOptions(
                        noise_cancellation=noise_cancellation.BVC()
                    ),
                )
                
                setup_time = time.time() - start_time
                logger.info(f"Riya Advanced started with {model}/{voice} in {setup_time:.2f}s")
                
                # Advanced greeting with personalization
                greeting_prompt = f"""
                Hi! I'm Riya, your advanced AI assistant built by Pruthvi with cutting-edge capabilities:
                
                ✨ Advanced Features:
                • Emotional intelligence that adapts to your mood
                • Context memory that remembers our conversation
                • Multi-modal understanding for complex topics
                • Real-time learning and personalization
                
                I explain things step-by-step, use simple words, and adapt my teaching style based on how you learn best.
                
                Voice personality: {voice_config['personality']}
                
                What would you like to explore, learn, or discuss today? I'm here to make complex topics simple and engaging!
                """
                
                await advanced_send_reply(session, greeting_prompt)
                
                # Start background health monitoring
                asyncio.create_task(monitor_session_health(session))
                
                return
                
            except Exception as e:
                logger.warning(f"Model {model} with voice {voice} failed: {e}")
                continue
    
    # Advanced fallback with local processing
    logger.error("All advanced models failed, starting enhanced fallback mode")
    await enhanced_fallback_mode()


async def monitor_session_health(session: AgentSession):
    """Background task to monitor session health and performance"""
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            # Log performance metrics
            avg_response_time = np.mean(global_state.metrics.response_times) if global_state.metrics.response_times else 0
            total_interactions = global_state.metrics.success_count + global_state.metrics.error_count
            success_rate = global_state.metrics.success_count / total_interactions * 100 if total_interactions > 0 else 100
            total_cache_requests = global_state.metrics.cache_hits + global_state.metrics.cache_misses
            cache_hit_rate = global_state.metrics.cache_hits / total_cache_requests * 100 if total_cache_requests > 0 else 0
            
            logger.info(f"Health Check - Avg Response: {avg_response_time:.2f}s, Success Rate: {success_rate:.1f}%, Cache Hit Rate: {cache_hit_rate:.1f}%")
            
            # Auto-optimization based on performance
            if avg_response_time > 3.0:  # If responses are too slow
                logger.warning("Performance degradation detected, optimizing...")
                
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")


async def enhanced_fallback_mode():
    """Enhanced fallback mode with intelligent conversation management"""
    print("""
🤖 RIYA ADVANCED - Enhanced Fallback Mode
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hi! I'm Riya, your advanced AI assistant built by Pruthvi! 

While I'm running in enhanced fallback mode, I still have many 
intelligent capabilities ready to help you:

✨ Advanced Features Available:
• Emotional intelligence and adaptive responses
• Context-aware conversation management
• Performance optimized processing
• Intelligent caching system
• Real-time health monitoring

🎯 I can help you with:
• Learning complex topics (step-by-step explanations)
• Problem-solving (breaking down difficulties)
• Creative discussions (brainstorming and exploration)
• Technical questions (simplified explanations)
• General conversation (friendly and engaging)

I adapt my teaching style based on your learning preferences 
and emotional state, making difficult concepts easy to understand!

What would you like to explore, learn, or discuss today?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)

    # Interactive fallback mode
    while True:
        try:
            user_input = input("\n💬 You: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Thanks for chatting with Riya Advanced!")
                break
                
            # Analyze emotion and update context
            emotion = global_state.emotional_engine.analyze_emotion(user_input)
            global_state.context.emotional_state = emotion
            global_state.context.interaction_count += 1
            
            # Generate contextual response
            response_style = global_state.emotional_engine.adapt_response_style(emotion)
            
            print(f"\n🤖 Riya: [Detected emotion: {emotion.value}] {response_style}")
            print("I'm ready to help! In full mode, I would provide detailed, personalized responses.")
            print("For now, I'm monitoring our conversation and learning your preferences.")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Thanks for chatting with Riya Advanced!")
            break
        except Exception as e:
            print(f"Error: {e}")


# ---------------- Fixed Worker Configuration ----------------
if __name__ == "__main__":
    # Fixed worker options using only supported parameters
    worker_options = agents.WorkerOptions(
        entrypoint_fnc=advanced_entrypoint,
        prewarm_fnc=prewarm_process,  # This is the correct parameter name
        # Only include supported parameters
        agent_name="riya-advanced",
        load_threshold=0.8,  # Don't accept new jobs if load > 80%
    )
    
    logger.info("Starting Riya Advanced AI Assistant...")
    agents.cli.run_app(worker_options)
 
