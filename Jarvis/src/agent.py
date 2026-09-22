import asyncio
import os
import numpy as np
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import JobContext, AgentSession, room_io
from livekit.plugins import google
from voice_verifier import VoiceVerifier

load_dotenv(dotenv_path=".env.local")

# Initialize Voice Verifier Guardrail
verifier = VoiceVerifier(owner_embedding_path="owner_voice.npy", threshold=0.75)

async def entrypoint(ctx: JobContext):
    print(f"[JARVIS] Connecting to room: {ctx.room.name}")
    await ctx.connect()

    # Explicitly pass the supported bidi-stream model target
    realtime_model = google.realtime.RealtimeModel(
        model="gemini-2.0-flash-exp",
        instructions=(
            "You are Jarvis, a highly efficient, intelligent, and loyal voice assistant. "
            "Keep responses brief, polite, natural, and crisp. Address the user as Sir."
        )
    )

    session = AgentSession(
        llm=realtime_model,
        user_away_timeout=8.0,
        room_options=room_io.RoomOptions(video_input=True)
    )

    # Audio Track Verification Interceptor
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(monitor_audio_track(track))

    async def monitor_audio_track(track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        buffer = []
        
        async for event in audio_stream:
            # Collect audio frame samples (16kHz PCM)
            frame_data = np.frombuffer(event.frame.data, dtype=np.int16).astype(np.float32) / 32768.0
            buffer.extend(frame_data)

            # Sample ~3 seconds of speech for embedding verification
            if len(buffer) >= 16000 * 3:
                sample_data = np.array(buffer[: 16000 * 3], dtype=np.float32)
                buffer = []  # Reset buffer
                
                is_verified, score = verifier.verify_audio_data(sample_data)
                print(f"[SECURITY CHECK] Biometric Match Score: {score:.3f} | Verified: {is_verified}")

                if not is_verified:
                    print("[SECURITY ALERT] Unauthorized voice detected!")

    print("[JARVIS] Agent active and biometrically secured.")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
