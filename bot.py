#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example with WS audio input.

- Audio IN: WebSocket (PCM16 LE mono @ 16 kHz) from ESP32
- Audio OUT: WebRTC (unchanged), rendered to the browser

Run:
    pip install websockets
    python bot.py
"""

import os
import asyncio

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading AI models (30-40 seconds first run, <2 seconds after)\n")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")
logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

logger.info("✅ Pipeline components loaded")

logger.info("Loading WebRTC transport...")
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

logger.info("✅ All components loaded successfully!")

from fruit_inventory_tools import tools, register_fruit_functions
from pipecat.processors.filters.wake_check_filter import WakeCheckFilter
from pipecat.frames.frames import StartFrame, InputAudioRawFrame, EndFrame

# NEW: lightweight WS server for audio input
import websockets
from websockets.server import WebSocketServerProtocol

# ---- Config for WS audio input ----
WS_AUDIO_HOST = os.environ.get("WS_AUDIO_HOST", "0.0.0.0")
WS_AUDIO_PORT = int(os.environ.get("WS_AUDIO_PORT", 8765))
INPUT_SAMPLE_RATE = int(os.environ.get("INPUT_SAMPLE_RATE", "16000"))  # ESP32 audio SR

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # Wake word filter (acts on transcriptions)
    wake_filter = WakeCheckFilter(
        wake_phrases=["Robin"],
        keepalive_timeout=15,
    )
    logger.info("Wake word filter initialized")

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1")

    # Tools
    register_fruit_functions(llm)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly AI assistant. Respond naturally and keep your answers conversational. "
                "Be aware that everything you say is being read out loud. You are a fruit inventory assistant. "
                "You can check the fruit inventory by calling the check_fruit_inventory function and update quantities "
                "using the update_fruit_inventory function. When users tell you about inventory changes "
                "(like 'we have only 2 bananas left' or 'we now have 10 apples'), use the update function to adjust the inventory. "
                "Keep your style good for a voice assistant, saying for example, we have three apples, five bananas, and three pears in this kind of order. "
                "Answer the questions concisely and don't suggest whether the user needs help with anything else."
            ),
        },
    ]

    context = OpenAILLMContext(messages, tools=tools)
    context_aggregator = llm.create_context_aggregator(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # --- PIPELINE ---
    # IMPORTANT: we remove transport.input() and feed audio via WebSocket handler below.
    pipeline = Pipeline(
        [
            # transport.input(),  # <-- REMOVED (we provide frames from WS)
            rtvi,                           # RTVI processor
            stt,                            # Speech to text
            wake_filter,                    # Wake word filter on transcriptions
            context_aggregator.user(),      # User responses
            llm,                            # LLM
            tts,                            # TTS
            transport.output(),             # Bot audio out (WebRTC)
            context_aggregator.assistant(), # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    # -------- WebSocket audio handler (PCM16 LE mono @ INPUT_SAMPLE_RATE) --------
    from pipecat.frames.frames import StartFrame, InputAudioRawFrame, EndFrame

    async def audio_ws_handler(ws):
        path = getattr(ws, "path", "/")
        logger.info(f"[WS-Audio] Client connected from {getattr(ws, 'remote_address', None)}, path={path}")

        # 1) Старт потока: ОДИН раз на подключение
        await task.queue_frame(
            StartFrame(
                audio_in_sample_rate=INPUT_SAMPLE_RATE,
                audio_out_sample_rate=INPUT_SAMPLE_RATE,
            )
        )

        try:
            async for data in ws:  # ждём bytes: int16 LE mono @ INPUT_SAMPLE_RATE
                if isinstance(data, (bytes, bytearray)) and data:
                    # 2) Аудио-чанки
                    await task.queue_frame(
                        InputAudioRawFrame(audio=bytes(data), sample_rate=INPUT_SAMPLE_RATE, num_channels=1)
                    )
        except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError) as e:
            logger.info(f"[WS-Audio] Closed: {e}")
        finally:
            # 3) Конец потока
            await task.queue_frame(EndFrame())
            logger.info("[WS-Audio] Stream ended")

    # Start WS server (runs alongside the pipeline runner)
    ws_server = await websockets.serve(audio_ws_handler, WS_AUDIO_HOST, WS_AUDIO_PORT, max_size=None, compression=None,
        origins=None)
    logger.info(f"[WS-Audio] Listening on ws://{WS_AUDIO_HOST}:{WS_AUDIO_PORT}")

    # ---- Optional: transport events (unchanged) ----
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick the pipeline so Agent leaves CONNECTING even before ESP WS connects
        await task.queue_frame(StartFrame(
            audio_in_sample_rate=INPUT_SAMPLE_RATE,
            audio_out_sample_rate=INPUT_SAMPLE_RATE,
        ))
        logger.info("WebRTC client connected")
        logger.info("Waiting for wake word 'Hey Chat' to activate...")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("WebRTC client disconnected")
        await task.cancel()

    # Run pipeline task (blocks until completion). WS server lives in the same loop.
    await runner.run(task)

    # Cleanup (if we ever exit runner)
    ws_server.close()
    await ws_server.wait_closed()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""
    # Keep WebRTC transport for OUTPUT only. Disable audio_in to avoid mic/browser input.
    transport = SmallWebRTCTransport(
        params=TransportParams(
            audio_in_enabled=False,  # <--- was True; we now feed audio from WS
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
        webrtc_connection=runner_args.webrtc_connection,
    )

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
