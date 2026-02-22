"""
Streaming Speech-to-Text Client for F1 Telemetry Dashboard.

Uses IBM Watson Speech-to-Text WebSocket API for real-time streaming
transcription with low latency. Transcribes audio while the user is
still speaking, providing interim results for faster response times.

Key features:
- WebSocket streaming (vs batch HTTP requests)
- Interim results for real-time feedback
- Callback-based architecture for async integration
- ~500-1000ms latency savings over batch approach
"""

import asyncio
import json
import logging
import base64
from typing import Callable, Optional, Union
from urllib.parse import urlencode

import numpy as np

logger = logging.getLogger(__name__)


class StreamingSTTClient:
    """
    Watson Speech-to-Text streaming client using WebSocket API.

    This client streams audio in real-time to Watson STT and receives
    transcription results as they become available, including interim
    (partial) results for lower latency.

    Usage:
        client = StreamingSTTClient(api_key="...", service_url="...")
        client.on_transcript(lambda text: print(f"Final: {text}"))
        client.on_partial(lambda text: print(f"Partial: {text}"))

        await client.start_streaming()
        await client.send_audio(audio_bytes)
        await client.stop_streaming()

    Attributes:
        api_key: IBM Watson API key
        service_url: Watson STT service URL
        model: STT model to use (default: en-US_BroadbandModel)
        sample_rate: Audio sample rate in Hz (default: 16000)
        interim_results: Whether to receive partial transcripts (default: True)
    """

    def __init__(
        self,
        api_key: str,
        service_url: str,
        model: str = "en-US_BroadbandModel",
        sample_rate: int = 16000,
        interim_results: bool = True
    ):
        """
        Initialize StreamingSTTClient.

        Args:
            api_key: IBM Watson API key
            service_url: Watson STT service URL (https://...)
            model: STT model (default: en-US_BroadbandModel)
            sample_rate: Audio sample rate in Hz (default: 16000)
            interim_results: Enable interim results for low latency (default: True)
        """
        self.api_key = api_key
        self.service_url = service_url.rstrip("/")
        self.model = model
        self.sample_rate = sample_rate
        self.interim_results = interim_results

        # Callbacks
        self._on_transcript_callback: Optional[Callable[[str], None]] = None
        self._on_partial_callback: Optional[Callable[[str], None]] = None
        self._on_error_callback: Optional[Callable[[str], None]] = None

        # State
        self._is_streaming = False
        self._websocket = None
        self._receive_task: Optional[asyncio.Task] = None

        logger.info(
            f"StreamingSTTClient initialized: model={model}, "
            f"sample_rate={sample_rate}, interim_results={interim_results}"
        )

    @property
    def is_streaming(self) -> bool:
        """Whether the client is currently streaming."""
        return self._is_streaming

    def on_transcript(self, callback: Callable[[str], None]) -> "StreamingSTTClient":
        """
        Register callback for final transcripts.

        Args:
            callback: Function called with final transcript text

        Returns:
            Self for method chaining
        """
        self._on_transcript_callback = callback
        return self

    def on_partial(self, callback: Callable[[str], None]) -> "StreamingSTTClient":
        """
        Register callback for partial (interim) transcripts.

        Args:
            callback: Function called with partial transcript text

        Returns:
            Self for method chaining
        """
        self._on_partial_callback = callback
        return self

    def on_error(self, callback: Callable[[str], None]) -> "StreamingSTTClient":
        """
        Register callback for errors.

        Args:
            callback: Function called with error message

        Returns:
            Self for method chaining
        """
        self._on_error_callback = callback
        return self

    def _get_websocket_url(self) -> str:
        """
        Construct WebSocket URL for Watson STT.

        Returns:
            WebSocket URL with model parameter
        """
        # Convert https:// to wss://
        base_url = self.service_url.replace("https://", "wss://")
        base_url = base_url.replace("http://", "ws://")

        # Build query parameters
        params = urlencode({"model": self.model})

        return f"{base_url}/v1/recognize?{params}"

    def _get_start_message(self) -> dict:
        """
        Get the start message to send to Watson after connection.

        Returns:
            Start message dict with audio configuration
        """
        return {
            "action": "start",
            "content-type": f"audio/l16;rate={self.sample_rate}",
            "interim_results": self.interim_results,
            "inactivity_timeout": -1,  # No timeout
        }

    async def _connect_websocket(self) -> None:
        """
        Connect to Watson STT WebSocket.

        Raises:
            RuntimeError: If connection fails
        """
        try:
            import websockets
            from websockets.client import connect

            ws_url = self._get_websocket_url()

            # Create auth header
            auth_string = f"apikey:{self.api_key}"
            auth_bytes = base64.b64encode(auth_string.encode()).decode()

            headers = {
                "Authorization": f"Basic {auth_bytes}"
            }

            logger.info(f"Connecting to Watson STT WebSocket: {ws_url}")

            self._websocket = await connect(
                ws_url,
                additional_headers=headers,
                ping_interval=30,
                ping_timeout=10
            )

            # Send start message
            start_msg = self._get_start_message()
            await self._websocket.send(json.dumps(start_msg))

            logger.info("Watson STT WebSocket connected")

        except ImportError:
            raise RuntimeError(
                "websockets package required for streaming STT. "
                "Install with: pip install websockets"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Watson STT: {e}")
            raise RuntimeError(f"WebSocket connection failed: {e}")

    async def start_streaming(self) -> None:
        """
        Start streaming session.

        Connects to Watson STT WebSocket and begins receiving transcripts.
        """
        if self._is_streaming:
            logger.warning("Already streaming, ignoring start request")
            return

        await self._connect_websocket()
        self._is_streaming = True

        # Start background task to receive messages
        self._receive_task = asyncio.create_task(self._receive_loop())

        logger.info("Streaming session started")

    async def stop_streaming(self) -> None:
        """
        Stop streaming session.

        Sends stop message and closes WebSocket connection.
        """
        if not self._is_streaming:
            return

        self._is_streaming = False

        # Send stop message
        if self._websocket:
            try:
                await self._websocket.send(json.dumps({"action": "stop"}))
                await self._websocket.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")

        # Cancel receive task
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        self._websocket = None
        self._receive_task = None

        logger.info("Streaming session stopped")

    async def send_audio(self, audio: Union[bytes, np.ndarray]) -> None:
        """
        Send audio chunk to Watson STT.

        Args:
            audio: Audio data as bytes or numpy int16 array

        Raises:
            RuntimeError: If not currently streaming
        """
        if not self._is_streaming:
            raise RuntimeError("Cannot send audio: not streaming")

        # Convert numpy array to bytes if needed
        if isinstance(audio, np.ndarray):
            audio = audio.astype(np.int16).tobytes()

        await self._websocket.send(audio)

    async def _receive_loop(self) -> None:
        """Background task to receive and process WebSocket messages."""
        try:
            async for message in self._websocket:
                if isinstance(message, str):
                    data = json.loads(message)
                    self._handle_message(data)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
            if self._on_error_callback:
                self._on_error_callback(str(e))

    def _handle_message(self, message: dict) -> None:
        """
        Handle incoming WebSocket message.

        Args:
            message: Parsed JSON message from Watson
        """
        # Check for error
        if "error" in message:
            error_msg = message.get("error", "Unknown error")
            logger.error(f"Watson STT error: {error_msg}")
            if self._on_error_callback:
                self._on_error_callback(error_msg)
            return

        # Check for results
        results = message.get("results", [])
        if not results:
            return

        for result in results:
            alternatives = result.get("alternatives", [])
            if not alternatives:
                continue

            transcript = alternatives[0].get("transcript", "").strip()
            if not transcript:
                continue

            is_final = result.get("final", False)

            if is_final:
                logger.info(f"[STT] Final transcript: {transcript}")
                if self._on_transcript_callback:
                    self._on_transcript_callback(transcript)
            else:
                logger.debug(f"[STT] Partial transcript: {transcript}")
                if self._on_partial_callback:
                    self._on_partial_callback(transcript)
