import asyncio
import audioop
import base64
import json
import logging
import os
import threading
import time
from urllib.parse import quote, urlparse, urlunparse

import websockets

logger = logging.getLogger(__name__)

OPENAI_REALTIME_SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2  # 16-bit PCM
CHANNELS = 1  # mono
OPENAI_APPEND_FRAME_BYTES = int(OPENAI_REALTIME_SAMPLE_RATE * SAMPLE_WIDTH * 0.1)  # 100ms


class OpenAIStreamingTranscriber:
    def __init__(
        self,
        *,
        openai_api_key,
        model,
        sample_rate,
        metadata=None,
        language=None,
        prompt=None,
        save_utterance_callback=None,
        max_retry_time=120,
    ):
        self.openai_api_key = openai_api_key
        self.model = model
        self.sample_rate = sample_rate
        self.metadata = metadata or {}
        self.language = language
        self.prompt = prompt
        self.save_utterance_callback = save_utterance_callback
        self.max_retry_time = max_retry_time

        self._participant_name = self.metadata.get("participant_full_name", "Unknown")
        self._resampler_state = None
        self._audio_buffer = bytearray()

        self.last_send_time = time.time()

        self._loop = None
        self._loop_thread = None
        self._send_queue = None
        self._ws_connection = None
        self._sender_task = None
        self._receiver_task = None

        self.connected = False
        self.reconnecting = True
        self.should_stop = False

        self.ws_url = self._realtime_ws_url()
        self._start_event_loop()

    def _realtime_ws_url(self):
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")

        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = parsed.path.rstrip("/")
        if not path:
            path = "/v1"

        realtime_path = f"{path}/realtime"
        query = f"model={quote(self.model)}"
        return urlunparse((scheme, parsed.netloc, realtime_path, "", query, ""))

    def _start_event_loop(self):
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True, name="openai-event-loop")
        self._loop_thread.start()

        time.sleep(0.1)
        asyncio.run_coroutine_threadsafe(self._connect(), self._loop)

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    async def _connect(self):
        attempt = 0
        start_time = time.time()
        backoff_seconds = [1, 2, 4, 8]

        while not self.should_stop:
            elapsed = time.time() - start_time
            if elapsed >= self.max_retry_time:
                logger.error(f"[{self._participant_name}] OpenAI realtime connection timed out after {self.max_retry_time}s")
                self.reconnecting = False
                return

            attempt += 1
            try:
                logger.info(f"[{self._participant_name}] Connecting to OpenAI realtime (attempt {attempt})")
                additional_headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "OpenAI-Beta": "realtime=v1",
                }

                async with websockets.connect(
                    self.ws_url,
                    additional_headers=additional_headers,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=16 * 1024 * 1024,
                ) as ws:
                    self._ws_connection = ws
                    self.connected = True
                    self.reconnecting = False
                    self._send_queue = asyncio.Queue()

                    await self._send_session_update()

                    self._receiver_task = asyncio.create_task(self._receiver_loop())
                    self._sender_task = asyncio.create_task(self._sender_loop())

                    await asyncio.gather(self._receiver_task, self._sender_task, return_exceptions=True)

                if self.should_stop:
                    return
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(f"[{self._participant_name}] OpenAI realtime connection error: {e}")

            self.connected = False
            self.reconnecting = True
            self._ws_connection = None
            self._send_queue = None

            delay = backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)]
            await asyncio.sleep(delay)

        self.reconnecting = False

    async def _send_session_update(self):
        if not self._ws_connection:
            return

        input_audio_transcription = {
            "model": self.model,
        }
        if self.language:
            input_audio_transcription["language"] = self.language
        if self.prompt:
            input_audio_transcription["prompt"] = self.prompt

        event = {
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_transcription": input_audio_transcription,
                "turn_detection": {
                    "type": "server_vad",
                    "silence_duration_ms": 500,
                },
            },
        }
        await self._ws_connection.send(json.dumps(event))

    async def _sender_loop(self):
        while not self.should_stop:
            try:
                payload = await asyncio.wait_for(self._send_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if not self.connected or not self._ws_connection:
                    return
                continue

            if not self.connected or not self._ws_connection:
                return

            try:
                await self._ws_connection.send(payload)
            except Exception as e:
                logger.warning(f"[{self._participant_name}] OpenAI realtime send failed: {e}")
                self.connected = False
                return

    async def _receiver_loop(self):
        try:
            async for raw_message in self._ws_connection:
                try:
                    message = json.loads(raw_message)
                except Exception:
                    logger.debug(f"[{self._participant_name}] Could not decode OpenAI realtime message")
                    continue

                self._handle_realtime_message(message)
        except Exception as e:
            if not self.should_stop:
                logger.warning(f"[{self._participant_name}] OpenAI realtime receiver closed unexpectedly: {e}")
        finally:
            self.connected = False

    def _extract_transcript_text(self, message):
        transcript = message.get("transcript")
        if isinstance(transcript, str) and transcript.strip():
            return transcript.strip()

        item = message.get("item")
        if not isinstance(item, dict):
            return None

        content = item.get("content")
        if not isinstance(content, list):
            return None

        text_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("transcript") or part.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())

        if text_parts:
            return " ".join(text_parts)
        return None

    def _handle_realtime_message(self, message):
        message_type = message.get("type")

        if message_type == "conversation.item.input_audio_transcription.completed":
            transcript_text = self._extract_transcript_text(message)
            if transcript_text:
                self._emit_utterance(transcript_text)
            return

        if message_type == "conversation.item.input_audio_transcription.failed":
            logger.warning(f"[{self._participant_name}] OpenAI realtime transcription failed: {message}")
            return

        if message_type == "error":
            logger.error(f"[{self._participant_name}] OpenAI realtime error event: {message}")

    def _emit_utterance(self, transcript_text):
        if not self.save_utterance_callback:
            return

        metadata = {
            "timestamp_ms": int(time.time() * 1000),
            "duration_ms": 0,
        }

        try:
            self.save_utterance_callback(transcript_text, metadata)
        except Exception as e:
            logger.error(f"[{self._participant_name}] Error in OpenAI save_utterance_callback: {e}", exc_info=True)

    def _enqueue_audio_append(self, chunk):
        if not self._loop or not self._send_queue:
            return

        payload = json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
            }
        )

        try:
            self._loop.call_soon_threadsafe(self._send_queue.put_nowait, payload)
        except Exception as e:
            logger.warning(f"[{self._participant_name}] Failed to enqueue OpenAI audio chunk: {e}")
            self.connected = False

    def send(self, audio_data):
        if not self.connected and not self.reconnecting and not self.should_stop:
            raise ConnectionError("OpenAI realtime WebSocket connection failed permanently")

        if not self.connected or self.should_stop:
            return

        self.last_send_time = time.time()

        try:
            if self.sample_rate != OPENAI_REALTIME_SAMPLE_RATE:
                audio_data, self._resampler_state = audioop.ratecv(
                    audio_data,
                    SAMPLE_WIDTH,
                    CHANNELS,
                    self.sample_rate,
                    OPENAI_REALTIME_SAMPLE_RATE,
                    self._resampler_state,
                )

            self._audio_buffer.extend(audio_data)

            while len(self._audio_buffer) >= OPENAI_APPEND_FRAME_BYTES:
                chunk = bytes(self._audio_buffer[:OPENAI_APPEND_FRAME_BYTES])
                del self._audio_buffer[:OPENAI_APPEND_FRAME_BYTES]
                self._enqueue_audio_append(chunk)
        except Exception as e:
            logger.error(f"[{self._participant_name}] Error while queuing OpenAI audio: {e}", exc_info=True)
            self.connected = False

    async def _flush_buffer_and_commit(self):
        if self._send_queue and self._ws_connection and self.connected:
            while True:
                try:
                    pending_payload = self._send_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                await self._ws_connection.send(pending_payload)

        if self._audio_buffer and self._ws_connection and self.connected:
            await self._ws_connection.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(bytes(self._audio_buffer)).decode("ascii"),
                    }
                )
            )
            self._audio_buffer = bytearray()

        if self._ws_connection and self.connected:
            await self._ws_connection.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await asyncio.sleep(0.2)

    def finish(self):
        if self.should_stop:
            return

        self.should_stop = True
        logger.info(f"Finishing OpenAI transcriber [{self._participant_name}]")

        try:
            if self._loop and self._loop.is_running():

                async def flush_and_close():
                    try:
                        await self._flush_buffer_and_commit()
                        if self._ws_connection:
                            await self._ws_connection.close()
                    except Exception as e:
                        logger.warning(f"[{self._participant_name}] Error closing OpenAI realtime connection: {e}")

                future = asyncio.run_coroutine_threadsafe(flush_and_close(), self._loop)
                try:
                    future.result(timeout=2)
                except Exception:
                    pass

                self._loop.call_soon_threadsafe(self._loop.stop)

            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=2)
        finally:
            self.connected = False
            self.reconnecting = False
