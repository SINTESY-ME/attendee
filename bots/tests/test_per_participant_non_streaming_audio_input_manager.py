from datetime import datetime, timedelta
from unittest import TestCase

from bots.bot_controller.per_participant_non_streaming_audio_input_manager import PerParticipantNonStreamingAudioInputManager


class TestPerParticipantNonStreamingAudioInputManager(TestCase):
    def setUp(self):
        self.saved_chunks = []
        self.participant = {
            "participant_uuid": "speaker-1",
            "participant_user_uuid": "user-1",
            "participant_full_name": "Speaker One",
            "participant_is_the_bot": False,
            "participant_is_host": False,
        }
        self.manager = PerParticipantNonStreamingAudioInputManager(
            save_audio_chunk_callback=lambda payload: self.saved_chunks.append(payload),
            get_participant_callback=lambda speaker_id: self.participant if speaker_id == "speaker-1" else None,
            sample_rate=32000,
            utterance_size_limit=1000000,
            silence_duration_limit=3,
            should_print_diagnostic_info=False,
        )

    @staticmethod
    def non_silent_chunk():
        return b"\xe8\x03" * 960  # int16 value 1000, 30ms at 32kHz

    @staticmethod
    def silent_chunk():
        return b"\x00\x00" * 960  # 30ms of silence at 32kHz

    def test_speech_stop_flushes_chunk_after_post_roll_window(self):
        start_time = datetime.utcnow()
        self.manager.add_speech_start_event("speaker-1", start_time + timedelta(milliseconds=500))
        self.manager.add_chunk("speaker-1", start_time, self.non_silent_chunk())
        self.manager.add_chunk("speaker-1", start_time + timedelta(milliseconds=600), self.non_silent_chunk())
        self.manager.add_speech_stop_event("speaker-1", start_time + timedelta(milliseconds=700))
        self.manager.add_chunk("speaker-1", start_time + timedelta(seconds=5), self.silent_chunk())

        self.manager.process_chunks()

        # No flush yet: we are still inside the 5s post-roll after SPEECH_STOP.
        self.assertEqual(len(self.saved_chunks), 0)

        self.manager.add_chunk("speaker-1", start_time + timedelta(seconds=6), self.silent_chunk())
        self.manager.process_chunks()

        self.assertEqual(len(self.saved_chunks), 1)
        self.assertEqual(self.saved_chunks[0]["flush_reason"], "speech_stop")
        # Pre-roll should move the start timestamp back to include the chunk before SPEECH_START.
        self.assertEqual(self.saved_chunks[0]["timestamp_ms"], int(start_time.timestamp() * 1000))

    def test_speech_start_within_stop_window_cancels_pending_flush(self):
        start_time = datetime.utcnow()
        self.manager.add_speech_start_event("speaker-1", start_time)
        self.manager.add_chunk("speaker-1", start_time + timedelta(milliseconds=100), self.non_silent_chunk())
        self.manager.add_speech_stop_event("speaker-1", start_time + timedelta(milliseconds=200))

        # Resume speaking before 5s passes, so the first stop should not flush.
        self.manager.add_speech_start_event("speaker-1", start_time + timedelta(seconds=3))
        self.manager.add_chunk("speaker-1", start_time + timedelta(seconds=3, milliseconds=100), self.non_silent_chunk())

        # Next stop should flush only after another 5s with no speech.
        self.manager.add_speech_stop_event("speaker-1", start_time + timedelta(seconds=3, milliseconds=200))
        self.manager.add_chunk("speaker-1", start_time + timedelta(seconds=9), self.silent_chunk())

        self.manager.process_chunks()

        self.assertEqual(len(self.saved_chunks), 1)
        self.assertEqual(self.saved_chunks[0]["flush_reason"], "speech_stop")

    def test_fallback_silence_still_flushes_when_no_speech_events_arrive(self):
        start_time = datetime.utcnow()
        self.manager.add_chunk("speaker-1", start_time, self.non_silent_chunk())
        self.manager.process_chunks()
        self.assertEqual(len(self.saved_chunks), 0)

        self.manager.add_chunk("speaker-1", start_time + timedelta(seconds=4), self.silent_chunk())
        self.manager.process_chunks()

        self.assertEqual(len(self.saved_chunks), 1)
        self.assertEqual(self.saved_chunks[0]["flush_reason"], "silence_limit")

    def test_queue_uses_timestamps_for_event_and_audio_ordering(self):
        start_time = datetime.utcnow()
        self.manager.add_speech_start_event("speaker-1", start_time)
        self.manager.add_speech_stop_event("speaker-1", start_time + timedelta(milliseconds=200))
        self.manager.add_chunk("speaker-1", start_time + timedelta(milliseconds=100), self.non_silent_chunk())
        self.manager.add_chunk("speaker-1", start_time + timedelta(seconds=6), self.silent_chunk())

        self.manager.process_chunks()

        self.assertEqual(len(self.saved_chunks), 1)
        self.assertEqual(self.saved_chunks[0]["flush_reason"], "speech_stop")
