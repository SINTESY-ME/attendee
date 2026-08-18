"""
Unit tests for GoogleMeetBotAdapter.should_retry_joining_meeting_that_requires_login_by_logging_in.

These tests verify the retry logic when a cached Chrome profile from S3 has
expired cookies. The core scenario: when login_mode="always" and a cached
profile was loaded (SSO skipped), the bot must still be able to retry with a
fresh SSO login instead of giving up.

Regression test for the bug where all Google Meet recordings failed with
"login_required" because the S3 profile's Google session expired and the bot
refused to retry, thinking it had already attempted login.
"""
from unittest.mock import MagicMock

from django.test import SimpleTestCase

from bots.google_meet_bot_adapter.google_meet_bot_adapter import GoogleMeetBotAdapter


class TestShouldRetryLoginWithExpiredS3Profile(SimpleTestCase):
    """Test the retry decision logic without needing a database or browser."""

    def _make_adapter(
        self,
        login_is_available=True,
        login_should_be_used=True,
        chrome_profile_loaded_from_s3=False,
        chrome_profile_s3_failed=False,
    ):
        """Create a bare adapter instance with only the fields needed for the retry logic."""
        adapter = object.__new__(GoogleMeetBotAdapter)
        adapter.google_meet_bot_login_is_available = login_is_available
        adapter.google_meet_bot_login_should_be_used = login_should_be_used
        adapter.google_meet_bot_login_session = None
        adapter.chrome_profile_loaded_from_s3 = chrome_profile_loaded_from_s3
        adapter.chrome_profile_s3_failed = chrome_profile_s3_failed
        return adapter

    def test_cached_s3_profile_with_always_mode_should_retry(self):
        """
        When login_mode='always' and a cached S3 profile was loaded (SSO skipped),
        the bot should retry with a fresh SSO login after the profile's session expires.

        This is the core regression scenario: before the fix, the bot refused to
        retry because google_meet_bot_login_should_be_used was True from the start
        (login_mode=always), even though SSO was never actually attempted.
        """
        adapter = self._make_adapter(
            login_is_available=True,
            login_should_be_used=True,
            chrome_profile_loaded_from_s3=True,
        )

        result = adapter.should_retry_joining_meeting_that_requires_login_by_logging_in()

        self.assertTrue(result, "Should retry with fresh SSO when cached profile expired")
        self.assertTrue(adapter.chrome_profile_s3_failed, "Should mark S3 profile as failed")
        self.assertFalse(adapter.chrome_profile_loaded_from_s3, "Should clear S3 loaded flag")
        self.assertTrue(adapter.google_meet_bot_login_should_be_used, "Should keep should_be_used True")

    def test_sso_already_attempted_should_not_retry(self):
        """
        When SSO was actually attempted (no S3 profile loaded) and login failed,
        the bot should NOT retry to avoid infinite loops.
        """
        adapter = self._make_adapter(
            login_is_available=True,
            login_should_be_used=True,
            chrome_profile_loaded_from_s3=False,
        )

        result = adapter.should_retry_joining_meeting_that_requires_login_by_logging_in()

        self.assertFalse(result, "Should not retry after a real SSO attempt failed")

    def test_no_login_available_should_not_retry(self):
        """When no login credentials are available, retry is impossible."""
        adapter = self._make_adapter(
            login_is_available=False,
            login_should_be_used=False,
            chrome_profile_loaded_from_s3=False,
        )

        result = adapter.should_retry_joining_meeting_that_requires_login_by_logging_in()

        self.assertFalse(result, "Should not retry without login credentials")

    def test_only_if_required_mode_without_s3_profile_should_retry(self):
        """
        When login_mode='only_if_required' (should_be_used=False initially) and
        no S3 profile was loaded, the bot should retry by activating login.
        This is the original 'only_if_required' flow that was already working.
        """
        adapter = self._make_adapter(
            login_is_available=True,
            login_should_be_used=False,
            chrome_profile_loaded_from_s3=False,
        )

        result = adapter.should_retry_joining_meeting_that_requires_login_by_logging_in()

        self.assertTrue(result, "Should retry by enabling login in only_if_required mode")
        self.assertTrue(adapter.google_meet_bot_login_should_be_used, "Should activate should_be_used")

    def test_cached_s3_profile_with_only_if_required_mode_should_retry(self):
        """
        When login_mode='only_if_required' and a cached S3 profile was loaded,
        the bot should retry with fresh SSO (same as always mode with S3 profile).
        """
        adapter = self._make_adapter(
            login_is_available=True,
            login_should_be_used=False,
            chrome_profile_loaded_from_s3=True,
        )

        result = adapter.should_retry_joining_meeting_that_requires_login_by_logging_in()

        self.assertTrue(result, "Should retry with fresh SSO regardless of login mode")
        self.assertTrue(adapter.chrome_profile_s3_failed, "Should mark S3 profile as failed")