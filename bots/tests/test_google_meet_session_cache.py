import json
from unittest.mock import MagicMock, patch

import redis
from django.conf import settings
from django.test import TransactionTestCase

from accounts.models import Organization
from bots.models import Bot, BotLogin, BotLoginGroup, BotLoginPlatform, Project


class GoogleMeetSessionCacheTest(TransactionTestCase):
    """Tests for the Google Meet session caching in Redis.

    These tests verify that after a successful SSO login, the Google auth cookies are cached
    in Redis and can be reused by subsequent bots to skip the SSO flow entirely.
    """

    def setUp(self):
        self.organization = Organization.objects.create(name="Test Org", centicredits=10000)
        self.project = Project.objects.create(name="Test Project", organization=self.organization)
        self.bot = Bot.objects.create(
            project=self.project,
            name="Test Bot",
            meeting_url="https://meet.google.com/abc-defg-hij",
        )

        self.google_meet_bot_login_group = BotLoginGroup.objects.create(
            project=self.project,
            platform=BotLoginPlatform.GOOGLE_MEET,
            name="Google Meet Group 1",
        )
        self.google_meet_bot_login = BotLogin.objects.create(
            group=self.google_meet_bot_login_group,
            workspace_domain="test-workspace.com",
            email="test-bot@test-workspace.com",
        )
        self.google_meet_bot_login.set_credentials(
            {
                "cert": "test-cert",
                "private_key": "test-private-key",
            }
        )

        self.redis_client = redis.from_url(settings.REDIS_URL_WITH_PARAMS)
        self.login_email = "test-bot@test-workspace.com"
        self.login_domain = "test-workspace.com"

    def tearDown(self):
        keys = self.redis_client.keys("google_meet_session:*")
        if keys:
            self.redis_client.delete(*keys)

    def _make_adapter(self):
        """Create a real GoogleMeetUIMethods instance with only the attributes we need set."""
        from bots.google_meet_bot_adapter.google_meet_ui_methods import GoogleMeetUIMethods

        adapter = GoogleMeetUIMethods.__new__(GoogleMeetUIMethods)
        adapter.google_meet_bot_login_session = {
            "session_id": "test-session-id",
            "login_email": self.login_email,
            "login_domain": self.login_domain,
        }
        adapter.used_cached_google_session = False
        adapter.driver = MagicMock()
        return adapter

    def _redis_key_for_email(self, email):
        import hashlib

        fingerprint = hashlib.sha256(email.encode()).hexdigest()
        return f"google_meet_session:{fingerprint}"

    def test_redis_key_is_scoped_per_login_email(self):
        """The Redis cache key should be unique per login email."""
        key1 = self._redis_key_for_email("test-bot@test-workspace.com")
        key2 = self._redis_key_for_email("other@example.com")
        self.assertNotEqual(key1, key2)
        self.assertTrue(key1.startswith("google_meet_session:"))

    def test_save_and_restore_session_from_redis(self):
        """After saving session cookies to Redis, _establish_cached_google_session should restore them."""
        adapter = self._make_adapter()

        auth_cookies = [
            {"name": "SID", "value": "test-sid", "domain": ".google.com", "path": "/", "secure": True},
            {"name": "HSID", "value": "test-hsid", "domain": ".google.com", "path": "/", "secure": True},
            {"name": "SAPISID", "value": "test-sapisid", "domain": ".google.com", "path": "/", "secure": True},
            {"name": "__Secure-1PSID", "value": "test-1psid", "domain": ".google.com", "path": "/", "secure": True},
            {"name": "NID", "value": "test-nid", "domain": ".google.com", "path": "/", "secure": True},
        ]
        adapter.driver.get_cookies.return_value = auth_cookies

        adapter._save_google_session_to_redis()

        cookie_key = self._redis_key_for_email(self.login_email)
        saved_data = self.redis_client.get(cookie_key)
        self.assertIsNotNone(saved_data)
        saved_cookies = json.loads(saved_data)
        self.assertEqual(len(saved_cookies), 5)

        with patch.object(
            type(adapter),
            "has_google_cookies_that_indicate_logged_in",
            return_value=True,
        ):
            result = adapter._establish_cached_google_session()

        self.assertTrue(result)
        self.assertTrue(adapter.used_cached_google_session)
        self.assertEqual(adapter.driver.add_cookie.call_count, 5)

    def test_clear_cached_session(self):
        """_clear_cached_google_session should remove the session from Redis."""
        adapter = self._make_adapter()
        cookie_key = self._redis_key_for_email(self.login_email)

        self.redis_client.setex(cookie_key, 1800, json.dumps([{"name": "SID", "value": "test"}]))
        self.redis_client.set(f"{cookie_key}:uses", 5)

        adapter._clear_cached_google_session()

        self.assertIsNone(self.redis_client.get(cookie_key))
        self.assertIsNone(self.redis_client.get(f"{cookie_key}:uses"))

    def test_establish_cached_session_returns_false_when_no_cache(self):
        """_establish_cached_google_session should return False when no cached session exists."""
        adapter = self._make_adapter()
        cookie_key = self._redis_key_for_email(self.login_email)
        self.redis_client.delete(cookie_key, f"{cookie_key}:uses")

        result = adapter._establish_cached_google_session()

        self.assertFalse(result)
        self.assertFalse(adapter.used_cached_google_session)

    def test_establish_cached_session_returns_false_when_session_invalid(self):
        """_establish_cached_google_session should return False and clear cache when session is invalid."""
        adapter = self._make_adapter()
        cookie_key = self._redis_key_for_email(self.login_email)

        self.redis_client.setex(
            cookie_key,
            1800,
            json.dumps([{"name": "SID", "value": "expired", "domain": ".google.com", "path": "/", "secure": True}]),
        )

        with patch.object(
            type(adapter),
            "has_google_cookies_that_indicate_logged_in",
            return_value=False,
        ):
            result = adapter._establish_cached_google_session()

        self.assertFalse(result)
        self.assertIsNone(self.redis_client.get(cookie_key))

    def test_session_expires_after_max_uses(self):
        """Cached session should be discarded after MAX_GOOGLE_MEET_SESSION_USES uses."""
        adapter = self._make_adapter()
        cookie_key = self._redis_key_for_email(self.login_email)

        self.redis_client.setex(
            cookie_key,
            1800,
            json.dumps([{"name": "SID", "value": "test", "domain": ".google.com", "path": "/", "secure": True}]),
        )

        max_uses = 20
        self.redis_client.set(f"{cookie_key}:uses", max_uses)

        result = adapter._establish_cached_google_session()

        self.assertFalse(result)
        self.assertIsNone(self.redis_client.get(cookie_key))
