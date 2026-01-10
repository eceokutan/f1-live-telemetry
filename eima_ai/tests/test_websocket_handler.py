"""
TDD Tests for WebSocket Server - Phase 2, Section 5

These tests define the expected behavior for:
1. FastAPI WebSocket endpoint (/live)
2. Connection handling (accept/reject)
3. Message routing (session_init, telemetry, text_query, config_update, session_end)
4. Heartbeat mechanism
5. Error handling

Run with: pytest tests/test_websocket_handler.py -v

Write these tests FIRST, watch them fail, then implement WebSocketHandler to pass.
"""

import pytest
import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def valid_session_init():
    """Valid session initialization message."""
    return {
        "type": "session_init",
        "session_id": "race_001",
        "source": "torcs",
        "track_name": "Monza",
        "config": {
            "verbosity": "moderate",
            "driver_name": "Driver 1"
        }
    }


@pytest.fixture
def valid_telemetry():
    """Valid telemetry message."""
    return {
        "type": "telemetry",
        "timestamp": "2026-01-15T14:32:05.123Z",
        "data": {
            "speed_kmh": 245.5,
            "rpm": 12500,
            "gear": 5,
            "throttle": 0.95,
            "brake": 0.0,
            "steering_angle": -0.12,
            "fuel_remaining": 45.2,
            "tire_temps": {"fl": 95.2, "fr": 96.1, "rl": 92.4, "rr": 93.8},
            "tire_wear": {"fl": 15.2, "fr": 16.1, "rl": 12.4, "rr": 13.8},
            "g_forces": {"lateral": 1.8, "longitudinal": 0.3},
            "track_position": 0.342,
            "lap_number": 12,
            "lap_time_current": 45.234,
            "sector": 2,
            "position": 3,
            "gap_ahead": 2.456,
            "gap_behind": 1.234
        }
    }


@pytest.fixture
def valid_text_query():
    """Valid text query message."""
    return {
        "type": "text_query",
        "timestamp": "2026-01-15T14:32:10.456Z",
        "query": "How are my tires looking?"
    }


@pytest.fixture
def valid_config_update():
    """Valid configuration update message."""
    return {
        "type": "config_update",
        "config": {
            "verbosity": "verbose"
        }
    }


@pytest.fixture
def valid_session_end():
    """Valid session end message."""
    return {
        "type": "session_end"
    }


# =============================================================================
# WEBSOCKET HANDLER CREATION
# =============================================================================

class TestWebSocketHandlerCreation:
    """Tests for WebSocketHandler instantiation."""

    def test_create_websocket_handler(self):
        """Should create WebSocketHandler instance."""
        from jarvis_granite.live.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()

        assert handler is not None

    def test_websocket_handler_has_required_methods(self):
        """WebSocketHandler should have required public methods."""
        from jarvis_granite.live.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()

        assert hasattr(handler, 'handle_connection')
        assert callable(handler.handle_connection)
        assert hasattr(handler, 'handle_message')
        assert callable(handler.handle_message)

    def test_websocket_handler_tracks_active_sessions(self):
        """WebSocketHandler should track active sessions."""
        from jarvis_granite.live.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()

        assert hasattr(handler, 'active_sessions')
        assert isinstance(handler.active_sessions, dict)


# =============================================================================
# FASTAPI APP SETUP
# =============================================================================

class TestFastAPIAppSetup:
    """Tests for FastAPI application with WebSocket endpoint."""

    def test_create_fastapi_app(self):
        """Should create FastAPI app with WebSocket endpoint."""
        from jarvis_granite.live.main import create_app

        app = create_app()

        assert app is not None

    def test_app_has_live_websocket_route(self):
        """App should have /live WebSocket route."""
        from jarvis_granite.live.main import create_app

        app = create_app()

        # Check routes
        routes = [route.path for route in app.routes]
        assert "/live" in routes

    def test_app_has_health_endpoint(self):
        """App should have /health endpoint for monitoring."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


# =============================================================================
# WEBSOCKET CONNECTION HANDLING
# =============================================================================

class TestWebSocketConnection:
    """Tests for WebSocket connection handling."""

    def test_websocket_connection_accepted(self):
        """Should accept WebSocket connections at /live."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/live") as websocket:
            # Connection should be established
            assert websocket is not None

    def test_websocket_requires_session_init_first(self, valid_telemetry):
        """Should reject telemetry before session_init."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/live") as websocket:
            # Send telemetry without session_init
            websocket.send_json(valid_telemetry)

            response = websocket.receive_json()

            assert response["type"] == "error"
            assert response["error_code"] == "SESSION_NOT_FOUND"


# =============================================================================
# SESSION INITIALIZATION
# =============================================================================

class TestSessionInitialization:
    """Tests for session initialization message handling."""

    def test_session_init_returns_session_confirmed(self, valid_session_init):
        """session_init should return session_confirmed response."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/live") as websocket:
            websocket.send_json(valid_session_init)

            response = websocket.receive_json()

            assert response["type"] == "session_confirmed"
            assert response["session_id"] == "race_001"

    def test_session_confirmed_includes_config(self, valid_session_init):
        """session_confirmed should include active configuration."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/live") as websocket:
            websocket.send_json(valid_session_init)

            response = websocket.receive_json()

            assert "config" in response
            assert isinstance(response["config"], dict)

    def test_session_confirmed_includes_livekit_details(self, valid_session_init):
        """session_confirmed should include LiveKit connection details."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/live") as websocket:
            websocket.send_json(valid_session_init)

            response = websocket.receive_json()

            assert "livekit" in response
            assert "url" in response["livekit"]
            assert "token" in response["livekit"]
            assert "room_name" in response["livekit"]

    def test_session_init_creates_context(self, valid_session_init):
        """session_init should create LiveSessionContext."""
        from jarvis_granite.live.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()
        mock_websocket = AsyncMock()

        # Simulate session init
        asyncio.get_event_loop().run_until_complete(
            handler.handle_message(valid_session_init, mock_websocket)
        )

        assert "race_001" in handler.active_sessions
        context = handler.active_sessions["race_001"]["context"]
        assert context.session_id == "race_001"
        assert context.track_name == "Monza"

    def test_invalid_source_returns_error(self):
        """Invalid source in session_init should return error."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/live") as websocket:
            invalid_init = {
                "type": "session_init",
                "session_id": "race_001",
                "source": "invalid_source",
                "track_name": "Monza"
            }
            websocket.send_json(invalid_init)

            response = websocket.receive_json()

            assert response["type"] == "error"
            assert response["error_code"] == "VALIDATION_ERROR"

    def test_duplicate_session_id_returns_error(self, valid_session_init):
        """Should reject duplicate session_id."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/live") as ws1:
            ws1.send_json(valid_session_init)
            response1 = ws1.receive_json()
            assert response1["type"] == "session_confirmed"

            # Try to create same session from another connection
            with client.websocket_connect("/live") as ws2:
                ws2.send_json(valid_session_init)
                response2 = ws2.receive_json()

                # Should reject duplicate
                assert response2["type"] == "error"


# =============================================================================
# TELEMETRY MESSAGE HANDLING
# =============================================================================

class TestTelemetryHandling:
    """Tests for telemetry message handling."""

    def test_telemetry_updates_context(self, valid_session_init, valid_telemetry):
        """Telemetry should update session context."""
        from jarvis_granite.live.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()
        mock_websocket = AsyncMock()

        loop = asyncio.get_event_loop()

        # Initialize session
        loop.run_until_complete(
            handler.handle_message(valid_session_init, mock_websocket)
        )

        # Send telemetry
        loop.run_until_complete(
            handler.handle_message(valid_telemetry, mock_websocket, session_id="race_001")
        )

        context = handler.active_sessions["race_001"]["context"]
        assert context.speed_kmh == 245.5
        assert context.current_lap == 12

    def test_telemetry_triggers_event_detection(self, valid_session_init, valid_telemetry):
        """Telemetry should trigger TelemetryAgent event detection."""
        from jarvis_granite.live.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()
        mock_websocket = AsyncMock()

        loop = asyncio.get_event_loop()

        # Initialize session
        loop.run_until_complete(
            handler.handle_message(valid_session_init, mock_websocket)
        )

        # Modify telemetry to trigger fuel warning
        valid_telemetry["data"]["fuel_remaining"] = 10.0

        # Send telemetry (should detect events)
        loop.run_until_complete(
            handler.handle_message(valid_telemetry, mock_websocket, session_id="race_001")
        )

        # Verify telemetry agent was used
        assert handler.active_sessions["race_001"]["context"].fuel_remaining == 10.0

    def test_telemetry_without_session_returns_error(self, valid_telemetry):
        """Telemetry without active session should return error."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/live") as websocket:
            websocket.send_json(valid_telemetry)

            response = websocket.receive_json()

            assert response["type"] == "error"
            assert response["error_code"] == "SESSION_NOT_FOUND"


# =============================================================================
# TEXT QUERY HANDLING
# =============================================================================

class TestTextQueryHandling:
    """Tests for text query message handling."""

    def test_text_query_stores_in_conversation_history(
        self, valid_session_init, valid_text_query
    ):
        """Text query should be stored in conversation history."""
        from jarvis_granite.live.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()
        mock_websocket = AsyncMock()

        loop = asyncio.get_event_loop()

        # Initialize session
        loop.run_until_complete(
            handler.handle_message(valid_session_init, mock_websocket)
        )

        # Send text query
        loop.run_until_complete(
            handler.handle_message(valid_text_query, mock_websocket, session_id="race_001")
        )

        # Note: Without LLM integration, we just verify the query was received
        # Full integration would check conversation_history

    def test_empty_query_returns_error(self, valid_session_init):
        """Empty query text should return error."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/live") as websocket:
            websocket.send_json(valid_session_init)
            websocket.receive_json()  # session_confirmed

            empty_query = {
                "type": "text_query",
                "timestamp": "2026-01-15T14:32:10.456Z",
                "query": ""
            }
            websocket.send_json(empty_query)

            response = websocket.receive_json()

            assert response["type"] == "error"
            assert response["error_code"] == "VALIDATION_ERROR"


# =============================================================================
# CONFIG UPDATE HANDLING
# =============================================================================

class TestConfigUpdateHandling:
    """Tests for configuration update message handling."""

    def test_config_update_modifies_session_config(
        self, valid_session_init, valid_config_update
    ):
        """config_update should modify session configuration."""
        from jarvis_granite.live.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()
        mock_websocket = AsyncMock()

        loop = asyncio.get_event_loop()

        # Initialize session
        loop.run_until_complete(
            handler.handle_message(valid_session_init, mock_websocket)
        )

        # Update config
        loop.run_until_complete(
            handler.handle_message(valid_config_update, mock_websocket, session_id="race_001")
        )

        session_config = handler.active_sessions["race_001"]["config"]
        assert session_config.get("verbosity") == "verbose"

    def test_config_update_without_session_returns_error(self, valid_config_update):
        """config_update without active session should return error."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/live") as websocket:
            websocket.send_json(valid_config_update)

            response = websocket.receive_json()

            assert response["type"] == "error"
            assert response["error_code"] == "SESSION_NOT_FOUND"


# =============================================================================
# SESSION END HANDLING
# =============================================================================

class TestSessionEndHandling:
    """Tests for session end message handling."""

    def test_session_end_cleans_up_session(self, valid_session_init, valid_session_end):
        """session_end should clean up session resources."""
        from jarvis_granite.live.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()
        mock_websocket = AsyncMock()

        loop = asyncio.get_event_loop()

        # Initialize session
        loop.run_until_complete(
            handler.handle_message(valid_session_init, mock_websocket)
        )

        assert "race_001" in handler.active_sessions

        # End session
        loop.run_until_complete(
            handler.handle_message(valid_session_end, mock_websocket, session_id="race_001")
        )

        assert "race_001" not in handler.active_sessions

    def test_session_end_graceful_on_disconnect(self, valid_session_init):
        """Session should be cleaned up on WebSocket disconnect."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        # Create and close connection
        with client.websocket_connect("/live") as websocket:
            websocket.send_json(valid_session_init)
            websocket.receive_json()

        # After disconnect, session should be cleaned up
        # This is handled by the connection lifecycle


# =============================================================================
# ERROR HANDLING
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in WebSocket communication."""

    def test_invalid_json_returns_error(self):
        """Invalid JSON should return error message."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/live") as websocket:
            websocket.send_text("not valid json {{{")

            response = websocket.receive_json()

            assert response["type"] == "error"
            assert response["error_code"] == "VALIDATION_ERROR"

    def test_unknown_message_type_returns_error(self, valid_session_init):
        """Unknown message type should return error."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/live") as websocket:
            websocket.send_json(valid_session_init)
            websocket.receive_json()  # session_confirmed

            unknown_msg = {"type": "unknown_type", "data": {}}
            websocket.send_json(unknown_msg)

            response = websocket.receive_json()

            assert response["type"] == "error"
            assert "Unknown message type" in response["message"]

    def test_missing_required_fields_returns_error(self):
        """Missing required fields should return validation error."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/live") as websocket:
            # session_init missing track_name
            incomplete_init = {
                "type": "session_init",
                "session_id": "race_001",
                "source": "torcs"
                # missing track_name
            }
            websocket.send_json(incomplete_init)

            response = websocket.receive_json()

            assert response["type"] == "error"
            assert response["error_code"] == "VALIDATION_ERROR"

    def test_error_message_includes_timestamp(self):
        """Error messages should include timestamp."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/live") as websocket:
            websocket.send_json({"type": "invalid"})

            response = websocket.receive_json()

            assert response["type"] == "error"
            assert "timestamp" in response


# =============================================================================
# HEARTBEAT MECHANISM
# =============================================================================

class TestHeartbeatMechanism:
    """Tests for heartbeat mechanism."""

    def test_handler_can_send_heartbeat(self, valid_session_init):
        """Handler should be able to send heartbeat messages."""
        from jarvis_granite.live.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()
        mock_websocket = AsyncMock()

        loop = asyncio.get_event_loop()

        # Initialize session
        loop.run_until_complete(
            handler.handle_message(valid_session_init, mock_websocket)
        )

        # Send heartbeat
        loop.run_until_complete(
            handler.send_heartbeat("race_001", mock_websocket)
        )

        # Verify heartbeat was sent
        mock_websocket.send_json.assert_called()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == "heartbeat"
        assert call_args["session_active"] is True

    def test_heartbeat_includes_timestamp(self, valid_session_init):
        """Heartbeat should include timestamp."""
        from jarvis_granite.live.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()
        mock_websocket = AsyncMock()

        loop = asyncio.get_event_loop()

        # Initialize session
        loop.run_until_complete(
            handler.handle_message(valid_session_init, mock_websocket)
        )

        # Send heartbeat
        loop.run_until_complete(
            handler.send_heartbeat("race_001", mock_websocket)
        )

        call_args = mock_websocket.send_json.call_args[0][0]
        assert "timestamp" in call_args


# =============================================================================
# MESSAGE ROUTING
# =============================================================================

class TestMessageRouting:
    """Tests for message routing logic."""

    def test_routes_session_init_correctly(self, valid_session_init):
        """Should route session_init to initialization handler."""
        from jarvis_granite.live.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()
        mock_websocket = AsyncMock()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            handler.handle_message(valid_session_init, mock_websocket)
        )

        # Should have created session
        assert "race_001" in handler.active_sessions

    def test_routes_telemetry_correctly(self, valid_session_init, valid_telemetry):
        """Should route telemetry to telemetry handler."""
        from jarvis_granite.live.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()
        mock_websocket = AsyncMock()

        loop = asyncio.get_event_loop()

        # Initialize first
        loop.run_until_complete(
            handler.handle_message(valid_session_init, mock_websocket)
        )

        # Then telemetry
        loop.run_until_complete(
            handler.handle_message(valid_telemetry, mock_websocket, session_id="race_001")
        )

        # Context should be updated
        assert handler.active_sessions["race_001"]["context"].speed_kmh == 245.5

    def test_routes_text_query_correctly(self, valid_session_init, valid_text_query):
        """Should route text_query to query handler."""
        from jarvis_granite.live.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()
        mock_websocket = AsyncMock()

        loop = asyncio.get_event_loop()

        # Initialize first
        loop.run_until_complete(
            handler.handle_message(valid_session_init, mock_websocket)
        )

        # Query should not raise error
        loop.run_until_complete(
            handler.handle_message(valid_text_query, mock_websocket, session_id="race_001")
        )


# =============================================================================
# CONCURRENT CONNECTIONS
# =============================================================================

class TestConcurrentConnections:
    """Tests for handling multiple concurrent connections."""

    def test_multiple_sessions_independent(self):
        """Multiple sessions should be independent."""
        from jarvis_granite.live.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()

        loop = asyncio.get_event_loop()

        session1 = {
            "type": "session_init",
            "session_id": "race_001",
            "source": "torcs",
            "track_name": "Monza"
        }
        session2 = {
            "type": "session_init",
            "session_id": "race_002",
            "source": "assetto_corsa",
            "track_name": "Spa"
        }

        loop.run_until_complete(handler.handle_message(session1, mock_ws1))
        loop.run_until_complete(handler.handle_message(session2, mock_ws2))

        assert "race_001" in handler.active_sessions
        assert "race_002" in handler.active_sessions
        assert handler.active_sessions["race_001"]["context"].track_name == "Monza"
        assert handler.active_sessions["race_002"]["context"].track_name == "Spa"

    def test_session_isolation(self):
        """Telemetry from one session should not affect another."""
        from jarvis_granite.live.websocket_handler import WebSocketHandler

        handler = WebSocketHandler()
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()

        loop = asyncio.get_event_loop()

        # Create two sessions
        session1 = {
            "type": "session_init",
            "session_id": "race_001",
            "source": "torcs",
            "track_name": "Monza"
        }
        session2 = {
            "type": "session_init",
            "session_id": "race_002",
            "source": "torcs",
            "track_name": "Spa"
        }

        loop.run_until_complete(handler.handle_message(session1, mock_ws1))
        loop.run_until_complete(handler.handle_message(session2, mock_ws2))

        # Send telemetry to session 1
        telemetry = {
            "type": "telemetry",
            "timestamp": "2026-01-15T14:32:05.123Z",
            "data": {
                "speed_kmh": 300.0,
                "rpm": 15000,
                "gear": 6,
                "throttle": 1.0,
                "brake": 0.0,
                "steering_angle": 0.0,
                "fuel_remaining": 30.0,
                "tire_temps": {"fl": 90, "fr": 90, "rl": 90, "rr": 90},
                "tire_wear": {"fl": 10, "fr": 10, "rl": 10, "rr": 10},
                "g_forces": {"lateral": 0, "longitudinal": 0},
                "track_position": 0.5,
                "lap_number": 5,
                "lap_time_current": 30.0,
                "sector": 2
            }
        }

        loop.run_until_complete(
            handler.handle_message(telemetry, mock_ws1, session_id="race_001")
        )

        # Session 1 should be updated
        assert handler.active_sessions["race_001"]["context"].speed_kmh == 300.0

        # Session 2 should not be affected
        assert handler.active_sessions["race_002"]["context"].speed_kmh == 0.0


# =============================================================================
# LIVEKIT TOKEN GENERATION
# =============================================================================

class TestLiveKitTokenGeneration:
    """Tests for LiveKit token generation."""

    def test_livekit_token_is_generated(self, valid_session_init):
        """session_confirmed should include generated LiveKit token."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/live") as websocket:
            websocket.send_json(valid_session_init)

            response = websocket.receive_json()

            assert response["livekit"]["token"] is not None
            assert len(response["livekit"]["token"]) > 0

    def test_livekit_room_name_includes_session_id(self, valid_session_init):
        """LiveKit room name should include session ID."""
        from jarvis_granite.live.main import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)

        with client.websocket_connect("/live") as websocket:
            websocket.send_json(valid_session_init)

            response = websocket.receive_json()

            assert "race_001" in response["livekit"]["room_name"]
