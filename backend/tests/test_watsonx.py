"""
Unit tests for WatsonxGraniteService and credential security.
Ensures credential masking, error sanitization, and fallback logic work correctly.
"""

import pytest
from watsonx_service import WatsonxGraniteService, WatsonxServiceError, _sanitize_error_message


def test_sanitize_error_message_masks_keys():
    """Verify that sensitive API keys and tokens are redacted from error logs."""
    raw_error = "Failed to authenticate with apikey: abc123def456ghi789jkl012mno345pqr678"
    sanitized = _sanitize_error_message(raw_error)
    assert "abc123def456ghi789jkl012mno345pqr678" not in sanitized
    assert "[REDACTED]" in sanitized or "[REDACTED_TOKEN]" in sanitized


def test_credential_status_does_not_reveal_keys():
    """Verify get_credential_status returns boolean flags rather than secret strings."""
    service = WatsonxGraniteService(api_key="super_secret_test_key_12345", project_id="proj-uuid-1234")
    status = service.get_credential_status()

    assert status["api_key_configured"] is True
    assert status["project_id_configured"] is True
    assert "super_secret_test_key_12345" not in str(status)


def test_empty_prompt_validation():
    """Verify that generating text with an empty prompt raises ValueError."""
    service = WatsonxGraniteService(api_key="dummy", project_id="dummy")
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        service.generate_text("")


def test_missing_credentials_categorization():
    """Verify that unconfigured service raises MISSING_CREDENTIALS error."""
    service = WatsonxGraniteService(api_key="", project_id="")
    assert service.is_configured() is False
    with pytest.raises(WatsonxServiceError) as exc_info:
        service._get_api_client()
    assert exc_info.value.category == "MISSING_CREDENTIALS"
    assert len(exc_info.value.suggestions) > 0


def test_error_categorization():
    """Verify error categorization helper detects status codes properly."""
    service = WatsonxGraniteService(api_key="dummy", project_id="dummy")

    cat_401, sugg_401 = service._categorize_error(Exception("401 Unauthorized: Invalid API key"))
    assert cat_401 == "AUTHENTICATION_FAILED"

    cat_403, sugg_403 = service._categorize_error(Exception("403 Forbidden: project access denied"))
    assert cat_403 == "PROJECT_ACCESS_DENIED"

    cat_404, sugg_404 = service._categorize_error(Exception("404 Not Found: resource missing"))
    assert cat_404 == "MODEL_NOT_FOUND"

    cat_unsup, sugg_unsup = service._categorize_error(Exception("model_no_support_for_function: unsupported"))
    assert cat_unsup == "MODEL_UNSUPPORTED_TASK"

    cat_429, sugg_429 = service._categorize_error(Exception("429 Too Many Requests: quota exceeded"))
    assert cat_429 == "QUOTA_OR_RATE_LIMIT"

    cat_net, sugg_net = service._categorize_error(Exception("Failed to connect to host"))
    assert cat_net == "NETWORK_CONNECTION_ERROR"
