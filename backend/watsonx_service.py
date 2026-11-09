"""
LeafSense - IBM watsonx.ai & Granite Service Module
===================================================
Provides a secure, robust integration with IBM watsonx.ai foundation models
using the official `ibm-watsonx-ai` SDK.

Security Principles:
- All credentials are read from backend environment variables (`backend/.env`).
- API keys, bearer tokens, and sensitive headers are never logged or exposed.
- All errors are sanitized into clear user-friendly diagnostic messages.
"""

import os
import re
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Configure module logger
logger = logging.getLogger("leafsense.watsonx")

# Load environment variables from backend/.env explicitly
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=False)
else:
    load_dotenv(override=False)


def _sanitize_error_message(message: str) -> str:
    """
    Strips potential API keys, UUIDs resembling bearer tokens,
    or Authorization headers from error strings to prevent accidental leaks.
    """
    if not message:
        return ""
    # Mask authorization headers or api keys
    cleaned = re.sub(
        r'(apikey|api_key|token|bearer|authorization)[:=]\s*["\']?[a-zA-Z0-9_\-]{16,}["\']?',
        r'\1: [REDACTED]',
        message,
        flags=re.IGNORECASE
    )
    # Mask potential standalone 32-64 character hex/base64 tokens
    cleaned = re.sub(r'\b[A-Za-z0-9_\-]{32,64}\b', '[REDACTED_TOKEN]', cleaned)
    return cleaned


class WatsonxServiceError(Exception):
    """Custom exception for watsonx service errors with safe categorization."""
    def __init__(self, message: str, category: str = "GENERAL_ERROR", suggestions: Optional[List[str]] = None):
        self.category = category
        self.suggestions = suggestions or []
        self.sanitized_message = _sanitize_error_message(message)
        super().__init__(self.sanitized_message)


class WatsonxGraniteService:
    """
    Service wrapper for IBM watsonx.ai foundation models, specifically tailored
    for IBM Granite instruct and chat models in LeafSense.
    """

    DEFAULT_URL = "https://us-south.ml.cloud.ibm.com"
    # Dallas watsonx.ai current Granite generative model:
    DEFAULT_MODEL = "ibm/granite-4-h-small"
    FALLBACK_MODELS = [
        "ibm/granite-4-h-small",
        "ibm/granite-3-8b-instruct",
        "ibm/granite-13b-instruct-v2",
        "ibm/granite-3-2b-instruct",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        url: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout: int = 45
    ):
        """
        Initialize the watsonx service.
        Parameters default to environment variables if not passed explicitly.
        """
        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            env_key = (
                os.getenv("WATSONX_API_KEY")
                or os.getenv("WATSONX_APIKEY")
                or os.getenv("IBM_CLOUD_API_KEY")
            )
            self.api_key = env_key.strip() if env_key else None

        if project_id is not None:
            self.project_id = project_id.strip()
        else:
            env_proj = os.getenv("WATSONX_PROJECT_ID")
            self.project_id = env_proj.strip() if env_proj else None

        if url is not None:
            self.url = url.strip().rstrip("/")
        else:
            env_url = os.getenv("WATSONX_URL") or self.DEFAULT_URL
            self.url = env_url.strip().rstrip("/")

        if model_id is not None:
            self.model_id = model_id.strip()
        else:
            env_model = os.getenv("WATSONX_MODEL_ID") or self.DEFAULT_MODEL
            self.model_id = env_model.strip()

        self.timeout = timeout
        self._client = None

    def is_configured(self) -> bool:
        """Check whether minimum required credentials exist."""
        return bool(self.api_key and self.api_key.strip() and self.project_id and self.project_id.strip())

    def get_credential_status(self) -> Dict[str, Any]:
        """
        Safely returns configuration status without printing secret values.
        """
        return {
            "api_key_configured": bool(self.api_key and self.api_key.strip()),
            "project_id_configured": bool(self.project_id and self.project_id.strip()),
            "url": self.url,
            "configured_model_id": self.model_id,
            "ready": self.is_configured(),
        }

    def _get_api_client(self):
        """Initializes and caches the ibm_watsonx_ai APIClient."""
        if not self.is_configured():
            raise WatsonxServiceError(
                "Missing IBM watsonx.ai configuration. Ensure WATSONX_API_KEY and WATSONX_PROJECT_ID are set in backend/.env.",
                category="MISSING_CREDENTIALS",
                suggestions=[
                    "Check backend/.env file exists.",
                    "Verify WATSONX_API_KEY contains a valid IBM Cloud API Key.",
                    "Verify WATSONX_PROJECT_ID contains your watsonx.ai project GUID."
                ]
            )

        if self._client is not None:
            return self._client

        try:
            from ibm_watsonx_ai import APIClient, Credentials
            credentials = Credentials(url=self.url, api_key=self.api_key)
            client = APIClient(credentials=credentials, project_id=self.project_id)
            self._client = client
            return self._client
        except ImportError:
            raise WatsonxServiceError(
                "Package 'ibm-watsonx-ai' is not installed in the current Python environment.",
                category="DEPENDENCY_MISSING",
                suggestions=["Run 'pip install ibm-watsonx-ai' inside your virtual environment."]
            )
        except Exception as e:
            category, suggestions = self._categorize_error(e)
            raise WatsonxServiceError(str(e), category=category, suggestions=suggestions)

    def _categorize_error(self, err: Exception) -> tuple[str, List[str]]:
        """Categorize watsonx errors and provide actionable troubleshooting suggestions."""
        err_str = str(err).lower()

        if "401" in err_str or "unauthorized" in err_str or "invalid api key" in err_str:
            return (
                "AUTHENTICATION_FAILED",
                [
                    "Verify your IBM Cloud API Key in backend/.env.",
                    "Ensure the API key has not been deleted or expired on cloud.ibm.com.",
                    "Confirm there are no leading/trailing spaces or quotes in WATSONX_API_KEY."
                ]
            )
        elif "403" in err_str or "forbidden" in err_str or "not authorized" in err_str:
            return (
                "PROJECT_ACCESS_DENIED",
                [
                    "Verify that WATSONX_PROJECT_ID in backend/.env matches your watsonx.ai project ID exactly.",
                    "Check on dataplatform.cloud.ibm.com that your IBM Cloud user is an 'Admin' or 'Editor' on the project.",
                    "Ensure a Watson Machine Learning (WML) service instance is associated with the project."
                ]
            )
        elif "model_no_support" in err_str or "unsupported" in err_str:
            return (
                "MODEL_UNSUPPORTED_TASK",
                [
                    f"Selected model does not support text generation or chat.",
                    "Use 'ibm/granite-4-h-small' or another Granite model with 'text_chat'/'text_generation' capability.",
                    "Set WATSONX_MODEL_ID=ibm/granite-4-h-small in backend/.env."
                ]
            )
        elif "404" in err_str or "not found" in err_str:
            return (
                "MODEL_NOT_FOUND",
                [
                    f"Model was not found in the {self.url} region.",
                    "Check available models using service.list_available_granite_models().",
                    "Set WATSONX_MODEL_ID=ibm/granite-4-h-small in backend/.env."
                ]
            )
        elif "429" in err_str or "rate limit" in err_str or "quota" in err_str or "too many requests" in err_str:
            return (
                "QUOTA_OR_RATE_LIMIT",
                [
                    "Your IBM Cloud account has reached its concurrency or monthly token limit on the Lite tier.",
                    "Wait a moment before retrying, or review plan limits at cloud.ibm.com/billing."
                ]
            )
        elif "connect" in err_str or "timeout" in err_str or "resolution" in err_str:
            return (
                "NETWORK_CONNECTION_ERROR",
                [
                    f"Could not connect to watsonx endpoint {self.url}.",
                    "Verify internet connectivity and check for proxy or firewall restrictions.",
                    "Ensure WATSONX_URL is set to a valid endpoint, e.g., https://us-south.ml.cloud.ibm.com"
                ]
            )
        else:
            return (
                "WATSONX_API_ERROR",
                [
                    "Inspect the sanitized error details.",
                    "Verify IBM Cloud service status at https://cloud.ibm.com/status."
                ]
            )

    def list_available_granite_models(self, filter_generative: bool = True) -> List[str]:
        """
        Lists available IBM Granite models accessible through the current watsonx endpoint.
        
        Parameters:
        - filter_generative: If True, only returns models supporting text_generation or text_chat.
        """
        try:
            client = self._get_api_client()
            models_dict = client.foundation_models.get_model_specs()
            resources = models_dict.get("resources", [])
            granite_models = []
            for spec in resources:
                mid = spec.get("model_id", "")
                if "granite" in mid.lower():
                    functions = [f.get("id") for f in spec.get("functions", [])]
                    if filter_generative:
                        if any(fn in ("text_generation", "text_chat") for fn in functions):
                            granite_models.append(mid)
                    else:
                        granite_models.append(mid)
            return sorted(granite_models)
        except WatsonxServiceError:
            raise
        except Exception as e:
            category, suggestions = self._categorize_error(e)
            raise WatsonxServiceError(str(e), category=category, suggestions=suggestions)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        model_id: Optional[str] = None
    ) -> str:
        """
        Execute multi-turn conversational chat with IBM Granite using `/ml/v1/text/chat`.

        Parameters:
        - messages: List of dicts with 'role' ('system', 'user', 'assistant') and 'content'.
        - max_tokens: Maximum tokens to generate.
        - temperature: Sampling temperature.
        - model_id: Override the model ID for this request.

        Returns:
        - Assistant's response string.
        """
        if not messages:
            raise ValueError("Messages list cannot be empty.")

        target_model = (model_id or self.model_id).strip()

        try:
            from ibm_watsonx_ai.foundation_models import ModelInference

            client = self._get_api_client()
            params: Dict[str, Any] = {
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            model = ModelInference(
                model_id=target_model,
                api_client=client,
                params=params,
                project_id=self.project_id
            )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                chat_res = model.chat(messages=messages)

            if isinstance(chat_res, dict):
                choices = chat_res.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
            return str(chat_res).strip()

        except WatsonxServiceError:
            raise
        except Exception as e:
            category, suggestions = self._categorize_error(e)
            raise WatsonxServiceError(str(e), category=category, suggestions=suggestions)

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        min_new_tokens: int = 1,
        temperature: float = 0.7,
        decoding_method: str = "greedy",
        repetition_penalty: float = 1.1,
        model_id: Optional[str] = None
    ) -> str:
        """
        Generate text inference using IBM Granite foundation model.
        Leverages chat endpoint under the hood for Granite chat/instruct models
        to provide robust compatibility with watsonx.ai v1 APIs.

        Parameters:
        - prompt: The user input or instruction.
        - system_prompt: Optional system instruction.
        - max_new_tokens: Maximum tokens to generate.
        - min_new_tokens: Minimum tokens to generate.
        - temperature: Sampling temperature.
        - decoding_method: 'greedy' or 'sample'.
        - repetition_penalty: Repetition penalty factor.
        - model_id: Override default model for this inference call.

        Returns:
        - Cleaned string response from IBM Granite.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        target_model = (model_id or self.model_id).strip()

        # Build messages for chat endpoint
        messages: List[Dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        try:
            return self.chat(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature if decoding_method == "sample" else 0.1,
                model_id=target_model
            )
        except WatsonxServiceError:
            raise
        except Exception as e:
            category, suggestions = self._categorize_error(e)
            raise WatsonxServiceError(str(e), category=category, suggestions=suggestions)


# Module-level singleton instance for convenient reuse across the application
watsonx_service = WatsonxGraniteService()
