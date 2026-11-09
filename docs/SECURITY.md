# LeafSense Security Policy & Best Practices

LeafSense is engineered with strict data hygiene, input validation, and agricultural safety guardrails to protect user privacy and prevent misuse.

---

## 1. Credential Hygiene & Environment Isolation

### Secret Separation
- **Backend Isolation:** All third-party API credentials (`WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `SARVAM_API_KEY`) reside exclusively in server-side environment variables (`backend/.env`).
- **Never Committed:** `backend/.env` is strictly excluded in `.gitignore`.
- **Zero Frontend Leakage:** The React frontend receives only sanitized diagnostic outputs and structured responses. API keys, authorization tokens, and cloud project IDs are never sent to the browser or embedded in client bundles.

### Automated Error Sanitization
All exceptions routed through `backend/watsonx_service.py` pass through regex sanitizers (`_sanitize_error_message`) that redact:
- API keys and Bearer tokens: `apikey: [REDACTED]`, `bearer: [REDACTED]`.
- Long hexadecimal or Base64 security tokens ($>32$ characters): `[REDACTED_TOKEN]`.
- Internal stack traces containing file paths or environment variables.

---

## 2. File Upload & Processing Security

### Pre-Flight Constraints
To prevent Denial-of-Service (DoS) and buffer exploitation:
- **Maximum File Size:** Limited to **15 MB** (`MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024`). Uploads exceeding this limit trigger an immediate `413 Payload Too Large` error.
- **Image Integrity Verification:** Uploaded bytes are verified by Pillow (`Image.open(io.BytesIO(file_bytes)).verify()`). Corrupted, truncated, or malicious non-image payloads are rejected immediately.
- **In-Memory Processing:** Images are transformed in memory buffers and not written to temporary disk storage, eliminating residual file traversal vulnerabilities.

---

## 3. Agricultural Safety Guardrails

Large language models can hallucinate incorrect or hazardous chemical dosages if not carefully constrained:
- **Zero Chemical Hallucination:** If a farmer asks for chemical pesticide dosages and no official, verified document in the Agriculture RAG knowledge base explicitly confirms the exact metric concentration, the assistant:
  1. Refuses to invent numerical dosages or metric application rates.
  2. Issues an explicit safety warning.
  3. Directs the farmer to inspect the chemical container label and consult their nearest Krishi Vigyan Kendra (KVK).
- **Prohibited Prescriptions:** The system explicitly forbids recommending unapproved chemical combinations or unauthorized tank mixes.

---

## 4. Cross-Origin Resource Sharing (CORS)

- **Development:** Permits `localhost:5173`, `127.0.0.1:5173`, `localhost:3000`.
- **Production:** Must be restricted to verified frontend domains via the `CORS_ORIGINS` environment variable (e.g., `CORS_ORIGINS=https://leafsense.yourdomain.com`). Wildcard origins (`*`) are disallowed when credentials or private cookies are enabled.

---

## 5. Credential Rotation Protocol

If an IBM Cloud API key or project ID is suspected of compromise:
1. Immediately revoke the key in the [IBM Cloud IAM Console](https://cloud.ibm.com/iam/apikeys).
2. Generate a new API key with scoped watsonx.ai editor privileges.
3. Update `backend/.env` with the new key.
4. Restart the backend service.

---

## 6. Reporting Security Vulnerabilities

If you discover a security issue or vulnerability within LeafSense:
- Do not create a public GitHub issue.
- Please notify the repository maintainer directly via GitHub security advisories or repository contact channels.
