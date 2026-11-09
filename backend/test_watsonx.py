#!/usr/bin/env python3
"""
LeafSense - IBM watsonx.ai & Granite Connection Verification Script
===================================================================
Verifies the end-to-end integration of IBM watsonx.ai and IBM Granite
for the LeafSense backend.

Ensures:
1. Credentials exist in backend/.env (without printing secret values).
2. IBM Cloud authentication succeeds.
3. Available Granite generative models in Dallas region are verified.
4. Granite generation executes successfully with the test prompt.
5. Errors (if any) are safely categorized with actionable steps.
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from watsonx_service import WatsonxGraniteService, WatsonxServiceError


def run_verification():
    print("=" * 70)
    print("  LeafSense - IBM watsonx.ai & Granite Connection Verification")
    print("=" * 70)

    service = WatsonxGraniteService()
    status = service.get_credential_status()

    # Step 1: Check environment variables
    print("\n[Step 1/4] Checking Environment Configuration...")
    print(f"  • WATSONX_URL:         {status['url']}")
    print(f"  • WATSONX_API_KEY:     {'[CONFIGURED]' if status['api_key_configured'] else '[MISSING]'}")
    print(f"  • WATSONX_PROJECT_ID:  {'[CONFIGURED]' if status['project_id_configured'] else '[MISSING]'}")
    print(f"  • Configured Model ID: {status['configured_model_id']}")

    if not status["ready"]:
        print("\n❌ Error: Missing configuration!")
        print("Please ensure both WATSONX_API_KEY and WATSONX_PROJECT_ID are set in backend/.env.")
        sys.exit(1)

    print("  ✔ Environment variables detected successfully.")

    # Step 2: Test IBM Cloud Authentication & Client Initialization
    print("\n[Step 2/4] Initializing IBM watsonx.ai Client...")
    try:
        client = service._get_api_client()
        print("  ✔ IBM watsonx.ai API client initialized successfully.")
    except WatsonxServiceError as e:
        print(f"\n❌ Client Initialization Failed [{e.category}]:")
        print(f"   {e}")
        if e.suggestions:
            print("\nRecommended Actions:")
            for s in e.suggestions:
                print(f"   - {s}")
        sys.exit(1)

    # Step 3: Query Available Granite Foundation Models (Generative)
    print("\n[Step 3/4] Querying Available Granite Generative Models in Region...")
    available_models = []
    try:
        available_models = service.list_available_granite_models(filter_generative=True)
        if available_models:
            print(f"  ✔ Found {len(available_models)} Granite generative model(s) available in this region:")
            for mid in available_models:
                marker = " -> [ACTIVE SELECTION]" if mid == service.model_id else ""
                print(f"     - {mid}{marker}")
        else:
            print("  ℹ No generative models matching 'granite' returned by model spec catalog.")
    except Exception as e:
        print(f"  ⚠ Could not query model list ({e}). Attempting direct inference with {service.model_id}...")

    # Determine active model
    active_model = service.model_id
    if available_models and active_model not in available_models:
        print(f"\n  ⚠ Notice: Configured model '{active_model}' is not in the active generative catalog.")
        active_model = available_models[0]
        print(f"  Automatically selecting available Granite model: '{active_model}'")

    # Step 4: Test IBM Granite Inference
    test_prompt = "Explain how AI can help farmers detect plant diseases."
    print(f"\n[Step 4/4] Sending Test Prompt to IBM Granite ({active_model})...")
    print(f"  Prompt: \"{test_prompt}\"")
    print("  Waiting for response from IBM watsonx.ai...")

    try:
        response_text = service.generate_text(
            prompt=test_prompt,
            system_prompt="You are a helpful agricultural AI assistant for LeafSense.",
            max_new_tokens=300,
            temperature=0.7,
            decoding_method="greedy",
            model_id=active_model
        )

        print("\n" + "=" * 70)
        print("  ✔ SUCCESS: Valid IBM Granite response received!")
        print("=" * 70)
        print(f"\nModel: {active_model}\n")
        print(response_text)
        print("\n" + "=" * 70)
        print("  Phase 2 IBM Granite integration verification completed successfully!")
        print("=" * 70)
        return True

    except WatsonxServiceError as e:
        print(f"\n❌ Inference Failed [{e.category}]:")
        print(f"   {e}")
        if e.suggestions:
            print("\nRecommended Actions:")
            for s in e.suggestions:
                print(f"   - {s}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error During Inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_verification()
