"""
LeafSense Lite Profile Entrypoint
Runs LeafSense in lightweight, memory-optimized single-model mode (DenseNet121).
Ideal for 512MB RAM cloud tiers (Render Free, Fly.io, etc.).
"""

import os

# Enforce lite mode before importing app
os.environ["LEAFSENSE_MODE"] = "lite"

from app import app  # noqa: E402

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
