"""
server/app.py – OpenEnv-required entry point.
Delegates everything to the root app.py so there is no duplication.
"""
import sys
import os

# Ensure the repo root is on the path so `app` can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # re-export the FastAPI instance
import uvicorn


def main():
    """Entry point referenced by openenv.yaml (server.app:main)."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
