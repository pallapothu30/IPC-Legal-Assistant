#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.api.app import app
from backend.core.config import settings

if __name__ == "__main__":
    print("Starting IPC Legal Assistant API Backend...")
    print(f"API will be available at: http://{settings.api_host}:{settings.api_port}")
    print(f"API Documentation at: http://{settings.api_host}:{settings.api_port}/docs")
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_level="info")
