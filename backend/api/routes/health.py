from __future__ import annotations

from fastapi import APIRouter

from backend.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "message": "IPC Legal Assistant API is running"}


@router.get("/health")
def health_check() -> dict[str, bool | str]:
    return {
        "status": "healthy" if settings.persist_directory.exists() else "warning",
        "vectorstore_initialized": settings.persist_directory.exists(),
    }

