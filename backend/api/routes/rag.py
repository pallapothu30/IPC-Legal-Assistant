from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.query import ConfigResponse, QueryRequest, QueryResponse
from backend.services.rag_service import get_app_config, get_answer

router = APIRouter(tags=["RAG"])


@router.get("/config", response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    return get_app_config()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return get_answer(request)

