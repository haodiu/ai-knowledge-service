"""Internal ingestion endpoint (Plan §6), deferred from Week 5 until a real auth context existed.

Service-token auth (get_service_auth), NOT the user JWT (get_auth) -- host-to-host, no
tier/AuthorizationContext involved. `app.ingestion.service` is synchronous (Week 5, shared with
the CLI and the Celery task -- app.ingestion.tasks), so every call into it here is wrapped in
`asyncio.to_thread()`, the same pattern already used for kombu's blocking AMQP probe
(app/api/dependencies.py::rabbitmq).
"""
import asyncio
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import Engine, text

from app.api.dependencies import get_service_auth, get_sync_engine
from app.ingestion.service import IngestionError, SupersededContentError, create_ingestion_job
from app.retrieval.schemas import Tier

router = APIRouter(
    prefix="/internal/ingestions", tags=["ingestions"], dependencies=[Depends(get_service_auth)]
)


class IngestionRequest(BaseModel):
    external_id: str
    title: str
    tier: Tier
    content: str
    embedding_model: str


class IngestionResponse(BaseModel):
    # None only for status="completed": content already matched the active version, so nothing
    # was queued and there is no job row to poll (Week 5/6 decision: no job row for a pure no-op).
    job_id: uuid.UUID | None
    status: str
    document_external_id: str
    version_no: int


class JobStatusResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    document_external_id: str
    version_no: int
    error_code: str | None
    retry_count: int


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=IngestionResponse)
async def submit_ingestion(
    body: IngestionRequest, engine: Annotated[Engine, Depends(get_sync_engine)]
) -> IngestionResponse:
    try:
        job = await asyncio.to_thread(
            create_ingestion_job,
            engine,
            external_id=body.external_id,
            title=body.title,
            tier=body.tier,
            content=body.content,
            embedding_model=body.embedding_model,
        )
    except SupersededContentError as exc:
        # No separate HTTP error branch (Week 5 decision): still 202, pointing at the job row
        # that records the rejection (status='failed', error_code='superseded_content').
        assert exc.job_id is not None
        return IngestionResponse(
            job_id=exc.job_id, status="failed",
            document_external_id=body.external_id, version_no=exc.version_no,
        )
    except IngestionError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if job.status == "queued":
        assert job.job_id is not None
        from app.ingestion.tasks import ingest_document_task  # local: needs RABBITMQ_URL resolvable

        await asyncio.to_thread(ingest_document_task.delay, str(job.job_id))

    return IngestionResponse(
        job_id=job.job_id, status=job.status,
        document_external_id=body.external_id, version_no=job.version_no,
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_ingestion(
    job_id: uuid.UUID, engine: Annotated[Engine, Depends(get_sync_engine)]
) -> JobStatusResponse:
    def _load() -> JobStatusResponse | None:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT j.status, j.error_code, j.retry_count, j.version_no, d.external_id "
                    "FROM ingestion_jobs j JOIN documents d ON d.id = j.document_id "
                    "WHERE j.id = :j"
                ),
                {"j": job_id},
            ).mappings().first()
        if row is None:
            return None
        return JobStatusResponse(
            job_id=job_id, status=row["status"], document_external_id=row["external_id"],
            version_no=row["version_no"], error_code=row["error_code"],
            retry_count=row["retry_count"],
        )

    result = await asyncio.to_thread(_load)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="job not found")
    return result
