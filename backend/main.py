"""
ICTV Virus Classification Agent — FastAPI backend.

Endpoints:
  POST /classify          Submit FASTA → returns job_id
  GET  /result/{job_id}   Poll job status / get result
  GET  /stream/{job_id}   SSE stream of reasoning steps
  GET  /family/{name}     Query ICTV criteria for a family
  GET  /species           Lookup species taxonomy
  GET  /health            Health check
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import classify_sequence
from .cache import cache_get, cache_put, cache_history, cache_get_by_hash, cache_delete, cache_clear
from .knowledge.criteria import get_criteria, list_families, get_demarcation_summary
from .models import ClassifyRequest, ClassifyResult, JobResponse, JobStatus
from .tools.taxonomy import family_summary, lookup_species, search_any_level

app = FastAPI(
    title="ICTV Virus Classifier",
    description="Classify virus sequences according to ICTV official demarcation criteria.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ──────────────────────────────────────────────────────
# In production, replace with Redis / database

_jobs: dict[str, JobResponse] = {}
_step_queues: dict[str, asyncio.Queue] = {}

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


# ── Background classification task ──────────────────────────────────────────

async def _run_classification(job_id: str, req: ClassifyRequest):
    _jobs[job_id].status = JobStatus.running
    queue = asyncio.Queue()
    _step_queues[job_id] = queue

    async def on_step(text: str):
        _jobs[job_id].steps.append(text)
        await queue.put(text)

    try:
        result, steps = await classify_sequence(
            fasta=req.fasta,
            max_steps=req.max_steps,
            step_callback=on_step,
            family_hint=req.family_hint,
        )
        _jobs[job_id].result = result
        _jobs[job_id].status = JobStatus.done
        # Save to cache
        if result:
            cache_put(req.fasta, result)
    except Exception as e:
        _jobs[job_id].status = JobStatus.error
        _jobs[job_id].error = str(e)
        _jobs[job_id].steps.append(f"[ERROR] {e}")
    finally:
        await queue.put(None)  # sentinel


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "families_in_criteria": len(list_families())}


@app.post("/classify")
async def classify(req: ClassifyRequest, background_tasks: BackgroundTasks):
    """Submit a FASTA sequence for classification. Returns a job_id.
    If the same sequence was classified before, returns the cached result instantly."""
    import logging
    logging.info(f"classify called: fasta_len={len(req.fasta)}")

    # Check cache first
    cached = cache_get(req.fasta)
    if cached:
        job_id = str(uuid.uuid4())
        _jobs[job_id] = JobResponse(
            job_id=job_id, status=JobStatus.done, result=cached,
            steps=["[Cache] Result loaded from local cache (identical sequence was classified before)."],
        )
        # Create a queue that immediately signals done, so SSE works
        queue = asyncio.Queue()
        _step_queues[job_id] = queue
        await queue.put("[Cache] Result loaded from local cache (identical sequence was classified before).")
        await queue.put(None)
        return {"job_id": job_id, "status": "done", "cached": True}

    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobResponse(job_id=job_id, status=JobStatus.pending)
    background_tasks.add_task(_run_classification, job_id, req)
    return {"job_id": job_id, "status": "pending"}


@app.get("/result/{job_id}", response_model=JobResponse)
def get_result(job_id: str):
    """Poll classification job status and result."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/stream/{job_id}")
async def stream_steps(job_id: str):
    """
    Server-Sent Events stream of reasoning steps for a job.
    Connect early (before or just after POST /classify) to receive steps in real time.
    """
    async def event_generator():
        # Wait for queue to be created (job may not have started yet)
        for _ in range(50):
            if job_id in _step_queues:
                break
            await asyncio.sleep(0.1)

        job = _jobs.get(job_id)
        if not job:
            yield "data: {\"error\": \"Job not found\"}\n\n"
            return

        # Replay already-logged steps and track how many we sent
        replayed = len(job.steps)
        for step in job.steps:
            yield f"data: {json.dumps({'step': step})}\n\n"

        queue = _step_queues.get(job_id)
        if queue is None:
            yield f"data: {json.dumps({'status': job.status.value})}\n\n"
            return

        skipped = 0
        while True:
            item = await queue.get()
            if item is None:  # sentinel — job finished
                final = _jobs.get(job_id)
                payload = {"status": final.status.value if final else "done"}
                if final and final.result:
                    payload["result"] = final.result.model_dump()
                if final and final.error:
                    payload["error"] = final.error
                yield f"data: {json.dumps(payload)}\n\n"
                break
            # Skip items already sent via replay
            if skipped < replayed:
                skipped += 1
                continue
            yield f"data: {json.dumps({'step': item})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/family/{family_name}")
def get_family_criteria(
    family_name: str,
    level: str = Query("all", description="species|genus|subfamily|all"),
):
    """Return ICTV demarcation criteria for a virus family."""
    crit = get_criteria(family_name)
    if not crit:
        families = list_families()
        raise HTTPException(
            status_code=404,
            detail=f"Family '{family_name}' not found. Available: {families[:20]}",
        )
    if level == "all":
        return crit
    return {
        "family": family_name,
        "level": level,
        "criteria": crit.get(f"{level}_demarcation"),
        "summary": get_demarcation_summary(family_name, level),
    }


@app.get("/families")
def list_families_endpoint():
    """List all families in the criteria knowledge base."""
    return {"families": list_families()}


@app.get("/species")
def species_lookup(
    q: str = Query(..., description="Species name or partial name"),
):
    """Look up virus species taxonomy from MSL40."""
    rows = lookup_species(q)
    if not rows:
        rows = search_any_level(q)
    if not rows:
        return {"results": [], "message": f"No results for '{q}'"}
    clean = [{k: v for k, v in r.items() if v is not None} for r in rows[:10]]
    return {"results": clean}


@app.get("/family/{family_name}/summary")
def family_summary_endpoint(family_name: str):
    """Return species and genus counts for a family."""
    s = family_summary(family_name)
    if not s:
        raise HTTPException(status_code=404, detail=f"Family '{family_name}' not in taxonomy DB")
    return {**s, "family": family_name}


# ── History / cache endpoints ────────────────────────────────────────────

@app.get("/history")
def get_history(limit: int = Query(20, le=100)):
    """Return recent classification results from cache."""
    return {"history": cache_history(limit)}


@app.get("/cache/{seq_hash}")
def get_cached_result(seq_hash: str):
    """Retrieve a cached result by sequence hash."""
    result = cache_get_by_hash(seq_hash)
    if not result:
        raise HTTPException(status_code=404, detail="Not found in cache")
    return result.model_dump()


@app.delete("/cache/{seq_hash}")
def delete_cached_result(seq_hash: str):
    """Delete a single cached result."""
    if cache_delete(seq_hash):
        return {"deleted": seq_hash}
    raise HTTPException(status_code=404, detail="Not found in cache")


@app.delete("/cache")
def clear_all_cache():
    """Delete all cached results."""
    count = cache_clear()
    return {"deleted_count": count}


# ── Serve frontend ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the frontend index.html."""
    html_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>ICTV Classifier</h1><p>Frontend not found.</p>")


# Mount static files (JS/CSS) if frontend dir exists
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
