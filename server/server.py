import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import onnxruntime as ort
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from PIL import Image
from uuid6 import uuid7
from werkzeug.utils import secure_filename

from vecdb.database import QdrantMultimodalDB
from vecdb.pipeline import Pipeline

load_dotenv()

DEFAULT_COLLECTION = "general"

# Ensure CUDA/cuDNN libs are preloaded when using GPU
PROVIDERS = os.getenv("ONNX_PROVIDERS", "CPUExecutionProvider").strip().split(",")
if "CUDAExecutionProvider" in PROVIDERS and hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

PROVIDERS = [
    (
        name,
        {
            "arena_extend_strategy": "kSameAsRequested",
            # "gpu_mem_limit": 1 * 1024 * 1024 * 1024,
        },
    )
    for name in PROVIDERS
]

executor = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict] = {}
jobs_lock = Lock()

# --- Storage & Path-to-VectorDB Manifest ---
FILES_ROOT = Path(os.getenv("STORAGE_DIR", "./uploaded_files")).resolve()
FILES_ROOT.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = FILES_ROOT / ".file_manifest.json"
manifest_lock = Lock()


def _load_manifest() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _save_manifest(data: dict[str, dict]) -> None:
    MANIFEST_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _safe_resolve(rel_path: str) -> tuple[Path, str]:
    """Resolve a user-supplied relative path safely inside FILES_ROOT."""
    cleaned = rel_path.strip().lstrip("/\\")
    target = (FILES_ROOT / cleaned).resolve()
    if not target.is_relative_to(FILES_ROOT):
        raise ValueError("Invalid path: escapes storage root")
    norm_rel = target.relative_to(FILES_ROOT).as_posix()
    return target, norm_rel


def _ensure_collection(collection: str) -> None:
    existing = {c.name for c in db.get_available_collections().collections}
    if collection not in existing:
        db.create_new_collection(collection)


def _set_job(job_id: str, **updates):
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(updates)


def _run_index_job(job_id: str, task_fn, **kwargs):
    _set_job(job_id, status="processing")
    try:
        file_id = task_fn(**kwargs)
        _set_job(job_id, status="completed", file_id=file_id)
    except Exception as e:
        _set_job(job_id, status="failed", error=str(e))


app = Flask(__name__)

db_host = os.getenv("QDRANT_HOST", None)
db_port = os.getenv("QDRANT_PORT", None)

if (db_host is None) or (db_port is None):
    db = QdrantMultimodalDB(
        db_path=os.getenv("QDRANT_PATH", "./qdrant_local"),
        models_cache=os.getenv("MODELS_CACHE", "./models"),
        providers=PROVIDERS,
    )
else:
    db = QdrantMultimodalDB(
        host=db_host,
        port=int(db_port),
        models_cache=os.getenv("MODELS_CACHE", "./models"),
        providers=PROVIDERS,
    )

pipeline = Pipeline(
    db=db,
    storage_dir=str(FILES_ROOT),
)
_ensure_collection(DEFAULT_COLLECTION)


# --- Collections ---

@app.get("/collections")
def list_collections():
    cols = [c.name for c in db.get_available_collections().collections]
    return jsonify({"collections": cols})


@app.post("/collections")
def create_collection():
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    if not name:
        return jsonify({"error": "Missing 'name'"}), 400
    db.create_new_collection(name)
    return jsonify({"status": "created", "collection": name}), 201


@app.delete("/collections/<collection>")
def delete_collection(collection: str):
    db.delete_collection(collection)
    return jsonify({"status": "deleted", "collection": collection})


# --- Directory CRUD ---

@app.get("/folders", defaults={"subpath": ""})
@app.get("/folders/<path:subpath>")
def list_directory(subpath: str):
    try:
        target, norm_rel = _safe_resolve(subpath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not target.exists() or not target.is_dir():
        return jsonify({"error": "Directory not found"}), 404

    with manifest_lock:
        manifest = _load_manifest()

    folders = []
    files = []
    for entry in sorted(target.iterdir(), key=lambda p: p.name):
        if entry.name == MANIFEST_PATH.name:
            continue
        rel = entry.relative_to(FILES_ROOT).as_posix()
        if entry.is_dir():
            folders.append({"name": entry.name, "path": rel})
        else:
            info = manifest.get(rel, {})
            files.append({
                "name": entry.name,
                "path": rel,
                "file_id": info.get("file_id"),
                "collection": info.get("collection"),
            })

    return jsonify({"path": norm_rel, "folders": folders, "files": files})


@app.post("/folders/<path:subpath>")
def create_directory(subpath: str):
    try:
        target, norm_rel = _safe_resolve(subpath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    target.mkdir(parents=True, exist_ok=True)
    return jsonify({"status": "created", "path": norm_rel}), 201


@app.delete("/folders/<path:subpath>")
def delete_directory(subpath: str):
    try:
        target, norm_rel = _safe_resolve(subpath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if target == FILES_ROOT:
        return jsonify({"error": "Cannot delete root directory"}), 400
    if not target.exists() or not target.is_dir():
        return jsonify({"error": "Directory not found"}), 404

    # Remove all indexed files under this folder from Qdrant
    prefix = f"{norm_rel}/"
    deleted_files = []
    with manifest_lock:
        manifest = _load_manifest()
        to_remove = [p for p in manifest if p.startswith(prefix)]
        for rel_path in to_remove:
            entry = manifest.pop(rel_path)
            pipeline.delete_file(file_id=entry["file_id"], collection=entry["collection"])
            deleted_files.append(rel_path)
        _save_manifest(manifest)

    shutil.rmtree(target)
    return jsonify({
        "status": "deleted",
        "path": norm_rel,
        "deleted_files": deleted_files,
    })


# --- File CRUD (Directory-based, optional collection) ---

@app.get("/files/<path:filepath>")
def get_file(filepath: str):
    try:
        target, _ = _safe_resolve(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not target.exists() or not target.is_file():
        return jsonify({"error": "File not found"}), 404
    return send_file(target)


@app.post("/files/<path:filepath>")
def create_file(filepath: str):
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in form-data"}), 400

    try:
        target, norm_rel = _safe_resolve(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if target.exists():
        return jsonify({"error": "File already exists; use PUT to update"}), 409

    uploaded = request.files["file"]
    collection = request.form.get("collection") or DEFAULT_COLLECTION
    metadata_raw = request.form.get("metadata")
    metadata = json.loads(metadata_raw) if metadata_raw else {}
    metadata.setdefault("original_filename", uploaded.filename or target.name)
    metadata["relative_path"] = norm_rel

    _ensure_collection(collection)

    # Save file into the structured directory (creating parent dirs automatically)
    target.parent.mkdir(parents=True, exist_ok=True)
    uploaded.save(target)

    job_id = str(uuid7())
    file_id = str(uuid7())

    with manifest_lock:
        manifest = _load_manifest()
        manifest[norm_rel] = {"file_id": file_id, "collection": collection}
        _save_manifest(manifest)

    _set_job(job_id, status="queued", collection=collection, file_id=file_id, path=norm_rel)
    executor.submit(
        _run_index_job,
        job_id,
        pipeline.upload_file,
        source_path=target,
        collection=collection,
        file_id=file_id,
        metadata=metadata,
    )

    return jsonify({
        "job_id": job_id,
        "file_id": file_id,
        "path": norm_rel,
        "collection": collection,
        "status": "queued",
    }), 202


@app.put("/files/<path:filepath>")
def update_file(filepath: str):
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in form-data"}), 400

    try:
        target, norm_rel = _safe_resolve(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    with manifest_lock:
        manifest = _load_manifest()
        existing = manifest.get(norm_rel)

    if not existing:
        return jsonify({"error": "File not found; use POST to create"}), 404

    uploaded = request.files["file"]
    old_collection = existing["collection"]
    file_id = existing["file_id"]
    new_collection = request.form.get("collection") or old_collection

    metadata_raw = request.form.get("metadata")
    metadata = json.loads(metadata_raw) if metadata_raw else {}
    metadata.setdefault("original_filename", uploaded.filename or target.name)
    metadata["relative_path"] = norm_rel

    _ensure_collection(new_collection)
    uploaded.save(target)

    job_id = str(uuid7())
    _set_job(job_id, status="queued", collection=new_collection, file_id=file_id, path=norm_rel)

    # If collection changed, remove from old collection and index into new one
    if new_collection != old_collection:
        def _move_and_reindex():
            pipeline.delete_file(file_id=file_id, collection=old_collection)
            return pipeline.upload_file(
                source_path=target,
                collection=new_collection,
                file_id=file_id,
                metadata=metadata,
            )

        with manifest_lock:
            manifest = _load_manifest()
            manifest[norm_rel]["collection"] = new_collection
            _save_manifest(manifest)

        executor.submit(_run_index_job, job_id, _move_and_reindex)
    else:
        executor.submit(
            _run_index_job,
            job_id,
            pipeline.update_file,
            file_id=file_id,
            new_source_path=target,
            collection=new_collection,
            metadata=metadata,
        )

    return jsonify({
        "job_id": job_id,
        "file_id": file_id,
        "path": norm_rel,
        "collection": new_collection,
        "status": "queued",
    }), 202


@app.delete("/files/<path:filepath>")
def delete_file(filepath: str):
    try:
        target, norm_rel = _safe_resolve(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    with manifest_lock:
        manifest = _load_manifest()
        entry = manifest.pop(norm_rel, None)
        _save_manifest(manifest)

    if not entry and not target.exists():
        return jsonify({"error": "File not found"}), 404

    if entry:
        pipeline.delete_file(file_id=entry["file_id"], collection=entry["collection"])
    target.unlink(missing_ok=True)

    return jsonify({
        "status": "deleted",
        "path": norm_rel,
        "file_id": entry["file_id"] if entry else None,
    })


@app.get("/jobs/<job_id>")
def get_job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"job_id": job_id, **job})


# --- Query ---

@app.post("/collections/<collection>/query")
def query_collection(collection: str):
    if "image" in request.files:
        query_input = Image.open(request.files["image"].stream).convert("RGB")
        limit = int(request.form.get("limit", 5))
        file_id = request.form.get("file_id")
        threshold = request.form.get("score_threshold")
        score_threshold = float(threshold) if threshold is not None else None
    else:
        data = request.get_json(silent=True) or {}
        query_input = data.get("query")
        if not query_input:
            return jsonify({"error": "Provide 'query' (JSON) or 'image' (form-data)"}), 400
        limit = int(data.get("limit", 5))
        file_id = data.get("file_id")
        score_threshold = data.get("score_threshold")

    hits = pipeline.query(
        query=query_input,
        collection=collection,
        limit=limit,
        file_id=file_id,
        score_threshold=score_threshold,
    )

    return jsonify({
        "results": [
            {"id": str(h.id), "score": h.score, "payload": h.payload}
            for h in hits
        ]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)