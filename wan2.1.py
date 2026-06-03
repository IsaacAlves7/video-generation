"""
Wan 2.1 Studio — FastAPI backend
Roda 100% local, sem API externa, sem custo.
"""

import os
import uuid
import asyncio
import logging
import time
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from diffusers import WanPipeline, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wan-studio")

# ── config ─────────────────────────────────────────────────────────────────────
# Modelos disponíveis:
#   Wan-AI/Wan2.1-T2V-1.3B   → ~8 GB VRAM  (RTX 3060/3070/4060)
#   Wan-AI/Wan2.1-T2V-14B    → ~24 GB VRAM (RTX 3090/4090 / A100)
#   Wan-AI/Wan2.1-I2V-14B-480P → para imagem→vídeo 480p
#   Wan-AI/Wan2.1-I2V-14B-720P → para imagem→vídeo 720p
T2V_MODEL = os.getenv("WAN_T2V_MODEL", "Wan-AI/Wan2.1-T2V-1.3B")
I2V_MODEL = os.getenv("WAN_I2V_MODEL", "Wan-AI/Wan2.1-I2V-14B-480P")

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# ── device ─────────────────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info(f"GPU detectada: {name} ({vram:.1f} GB VRAM)")
        return "cuda"
    if torch.backends.mps.is_available():
        log.info("Apple Silicon MPS detectado")
        return "mps"
    log.warning("Nenhuma GPU detectada — usando CPU (muito lento!)")
    return "cpu"

DEVICE = get_device()
DTYPE  = torch.bfloat16 if DEVICE in ("cuda", "mps") else torch.float32

# ── model loader ───────────────────────────────────────────────────────────────
t2v_pipe: Optional[WanPipeline] = None
i2v_pipe: Optional[WanImageToVideoPipeline] = None
models_loaded = {"t2v": False, "i2v": False}

def load_t2v():
    global t2v_pipe
    log.info(f"Carregando T2V: {T2V_MODEL}")
    t2v_pipe = WanPipeline.from_pretrained(T2V_MODEL, torch_dtype=DTYPE)
    t2v_pipe.to(DEVICE)
    # Otimizações de memória
    if DEVICE == "cuda":
        t2v_pipe.enable_model_cpu_offload()
        t2v_pipe.vae.enable_slicing()
        t2v_pipe.vae.enable_tiling()
    models_loaded["t2v"] = True
    log.info("✅ Modelo T2V carregado")

def load_i2v():
    global i2v_pipe
    log.info(f"Carregando I2V: {I2V_MODEL}")
    i2v_pipe = WanImageToVideoPipeline.from_pretrained(I2V_MODEL, torch_dtype=DTYPE)
    i2v_pipe.to(DEVICE)
    if DEVICE == "cuda":
        i2v_pipe.enable_model_cpu_offload()
        i2v_pipe.vae.enable_slicing()
        i2v_pipe.vae.enable_tiling()
    models_loaded["i2v"] = True
    log.info("✅ Modelo I2V carregado")

# Executor para rodar inferência em thread separada (não bloqueia o event loop)
executor = ThreadPoolExecutor(max_workers=1)  # 1 job por vez (GPU é single-tenant)

# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Wan 2.1 Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve os vídeos gerados como arquivos estáticos
app.mount("/videos", StaticFiles(directory=str(OUTPUTS_DIR)), name="videos")

# ── schemas ────────────────────────────────────────────────────────────────────
class T2VRequest(BaseModel):
    prompt: str
    negative_prompt: str = "nsfw, low quality, blur, watermark, distorted"
    num_frames: int = 81          # 81 frames ÷ 16 fps ≈ 5s | 161 ≈ 10s
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    width: int = 832
    height: int = 480
    seed: int = -1                # -1 = aleatório

class I2VRequest(BaseModel):
    prompt: str
    image_url: str
    negative_prompt: str = "nsfw, low quality, blur, watermark, distorted"
    num_frames: int = 81
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    seed: int = -1

class JobResponse(BaseModel):
    job_id: str
    status: str                   # queued | processing | completed | failed
    video_url: Optional[str] = None
    error: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    prompt: Optional[str] = None
    mode: Optional[str] = None

# In-memory store
jobs: dict[str, dict] = {}

# ── geração T2V ────────────────────────────────────────────────────────────────
def _run_t2v(job_id: str, req: T2VRequest):
    start = time.time()
    try:
        if not models_loaded["t2v"]:
            load_t2v()

        generator = None
        if req.seed != -1:
            generator = torch.Generator(device=DEVICE).manual_seed(req.seed)

        log.info(f"[{job_id}] Gerando T2V: '{req.prompt[:60]}…'")
        output = t2v_pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_frames=req.num_frames,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            width=req.width,
            height=req.height,
            generator=generator,
        )

        filename = f"{job_id}.mp4"
        filepath = OUTPUTS_DIR / filename
        export_to_video(output.frames[0], str(filepath), fps=16)

        elapsed = round(time.time() - start, 1)
        jobs[job_id].update({
            "status": "completed",
            "video_url": f"/videos/{filename}",
            "elapsed_seconds": elapsed,
        })
        log.info(f"[{job_id}] ✅ Concluído em {elapsed}s → {filename}")

    except Exception as e:
        log.error(f"[{job_id}] ❌ Erro: {e}")
        jobs[job_id].update({"status": "failed", "error": str(e)})

# ── geração I2V ────────────────────────────────────────────────────────────────
def _run_i2v(job_id: str, req: I2VRequest):
    start = time.time()
    try:
        if not models_loaded["i2v"]:
            load_i2v()

        image = load_image(req.image_url)
        generator = None
        if req.seed != -1:
            generator = torch.Generator(device=DEVICE).manual_seed(req.seed)

        log.info(f"[{job_id}] Gerando I2V: '{req.prompt[:60]}…'")
        output = i2v_pipe(
            image=image,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_frames=req.num_frames,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            generator=generator,
        )

        filename = f"{job_id}.mp4"
        filepath = OUTPUTS_DIR / filename
        export_to_video(output.frames[0], str(filepath), fps=16)

        elapsed = round(time.time() - start, 1)
        jobs[job_id].update({
            "status": "completed",
            "video_url": f"/videos/{filename}",
            "elapsed_seconds": elapsed,
        })
        log.info(f"[{job_id}] ✅ Concluído em {elapsed}s → {filename}")

    except Exception as e:
        log.error(f"[{job_id}] ❌ Erro: {e}")
        jobs[job_id].update({"status": "failed", "error": str(e)})

# ── routes ─────────────────────────────────────────────────────────────────────
@app.post("/api/generate/text-to-video", response_model=JobResponse)
async def text_to_video(req: T2VRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt não pode estar vazio")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id, "status": "queued",
        "video_url": None, "error": None,
        "elapsed_seconds": None,
        "prompt": req.prompt, "mode": "t2v",
    }
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_t2v, job_id, req)
    return JobResponse(**jobs[job_id])


@app.post("/api/generate/image-to-video", response_model=JobResponse)
async def image_to_video(req: I2VRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt não pode estar vazio")
    if not req.image_url.strip():
        raise HTTPException(400, "image_url não pode estar vazio")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id, "status": "queued",
        "video_url": None, "error": None,
        "elapsed_seconds": None,
        "prompt": req.prompt, "mode": "i2v",
    }
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_i2v, job_id, req)
    return JobResponse(**jobs[job_id])


@app.get("/api/status/{job_id}", response_model=JobResponse)
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job não encontrado")
    return JobResponse(**jobs[job_id])


@app.get("/api/jobs")
async def list_jobs():
    return list(reversed(list(jobs.values())))


@app.get("/health")
async def health():
    gpu_info = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info = {
            "name": props.name,
            "vram_total_gb": round(props.total_memory / 1024**3, 1),
            "vram_free_gb": round(
                (props.total_memory - torch.cuda.memory_allocated()) / 1024**3, 1
            ),
        }
    return {
        "status": "ok",
        "device": DEVICE,
        "gpu": gpu_info,
        "models_loaded": models_loaded,
        "t2v_model": T2V_MODEL,
        "i2v_model": I2V_MODEL,
        "jobs_total": len(jobs),
        "jobs_processing": sum(1 for j in jobs.values() if j["status"] == "processing"),
    }


@app.on_event("startup")
async def startup():
    log.info("🚀 Wan 2.1 Studio iniciando…")
    log.info(f"   Device: {DEVICE} | Dtype: {DTYPE}")
    log.info(f"   T2V: {T2V_MODEL}")
    log.info(f"   I2V: {I2V_MODEL}")
    log.info("   Modelos serão carregados na primeira requisição (lazy loading)")
    log.info("   Para pré-carregar: GET /api/preload")


@app.get("/api/preload")
async def preload():
    """Pré-carrega ambos os modelos em background."""
    loop = asyncio.get_event_loop()
    if not models_loaded["t2v"]:
        loop.run_in_executor(executor, load_t2v)
    return {"message": "Carregamento iniciado. Acompanhe em /health"}
