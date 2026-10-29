"""Local embedding backend for ``embedding_api_base: local``.

Part of the built-in **embedding plugin** (``plugins/embedding``); moved here
from ``jindai/local_embedding.py`` (2026-10-29) and loaded lazily by
``jindai.models.TextEmbeddings`` when the config asks for it.

When the config sets ``embedding_api_base`` to the literal value
``local``, the app stops POSTing to an OpenAI-compatible
``/v1/embeddings`` server and instead runs the Hugging Face model named
by ``embedding_model`` (e.g. ``BAAI/bge-m3``) in-process, on this
machine's GPU.  The model is fetched from the Hugging Face Hub on first
use and cached under ``HF_HOME`` (mounted to ``./hf-cache`` in the
compose file so it survives rebuilds).

Implementation: the Hub's exported ``onnx/model.onnx`` through
``onnxruntime-gpu`` (already a project dependency, CUDA EP with a CPU
fallback), plus the Hub tokenizer through ``transformers``.  No torch in
the image, no separate embedding server, no HTTP hop.

**Precision**: the Hub export is fp32 (2.11 GiB).  ``config.embedding_onnx_path``
can point at a lower-precision conversion; the shipped one is
``quant/bge-m3-fp16.onnx`` (1.06 GiB, built 2026-10-07 and checked to
``cos >= 0.99998`` against both fp32 and the old llama-server's fp16 GGUF
vectors).  int8 was measured and **rejected**: dynamic int8 lands at
cos 0.976-0.980 against the existing vectors (breaks index consistency) and
its MatMulInteger ops are not implemented by the CUDA EP; the static QDQ
route emits an invalid model for this opset-11 export.

**Output parity is a hard requirement.**  ``text_embeddings`` was seeded
by a llama-server bge-m3 (``--pooling cls``, L2-normalised output), so
this backend must keep producing the same thing:

* CLS pooling - the first token's hidden state, not mean pooling
  (BAAI/bge-m3's own ``1_Pooling/config.json`` also says cls);
* L2 normalisation to unit length;
* ``config.embedding_dims`` components (bge-m3: 1024).

Mixing unnormalised or mean-pooled vectors into ``text_embeddings``
silently degrades cosine search (new rows score ~0.9 against old ones
instead of matching), so verify with a cosine comparison against a
reference server before flipping the config - and after any change here.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import sysconfig
import threading
from pathlib import Path
from typing import Optional

from jindai.config import config

# Sentinel value: embedding_api_base == "local" selects this backend.
LOCAL = "local"

_load_lock = threading.Lock()   # serialises first-time model loading
_infer_lock = threading.Lock()  # one inference at a time (shared session)
_state: Optional[dict] = None
_cuda_preloaded = False

# Order matters for the first pass: things that depend on the CUDA runtime
# must find it already loaded (RTLD_GLOBAL).
_LIB_ORDER = [
    "cuda_runtime", "cuda_nvrtc", "cublas", "cudnn", "cusparse", "cusolver",
    "curand", "cufft", "cuda_cupti", "nvjitlink", "cusparselt", "nccl",
    "nvshmem", "cufile", "nvtx",
]


def is_local(base: Optional[str]) -> bool:
    """True when the configured embedding API base selects this backend."""
    return (base or "").strip().lower().rstrip("/") == LOCAL


def _preload_cuda_libs() -> None:
    """Make the venv's nvidia pip packages visible to the dynamic loader.

    onnxruntime-gpu's CUDA EP dlopens libcudart/libcudnn/libcublas, but those
    live in ``<site-packages>/nvidia/*/lib`` which is not on the loader's
    path.  Preload them with RTLD_GLOBAL so ORT's own dlopen resolves.
    """
    global _cuda_preloaded
    if _cuda_preloaded:
        return
    _cuda_preloaded = True
    nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if not nvidia.is_dir():
        logging.warning("local embedding: no venv nvidia packages at %s", nvidia)
        return
    seen: set[Path] = set()

    def _load(lib: Path) -> None:
        if lib in seen:
            return
        seen.add(lib)
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass  # unresolvable sibling; a later pass may pick it up

    names = _LIB_ORDER + sorted(p.name for p in nvidia.iterdir() if p.is_dir())
    for name in names:
        libdir = nvidia / name / "lib"
        if libdir.is_dir():
            for lib in sorted(libdir.glob("*.so*")):
                _load(lib)
    # second pass: anything that failed because its dependency came later
    for name in names:
        libdir = nvidia / name / "lib"
        if libdir.is_dir():
            for lib in sorted(libdir.glob("*.so*")):
                _load(lib)


def _make_session(onnx_path: str):
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers: list = []
    if "CUDAExecutionProvider" in available:
        _preload_cuda_libs()
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    else:
        logging.warning(
            "local embedding: this onnxruntime build has no CUDA provider (%s); "
            "running on the CPU", available,
        )
    providers.append("CPUExecutionProvider")

    def _make(level):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = level
        return ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)

    try:
        sess = _make(ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    except Exception:   # noqa: BLE001 - ORT raises its own RuntimeException type,
        # which is NOT a subclass of RuntimeError: fp16 conversions insert
        # precision-free casts that crash ORT's transformer fusions
        # (SimplifiedLayerNormFusion); BASIC keeps the generic fusions and loads.
        logging.info("local embedding: ORT_ENABLE_ALL failed, retrying at BASIC")
        sess = _make(ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
    if sess.get_providers()[0] != "CUDAExecutionProvider":
        logging.warning(
            "local embedding: CUDA provider failed to initialise, running on the "
            "CPU (providers=%s)", sess.get_providers(),
        )
    else:
        logging.info("local embedding: onnxruntime CUDA provider active")
    return sess


def _ensure_loaded() -> dict:
    """Load tokenizer + ONNX session once (lazy: only on first local embed)."""
    global _state
    if _state is not None:
        return _state
    with _load_lock:
        if _state is not None:
            return _state
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        name = config.embedding_model
        if not name:
            raise RuntimeError(
                "embedding_api_base is 'local' but embedding_model is empty "
                "(expected a Hugging Face model id, e.g. BAAI/bge-m3)"
            )
        logging.info("local embedding: preparing %s ...", name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        # ONNX file: config.embedding_onnx_path optionally points at a
        # lower-precision conversion (fp16 lives at HF_HOME/quant/...), an
        # absolute path or one relative to HF_HOME; empty/missing -> the Hub's
        # fp32 export (hf_hub_download resolves the HF_HOME cache, which the
        # compose file mounts as ./hf-cache so it survives rebuilds).
        onnx_path = None
        override = (getattr(config, "embedding_onnx_path", "") or "").strip()
        if override:
            p = Path(override)
            if not p.is_absolute():
                hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
                p = Path(hf_home) / override
            if p.exists():
                onnx_path = str(p)
                # size = graph file + its external data file(s)
                blob = p.stat().st_size + sum(
                    f.stat().st_size for f in p.parent.glob(p.name + ".*"))
                logging.info("local embedding: using %s (%.2f GiB)", p, blob / 2**30)
            else:
                logging.warning(
                    "local embedding: embedding_onnx_path %s not found, "
                    "falling back to the Hub fp32 export", p)
        if onnx_path is None:
            onnx_path = hf_hub_download(repo_id=name, filename="onnx/model.onnx")
        sess = _make_session(onnx_path)
        dims = getattr(config, "embedding_dims", None)
        if dims:
            out_name = sess.get_outputs()[0].shape
            got = out_name[-1] if isinstance(out_name, (list, tuple)) else None
            if isinstance(got, int) and got != dims:
                logging.warning(
                    "local embedding: model %s outputs %s dims, config embedding_dims=%d",
                    name, got, dims,
                )
        _state = {"tokenizer": tokenizer, "session": sess, "name": name}
        logging.info("local embedding: %s ready (%s)", name, sess.get_providers()[0])
        return _state


def _encode_sync(texts: list[str], batch_size: int = 128) -> list[list[float]]:
    """CLS-pooled, L2-normalised embeddings - same shape as the old server."""
    if not texts:
        return []
    import numpy as np

    st = _ensure_loaded()
    tokenizer, sess = st["tokenizer"], st["session"]
    input_names = {i.name for i in sess.get_inputs()}
    # An empty string is what a blank paragraph chunk looks like; the
    # tokenizer still needs something to bite on.
    batch_texts = [t.strip() or " " for t in texts]
    out: list[list[float]] = []
    with _infer_lock:
        for start in range(0, len(batch_texts), batch_size):
            enc = tokenizer(
                batch_texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="np",
            )
            feed = {
                k: np.ascontiguousarray(v, dtype=np.int64)
                for k, v in enc.items() if k in input_names
            }
            hidden = sess.run(None, feed)[0]      # [B, L, H] last_hidden_state
            cls = hidden[:, 0, :]                 # CLS pooling
            cls = cls / np.linalg.norm(
                cls, axis=1, keepdims=True
            ).clip(min=1e-12)                     # L2 normalise (llama-server parity)
            out.extend(cls.astype(np.float64).tolist())
    return out


async def encode(texts: list[str], batch_size: int = 128) -> list[list[float]]:
    """Embed texts on the local GPU (runs in a worker thread: don't block the loop)."""
    return await asyncio.to_thread(_encode_sync, texts, batch_size)


async def encode_one(text: str) -> list[float]:
    """Embed a single text (query path)."""
    return (await encode([text]))[0]
