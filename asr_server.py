# -*- encoding: utf-8 -*-
"""FunASR 本地 ASR 服务（接口与 RapidASR 版 asr_server.py 完全同构）。

提供两类接口：

1. 文件上传识别（通用 REST，浏览器/Edge 文件表单、curl -F 均可直传）
    POST /v1/asr/transcribe        multipart/form-data，字段 file=音频文件
    格式按文件头嗅探：wav/mp3/flac/ogg 容器，或固件直传的裸 PCM16LE
    （无 RIFF 头，如 EasyInputStudio 的 capture.pcm；采样率用 sample_rate 传，
    默认 16000 即固件协议值）。
    返回 {"text": ..., "segments": ..., "elapsed_ms": ...}

2. DashScope 兼容识别（与 easyinput-studio 的 remoteHTTPASR 同构）
    POST /compatible-mode/v1/chat/completions
    messages[].content[].type=input_audio + base64 data URL，
    响应取 choices[0].message.content；错误统一 {"error": {"message": ...}}

辅助：
    GET  /health                   存活与模型加载状态探针
    GET  /openapi.json             机器可读接口描述（AI 对接直接读这份）

启动：
    .venv/Scripts/python.exe asr_server.py             # 默认 127.0.0.1:18466，断句(标点)默认开启
    .venv/Scripts/python.exe asr_server.py --no-punc   # 关闭标点恢复
    .venv/Scripts/python.exe asr_server.py --port 9000
    .venv/Scripts/python.exe asr_server.py --vad       # 加载 fsmn-vad，支持长音频
    .venv/Scripts/python.exe asr_server.py --no-silence-gate   # 关闭静音门限

静音门限（默认开启，基于 VAD）：
    Paraformer 对「几乎没有语音」的输入会产生幻觉文本（实测纯静音输出
    「没有没有有」、按键咔哒输出「退出hello hello」）。服务端在推理前先用
    FunASR 官方 fsmn-vad 判断是否含人声（训练模型，非能量/频率阈值猜测），
    语音总时长低于门限的整段直接返回空文本，不喂给模型。
    门限可用 --min-speech-sec 调整，或 --no-silence-gate 整体关闭。

与 RapidASR 版（默认 18465）端口错开，两台服务可同时运行；
客户端只需改端口号即可在两个引擎之间切换。

鉴权：默认不校验。设环境变量 ASR_SERVER_KEY 后，两个识别接口都要求
Authorization: Bearer <key>。
"""
import argparse
import base64
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import librosa
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)
logger = logging.getLogger("funasr_server")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# 模型缓存目录固定到项目内（默认 modelscope 会下到 C:\Users\<user>\.cache\modelscope，
# 2GB+ 容易撑爆 C 盘）。必须在首次调用 funasr.AutoModel（模型下载/加载）之前设置。
# 允许外部用 MODELSCOPE_CACHE 覆盖。
os.environ.setdefault("MODELSCOPE_CACHE", str(BASE_DIR / ".models_cache"))

# 单段音频解码后上限：300s × 16kHz × 4B(float32) ≈ 19.2MB，留余量挡异常大文件
MAX_AUDIO_BYTES = 32 * 1024 * 1024
TARGET_SR = 16000  # 模型输入采样率；固件协议同为 16kHz（16000Hz/320样本/20ms）

DEFAULT_MODEL = "paraformer-zh"

# ---------------------------------------------------------------- 模型加载

_infer_lock = threading.Lock()
_model = None
_model_name = DEFAULT_MODEL
_vad_model = None  # 静音门限用的 fsmn-vad（在 _lifespan 中按需加载）


@asynccontextmanager
async def _lifespan(_app):
    global _model, _model_name, _vad_model
    from funasr import AutoModel

    t0 = time.time()
    # 静音门限用的 VAD（fsmn-vad，~1.7MB）：训练好的语音/非语音判别模型，
    # 用于挡住按键咔哒/底噪等「有能量但非语音」的输入，同时不误杀弱语音。
    if _GATE_ENABLED:
        _vad_model = AutoModel(
            model="fsmn-vad", device="cpu", disable_update=True, disable_pbar=True
        )
    kwargs = dict(
        model=_model_name,
        device="cpu",
        disable_update=True,
        disable_pbar=True,
    )
    if _LOAD_VAD:
        kwargs["vad_model"] = "fsmn-vad"  # 长音频自动切分（首次运行额外下载 ~5MB）
    if _LOAD_PUNC:
        # ct-punc：CT-Transformer 标点恢复（中英文，~290MB，首次运行自动下载），
        # 挂进 pipeline 后 generate 输出自动带标点断句
        kwargs["punc_model"] = "ct-punc"
    _model = AutoModel(**kwargs)
    logger.info(
        "模型加载完成（%s，vad=%s，punc=%s），耗时 %.1fs",
        _model_name, _LOAD_VAD, _LOAD_PUNC, time.time() - t0,
    )
    yield


_LOAD_VAD = False
_LOAD_PUNC = True  # 默认开启 ct-punc 标点恢复（断句），RapidASR 版不具备此能力

APP_TITLE = "FunASR 本地语音识别服务"
APP_DESC = """FunASR Paraformer 离线中文语音识别（CPU）。

## 接入方式二选一

- **文件上传**（推荐通用场景）：`POST /v1/asr/transcribe`，multipart 表单直传音频文件
- **DashScope 兼容**（OpenAI 风格客户端 / easyinput-studio）：`POST /compatible-mode/v1/chat/completions`

## AI 对接

直接读 `/openapi.json` 即可获得完整接口描述；音频采样率不限，服务端自动重采样到 16kHz。
"""

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESC,
    version="1.1.0",
    lifespan=_lifespan,
    docs_url=None,  # Swagger UI 从 CDN 拉资源国内不可达，仅保留机器可读的 openapi.json
    oauth2_redirect_url=None,
)

# 与 RapidASR 版保持一致：static/ 下有离线 swagger 资源就挂载
if STATIC_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/docs", include_in_schema=False)
    def swagger_ui():
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=f"{APP_TITLE} - Swagger UI",
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
        )

    @app.get("/docs/oauth2-redirect", include_in_schema=False)
    def swagger_ui_redirect():
        return get_swagger_ui_oauth2_redirect_html()


# ---------------------------------------------------------------- 公共逻辑


class TranscribeResponse(BaseModel):
    """识别结果。"""

    text: str = Field(description="识别出的文本，UTF-8")
    segments: int = Field(description="本次请求包含的音频段数")
    elapsed_ms: int = Field(description="纯推理耗时（毫秒，不含上传与解码）")


class HealthResponse(BaseModel):
    status: str = Field(description='"ok"=模型就绪；"loading"=仍在加载')
    model: str = Field(description="模型标识")


def _check_auth(request: Request) -> JSONResponse | None:
    """设了 ASR_SERVER_KEY 就强制 Bearer 校验，否则放行（本地服务默认无鉴权）。"""
    expected = os.environ.get("ASR_SERVER_KEY", "")
    if not expected:
        return None
    if request.headers.get("Authorization", "") != f"Bearer {expected}":
        return _error("密钥不正确", status_code=401, err_type="authentication_error")
    return None


def _error(message: str, status_code: int = 400, err_type: str = "invalid_request_error") -> JSONResponse:
    """按 DashScope 风格返回错误，保证客户端 remoteASRErrorText 能抽出 message。"""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": err_type, "code": None}},
    )


def _decode_audio_data(data_url: str) -> tuple[bytes, str]:
    """解析 data URL（或裸 base64），返回 (音频字节, mime)。"""
    mime = "audio/wav"
    payload = data_url.strip()
    if payload.startswith("data:"):
        header, _, payload = payload.partition(",")
        mime = header[5:].split(";", 1)[0] or mime
    try:
        audio = base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise ValueError(f"音频 base64 解码失败：{exc}") from exc
    if not audio:
        raise ValueError("音频内容为空")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError(f"音频 {len(audio)} 字节，超过服务端上限 {MAX_AUDIO_BYTES}")
    return audio, mime


def _sniff_container(audio: bytes) -> str:
    """按文件头 magic bytes 嗅探容器类型。

    固件直传的裸 PCM16LE 没有 RIFF 头（实测 capture.pcm 开头 00 00 ...），
    一律归入 raw_pcm，不能按后缀猜 —— .wav 后缀也可能装着裸 PCM。
    """
    if audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        return "wav"
    if audio[:4] == b"fLaC":
        return "flac"
    if audio[:4] == b"OggS":
        return "ogg"
    if audio[:3] == b"ID3" or audio[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    return "raw_pcm"


def _load_waveform(audio: bytes, sample_rate: int = TARGET_SR) -> np.ndarray:
    """音频字节 → 16kHz 单声道 float32（shape [1, n]）。

    有容器头（wav/mp3/flac/ogg）用 soundfile 内存解码（多声道平均成单声道）；
    裸 PCM16LE 按 RAW 格式解，采样率由调用方指定（默认 16000，固件协议值），
    最后统一重采样到模型输入采样率。
    """
    import io

    import soundfile as sf

    container = _sniff_container(audio)
    try:
        if container == "raw_pcm":
            waveform, sr = sf.read(
                io.BytesIO(audio),
                samplerate=sample_rate,
                channels=1,
                format="RAW",
                subtype="PCM_16",
                dtype="float32",
            )
        else:
            info = sf.info(io.BytesIO(audio))
            sr = info.samplerate
            waveform, _ = sf.read(io.BytesIO(audio), dtype="float32", always_2d=False)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
    except Exception as exc:
        raise ValueError(f"音频解码失败（不支持的格式或文件损坏）：{exc}") from exc
    if waveform.size == 0:
        raise ValueError("音频解码后为空（可能是静音或格式不被接受）")
    if sr != TARGET_SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
    return waveform.astype(np.float32)[None, ...]


# ---------------------------------------------------------------- 静音门限（VAD）
# Paraformer 对「几乎没有语音」的输入会产生幻觉文本：实测纯静音输出「没有没有有」、
# 按键咔哒输出「退出hello hello」。仅靠能量阈值只能挡住「绝对安静」，挡不住按键
# 咔哒这类**有能量但非语音**的输入，还可能误杀弱语音——所以改用训练好的 VAD。
#
# 正解：用 FunASR 官方 fsmn-vad 判「是不是人声」（基于 FBank 时频特征学习，而非
# 拍脑袋的能量/频率阈值）。实测能拦下按键咔哒/底噪/纯静音，对弱语音(-25dB)、
# 短语音仍灵敏不误杀。推理前先过 VAD：语音总时长 < 门限的整段判为无语音。

MIN_SPEECH_SEC = 0.2  # VAD 判定「有语音」所需的最短语音总时长（秒）

_GATE_ENABLED = True


def _vad_speech_sec(waveform: np.ndarray) -> float:
    """用 fsmn-vad 求这段音频的语音总时长（秒）。

    无 VAD 模型（未加载/已关闭门限）时返回 inf，表示「不拦截」。
    """
    if _vad_model is None:
        return float("inf")
    x = waveform[0] if waveform.ndim > 1 else waveform
    if x.size == 0:
        return 0.0
    res = _vad_model.generate(input=x)
    segs = res[0].get("value", []) if res else []
    return sum(e - s for s, e in segs) / 1000.0


def _has_speech(waveform: np.ndarray) -> bool:
    """判断这段音频是否含人声；纯静音/噪声/按键声直接跳过推理，防幻觉。"""
    if not _GATE_ENABLED:
        return True
    sec = _vad_speech_sec(waveform)
    if sec >= MIN_SPEECH_SEC:
        return True
    logger.info(
        "fsmn-vad 判定为无有效语音，跳过推理：语音 %.2fs（门限 %.2fs）", sec, MIN_SPEECH_SEC
    )
    return False


# ---- 以下为【旧方案：能量门限】保留供对照/回退，默认不再调用 ----
# 实测缺点：挡不住有能量的按键咔哒（会输出幻觉），弱语音时占比偏低易误杀。
SILENCE_DBFS = -45.0  # 绝对门限：帧能量低于此值视为静音
SPEECH_MARGIN_DB = 8.0  # 相对门限：需高出噪声底这么多 dB 才算有效语音
MIN_SPEECH_RATIO = 0.05  # 有效语音最低占比


def _speech_stats(waveform: np.ndarray) -> tuple[float, float]:
    """【旧】能量门限统计。返回 (有效语音总时长秒, 有效语音帧占比)。"""
    x = (waveform[0] if waveform.ndim > 1 else waveform).astype(np.float64)
    if x.size == 0:
        return 0.0, 0.0

    frame = max(1, int(0.025 * TARGET_SR))  # 25ms
    hop = max(1, int(0.010 * TARGET_SR))  # 10ms

    if x.size < frame:
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        voiced = (20 * np.log10(rms + 1e-10)) > SILENCE_DBFS
        return (x.size / TARGET_SR if voiced else 0.0), (1.0 if voiced else 0.0)

    csum = np.concatenate(([0.0], np.cumsum(x * x)))
    starts = np.arange(0, x.size - frame + 1, hop)
    ends = starts + frame
    rms = np.sqrt((csum[ends] - csum[starts]) / frame + 1e-12)
    dbfs = 20 * np.log10(rms)

    # 以低分位数作噪声底，避免整段偏噪时把噪声当语音
    threshold = max(SILENCE_DBFS, float(np.percentile(dbfs, 10)) + SPEECH_MARGIN_DB)
    voiced = dbfs > threshold
    return float(voiced.sum()) * hop / TARGET_SR, float(voiced.mean())


def _has_speech_energy(waveform: np.ndarray) -> bool:
    """【旧】能量门限判定，保留供对照；默认不再调用。"""
    speech_sec, ratio = _speech_stats(waveform)
    return speech_sec >= MIN_SPEECH_SEC and ratio >= MIN_SPEECH_RATIO


def _infer(audios: list[np.ndarray]) -> tuple[list[str], int]:
    """串行推理（funasr 模型非线程安全，加锁）。返回 (各段文本, 纯推理毫秒)。

    静音段（VAD 门限未通过）不喂给模型，直接给空文本，避免幻觉输出。
    """
    assert _model is not None
    texts: list[str] = []
    t0 = time.time()
    with _infer_lock:
        for wav in audios:
            if not _has_speech(wav):
                texts.append("")
                continue
            res = _model.generate(input=wav[0])  # funasr 接收一维 float32 波形
            texts.append(res[0].get("text", "") if res else "")
    return texts, int((time.time() - t0) * 1000)


# ---------------------------------------------------------------- 接口一：文件上传


@app.post(
    "/v1/asr/transcribe",
    summary="上传音频文件识别（multipart 表单）",
    description=(
        "浏览器/Edge 文件上传的标准方式：`<form enctype=\"multipart/form-data\">` "
        "或 `<input type=\"file\">` 选文件后提交，`file` 字段即音频。\n\n"
        "命令行：`curl -F file=@test.wav http://127.0.0.1:18466/v1/asr/transcribe`\n\n"
        "格式按**文件头**自动嗅探：wav/mp3/flac/ogg 容器，或固件直传的**裸 PCM16LE**"
        "（如 EasyInputStudio 的 capture.pcm，无 RIFF 头）；采样率不限，自动重采样到 16kHz。"
        "裸 PCM 的采样率用 `sample_rate` 字段指定（默认 16000，即固件协议值）。"
    ),
    response_model=TranscribeResponse,
    responses={400: {"description": "音频缺失、超限或解码失败"}},
    tags=["文件上传识别"],
)
async def transcribe_upload(
    request: Request,
    file: UploadFile = File(description="音频文件（wav/mp3/flac/ogg，或裸 PCM16LE）"),
    sample_rate: int = Form(
        default=TARGET_SR,
        description="裸 PCM 的采样率，默认 16000（固件协议值）；带容器头的文件忽略此参数",
    ),
    joiner: str = Form(
        default="",
        description="多段结果拼接符，默认直接相连；传 '\\n' 可按行分隔",
    ),
):
    auth = _check_auth(request)
    if auth is not None:
        return auth

    audio = await file.read()
    if not audio:
        return _error("音频文件为空")
    if len(audio) > MAX_AUDIO_BYTES:
        return _error(f"音频 {len(audio)} 字节，超过服务端上限 {MAX_AUDIO_BYTES}")

    try:
        wav = _load_waveform(audio, sample_rate)
    except ValueError as exc:
        return _error(str(exc))

    try:
        texts, elapsed_ms = _infer([wav])
    except Exception as exc:
        logger.exception("推理失败")
        return _error(f"识别失败：{exc}", status_code=500, err_type="server_error")

    text = joiner.join(t for t in texts if t).strip()
    logger.info("文件识别完成：%s，结果 %d 字，推理 %dms", file.filename, len(text), elapsed_ms)
    return TranscribeResponse(text=text, segments=1, elapsed_ms=elapsed_ms)


# ---------------------------------------------------------------- 接口二：DashScope 兼容


@app.post(
    "/compatible-mode/v1/chat/completions",
    summary="DashScope 兼容识别（base64 音频）",
    description=(
        "与 DashScope qwen3-asr 请求同构，OpenAI 风格客户端可直接切换端点。\n\n"
        "```json\n"
        '{"model": "paraformer-v2", "messages": [{"role": "user", "content": ['
        '{"type": "input_audio", "input_audio": {"data": "data:audio/wav;base64,<...>"}}]}]}\n'
        "```\n\n"
        "识别文本在 `choices[0].message.content`。"
    ),
    responses={400: {"description": "请求缺段或音频无效"}, 401: {"description": "鉴权失败"}},
    tags=["DashScope 兼容"],
)
async def chat_completions(request: Request):
    auth = _check_auth(request)
    if auth is not None:
        return auth

    try:
        payload = await request.json()
    except Exception:
        return _error("请求体不是合法 JSON")

    messages = payload.get("messages") or []
    if not messages:
        return _error("请求缺少 messages")

    audios: list[np.ndarray] = []
    for message in messages:
        content = message.get("content")
        # content 可能是字符串（纯文本消息，不处理）或分段列表
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "input_audio":
                continue
            inner = part.get("input_audio") or {}
            data = inner.get("data") or ""
            if not data:
                return _error("input_audio 缺少 data 字段")
            try:
                audio_bytes, _mime = _decode_audio_data(data)
            except ValueError as exc:
                return _error(str(exc))
            try:
                audios.append(_load_waveform(audio_bytes))
            except ValueError as exc:
                return _error(str(exc))

    if not audios:
        return _error("请求中没有可识别的 input_audio 分段")

    try:
        texts, _elapsed = _infer(audios)
    except Exception as exc:
        logger.exception("推理失败")
        return _error(f"识别失败：{exc}", status_code=500, err_type="server_error")

    text = "".join(t for t in texts if t).strip()
    logger.info("识别完成：%d 段音频，结果 %d 字", len(audios), len(text))
    return {
        "id": f"local-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model") or _model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ---------------------------------------------------------------- 辅助接口


@app.get("/health", summary="存活探针", response_model=HealthResponse, tags=["辅助"])
def health():
    return HealthResponse(
        status="ok" if _model is not None else "loading",
        model=(
            f"FunASR({_model_name}"
            f"{' + fsmn-vad' if _LOAD_VAD else ''}"
            f"{' + ct-punc' if _LOAD_PUNC else ''})"
        ),
    )


def main() -> None:
    global _model_name, _LOAD_VAD, _LOAD_PUNC, _GATE_ENABLED, MIN_SPEECH_SEC

    parser = argparse.ArgumentParser(description="FunASR 本地 ASR 服务（CPU）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机")
    parser.add_argument("--port", type=int, default=18466, help="监听端口，默认 18466（错开 RapidASR 版的 18465）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"funasr 模型名，默认 {DEFAULT_MODEL}")
    parser.add_argument("--vad", action="store_true", help="加载 fsmn-vad 支持长音频自动切分")
    parser.add_argument("--no-punc", action="store_true", help="关闭 ct-punc 标点恢复（默认开启断句）")
    parser.add_argument(
        "--no-silence-gate",
        action="store_true",
        help="关闭静音门限（默认开启：纯静音/噪声片段直接返回空文本，防模型幻觉）",
    )
    parser.add_argument(
        "--min-speech-sec",
        type=float,
        default=MIN_SPEECH_SEC,
        help=f"判定有语音所需的最短有效语音时长（秒），默认 {MIN_SPEECH_SEC}",
    )
    args = parser.parse_args()

    _model_name = args.model
    _LOAD_VAD = args.vad
    _LOAD_PUNC = not args.no_punc
    _GATE_ENABLED = not args.no_silence_gate
    MIN_SPEECH_SEC = args.min_speech_sec

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
