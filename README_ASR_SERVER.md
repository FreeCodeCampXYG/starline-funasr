# FunASR CPU 版 ASR 服务（带断句 + 一键启动）

基于官方 [modelscope/FunASR](https://github.com/modelscope/FunASR)，
在本仓库基础上增加了**纯 CPU、Windows 可用**的本地语音识别 HTTP 服务，
支持中文**断句（标点恢复）**、固件直传 **PCM 裸流**，以及一键启动脚本。

> 适用场景：本地离线、无显卡（CPU）的 Windows 电脑；需要把麦克风/固件采集的
> 音频送到一个 HTTP 接口做识别，并且希望输出是**带标点、通顺的句子**。

---

## 1. 硬件要求

| 项目 | 最低建议 | 说明 |
| --- | --- | --- |
| CPU | 任意近 5 年 x86_64，4 核+ | 本机实测 Intel Core Ultra 5 225H（14 核）流畅 |
| 内存 | 4 GB 空闲 | 推理峰值约 1~2 GB（Paraformer + ct-punc） |
| 磁盘 | 2 GB+ 空闲 | torch CPU 版 + 模型权重约 1.5~2 GB |
| 系统 | Windows 10/11（64 位） | 已验证；Linux/macOS 同理，命令略不同 |

---

## 2. 搭建 Python 虚拟环境（不影响全局）

> 全程在项目根目录 `D:\MyPro\FunASR` 下操作，所有依赖只装进 `.venv`，
> **不会污染系统 Python**。

### 2.1 用系统 Python 创建虚拟环境

```bat
:: 打开“命令提示符”或 PowerShell，进入项目目录
cd D:\MyPro\FunASR

:: 用你机器上的 python 3.8~3.12 创建虚拟环境（目录名 .venv）
python -m venv .venv
```

> 注意：本仓库的 `.venv` 是用 Python 3.13 创建的，但 FunASR 官方推荐 3.8~3.12。
> 如果你的 `python` 是 3.13，也能跑（本机已验证）；追求稳妥可用 3.10/3.11。

### 2.2 激活虚拟环境

```bat
.venv\Scripts\activate
```

激活后命令行前面会出现 `(.venv)` 前缀，之后所有 `pip install` / `python`
都只作用于这个环境。

### 2.3 安装依赖

Windows 上 PyPI 默认提供的 `torch` 就是 **CPU 版**，无需额外源。
如果 pip 默认源慢/被拦，改用阿里云镜像（本机实测可通）：

```bat
:: 1) torch / torchaudio（CPU 版）
pip install torch torchaudio -i https://mirrors.aliyun.com/pypi/simple/

:: 2) 以“可编辑”方式安装 FunASR 本体及全部依赖（librosa / modelscope / transformers ...）
pip install -e . -i https://mirrors.aliyun.com/pypi/simple/

:: 3) HTTP 服务依赖
pip install fastapi "uvicorn[standard]" python-multipart -i https://mirrors.aliyun.com/pypi/simple/
```

### 2.4（可选）接入 RapidASR 引擎

如果你想用 ONNXRuntime 版的 Paraformer（更快、更轻），可额外装 `rapid_paraformer`：
模型仓库在 HuggingFace，国内需走镜像。

```bat
pip install rapid_paraformer --ignore-requires-python -i https://mirrors.aliyun.com/pypi/simple/

:: 下载模型（国内镜像 + 关闭 Xet 通道）
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_DISABLE_XET=1
python -c "from rapid_paraformer import download_hf_model; download_hf_model(repo_id='SWHL/RapidParaformer', save_dir='rapid_asr_models')"
```

> 这一步是可选的。不装也能用 FunASR 原生引擎跑全部功能。

---

## 3. 启动 ASR 服务

### 3.1 直接启动（推荐先看这个）

激活虚拟环境后：

```bat
.venv\Scripts\python.exe asr_server.py
```

默认监听 `127.0.0.1:18466`（**故意错开 RapidASR 版的 18465**，两边可同时跑）。
首次启动会自动联网下载模型权重（Paraformer ~990MB、ct-punc 标点 ~290MB，
仅需一次，之后缓存到 `~\.cache\modelscope` 离线可用），加载约 20~30 秒。

可用参数：

| 参数 | 说明 |
| --- | --- |
| `--host` | 监听地址，默认 `127.0.0.1`（仅本机） |
| `--port` | 端口，默认 `18466` |
| `--model` | FunASR 模型名，默认 `paraformer-zh` |
| `--no-punc` | 关闭断句（默认**开启**标点恢复） |
| `--vad` | 加载 fsmn-vad，支持长音频自动切分 |
| `--no-silence-gate` | 关闭**静音门限**（默认开启，防幻觉，见第 7 节） |
| `--min-speech-sec` | 判定「有语音」所需最短有效语音时长（秒），默认 `0.2` |

示例：

```bat
.venv\Scripts\python.exe asr_server.py --port 9000      :: 换端口
.venv\Scripts\python.exe asr_server.py --no-punc        :: 不要标点
.venv\Scripts\python.exe asr_server.py --vad            :: 长音频切分
.venv\Scripts\python.exe asr_server.py --no-silence-gate :: 关闭静音门限
```

### 3.2 一键启动（演示用）

项目根目录提供了 `start_asr_server.bat`：双击即可

1. 自动杀掉 18466 端口上的旧进程（不用手动开任务管理器）
2. 前台启动 `asr_server.py`（窗口显示日志，不会一闪而过）
3. 自动打开浏览器到 Swagger 页面
4. 服务停止后 `pause` 挂住窗口，方便你看日志

> 演示当天直接双击这个 bat 即可。

---

## 4. 怎么看服务 / Swagger 地址

服务起来后，在浏览器打开：

| 地址 | 用途 |
| --- | --- |
| **http://127.0.0.1:18466/docs** | **Swagger 接口文档**（已内置离线资源，无需联网） |
| http://127.0.0.1:18466/health | 探活，返回 `{"status":"ok","model":"FunASR(paraformer-zh + ct-punc)"}` |
| http://127.0.0.1:18466/openapi.json | OpenAPI 描述 |

> 若 `/docs` 打不开：本项目已把 Swagger 的 `swagger-ui-bundle.js / .css`
> 放到 `static/` 目录做离线托管，只要 `static/` 在 `asr_server.py` 旁边即可。
> 浏览器**刷新一次**再试。

---

## 5. 接口说明

服务与 RapidASR 的 `asr_server.py` **接口同构**，客户端只换端口号即可切换引擎。

### 5.1 文件 / 裸 PCM 上传

`POST /v1/asr/transcribe`（multipart 表单）

- `file`：音频文件。支持 **wav / mp3 / flac / ogg**（按文件头嗅探，不靠后缀），
  以及**固件直传的 PCM16LE 裸流**。
- `sample_rate`：裸 PCM 时必填（默认 16000），任意采样率会自动重采样到 16kHz。
- `joiner`：分句连接符，默认空。

请求示例：

```bat
:: WAV 文件
curl -F file=@asr_example_zh.wav http://127.0.0.1:18466/v1/asr/transcribe

:: 固件裸 PCM16LE（sample_rate 必须给）
curl -F file=@capture.pcm -F sample_rate=16000 http://127.0.0.1:18466/v1/asr/transcribe
```

返回：

```json
{"text": "欢迎大家来体验达摩院推出的语音识别模型。", "code": 0}
```

### 5.2 DashScope 兼容（easyinput-studio 等可直接用）

`POST /compatible-mode/v1/chat/completions`

```bat
curl http://127.0.0.1:18466/compatible-mode/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"paraformer-v2\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"data:audio/wav;base64,<BASE64>\"}}]}]}"
```

### 5.3 鉴权

若设置了环境变量 `ASR_SERVER_KEY`，则请求头需带 `Authorization: Bearer <key>`，
未带会返回 `{"error":{"message":"..."}}`。不设置则免鉴权（本机演示足够）。

---

## 6. 断句（标点恢复）

服务默认开启 **ct-punc**（CT-Transformer 标点恢复模型，中英文）。
它会在识别完成后自动补回标点，并把字间空格归并成通顺句子：

```
未开断句：欢 迎 大 家 来 体 验 达 摩 院 推 出 的 语 音 识 别 模 型
开启断句：欢迎大家来体验达摩院推出的语音识别模型。
```

RapidASR 版不具备此能力，**需要通顺句子时优先用本 FunASR 版（18466）**。
不想用断句时加 `--no-punc` 关闭。

---

## 7. 静音门限（防「幻觉文本」，基于 VAD）

**问题**：Paraformer 对**几乎没有语音**的输入会产生**幻觉文本**——空白、底噪、
按键咔哒声都可能被「编」成几个字。实测纯静音被识别成 `没有没有有`、按键咔哒被
识别成 `退出hello hello`，等于凭空造出一句话，下游拿到就是脏数据。

**解决**：服务端在**推理之前**先用 **FunASR 官方 `fsmn-vad`**（训练好的语音/非语音
判别模型，~1.7MB，首次运行自动下载到 `.models_cache`）判断这段音频是否含**人声**：

1. 对整段音频跑 fsmn-vad，得到若干语音区间（毫秒边界）；
2. 累加语音区间总时长；
3. 若 语音总时长 < `--min-speech-sec`（默认 0.2s），判定该段无语音 → **直接返回空文本，不喂给模型**。

> **为什么不用能量阈值**：早期版本用「分帧 RMS + 噪声底」做门限，实测**挡不住按键
> 咔哒**这类「有能量但不是人声」的输入（持续咔哒 3s 会被放行，照样输出幻觉），
> 也可能误杀弱语音。手写的能量/频率阈值本质是猜测；改用**训练好的 VAD** 才是正解。

实测效果（`.venv\Scripts\python.exe test_silence_gate.py`，8 个用例全部 PASS）：

| 输入 | 旧能量门限 | fsmn-vad（现方案） |
| --- | --- | --- |
| 纯静音 540ms | 拦 ✅ | 拦 ✅ |
| 低幅底噪 2s | 拦 ✅ | 拦 ✅ |
| 按键咔哒 40ms | 拦 ✅ | 拦 ✅ |
| **持续按键咔哒 3s** | **放行 ❌（输出幻觉）** | **拦 ✅** |
| 真实语音 5.5s | 放行 ✅ | 放行 ✅ |
| 真实语音 0.25s + 静音 1s | 放行（勉强） | 放行 ✅（更灵敏、不易误杀） |

**调节方式**：

- 说话很轻、很短被判成静音 → 调低阈值：`--min-speech-sec 0.1`
- 想完全关掉门限 → `--no-silence-gate`（此时不再加载 VAD 模型）
- 离线自测（不启动服务）：`.venv\Scripts\python.exe test_silence_gate.py`
  （8 个用例，全部 PASS 即正常）

---

## 8. 命令行脚本（批量 / 手工处理）

| 脚本 | 说明 |
| --- | --- |
| `funasr_cli.py` | FunASR 原生引擎（torch）识别，支持单文件 / 目录批量、`-o` 结果落盘 |
| `rapid_asr_cli.py` | RapidASR（ONNXRuntime）引擎识别，参数完全一致 |

用法（参数两版通用）：

```bat
.venv\Scripts\python.exe funasr_cli.py asr_example_zh.wav                :: 单文件
.venv\Scripts\python.exe funasr_cli.py 音频目录 -o 结果.txt              :: 批量 + 落盘
.venv\Scripts\python.exe funasr_cli.py a.wav --model paraformer-zh       :: 指定模型

.venv\Scripts\python.exe rapid_asr_cli.py asr_example_zh.wav
```

输出格式：`文件路径<TAB>识别文本`，一行一个文件。

---

## 9. 目录结构（本仓库新增 / 改动部分）

```
FunASR/
├── asr_server.py          # FastAPI 服务（18466，断句默认开，支持 PCM，含静音门限）
├── funasr_cli.py          # FunASR 引擎命令行识别
├── rapid_asr_cli.py       # RapidASR 引擎命令行识别
├── test_cpu_asr.py        # 最小冒烟测试
├── test_silence_gate.py   # 静音门限离线自测（VAD，8 用例）
├── start_asr_server.bat   # 一键启动（杀旧进程→启动→等就绪→开 Swagger）
├── push_to_github.bat     # 一键推送到本 fork（本机 git 直推）
├── push_via_api.py        # 走 GitHub API 推送（git 协议被墙时的备用通道）
├── cleanup_branches.py    # 清理 fork 中继承自上游的多余分支
├── static/                # 离线 Swagger 资源（swagger-ui-bundle.js / .css）
├── asr_example_zh.wav     # 示例音频（约 5.5s 中文）
├── README_ASR_SERVER.md   # 本文档
├── .models_cache/         # modelscope 模型缓存（2GB+，已 gitignore，不入库）
├── .venv/                 # 虚拟环境（已 gitignore，不入库）
└── rapid_asr_models/      # RapidASR 模型（可选，已 gitignore，不入库）
```

---

## 10. 常见问题

- **Q：麦克风没说话，也识别出文字？**
  这是 Paraformer 的幻觉输出，本服务已用**基于 fsmn-vad 的静音门限**解决（见第 7 节）。
  若仍出现，可把阈值调高：`--min-speech-sec 0.3`；并确认没加 `--no-silence-gate`。
- **Q：模型缓存跑到 C 盘、把 C 盘撑爆？**
  本服务已把缓存固定到项目内 `D:\MyPro\FunASR\.models_cache`（modelscope 默认是
  `C:\Users\<user>\.cache\modelscope`）。`asr_server.py` 用 `os.environ.setdefault(...)`
  设定，`start_asr_server.bat` 用 `set "MODELSCOPE_CACHE=%~dp0.models_cache"` 设定；
  如需换位置，外部设 `MODELSCOPE_CACHE` 即可覆盖（bat 会强制为项目目录）。
- **Q：提示 `torch` 不是 CPU 版 / 想确认？**
  跑 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`，
  版本带 `+cpu` 且 `cuda_available=False` 即为 CPU 版。
- **Q：pip 安装慢或 403？**
  加 `-i https://mirrors.aliyun.com/pypi/simple/`；本机清华镜像对 pip 返回 403，已改用阿里云。
- **Q：模型下载不动 / 401？**
  FunASR 走 ModelScope 一般没问题；RapidASR 模型走
  `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`。
- **Q：端口被占用？**
  启动加 `--port 新端口`，或先停掉占用进程；一键 bat 会自动杀 18466 旧进程。

---

> 仓库：`FreeCodeCampXYG/starline-funasr`（Fork 自 modelscope/FunASR，仅保留 `main` 一个分支）。
> 本仓库的 **`main` 分支即包含上述全部改动**（默认分支，访客进来看到的就是带 ASR 服务的版本），
> 不向上游 modelscope/FunASR 提 PR/issue。
> 推送命令见根目录 `push_to_github.bat`（本机 git 直推）；
> 若本机代理挡掉了 `github.com` 的 git 协议，改用 `python push_via_api.py --squash`。
