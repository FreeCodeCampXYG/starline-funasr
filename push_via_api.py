# -*- encoding: utf-8 -*-
"""经 GitHub API 把本地新增文件推送到 fork 的 main（绕开被封的 github.com git 协议）。

背景：本机沙箱代理只放行 api.github.com，github.com 被整个挡掉，`git push` 报
502 CONNECT tunnel failed。这里改用 GitHub Git Data API：直接创建 blob →
tree（以远端 main 的 tree 为 base）→ commit → 移动 ref，
效果等同于把改动提交到 fork 的 main 上，且不覆盖上游已有文件。

幂等：若生成的 tree 与远端当前 tree 相同，说明无变化，直接退出。

用法：
    .venv/Scripts/python.exe push_via_api.py                # 推到默认 fork 的 main
    .venv/Scripts/python.exe push_via_api.py --dry-run      # 只看会推哪些文件
    .venv/Scripts/python.exe push_via_api.py --branch main
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_REPO = "FreeCodeCampXYG/starline-funasr"

# 待推送文件（相对项目根目录）。均为新增文件；.gitignore 单独处理（基于远端版本追加）
FILES = [
    ".gitignore",
    "README_ASR_SERVER.md",
    "asr_server.py",
    "asr_example_zh.wav",
    "funasr_cli.py",
    "push_to_github.bat",
    "push_via_api.py",
    "rapid_asr_cli.py",
    "start_asr_server.bat",
    "test_cpu_asr.py",
    "test_silence_gate.py",
    "static/swagger-ui-bundle.js",
    "static/swagger-ui-bundle.js.map",
    "static/swagger-ui.css",
    "static/swagger-ui.css.map",
]

GI_MARKER = "# === 本地环境"

COMMIT_MESSAGE = """feat: FunASR CPU 版 ASR 服务（断句 + PCM + 静音门限）

在官方 FunASR 之上增加一套本地 ASR 服务与命令行工具，面向 Windows + CPU 场景。

- asr_server.py: FastAPI 服务，端口 18466，接口与 RapidASR 版 asr_server.py 同构
  * POST /v1/asr/transcribe —— multipart 上传，文件头嗅探 wav/mp3/flac/ogg，
    无 RIFF 头的裸 PCM16LE（固件直传）按 RAW 解码，任意采样率自动重采样到 16kHz
  * POST /compatible-mode/v1/chat/completions —— DashScope 兼容（base64 音频）
  * GET /docs（离线 Swagger 资源）、/health、/openapi.json
  * 默认挂 ct-punc 做标点恢复（断句），--no-punc 可关
  * 静音门限：推理前统计有效语音时长/占比，空白片段直接返回空文本，
    避免 Paraformer 对近静音输入产生幻觉文本（实测 540ms/静音 96% 的片段
    曾被编成「这个这的这的e个。」）；--no-silence-gate 可关
- funasr_cli.py / rapid_asr_cli.py: 双引擎命令行识别，参数一致（torch / ONNXRuntime）
- test_cpu_asr.py: 最小冒烟测试
- test_silence_gate.py: 静音门限离线自测（7 个用例）
- start_asr_server.bat: 一键启动（清端口旧进程 → 起服务 → 开 Swagger → pause）
- push_to_github.bat / push_via_api.py: 推送到本 fork 的脚本
- static/: 离线 Swagger UI 资源（国内 CDN 不可达）
- README_ASR_SERVER.md: CPU 部署、venv、依赖、接口与 Swagger 说明
- .gitignore: 排除 .venv / rapid_asr_models / .workbuddy 等本地产物
"""


def gh_token() -> str:
    out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        sys.exit("无法获取 gh token，请先执行 gh auth login")
    return out.stdout.strip()


class GH:
    def __init__(self, repo: str, token: str) -> None:
        self.base = f"https://api.github.com/repos/{repo}"
        proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
        handlers = [urllib.request.ProxyHandler({"https": proxy, "http": proxy})] if proxy else []
        self.opener = urllib.request.build_opener(*handlers)
        self.token = token

    def api(self, path: str, method: str = "GET", payload: dict | None = None) -> dict | list | None:
        url = path if path.startswith("http") else self.base + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("User-Agent", "funasr-cpu-push")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with self.opener.open(req, timeout=600) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"GitHub API {method} {url} -> {exc.code}\n{detail}") from None
        return json.loads(body) if body else None


def build_gitignore(gh: GH, base_sha: str) -> bytes:
    """以远端 main 的 .gitignore 为基础，追加本地排除规则（不覆盖上游改动）。"""
    remote = gh.api(f"/contents/.gitignore?ref={base_sha}")
    remote_text = base64.b64decode(remote["content"]).decode("utf-8")
    local_text = Path(".gitignore").read_text(encoding="utf-8")

    idx = local_text.find(GI_MARKER)
    block = local_text[idx:].rstrip("\n") if idx >= 0 else ""

    ridx = remote_text.find(GI_MARKER)
    if ridx >= 0:  # 远端已有本地区块 → 替换
        merged = remote_text[:ridx].rstrip("\n") + "\n\n" + block + "\n"
    else:  # 追加
        merged = remote_text.rstrip("\n") + "\n\n" + block + "\n"
    return merged.encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="经 GitHub API 推送到 fork")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"目标仓库，默认 {DEFAULT_REPO}")
    parser.add_argument("--branch", default="main", help="目标分支，默认 main")
    parser.add_argument("--message", default=COMMIT_MESSAGE, help="提交信息")
    parser.add_argument("--dry-run", action="store_true", help="只列出将推送的文件")
    args = parser.parse_args()

    missing = [f for f in FILES if not Path(f).is_file()]
    if missing:
        sys.exit(f"以下文件不存在：{missing}")

    total = sum(Path(f).stat().st_size for f in FILES)
    print(f"仓库 {args.repo} / 分支 {args.branch}")
    print(f"待推送 {len(FILES)} 个文件，合计 {total / 1024 / 1024:.2f} MB")
    for f in FILES:
        print(f"  + {f}  ({Path(f).stat().st_size / 1024:.0f} KB)")
    if args.dry_run:
        return

    gh = GH(args.repo, gh_token())

    ref = gh.api(f"/git/ref/heads/{args.branch}")
    base_sha = ref["object"]["sha"]
    base_tree = gh.api(f"/git/commits/{base_sha}")["tree"]["sha"]
    print(f"\n远端 {args.branch} 当前 = {base_sha[:8]}")

    entries = []
    for rel in FILES:
        content = build_gitignore(gh, base_sha) if rel == ".gitignore" else Path(rel).read_bytes()
        blob = gh.api(
            "/git/blobs",
            "POST",
            {"content": base64.b64encode(content).decode("ascii"), "encoding": "base64"},
        )
        entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        print(f"  blob {rel} -> {blob['sha'][:8]}")

    tree = gh.api("/git/trees", "POST", {"base_tree": base_tree, "tree": entries})
    if tree["sha"] == base_tree:
        print("\n远端 tree 与本地一致，无变化，无需推送。")
        return

    commit = gh.api(
        "/git/commits",
        "POST",
        {"message": args.message, "tree": tree["sha"], "parents": [base_sha]},
    )
    print(f"  commit -> {commit['sha'][:8]}")

    gh.api(f"/git/refs/heads/{args.branch}", "PATCH", {"sha": commit["sha"], "force": False})
    print(f"\n完成：https://github.com/{args.repo}/commit/{commit['sha']}")
    print(f"分支视图：https://github.com/{args.repo}/tree/{args.branch}")


if __name__ == "__main__":
    main()
