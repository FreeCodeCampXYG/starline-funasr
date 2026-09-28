# -*- coding: utf-8 -*-
"""FunASR 中文语音识别命令行脚本（CPU，虚拟环境 .venv 下运行）

用法：
  .venv/Scripts/python.exe funasr_cli.py 音频.wav            # 识别单个文件
  .venv/Scripts/python.exe funasr_cli.py 音频目录            # 批量识别目录下全部音频
  .venv/Scripts/python.exe funasr_cli.py a.wav b.wav -o 结果.txt

说明：
  - 首次运行会自动从 ModelScope 下载模型（约 990MB），缓存于 ~/.cache/modelscope
  - 输出格式：文件路径 <TAB> 识别文本，一行一个文件
"""
import argparse
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def collect_audios(inputs):
    """把命令行输入（文件/目录混合）展开成音频文件列表"""
    files = []
    for p in inputs:
        p = Path(p)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            print(f"[警告] 路径不存在: {p}")
    return files


def main():
    ap = argparse.ArgumentParser(description="FunASR 中文语音识别（CPU）")
    ap.add_argument("inputs", nargs="+", help="音频文件或目录（可多个）")
    ap.add_argument("-o", "--output", help="结果保存到文本文件（UTF-8）")
    ap.add_argument("--model", default="paraformer-zh", help="模型名，默认 paraformer-zh")
    args = ap.parse_args()

    files = collect_audios(args.inputs)
    if not files:
        print("没有找到可识别的音频文件")
        return
    print(f"共 {len(files)} 个文件，加载模型: {args.model} ...")

    from funasr import AutoModel
    model = AutoModel(model=args.model, device="cpu", disable_update=True, disable_pbar=True)

    results = []
    for f in files:
        try:
            res = model.generate(input=str(f))
            text = res[0]["text"]
        except Exception as e:
            text = f"<识别失败: {e}>"
        print(f"{f}\t{text}")
        results.append(f"{f}\t{text}")

    if args.output:
        Path(args.output).write_text("\n".join(results) + "\n", encoding="utf-8")
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
