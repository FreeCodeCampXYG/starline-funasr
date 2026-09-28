# -*- coding: utf-8 -*-
"""RapidASR（rapid_paraformer，ONNXRuntime 推理）中文语音识别命令行脚本

用法（与 funasr_cli.py 完全一致的接口）：
  .venv/Scripts/python.exe rapid_asr_cli.py 音频.wav            # 识别单个文件
  .venv/Scripts/python.exe rapid_asr_cli.py 音频目录            # 批量识别目录下全部音频
  .venv/Scripts/python.exe rapid_asr_cli.py a.wav b.wav -o 结果.txt

说明：
  - 模型已下载到 rapid_asr_models/resources/（ONNX 格式，约 824MB）
  - 与 FunASR 版相比：纯 ONNXRuntime 推理、不依赖 torch、单文件启动更快，
    但只支持 Paraformer 这一个模型，长音频建议先用 RapidVAD 切分
"""
import argparse
import os
from pathlib import Path

# 模型配置文件路径（download_hf_model 下载的目录结构）
CONFIG_PATH = Path(__file__).parent / "rapid_asr_models" / "resources" / "config.yaml"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def collect_audios(inputs):
    """把命令行输入（文件/目录混合）展开成音频文件列表（统一转绝对路径，
    因为加载模型时会切换工作目录）"""
    files = []
    for p in inputs:
        p = Path(p)
        if p.is_dir():
            files += sorted(f.resolve() for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        elif p.is_file():
            files.append(p.resolve())
        else:
            print(f"[警告] 路径不存在: {p}")
    return files


def main():
    ap = argparse.ArgumentParser(description="RapidASR 中文语音识别（ONNXRuntime CPU）")
    ap.add_argument("inputs", nargs="+", help="音频文件或目录（可多个）")
    ap.add_argument("-o", "--output", help="结果保存到文本文件（UTF-8）")
    ap.add_argument("--config", default=str(CONFIG_PATH), help="resources/config.yaml 路径")
    args = ap.parse_args()

    files = collect_audios(args.inputs)
    if not files:
        print("没有找到可识别的音频文件")
        return
    print(f"共 {len(files)} 个文件，加载模型: {args.config}")

    from rapid_paraformer import RapidParaformer
    # config.yaml 里的模型路径是 "resources/models/xxx" 相对路径，
    # 需切到 config 的上上级（即包含 resources 的目录）才能解析
    os.chdir(Path(args.config).parent.parent)
    paraformer = RapidParaformer(args.config)

    results = []
    for f in files:
        try:
            # rapid_paraformer 接口：传入路径列表，返回文本列表（支持批量）
            res = paraformer([str(f)])
            text = res[0] if res else "<识别失败: 无结果>"
        except Exception as e:
            text = f"<识别失败: {e}>"
        print(f"{f}\t{text}")
        results.append(f"{f}\t{text}")

    if args.output:
        out = Path(args.output).resolve()  # 切目录前先解析输出路径
        out.write_text("\n".join(results) + "\n", encoding="utf-8")
        print(f"\n结果已保存到: {out}")


if __name__ == "__main__":
    main()
