# -*- encoding: utf-8 -*-
"""静音门限离线自测：加载 fsmn-vad，验证 _has_speech 判定，并与【旧能量门限】对照。

用法：.venv/Scripts/python.exe test_silence_gate.py
"""
import librosa
import numpy as np

import asr_server as srv

SR = srv.TARGET_SR
rng = np.random.default_rng(0)


def make_silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SR), dtype=np.float32)[None, :]


def make_noise(sec: float, amp: float) -> np.ndarray:
    return (rng.normal(0, amp, int(sec * SR))).astype(np.float32)[None, :]


def make_click_only(sec: float = 0.54, click_sec: float = 0.04, amp: float = 0.27) -> np.ndarray:
    """复刻用户案例：空白音频里只有一小段按键咔哒。"""
    x = np.zeros(int(sec * SR), dtype=np.float32)
    n = int(click_sec * SR)
    x[:n] = rng.normal(0, amp, n).astype(np.float32)
    return x[None, :]


def make_keyboard(sec: float = 3.0) -> np.ndarray:
    """持续按键咔哒：能量门限挡不住（会输出幻觉），VAD 应能拦下。"""
    x = np.zeros(int(sec * SR), dtype=np.float32)
    for t in np.arange(0.2, sec, 0.18):
        i = int(t * SR)
        L = int(0.01 * SR)
        x[i:i + L] += (rng.standard_normal(L) * np.exp(-np.arange(L) / (0.002 * SR)) * 0.7).astype(np.float32)
    return x[None, :]


def make_speech(speech_sec: float, total_sec: float, offset: float = 2.0) -> np.ndarray:
    """从中段取 speech_sec 秒真实语音，其余补静音。"""
    wav, _ = librosa.load("asr_example_zh.wav", sr=SR, mono=True)
    seg = wav[int(offset * SR):]
    n = min(int(speech_sec * SR), seg.size)
    x = np.zeros(int(total_sec * SR), dtype=np.float32)
    x[:n] = seg[:n]
    return x[None, :]


CASES = [
    ("纯静音 0.54s", make_silence(0.54), False),
    ("低幅噪声 0.54s（-46dB）", make_noise(0.54, 0.005), False),
    ("更低的底噪 2s", make_noise(2.0, 0.002), False),
    ("按键咔哒 0.04s（用户案例）", make_click_only(), False),
    ("持续按键咔哒 3s", make_keyboard(), False),
    ("真实语音 5.5s", make_speech(5.5, 5.5), True),
    ("真实语音 0.5s + 静音 2s", make_speech(0.5, 2.0), True),
    ("真实语音 0.25s + 静音 1s", make_speech(0.25, 1.0), True),
]


def main() -> None:
    print("加载 fsmn-vad ...", flush=True)
    from funasr import AutoModel

    srv._vad_model = AutoModel(
        model="fsmn-vad", device="cpu", disable_update=True, disable_pbar=True
    )
    srv._GATE_ENABLED = True
    print(f"门限：fsmn-vad 语音总时长 >= {srv.MIN_SPEECH_SEC}s\n")
    print(f"{'用例':<28}{'旧能量':>9}{'VAD语音':>9}{'判定':>8}{'期望':>8}{'':>5}")
    print("-" * 72)
    failed = 0
    for name, wav, expect in CASES:
        esec, _ = srv._speech_stats(wav)
        vsec = srv._vad_speech_sec(wav)
        got = srv._has_speech(wav)
        ok = "PASS" if got == expect else "FAIL"
        if got != expect:
            failed += 1
        print(f"{name:<28}{esec:>7.2f}s{vsec:>8.2f}s"
              f"{'有语音' if got else '无语音':>10}{'有语音' if expect else '无语音':>10}{ok:>6}")
    print("-" * 72)
    print(f"结果：{'全部通过' if failed == 0 else f'{failed} 个用例失败'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
