# -*- encoding: utf-8 -*-
"""静音门限离线自测：不加载模型，只验证 _speech_stats / _has_speech 的判定。

用法：.venv/Scripts/python.exe test_silence_gate.py
"""
import numpy as np

import asr_server as srv

SR = srv.TARGET_SR
rng = np.random.default_rng(0)


def make_silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SR), dtype=np.float32)[None, :]


def make_noise(sec: float, amp: float) -> np.ndarray:
    return (rng.normal(0, amp, int(sec * SR))).astype(np.float32)[None, :]


def make_click_only(sec: float = 0.54, click_sec: float = 0.04, amp: float = 0.27) -> np.ndarray:
    """复刻用户案例：空白音频里只有一小段按键咔哒（峰值高、时长短）。"""
    x = np.zeros(int(sec * SR), dtype=np.float32)
    n = int(click_sec * SR)
    x[:n] = rng.normal(0, amp, n).astype(np.float32)
    return x[None, :]


def make_speech_then_silence(speech_sec: float, total_sec: float, offset: float = 2.0) -> np.ndarray:
    """从中段取 speech_sec 秒真实语音（开头是静音，避开），其余补静音。"""
    import librosa

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
    ("按键咔哒 only（用户案例）", make_click_only(), False),
    ("真实语音 5.5s", make_speech_then_silence(5.5, 5.5), True),
    ("真实语音 0.5s + 静音 2s", make_speech_then_silence(0.5, 2.0), True),
    ("真实语音 0.25s + 静音 1s", make_speech_then_silence(0.25, 1.0), True),
]


def main() -> None:
    print(f"门限参数：绝对 {srv.SILENCE_DBFS}dB / 相对噪声底 +{srv.SPEECH_MARGIN_DB}dB / "
          f"最短 {srv.MIN_SPEECH_SEC}s / 最低占比 {srv.MIN_SPEECH_RATIO:.0%}\n")
    print(f"{'用例':<28}{'有效时长':>10}{'占比':>9}{'判定':>8}{'期望':>8}{'':>4}")
    print("-" * 72)
    failed = 0
    for name, wav, expect in CASES:
        sec, ratio = srv._speech_stats(wav)
        got = srv._has_speech(wav)
        ok = "PASS" if got == expect else "FAIL"
        if got != expect:
            failed += 1
        print(f"{name:<28}{sec:>9.2f}s{ratio:>8.0%}"
              f"{'有语音' if got else '无语音':>10}{'有语音' if expect else '无语音':>10}{ok:>6}")
    print("-" * 72)
    print(f"结果：{'全部通过' if failed == 0 else f'{failed} 个用例失败'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
