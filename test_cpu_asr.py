# -*- coding: utf-8 -*-
"""CPU 环境下 FunASR 中文识别冒烟测试"""
import time
from funasr import AutoModel

t0 = time.time()
# paraformer-zh 会自动从 ModelScope 下载并缓存到本地
model = AutoModel(
    model="paraformer-zh",   # 中文 Paraformer 大模型
    device="cpu",
    disable_update=True,     # 跳过版本检查
    disable_pbar=True,
)
print("模型加载耗时 %.1fs" % (time.time() - t0))

t1 = time.time()
res = model.generate(input="asr_example_zh.wav")
print("推理耗时 %.1fs" % (time.time() - t1))
print("识别结果:", res[0]["text"])
