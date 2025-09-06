
import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.chatTTS import ENV_DIR
from Inspection.adapters.custom_adapters.chatTTS import *
exe = Executor('chatTTS','simulation')
FILE_RECORD_PATH = exe.now_record_path

import os
import sys
import logging
import ChatTTS

# 设置环境变量以支持 MPS fallback
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 当前工作目录
now_dir = os.getcwd()
sys.path.append(now_dir)

# 设置日志记录器

# 初始化 CustomAdapter 对象

# 加载模型
exe.create_interface_objects(source="local", custom_path=ENV_DIR)  # 使用 load_model 方法加载模型

# 下载模型
exe.run("download_models", source="local", force_redownload=False, custom_path=ENV_DIR)

# 检查是否加载
exe.run("has_loaded")

# 准备文本输入
texts = [
    "的 话 语 音 太 短 了 会 造 成 生 成 音 频 错 误 ， 这 是 占 位 占 位 ， 老 大 爷 觉 得 车 夫 的 想 法 很 有 道 理 [uv_break]",
    "的 话 评 分 只 是 衡 量 音 色 的 稳 定 性 ， 不 代 表 音 色 的 好 坏 ， 可 以 根 据 自 己 的 需 求 选 择 [uv_break] 合 适 的 音 色",
    "然 后 举 个 简 单 的 例 子 ， 如 果 一 个 [uv_break] 沙 哑 且 结 巴 的 音 色 一 直 很 稳 定 ， 那 么 它 的 评 分 就 会 很 高 。",
    "语 音 太 短 了 会 造 成 生 成 音 频 错 误 ， 这 是 占 位 [uv_break] 占 位 。 我 使 用 seed id 去 生 成 音 频 ， 但 是 生 成 的 音 频 不 稳 定",
    "在d id 只 是 一 个 参 考 id [uv_break] 不 同 的 环 境 下 音 色 不 一 定 一 致 。 还 是 推 荐 使 用 。 pt 文 件 载 入 音 色",
    "的 话 语 音 太 短 了 会 造 成 生 成 音 频 错 误 ， 这 是 占 位 占 位 。 音 色 标 的 男 女 [uv_break] 准 确 吗",
    "， 当 前 第 一 批 测 试 的 音 色 有 两 千 条 [uv_break] ， 根 据 声 纹 相 似 性 简 单 打 标 ， 准 确 度 不 高 ， 特 别 是 特 征 一 项",
    "语 音 太 短 了 会 造 成 生 成 音 频 错 误 ， 这 是 占 位 占 位 。 仅 供 参 考 。 如 果 大 家 有 更 好 的 标 注 方 法 ， 欢 迎 pr [uv_break] 。",
]

# 创建 InferCodeParams 对象
params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb=exe.run("sample_random_speaker"),  # Ensure this method is available
    temperature=0.3,
    top_P=0.005,
    top_K=1,
    show_tqdm=False,
)

fail = False

# 推理
wavs = exe.run("infer", text=texts, skip_refine_text=True, split_text=False, params_infer_code=params_infer_code)

for k, wav in enumerate(wavs):
    if wav is None:
        fail = True

if fail:
    print("推理失败")

# 清理资源
exe.run("unload")
