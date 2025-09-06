
import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.FishSpeech_fixed import ENV_DIR
from Inspection.adapters.custom_adapters.FishSpeech_fixed import *
exe = Executor('FishSpeech_fixed','simulation')
FILE_RECORD_PATH = exe.now_record_path

from pathlib import Path
import numpy as np
import pyrootutils
import soundfile as sf
import torch
import torchaudio
from loguru import logger
from omegaconf import OmegaConf

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# 模拟输入参数
input_path = Path(ENV_DIR) / "test.wav"  # 假设存在的音频文件
output_path = Path(FILE_RECORD_PATH) / "fake.wav"  # 输出文件路径
config_name = ENV_DIR / "my_model_config"  # 配置名称
checkpoint_path = ENV_DIR / "path/to/checkpoint.pth"  # 检查点路径
device = "cuda"  # 使用的设备

# 加载模型
model = exe.create_interface_objects(config_name=config_name, checkpoint_path=checkpoint_path, device=device)

# 处理音频
logger.info(f"Processing in-place reconstruction of {input_path}")

# 加载音频
audio, sr = torchaudio.load(str(input_path))
if audio.shape[0] > 1:
    audio = audio.mean(0, keepdim=True)
audio = torchaudio.functional.resample(audio, sr, model.spec_transform.sample_rate)

audios = audio[None].to(device)
logger.info(f"Loaded audio with {audios.shape[2] / model.spec_transform.sample_rate:.2f} seconds")

# VQ Encoder
audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)

# 替换为 exe.run 调用
indices = exe.run("encode", audios=audios, audio_lengths=audio_lengths)[0][0]

logger.info(f"Generated indices of shape {indices.shape}")

# 保存 indices
np.save(output_path.with_suffix(".npy"), indices.cpu().numpy())

# Restore
feature_lengths = torch.tensor([indices.shape[1]], device=device)

# 替换为 exe.run 调用
fake_audios, _ = exe.run("decode", indices=indices[None], feature_lengths=feature_lengths)

audio_time = fake_audios.shape[-1] / model.spec_transform.sample_rate

logger.info(f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.2f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}")

# 保存音频
fake_audio = fake_audios[0, 0].float().cpu().numpy()
sf.write(output_path, fake_audio, model.spec_transform.sample_rate)
logger.info(f"Saved audio to {output_path}")
