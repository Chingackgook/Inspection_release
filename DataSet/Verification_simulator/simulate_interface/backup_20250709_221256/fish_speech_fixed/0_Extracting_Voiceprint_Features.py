import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.fish_speech_fixed import ENV_DIR
from Inspection.adapters.custom_adapters.fish_speech_fixed import *
exe = Executor('fish_speech_fixed', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from loguru import logger

#pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.utils.file import AUDIO_EXTENSIONS

# register eval resolver
#OmegaConf.register_new_resolver("eval", eval)



@torch.no_grad()
def main():
    # 使用默认参数
    input_path = Path(ENV_DIR + "/test.wav")
    output_path = Path(FILE_RECORD_PATH + "/fake.wav")
    config_name = "firefly_gan_vq"
    checkpoint_path = ENV_DIR + "/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    device = "cuda"

    exe.create_interface_objects(
        config_name=config_name,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model = exe.adapter.model

    if input_path.suffix in AUDIO_EXTENSIONS:
        logger.info(f"Processing in-place reconstruction of {input_path}")

        # Load audio
        audio, sr = torchaudio.load(str(input_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(
            audio, sr, model.spec_transform.sample_rate
        )

        audios = audio[None].to(device)
        logger.info(
            f"Loaded audio with {audios.shape[2] / model.spec_transform.sample_rate:.2f} seconds"
        )

        # VQ Encoder
        audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)
        indices = exe.run("encode", audios=audios, audio_lengths=audio_lengths)[0][0]
        logger.info(f"Generated indices of shape {indices.shape}")

        # Save indices
        np.save(output_path.with_suffix(".npy"), indices.cpu().numpy())
    elif input_path.suffix == ".npy":
        logger.info(f"Processing precomputed indices from {input_path}")
        indices = np.load(input_path)
        indices = torch.from_numpy(indices).to(device).long()
        assert indices.ndim == 2, f"Expected 2D indices, got {indices.ndim}"
    else:
        raise ValueError(f"Unknown input type: {input_path}")

    # Restore
    feature_lengths = torch.tensor([indices.shape[1]], device=device)
    fake_audios, _ = exe.run("decode", indices=indices[None], feature_lengths=feature_lengths)
    audio_time = fake_audios.shape[-1] / model.spec_transform.sample_rate

    logger.info(
        f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.2f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}"
    )

    # Save audio
    fake_audio = fake_audios[0, 0].float().cpu().numpy()
    sf.write(output_path, fake_audio, model.spec_transform.sample_rate)
    logger.info(f"Saved audio to {output_path}")

# 直接运行主逻辑
exe.run("GlobalHydra_clear")
main()
