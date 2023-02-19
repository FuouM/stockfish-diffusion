import argparse
import json
import os
from typing import Union
import librosa
import numpy as np
import soundfile as sf
import torch
from mmengine import Config

from audio_stuff.audio_utils import loudness_norm
from audio_stuff.audio_processing import slice_audio

from loguru import logger

from feature_extractors import FEATURE_EXTRACTORS, PITCH_EXTRACTORS

from model_stuff.inference import load_checkpoint
from audio_stuff.audio_processing import get_mel_from_audio, slice_audio
from util_stuff.tensor import repeat_expand


def parse_args():
    parser = argparse.ArgumentParser(
        prog="stockfish-diffusion",
        description="""Fast Singing Voice Conversion""")
    
    parser.add_argument("--config", dest="config_path", type=str, required=True,
                        help="Path to config file (finetune.py).")
    parser.add_argument("--model", dest="model_path", type=str, required=True,
                        help="Path to model file (ckpt).")
    parser.add_argument("--input", dest="input_path", type=str, required=True,
                        help="Path to input file (wav).")
    parser.add_argument("--output", dest="output_path", type=str,
                        help="Path to output file (folder).")
    parser.add_argument("--transpose", type=int, default=0,
                        help="Semitones to transpose. Default=0.")
    args = parser.parse_args()
    return args

@torch.no_grad()
def main():
    args = parse_args()
    
    if torch.cuda.is_available():
        print('GPU is available')
        device = "cuda"
    else:
        print('GPU is not available. CPU mode.')
        device = "cpu"
    
    # CONSTANTS
    speaker_id = 0
    silence_threshold=60
    max_slice_duration=30.0
    sampler_interval=None
    
    sampler_progress=False
    vocals_loudness_gain=0.0
    
    config = Config.fromfile(args.config_path)
    model_path = args.model_path
    # config.model.diffusion.sampler_interval = sampler_interval
    input_path = args.input_path
    output_path = args.output_path
    
    transpose = args.transpose
    
    # Load audio
    audio, sr = librosa.load(input_path, sr=config.sampling_rate, mono=True)
    
    # Normalize loudness
    audio = loudness_norm(audio, sr)
    
    segments = list(
        slice_audio(
            audio, sr, max_duration=max_slice_duration, top_db=silence_threshold
        )
    )
    logger.info(f"Sliced into {len(segments)} segments")
    
    # Load models
    text_features_extractor = FEATURE_EXTRACTORS.build(
        config.preprocessing.text_features_extractor
    ).to(device)
    text_features_extractor.eval()
    model = load_checkpoint(config, model_path, device=device)
    
    pitch_extractor = PITCH_EXTRACTORS.build(config.preprocessing.pitch_extractor)
    assert pitch_extractor is not None, "Pitch extractor not found"
    
    generated_audio = np.zeros_like(audio)
    audio_torch = torch.from_numpy(audio).to(device)[None]
    
    for idx, (start, end) in enumerate(segments):
        segment = audio_torch[:, start:end]
        logger.info(
            f"Processing segment {idx + 1}/{len(segments)}, duration: {segment.shape[-1] / sr:.2f}s"
        )
        # Extract mel
        mel = get_mel_from_audio(segment, sr)

        # Extract pitch (f0)
        pitch = pitch_extractor(segment, sr, pad_to=mel.shape[-1]).float()
        pitch *= 2 ** (transpose / 12)

        # Extract text features
        text_features = text_features_extractor(segment, sr)[0]
        text_features = repeat_expand(text_features, mel.shape[-1]).T

        # Predict
        src_lens = torch.tensor([mel.shape[-1]]).to(device)
        
        features = model.model.forward_features(
            speakers=torch.tensor([speaker_id]).long().to(device),
            contents=text_features[None].to(device),
            src_lens=src_lens,
            max_src_len=max(src_lens),
            mel_lens=src_lens,
            max_mel_len=max(src_lens),
            pitches=pitch[None].to(device),
        )
        
        result = model.model.diffusion(features["features"], progress=sampler_progress)
        wav = model.vocoder.spec2wav(result[0].T, f0=pitch).cpu().numpy()
        max_wav_len = generated_audio.shape[-1] - start
        generated_audio[start : start + wav.shape[-1]] = wav[:max_wav_len]
    
    # Loudness normalization
    generated_audio = loudness_norm(generated_audio, sr)

    # Loudness gain
    loudness_float = 10 ** (vocals_loudness_gain / 20)
    generated_audio = generated_audio * loudness_float

    logger.info("Done")

    if output_path is not None:
        sf.write(output_path, generated_audio, sr)
    else:
        sf.write(f"./output/{os.path.basename(input_path).split('.')[0]}.wav", generated_audio, sr)

    print("Audio sucessfully writen")

if __name__ == '__main__':
    main()
    