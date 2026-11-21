"""Audio preprocessing: load, resample to 16 kHz mono, energy VAD and log-Mel filterbank features."""

import argparse
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchaudio
import yaml


def load_audio(path, sample_rate):
    """Read an audio file as a mono float32 tensor in [-1, 1] at `sample_rate`."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(data.mean(axis=1))
    if sr != sample_rate:
        wave = torchaudio.functional.resample(wave, sr, sample_rate)
    return wave


def add_noise(wave, snr_db, seed):
    """Additive white Gaussian noise at `snr_db` relative to the power of the whole utterance."""
    noise = torch.randn(len(wave), generator=torch.Generator().manual_seed(seed))
    scale = (wave.pow(2).mean() / (noise.pow(2).mean() * 10 ** (snr_db / 10))).sqrt()
    return wave + scale * noise


def speech_mask(wave, sample_rate, cfg):
    """Per-sample speech mask from frame energies relative to the loudest frame."""
    frame = int(sample_rate * cfg["frame_ms"] / 1000)
    hop = int(sample_rate * cfg["hop_ms"] / 1000)
    if len(wave) < frame:
        return torch.zeros(len(wave), dtype=torch.bool)
    frames = wave.unfold(0, frame, hop)
    energy_db = 10 * torch.log10(frames.pow(2).mean(dim=1) + 1e-10)
    mask = energy_db > energy_db.max() - cfg["threshold_db"]
    hangover = int(cfg["hangover_ms"] / cfg["hop_ms"])
    if hangover:
        mask = torch.nn.functional.max_pool1d(
            mask[None, None].float(), 2 * hangover + 1, stride=1, padding=hangover
        )[0, 0].bool()
    frame_of_sample = torch.clamp(torch.arange(len(wave)) // hop, max=len(mask) - 1)
    return mask[frame_of_sample]


def fbank(wave, sample_rate, cfg):
    """Kaldi log-Mel filterbank with per-utterance mean normalisation (WeSpeaker recipe)."""
    feats = torchaudio.compliance.kaldi.fbank(
        wave[None] * (1 << 15),
        num_mel_bins=cfg["num_mel_bins"],
        frame_length=cfg["frame_length_ms"],
        frame_shift=cfg["frame_shift_ms"],
        dither=0.0,
        sample_frequency=sample_rate,
        window_type="hamming",
    )
    return feats - feats.mean(dim=0)


def preprocess(path, cfg):
    """Full pipeline for one file: (frames, mel bins) features, or None on failure to acquire."""
    sr = cfg["audio"]["sample_rate"]
    wave = load_audio(path, sr)
    if "noise" in cfg:
        # Seeded by file name, so every model and worker sees the same noise for a given file.
        seed = zlib.crc32(Path(path).name.encode()) ^ cfg["noise"]["seed"]
        wave = add_noise(wave, cfg["noise"]["snr_db"], seed)
    if cfg["vad"]["enabled"]:
        wave = wave[speech_mask(wave, sr, cfg["vad"])]
    if len(wave) < cfg["audio"]["min_speech_s"] * sr:
        return None
    return fbank(wave, sr, cfg["features"])


def plot_pipeline(path, cfg, out):
    """Figure with the raw waveform, the VAD decision and the resulting features."""
    sr = cfg["audio"]["sample_rate"]
    wave = load_audio(path, sr)
    mask = speech_mask(wave, sr, cfg["vad"])
    feats = fbank(wave[mask], sr, cfg["features"])
    t = torch.arange(len(wave)) / sr

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(t, wave, linewidth=0.5)
    axes[0].set(title=f"Raw waveform, {sr} Hz mono ({Path(path).name})", xlabel="time (s)")
    axes[1].plot(t, wave, linewidth=0.5, color="0.7")
    axes[1].fill_between(t, -1, 1, where=mask.numpy(), alpha=0.25, label="speech")
    axes[1].set(title="Energy VAD", xlabel="time (s)", ylim=(wave.min(), wave.max()))
    axes[1].legend(loc="upper right")
    image = axes[2].imshow(feats.T, origin="lower", aspect="auto", interpolation="nearest")
    axes[2].set(title="Log-Mel filterbank after VAD, mean-normalised", xlabel="frame (10 ms)", ylabel="Mel bin")
    fig.colorbar(image, ax=axes[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot the preprocessing stages for one audio file.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    plot_pipeline(args.audio, cfg, args.out)


if __name__ == "__main__":
    main()
