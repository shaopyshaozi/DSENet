import os
import re
import sys
import torch
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from DOATrainer import TrainModule
from models.arch.DSENet import DSENet
from models.utils.metrics import recover_scale


def load_audio_file(path: str, target_sr: int):
    wav, sr = sf.read(path, always_2d=True)   # [T, C]
    wav = wav.T.astype(np.float32)            # [C, T]

    if sr != target_sr:
        resampled = []
        gcd = np.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd

        for ch in wav:
            resampled_ch = resample_poly(ch, up, down).astype(np.float32)
            resampled.append(resampled_ch)

        wav = np.stack(resampled, axis=0)
        sr = target_sr

    return torch.from_numpy(wav), sr


def parse_doa_width(filename: str):
    doa_match = re.search(r"doa(\d+)", filename)
    width_match = re.search(r"width(\d+)", filename)

    doa_val = int(doa_match.group(1)) if doa_match else 0
    width_val = int(width_match.group(1)) if width_match else 30

    return doa_val, width_val


def main():
    ckpt_path = r"D:\邵鹏远\UCL\博1\code\DSENet\logs\DSENet\version_11\checkpoints\epoch96_loss-3.4198_neg_si_sdr-7.3765.ckpt"

    input_dir = r"D:\邵鹏远\UCL\博1\code\usb_4_mic_array\data\Respeaker"
    save_enhanced_dir = r"D:\邵鹏远\UCL\博1\code\usb_4_mic_array\data\Respeaker\enhanced"

    os.makedirs(save_enhanced_dir, exist_ok=True)

    sample_rate = 16000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    arch = DSENet(
        dim_input=8,
        dim_output=2,
        dim_squeeze=8,
        num_layers=8,
        num_freqs=129,
        encoder_kernel_size=5,
        dim_hidden=192,
        dim_ffn=192,
        num_heads=4,
        dropout=(0.0, 0.0, 0.0),
        kernel_size=(5, 3),
        conv_groups=(8, 8),
        norms=("LN", "LN", "GN", "LN", "LN", "LN"),
        padding="zeros",
        full_share=0,
        d_embedding=40,
        d_alpha=20,
        width_emb_dim=3,
        width_stage=15,
        width_control=True,
    )

    print("Loading checkpoint...")
    model = TrainModule.load_from_checkpoint(
        ckpt_path,
        arch=arch,
        map_location=device,
    )

    model.eval()
    model.to(device)
    model.float()

    wav_files = sorted(
        [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(".wav")
            and not f.lower().startswith("enhanced_")
        ]
    )

    print(f"Found {len(wav_files)} wav files")

    for idx, wav_name in enumerate(tqdm(wav_files), 1):
        try:
            wav_path = os.path.join(input_dir, wav_name)

            noisy_ds, _ = load_audio_file(wav_path, sample_rate)

            # Expected ReSpeaker raw file: [4, T]
            if noisy_ds.shape[0] != 4:
                print(f"[{idx}/{len(wav_files)}] Skip {wav_name}: expected 4 channels, got {noisy_ds.shape[0]}")
                continue

            x = noisy_ds.unsqueeze(0).float().to(device)  # [1, 4, T]

            doa_value, width_value = parse_doa_width(wav_name)

            DOA = torch.tensor([doa_value], dtype=torch.long, device=device)
            width = torch.tensor([width_value], dtype=torch.long, device=device)

            ref_channel = model.ref_channel

            with torch.no_grad():
                yr_hat = model.forward(x, DOA, width)  # [1, 1, T]

                if model.loss.is_scale_invariant_loss:
                    yr_hat = recover_scale(
                        preds=yr_hat,
                        mixture=x[:, ref_channel, :],
                        scale_src_together=True,
                        norm_if_exceed_1=False,
                    )

            enhanced = yr_hat[0, 0].detach().cpu().numpy()

            save_name = wav_name.replace(".wav", "_enhanced.wav")
            save_path = os.path.join(save_enhanced_dir, save_name)

            sf.write(save_path, enhanced, sample_rate)

            print(f"[{idx}/{len(wav_files)}] Enhanced: {wav_name} -> {save_name}")

        except Exception as e:
            print(f"[{idx}/{len(wav_files)}] Failed: {wav_name} | {e}")

    print(f"\nSaved enhanced wavs to: {save_enhanced_dir}")


if __name__ == "__main__":
    main()