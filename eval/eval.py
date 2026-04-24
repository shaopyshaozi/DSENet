import os
import re
import sys
import torch
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from DOATrainer import TrainModule
from models.arch.DSENet import DSENet
from models.utils.metrics import cal_metrics_functional, recover_scale


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

def to_printable(v):
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return f"{v.item():.4f}"
        return str(v.detach().cpu().tolist())

    if isinstance(v, np.ndarray):
        if v.size == 1:
            return f"{float(v.item()):.4f}"
        return str(v.tolist())

    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            try:
                return f"{float(v[0]):.4f}"
            except:
                return str(v)
        return str(v)

    try:
        return f"{float(v):.4f}"
    except:
        return str(v)


def main():
    ckpt_path = r"D:\邵鹏远\UCL\博1\code\DSENet\logs\DSENet\version_9\checkpoints\epoch98_loss1.0465_neg_si_sdr-0.5227.ckpt"
    noisy_path = r"D:\邵鹏远\UCL\博1\code\DSENet\eval\mic_fileid_3_doa67_6spk.wav"
    clean_path = r"D:\邵鹏远\UCL\博1\code\DSENet\eval\clean_fileid_3_doa67_spk6.wav"
    save_enhanced_path = r"D:\邵鹏远\UCL\博1\code\DSENet\eval\enhanced_test.wav"

    sample_rate = 16000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    arch = DSENet(
        dim_input=6,
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
        map_location=device
    )
    model.eval()
    model.to(device)
    model.float()

    noisy_ds, _ = load_audio_file(noisy_path, sample_rate)
    clean_ds, _ = load_audio_file(clean_path, sample_rate)

    print(f"Noisy shape: {tuple(noisy_ds.shape)}")
    print(f"Clean shape: {tuple(clean_ds.shape)}")

    assert noisy_ds.shape[1] == clean_ds.shape[1], \
        f"Length mismatch: noisy={noisy_ds.shape[1]}, clean={clean_ds.shape[1]}"

    if clean_ds.shape[0] == 1 and noisy_ds.shape[0] > 1:
        clean_ds = clean_ds.repeat(noisy_ds.shape[0], 1)

    x = noisy_ds.unsqueeze(0).float().to(device)   # [1, C, T]
    yr = clean_ds.unsqueeze(0).float().to(device)  # [1, C, T]

    filename = os.path.basename(noisy_path)
    doa_value, width_value = parse_doa_width(filename)

    DOA = torch.tensor([doa_value], dtype=torch.long, device=device)
    width = torch.tensor([width_value], dtype=torch.long, device=device)

    print(f"Using DOA={doa_value}, width={width_value}")

    ref_channel = model.ref_channel

    with torch.no_grad():
        yr_hat = model.forward(x, DOA, width)           # [1, 1, T]
        yr_ref = yr[:, ref_channel, :].unsqueeze(1)     # [1, 1, T]
        x_ref = x[:, ref_channel, :].unsqueeze(1)       # [1, 1, T]

        if model.loss.is_scale_invariant_loss:
            yr_hat = recover_scale(
                preds=yr_hat,
                mixture=x[:, ref_channel, :],
                scale_src_together=True,
                norm_if_exceed_1=False
            )

    metrics_list = ["SDR", "SI_SDR", "WB_PESQ"]

    metrics, input_metrics, imp_metrics = cal_metrics_functional(
        metrics_list,
        yr_hat[0],
        yr_ref[0],
        x_ref[0],
        sample_rate,
        device_only=None
    )

    print("\n===== Metric Types =====")
    for name, d in [("input", input_metrics), ("output", metrics), ("improve", imp_metrics)]:
        print(f"\n{name}:")
        for k, v in d.items():
            print(k, type(v), v)

    print("\n===== Input Metrics =====")
    for k, v in input_metrics.items():
        print(f"{k}: {to_printable(v)}")

    print("\n===== Output Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {to_printable(v)}")

    print("\n===== Improvement Metrics =====")
    for k, v in imp_metrics.items():
        print(f"{k}: {to_printable(v)}")

    enhanced = yr_hat[0, 0].detach().cpu().numpy()
    sf.write(save_enhanced_path, enhanced, sample_rate)
    print(f"\nSaved enhanced audio to: {save_enhanced_path}")


if __name__ == "__main__":
    main()