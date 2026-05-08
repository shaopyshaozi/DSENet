import os
import re
import sys
import csv
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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


def to_float_scalar(v):
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return float(v.item())
        return float(v.flatten()[0].item())

    if isinstance(v, np.ndarray):
        if v.size == 1:
            return float(v.item())
        return float(v.flatten()[0])

    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return np.nan
        return float(v[0])

    try:
        return float(v)
    except Exception:
        return np.nan


def build_model(device, ckpt_path):
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
        map_location=device
    )
    model.eval()
    model.to(device)
    model.float()
    return model


def save_wave(path, wav, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, wav, sr)


def plot_curve(doas, values, title, xlabel, ylabel, save_path, gt_doa=None):
    plt.figure(figsize=(10, 5))
    plt.plot(doas, values, linewidth=2, label="Pseudo SI-SDRi")

    if gt_doa is not None:
        plt.axvline(
            gt_doa,
            linestyle="--",
            linewidth=1.5,
            label=f"Reference DOA = {gt_doa}°"
        )

    plt.xlim(0, 359)
    plt.xticks(np.arange(0, 361, 30))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    


def main():
    ckpt_path = r"D:\邵鹏远\UCL\博1\code\DSENet\logs\DSENet\version_11\checkpoints\epoch98_loss-3.4107_neg_si_sdr-7.3624.ckpt"
    noisy_path = r"D:\邵鹏远\UCL\博1\code\usb_4_mic_array\data\Respeaker\test3_doa301_spk1.wav"
    clean_path = r"D:\邵鹏远\UCL\博1\code\usb_4_mic_array\data\Respeaker\enhanced\test3_doa301_spk1_enhanced.wav"

    output_root = r"D:\邵鹏远\UCL\博1\code\DSENet\eval\doa_sweep"
    csv_path = os.path.join(output_root, "doa_sweep_metrics.csv")
    plot_path = os.path.join(output_root, "doa_sweep_sisdri.png")

    sample_rate = 16000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    os.makedirs(output_root, exist_ok=True)

    model = build_model(device, ckpt_path)

    noisy_ds, _ = load_audio_file(noisy_path, sample_rate)   # [C, T]
    clean_ds, _ = load_audio_file(clean_path, sample_rate)   # [C, T] or [1, T]

    print(f"Noisy shape: {tuple(noisy_ds.shape)}")
    print(f"Clean shape: {tuple(clean_ds.shape)}")

    assert noisy_ds.shape[1] == clean_ds.shape[1], \
        f"Length mismatch: noisy={noisy_ds.shape[1]}, clean={clean_ds.shape[1]}"

    filename = os.path.basename(noisy_path)
    gt_doa, default_width = parse_doa_width(filename)

    print(f"Parsed GT DOA from filename: {gt_doa}")
    print(f"Parsed width from filename: {default_width}")

    # Keep one target reference only
    # If clean is single channel, good.
    # If clean has multiple channels, we will still use the model ref channel as target reference.
    x = noisy_ds.unsqueeze(0).float().to(device)   # [1, C, T]
    yr = clean_ds.unsqueeze(0).float().to(device)  # [1, C, T] or [1, 1, T]

    ref_channel = model.ref_channel

    if yr.shape[1] == 1:
        yr_ref = yr[:, 0:1, :]
    else:
        yr_ref = yr[:, ref_channel:ref_channel+1, :]

    x_ref = x[:, ref_channel:ref_channel+1, :]

    fixed_width = default_width   # you can also manually set, e.g. 30

    metrics_list = ["SDR", "SI_SDR", "WB_PESQ"]

    results = []

    print("\nStarting DOA sweep...")
    for doa_value in tqdm(range(360), desc="DOA sweep"):
        DOA = torch.tensor([doa_value], dtype=torch.long, device=device)
        width = torch.tensor([fixed_width], dtype=torch.long, device=device)

        with torch.no_grad():
            yr_hat = model.forward(x, DOA, width)   # [1, 1, T]

            if model.loss.is_scale_invariant_loss:
                yr_hat = recover_scale(
                    preds=yr_hat,
                    mixture=x[:, ref_channel, :],
                    scale_src_together=True,
                    norm_if_exceed_1=False
                )

        metrics, input_metrics, imp_metrics = cal_metrics_functional(
            metrics_list,
            yr_hat[0],
            yr_ref[0],
            x_ref[0],
            sample_rate,
            device_only=None
        )

        # print(metrics)
        # print(input_metrics)
        # print(imp_metrics)

        row = {
            "doa": doa_value,
            "width": fixed_width,
            "input_SDR": to_float_scalar(input_metrics.get("input_sdr", np.nan)),
            "input_SI_SDR": to_float_scalar(input_metrics.get("input_si_sdr", np.nan)),
            "input_WB_PESQ": to_float_scalar(input_metrics.get("input_wb_pesq", np.nan)),
            "output_SDR": to_float_scalar(metrics.get("sdr", np.nan)),
            "output_SI_SDR": to_float_scalar(metrics.get("si_sdr", np.nan)),
            "output_WB_PESQ": to_float_scalar(metrics.get("wb_pesq", np.nan)),
            "improve_SDR": to_float_scalar(imp_metrics.get("sdr_i", np.nan)),
            "improve_SI_SDR": to_float_scalar(imp_metrics.get("si_sdr_i", np.nan)),
            "improve_WB_PESQ": to_float_scalar(imp_metrics.get("wb_pesq_i", np.nan)),
        }
        results.append(row)

    print("\nSaving CSV...")
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    doa_list = [r["doa"] for r in results]
    sisdri_list = [r["improve_SI_SDR"] for r in results]

    print("Saving plot...")
    plot_curve(
    doas=doa_list,
    values=sisdri_list,
    title=f"Pseudo-Reference DOA Sweep Similarity (reference DOA = {gt_doa}°)",
    xlabel="Input DOA φ(°)",
    ylabel="Pseudo SI-SDRi to Reference (dB)",
    save_path=plot_path,
    gt_doa=gt_doa
)

    best_idx = int(np.nanargmax(sisdri_list))
    best_doa = doa_list[best_idx]
    best_val = sisdri_list[best_idx]

    print("\n===== Sweep Summary =====")
    print(f"GT DOA from filename: {gt_doa}°")
    print(f"Fixed width: {fixed_width}°")
    print(f"Best DOA by SI-SDRi: {best_doa}°")
    print(f"Best SI-SDRi: {best_val:.4f} dB")
    print(f"CSV saved to: {csv_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()