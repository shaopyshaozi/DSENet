import os
import re
import time
import torch
import soundfile as sf
import numpy as np

from DOATrainer import TrainModule

from models.arch.DSENet import DSENet

# =========================
# CONFIG
# =========================
ckpt_path = "last.ckpt"  
audio_path = "mic_fileid_0_doa48_6spk.wav"  
#device = "cpu"
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

# =========================
# LOAD TRAINED MODEL
# =========================
print("Loading checkpoint...")
model = TrainModule.load_from_checkpoint(ckpt_path, arch=arch,map_location=device)
model.eval()
model.to(device)

# =========================
# LOAD AUDIO
# =========================
wav, sr = sf.read(audio_path, always_2d=True)   # [T, C]
wav = wav.T.astype(np.float32)                  # [C, T]

x = torch.from_numpy(wav).unsqueeze(0).to(device)  # [1, C, T]

# =========================
# GET DOA / WIDTH
# =========================
# Default width for stage 1 is 30, matching the dataset loader logic. :contentReference[oaicite:0]{index=0}
filename = os.path.basename(audio_path)

doa_match = re.search(r"doa(\d+)", filename)
DOA_val = int(doa_match.group(1)) if doa_match else 0

width_match = re.search(r"width(\d+)", filename)
width_val = int(width_match.group(1)) if width_match else 30

DOA = torch.tensor([DOA_val], dtype=torch.long, device=device)
width = torch.tensor([width_val], dtype=torch.long, device=device)

print(f"Using DOA={DOA_val}, width={width_val}")

# =========================
# WARMUP
# =========================
print("Warmup...")
with torch.no_grad():
    for _ in range(5):
        _ = model.forward(x, DOA, width)

if device == "cuda":
    torch.cuda.synchronize()

# =========================
# REAL INFERENCE
# =========================
print("Running inference...")
with torch.no_grad():
    y_hat = model.forward(x, DOA, width)   # [1, 1, T]

if device == "cuda":
    torch.cuda.synchronize()

enhanced = y_hat.squeeze(0).squeeze(0).detach().cpu().numpy()

# Save result
save_path = "inference_output.wav"  # optional
sf.write(save_path, enhanced, sr)
print(f"Saved enhanced audio to: {save_path}")

# =========================
# TIMING
# =========================
print("Benchmarking...")
num_runs = 20

start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = model.forward(x, DOA, width)

if device == "cuda":
    torch.cuda.synchronize()
end = time.time()

avg_time = (end - start) / num_runs
audio_length = x.shape[-1] / sr
rtf = avg_time / audio_length

print("\n===== RESULT =====")
print(f"Average inference time: {avg_time:.4f} sec")
print(f"Audio length: {audio_length:.2f} sec")
print(f"RTF: {rtf:.3f}")
print("Faster than real-time" if rtf < 1 else "Slower than real-time")