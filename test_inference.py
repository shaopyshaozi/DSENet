import time
import torch
import soundfile as sf
import numpy as np

from DOATrainer import TrainModule
from models.arch.DSENet import DSENet

# =========================
# CONFIG
# =========================
audio_path = "mic_fileid_0_doa48_6spk.wav"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

# =========================
# BUILD MODEL EXPLICITLY
# =========================
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

model = TrainModule(arch=arch)
model.eval()
model.to(device)

# =========================
# LOAD AUDIO
# =========================
wav, sr = sf.read(audio_path, always_2d=True)   # [T, C]
wav = wav.T.astype(np.float32)                  # [C, T]

x = torch.from_numpy(wav).unsqueeze(0).to(device)  # [1, C, T]

# Use the DOA encoded in the filename if you want; here 48 and default width 30
DOA = torch.tensor([48], dtype=torch.long, device=device)
width = torch.tensor([30], dtype=torch.long, device=device)

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