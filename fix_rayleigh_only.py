import os
import json
import matplotlib.pyplot as plt

# ==============================================================================
# 🌪️ CHIẾN DỊCH: THUẦN HÓA KÊNH RAYLEIGH
# ==============================================================================
SNR_MIN = -10
SNR_MAX = 13
TRAIN_SNR_LIST = "-10 -7 -4 -1 1 4 7 10 13" 
EVAL_SNR_LIST = "-10 -7 -4 -1 1 4 7 10 13"

# --- ĐƠN THUỐC SIÊU LIỀU ---
WARMSTART_EPOCHS = 40  # Tăng gấp đôi thời gian khởi động (20 -> 40)
FINETUNE_LR = 0.0002   # Tăng lực học để phá vỡ sức ỳ (0.0001 -> 0.0002)
# ---------------------------

RATIO = 0.1667
DATASET = "cifar10"
IMAGE_SIZE = 32
BUDGET = 1.0

CHANNEL = "Rayleigh"
MODES = ["linear", "importance_only", "snr_only", "full"]

print("\n" + "="*80)
print(f" 🚀 ĐANG XỬ LÝ KÊNH KHẮC NGHIỆT: {CHANNEL}")
print(f"   - Warmstart: {WARMSTART_EPOCHS} epochs | LR: {FINETUNE_LR}")
print("="*80 + "\n")

# Lấy nền móng baseline đã train lúc nãy
base_ckpt = f"exp_ctx/ckpts_baseline_{CHANNEL}/baseline_best.pth"
eq_flag = "--rayleigh_equalize"

DONE = 0
TOTAL_JOBS = 4

# 1. Train lại 4 mô hình FIS với chiến thuật mới
for mode in MODES:
    fis_dir = f"exp_ctx/ckpts_eq_{mode}_{CHANNEL}_round2"
    fis_ckpt = f"{fis_dir}/fis_power_best.pth"

    # XÓA TẠ CŨ ĐỂ ÉP TRAIN LẠI BẢN MỚI
    if os.path.exists(fis_ckpt):
        print(f"🗑️ Đang xóa tạ cũ bị dính chùm của {mode.upper()}...")
        os.remove(fis_ckpt)

    DONE += 1
    print(f"\n🔧 [{DONE}/{TOTAL_JOBS}] Đang train lại {mode.upper()} (Warm-start {WARMSTART_EPOCHS})...")

    cmd_fis = (
        f"python train_fis_power.py \
        --dataset {DATASET} --image_size {IMAGE_SIZE} \
        --channel {CHANNEL} --mode {mode} --budget {BUDGET} \
        --snr_min {SNR_MIN} --snr_max {SNR_MAX} \
        --train_snr_list {TRAIN_SNR_LIST} --eval_snr_list {EVAL_SNR_LIST} \
        --save_dir {fis_dir} \
        --baseline_ckpt {base_ckpt} \
        --warmstart_controller_only_epochs {WARMSTART_EPOCHS} \
        --finetune_lr {FINETUNE_LR} \
        {eq_flag}"
    )
    os.system(cmd_fis)

print("\n📊 Đang chấm điểm và vẽ lại biểu đồ...")

# 2. Vẽ lại biểu đồ
map_file = f".tmp_fis_map_{CHANNEL}.json"
fis_map = {
    "linear": f"exp_ctx/ckpts_eq_linear_{CHANNEL}_round2/fis_power_best.pth",
    "importance_only": f"exp_ctx/ckpts_eq_importance_only_{CHANNEL}_round2/fis_power_best.pth",
    "snr_only": f"exp_ctx/ckpts_eq_snr_only_{CHANNEL}_round2/fis_power_best.pth",
    "full": f"exp_ctx/ckpts_eq_full_{CHANNEL}_round2/fis_power_best.pth"
}
with open(map_file, "w") as f:
    json.dump(fis_map, f)

save_dir = f"paper_sims_FIXED_{CHANNEL}"
cmd_sim = (
    f"python run_paper_sims.py \
    --baseline_ckpt {base_ckpt} \
    --fis_ckpt_map_json {map_file} \
    --channel {CHANNEL} {eq_flag} \
    --ratio {RATIO} \
    --snrs {TRAIN_SNR_LIST} \
    --modes baseline,linear,importance_only,snr_only,full \
    --dataset {DATASET} --image_size {IMAGE_SIZE} \
    --batch_size 128 \
    --save_dir {save_dir}"
)
os.system(cmd_sim)
if os.path.exists(map_file): os.remove(map_file)

# Đọc và vẽ Plot
plot_methods = ["baseline", "linear", "importance_only", "snr_only", "full"]
labels = ["Baseline", "Linear", "Importance Only", "SNR Only", "Full FIS (Channel-Aware)"]
markers = ['o', 's', '^', 'x', '*']

try:
    with open(f"{save_dir}/paper_sims_results.json", 'r') as f:
        data = json.load(f)
        
    ratio_key = list(data['results'].keys())[0]
    results = data['results'][ratio_key]

    plt.figure(figsize=(10, 6))
    snrs_list = sorted([float(s) for s in list(results.keys())])

    for idx, m in enumerate(plot_methods):
        y_psnr = [results[f"{snr}"][m]["psnr"] for snr in snrs_list if f"{snr}" in results and m in results[f"{snr}"]]
        x_snrs = [snr for snr in snrs_list if f"{snr}" in results and m in results[f"{snr}"]]
        plt.plot(x_snrs, y_psnr, marker=markers[idx], linestyle='-', linewidth=2, markersize=7, label=labels[idx])

    plt.title(f"PSNR vs SNR under {CHANNEL} Channel (Deep Fading Fixed)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("SNR (dB)", fontsize=12, fontweight='bold')
    plt.ylabel("PSNR (dB)", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(snrs_list)
    plt.legend(fontsize=10)

    img_name = f"Chart_5_Methods_FIXED_{CHANNEL}_V2.png"
    plt.savefig(img_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎉 Hoàn thành! Mở file ảnh mới: {img_name} để xem đường Tím bứt tốc nhé!")
except Exception as e:
    print(f"Lỗi: {e}")