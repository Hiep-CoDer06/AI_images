import os
import json

# 1. THIẾT LẬP ĐƯỜNG DẪN KÊNH RAYLEIGH (NO-EQ)
base_ckpt = "exp_ctx/ckpts_noeq_baseline_Rayleigh/baseline_best.pth"
full_ckpt = "exp_ctx/ckpts_noeq_full_Rayleigh_round2/fis_power_best.pth"
snrs = [-10, -7, -4, -1]

print("████████████████████████████████████████████████████████")
print(" 1️⃣ ĐANG ĐO HỆ SỐ KÊNH (SIGMA_A) RAYLEIGH DẢI ÂM...")
print("████████████████████████████████████████████████████████")

for snr in snrs:
    print(f"\n>>> Đang quét SNR = {snr} dB...")
    cmd_diag = (
        f"python diagnose_controller.py --dataset cifar10 --image_size 32 "
        f"--channel Rayleigh "
        f"--linear_ckpt {full_ckpt} "
        f"--importance_only_ckpt {full_ckpt} "
        f"--snr_only_ckpt {full_ckpt} "
        f"--full_ckpt {full_ckpt} "
        f"--snr_db {snr} --ratio 0.1667 --modes full"
    )
    os.system(cmd_diag)

print("\n████████████████████████████████████████████████████████")
print(" 2️⃣ ĐANG CHẤM ĐIỂM PSNR: FULL VS BASELINE RAYLEIGH...")
print("████████████████████████████████████████████████████████")

map_file = ".tmp_fis_map_rayleigh_am.json"
with open(map_file, "w") as f:
    json.dump({"full": full_ckpt}, f)

cmd_sim = (
    f"python run_paper_sims.py "
    f"--baseline_ckpt {base_ckpt} "
    f"--fis_ckpt_map_json {map_file} "
    f"--channel Rayleigh "
    f"--ratio 0.1667 "
    f"--snrs -10 -7 -4 -1 "
    f"--modes baseline,full "
    f"--dataset cifar10 --image_size 32 --batch_size 128 "
    f"--save_dir paper_sims_rayleigh_am"
)
os.system(cmd_sim)

if os.path.exists(map_file): os.remove(map_file)
print("\n✅ ĐÃ HOÀN THÀNH TOÀN BỘ YÊU CẦU CỦA THẦY!")