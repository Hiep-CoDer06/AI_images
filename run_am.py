import os
ckpt_path = "exp_ctx/ckpts_full_AWGN_round2/fis_power_best.pth"
snrs = [-10, -7, -4, -1]
print("=== BẮT ĐẦU CHẠY KHẢO SÁT DẢI ÂM (-10 ĐẾN -1 DB) ===")
for snr in snrs:
    print(f"\n{'='*40}\n>>> Đang test hệ số kênh cho SNR = {snr} dB...\n{'='*40}")
    cmd = f"python diagnose_controller.py --dataset cifar10 --image_size 32 --channel AWGN --linear_ckpt {ckpt_path} --importance_only_ckpt {ckpt_path} --snr_only_ckpt {ckpt_path} --full_ckpt {ckpt_path} --snr_db {snr} --ratio 0.1667 --modes full"
    os.system(cmd)
print("\n✅ ĐÃ CHẠY XONG TOÀN BỘ DẢI ÂM!")