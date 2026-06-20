"""
Chạy diagnostic cho Rayleigh Equalized với nhiều SNR
Tạo dữ liệu cho báo cáo Ablation Study
"""
import os

# Các checkpoint Rayleigh Equalized round2
ckpt_full = "exp_ctx/ckpts_eq_full_Rayleigh_round2/fis_power_best.pth"
ckpt_linear = "exp_ctx/ckpts_eq_linear_Rayleigh_round2/fis_power_best.pth"
ckpt_importance = "exp_ctx/ckpts_eq_importance_only_Rayleigh_round2/fis_power_best.pth"
ckpt_snr = "exp_ctx/ckpts_eq_snr_only_Rayleigh_round2/fis_power_best.pth"

# SNR values cần chạy
snrs = [1.0, 4.0, 7.0, 10.0, 13.0]

print("=" * 60)
print("CHAY DIAGNOSTIC CHO KENH RAYLEIGH EQUALIZED")
print("=" * 60)

for snr in snrs:
    output_dir = f"diag_eq_snr{int(snr)}"
    
    # Tạo thư mục nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n>>> SNR = {snr} dB")
    print(f"    Output: {output_dir}/")
    
    # Chạy diagnostic với tất cả modes
    cmd = (
        f'python diagnose_controller.py '
        f'--dataset cifar10 '
        f'--image_size 32 '
        f'--channel Rayleigh '
        f'--linear_ckpt {ckpt_linear} '
        f'--importance_only_ckpt {ckpt_importance} '
        f'--snr_only_ckpt {ckpt_snr} '
        f'--full_ckpt {ckpt_full} '
        f'--snr_db {snr} '
        f'--ratio 0.1667 '
        f'--modes baseline,linear,importance_only,snr_only,full '
        f'--save_dir {output_dir}'
    )
    
    print(f"    Running: {cmd[:80]}...")
    os.system(cmd)

print("\n" + "=" * 60)
print("HOAN TAT DIAGNOSTIC!")
print("=" * 60)
print("\nSau khi chay xong, chay script nay de tao bao cao:")
print("  python generate_rayleigh_report.py")
