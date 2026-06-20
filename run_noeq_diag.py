import os

# No-EQ Rayleigh checkpoints
linear_ckpt = "exp_ctx/ckpts_noeq_linear_Rayleigh_round2/fis_power_best.pth"
importance_only_ckpt = "exp_ctx/ckpts_noeq_importance_only_Rayleigh_round2/fis_power_best.pth"
snr_only_ckpt = "exp_ctx/ckpts_noeq_snr_only_Rayleigh_round2/fis_power_best.pth"
full_ckpt = "exp_ctx/ckpts_noeq_full_Rayleigh_round2/fis_power_best.pth"

snrs = [1.0, 4.0, 7.0, 10.0, 13.0]

print("=" * 60)
print("CHAY DIAGNOSTIC CHO KENH RAYLEIGH (KHONG EQUALIZE)")
print("=" * 60)

for snr in snrs:
    out_dir = f"diag_noeq_snr{int(snr)}"
    print(f"\n>>> SNR = {snr} dB")
    print(f"    Output: {out_dir}/")
    
    cmd = (
        f'python diagnose_controller.py --dataset cifar10 --image_size 32 '
        f'--channel Rayleigh '
        f'--linear_ckpt {linear_ckpt} '
        f'--importance_only_ckpt {importance_only_ckpt} '
        f'--snr_only_ckpt {snr_only_ckpt} '
        f'--full_ckpt {full_ckpt} '
        f'--snr_db {snr} --ratio 0.1667 --save_dir {out_dir} '
        f'--modes linear,importance_only,snr_only,full'
    )
    
    print(f"    Running: python diagnose_controller.py ...")
    os.system(cmd)
    print(f"    Done!")

print("\n" + "=" * 60)
print("HOAN TAT DIAGNOSTIC!")
print("=" * 60)
print("\nSau khi chay xong, chay script nay de tao bao cao:")
print("  python generate_noeq_report.py")
