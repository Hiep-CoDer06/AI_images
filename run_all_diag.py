"""
Script chạy tất cả diagnostic cho Ablation Study
Chạy cho: AWGN, Rayleigh No-EQ, Rayleigh EQ với SNR 1,4,7,10,13
"""
import os
import subprocess
import time

# Checkpoint paths (round2)
EXP_CTX = "exp_ctx"

# AWGN Checkpoints
AWGN_CKPTS = {
    "baseline": f"{EXP_CTX}/ckpts_baseline_AWGN/baseline_best.pth",
    "linear": f"{EXP_CTX}/ckpts_linear_AWGN_round2/fis_power_best.pth",
    "importance_only": f"{EXP_CTX}/ckpts_importance_only_AWGN_round2/fis_power_best.pth",
    "snr_only": f"{EXP_CTX}/ckpts_snr_only_AWGN_round2/fis_power_best.pth",
    "full": f"{EXP_CTX}/ckpts_full_AWGN_round2/fis_power_best.pth",
}

# Rayleigh No-EQ Checkpoints
NOEQ_CKPTS = {
    "baseline": f"{EXP_CTX}/ckpts_noeq_baseline_Rayleigh/baseline_best.pth",
    "linear": f"{EXP_CTX}/ckpts_noeq_linear_Rayleigh_round2/fis_power_best.pth",
    "importance_only": f"{EXP_CTX}/ckpts_noeq_importance_only_Rayleigh_round2/fis_power_best.pth",
    "snr_only": f"{EXP_CTX}/ckpts_noeq_snr_only_Rayleigh_round2/fis_power_best.pth",
    "full": f"{EXP_CTX}/ckpts_noeq_full_Rayleigh_round2/fis_power_best.pth",
}

# Rayleigh EQ Checkpoints
EQ_CKPTS = {
    "baseline": f"{EXP_CTX}/ckpts_eq_baseline_Rayleigh/baseline_best.pth",
    "linear": f"{EXP_CTX}/ckpts_eq_linear_Rayleigh_round2/fis_power_best.pth",
    "importance_only": f"{EXP_CTX}/ckpts_eq_importance_only_Rayleigh_round2/fis_power_best.pth",
    "snr_only": f"{EXP_CTX}/ckpts_eq_snr_only_Rayleigh_round2/fis_power_best.pth",
    "full": f"{EXP_CTX}/ckpts_eq_full_Rayleigh_round2/fis_power_best.pth",
}

SNRS = [1.0, 4.0, 7.0, 10.0, 13.0]

def run_diagnostic(channel, snr, ckpts, save_dir, extra_args=""):
    """Run diagnose_controller.py for a specific configuration"""
    
    # Build command
    cmd = f'python diagnose_controller.py '
    cmd += f'--channel {channel} '
    cmd += f'--snr_db {snr} '
    cmd += f'--save_dir {save_dir} '
    cmd += f'--ratio 0.5 '  # Placeholder - will be inferred from checkpoint
    cmd += f'--modes baseline,linear,importance_only,snr_only,full '
    
    # Add baseline if exists
    if "baseline" in ckpts and os.path.exists(ckpts["baseline"]):
        cmd += f'--baseline_ckpt {ckpts["baseline"]} '
    
    # Add FIS checkpoints
    for mode in ["linear", "importance_only", "snr_only", "full"]:
        if mode in ckpts and os.path.exists(ckpts[mode]):
            cmd += f'--{mode}_ckpt {ckpts[mode]} '
    
    if extra_args:
        cmd += f'{extra_args} '
    
    print(f"\n{'='*60}")
    print(f"Running: {channel} SNR={snr}")
    print(f"Output: {save_dir}/")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Run command
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"[OK] SUCCESS ({elapsed:.1f}s)")
    else:
        print(f"[X] FAILED (exit code: {result.returncode})")
    
    return result.returncode == 0

def main():
    print("="*60)
    print("CHAY TAT CA DIAGNOSTIC CHO ABLATION STUDY")
    print("="*60)
    
    success_count = 0
    fail_count = 0
    
    # ============================================================
    # 1. AWGN Channel
    # ============================================================
    print("\n" + "="*60)
    print("PHAN 1: AWGN CHANNEL")
    print("="*60)
    
    for snr in SNRS:
        save_dir = f"diag_awgn_snr{int(snr)}"
        
        # ĐÃ THÁO BẪY SKIP: Cảnh báo ghi đè thay vì bỏ qua
        if os.path.exists(os.path.join(save_dir, "diagnostics.json")):
            print(f"\n[WARNING] {save_dir}/ Đã tồn tại -> Sẽ chạy đè để update số liệu mới!")
        
        success = run_diagnostic("AWGN", snr, AWGN_CKPTS, save_dir)
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # ============================================================
    # 2. Rayleigh No-EQ
    # ============================================================
    print("\n" + "="*60)
    print("PHAN 2: RAYLEIGH NO-EQUALIZER")
    print("="*60)
    
    for snr in SNRS:
        save_dir = f"diag_noeq_snr{int(snr)}"
        
        if os.path.exists(os.path.join(save_dir, "diagnostics.json")):
            print(f"\n[WARNING] {save_dir}/ Đã tồn tại -> Sẽ chạy đè để update số liệu mới!")
        
        success = run_diagnostic("Rayleigh", snr, NOEQ_CKPTS, save_dir)
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # ============================================================
    # 3. Rayleigh EQ
    # ============================================================
    print("\n" + "="*60)
    print("PHAN 3: RAYLEIGH EQUALIZER")
    print("="*60)
    
    for snr in SNRS:
        save_dir = f"diag_eq_snr{int(snr)}"
        
        if os.path.exists(os.path.join(save_dir, "diagnostics.json")):
            print(f"\n[WARNING] {save_dir}/ Đã tồn tại -> Sẽ chạy đè để update số liệu mới!")
        
        success = run_diagnostic("Rayleigh", snr, EQ_CKPTS, save_dir, extra_args="--rayleigh_equalize")
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("KET QUA CHAY TỔNG HỢP")
    print("="*60)
    print(f"Thành công: {success_count}")
    print(f"Thất bại: {fail_count}")
    print("\nSau khi hệ thống xả xong data, bạn hãy gõ tiếp lệnh sau để xuất báo cáo nộp thầy Nam:")
    print(">>> python generate_report.py")

if __name__ == "__main__":
    main()