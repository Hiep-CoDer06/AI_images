"""
Script tổng hợp kết quả cho TẤT CẢ SNR trên cùng 1 ảnh.
Chạy diagnose cho nhiều SNR và vẽ line plots so sánh.
"""
import os
import argparse
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt


# Các SNR cần test
DEFAULT_SNR_VALUES = [1.0, 4.0, 7.0, 10.0, 13.0]

# Đường dẫn checkpoint mặc định
DEFAULT_PATHS = {
    'baseline': 'exp_ctx/ckpts_baseline_Rayleigh/baseline_best.pth',
    'linear': 'exp_ctx/ckpts_linear_AWGN_round2/fis_power_best.pth',
    'importance_only': 'exp_ctx/ckpts_eq_importance_only_Rayleigh_round2/fis_power_best.pth',
    'snr_only': 'exp_ctx/ckpts_eq_snr_only_Rayleigh_round2/fis_power_best.pth',
    'full': 'exp_ctx/ckpts_eq_full_Rayleigh_round2/fis_power_best.pth',
}


def run_diagnose_for_snr(snr_db, modes, save_base_dir, checkpoints):
    """Chạy diagnose cho 1 giá trị SNR"""
    snr_dir = os.path.join(save_base_dir, f'snr_{snr_db}')
    os.makedirs(snr_dir, exist_ok=True)
    
    mode_args = ','.join(modes)
    checkpoint_args = []
    
    for mode in modes:
        if mode == 'baseline':
            checkpoint_args.extend(['--baseline_ckpt', checkpoints.get('baseline', '')])
        else:
            ckpt = checkpoints.get(mode, '')
            arg_name = f'--{mode}_ckpt'
            checkpoint_args.extend([arg_name, ckpt])
    
    cmd = [
        'python', 'diagnose_controller.py',
        '--save_dir', snr_dir,
        '--modes', mode_args,
        '--snr_db', str(snr_db),
        '--batch_index', '0',
        '--sample_index', '0',
    ] + checkpoint_args
    
    print(f"\n{'='*60}")
    print(f"Running SNR = {snr_db} dB...")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
    else:
        print(f"Completed SNR = {snr_db} dB")
    
    return snr_dir


def parse_metrics_from_log(log_path):
    """Đọc metrics từ log file"""
    metrics = {
        'sigma_A': None,
        'E_top20': None,
        'H_rule': None,
    }
    
    if not os.path.exists(log_path):
        return metrics
    
    with open(log_path, 'r') as f:
        content = f.read()
        
    for line in content.split('\n'):
        if 'sigma_A' in line.lower():
            try:
                val = float(line.split(':')[-1].strip())
                metrics['sigma_A'] = val
            except:
                pass
        elif 'E_top' in line:
            try:
                val = float(line.split(':')[-1].strip().replace('%', '')) / 100
                metrics['E_top20'] = val
            except:
                pass
        elif 'H_rule' in line:
            try:
                val = float(line.split(':')[-1].strip())
                metrics['H_rule'] = val
            except:
                pass
    
    return metrics


def plot_all_snr_comparison(save_base_dir, modes, snr_values):
    """Vẽ tất cả SNR trên cùng 1 ảnh - LINE PLOTS"""
    
    # Màu sắc cho các mode
    mode_colors = {
        'baseline': '#1f77b4',      # blue
        'linear': '#ff7f0e',         # orange
        'importance_only': '#2ca02c', # green
        'snr_only': '#d62728',       # red
        'full': '#9467bd',           # purple
    }
    
    mode_labels = {
        'baseline': 'Baseline',
        'linear': 'Linear',
        'importance_only': 'Importance-Only',
        'snr_only': 'SNR-Only',
        'full': 'Full-FIS (Ours)',
    }
    
    # Thu thập dữ liệu
    data = {mode: {'snr': [], 'sigma_A': [], 'E_top20': [], 'H_rule': []} for mode in modes}
    
    for snr in snr_values:
        snr_dir = os.path.join(save_base_dir, f'snr_{snr}')
        
        for mode in modes:
            if mode == 'baseline':
                log_path = os.path.join(snr_dir, f'baseline_metrics_log.txt')
            else:
                log_path = os.path.join(snr_dir, f'{mode}_metrics_log.txt')
            
            metrics = parse_metrics_from_log(log_path)
            
            if metrics['sigma_A'] is not None or metrics['E_top20'] is not None:
                data[mode]['snr'].append(snr)
                data[mode]['sigma_A'].append(metrics['sigma_A'])
                data[mode]['E_top20'].append(metrics['E_top20'])
                data[mode]['H_rule'].append(metrics['H_rule'])
    
    # =========================================================
    # VẼ 1 ẢNH TỔNG HỢP - 3 subplots
    # =========================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ----- Plot 1: Dispersion (σA) -----
    for mode in modes:
        if data[mode]['snr']:
            axes[0].plot(data[mode]['snr'], data[mode]['sigma_A'], 
                        marker='o', linewidth=2, markersize=8,
                        color=mode_colors.get(mode, 'gray'),
                        label=mode_labels.get(mode, mode))
    
    axes[0].set_xlabel('SNR (dB)', fontsize=12)
    axes[0].set_ylabel('Dispersion (σA)', fontsize=12)
    axes[0].set_title('Amplitude Dispersion vs SNR', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # ----- Plot 2: Energy Concentration (E_top20) -----
    for mode in modes:
        if data[mode]['snr']:
            axes[1].plot(data[mode]['snr'], data[mode]['E_top20'], 
                        marker='s', linewidth=2, markersize=8,
                        color=mode_colors.get(mode, 'gray'),
                        label=mode_labels.get(mode, mode))
    
    axes[1].set_xlabel('SNR (dB)', fontsize=12)
    axes[1].set_ylabel('Energy Concentration (%)', fontsize=12)
    axes[1].set_title('Energy Concentration vs SNR', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # ----- Plot 3: Rule Entropy (H_rule) -----
    for mode in modes:
        if data[mode]['snr']:
            axes[2].plot(data[mode]['snr'], data[mode]['H_rule'], 
                        marker='^', linewidth=2, markersize=8,
                        color=mode_colors.get(mode, 'gray'),
                        label=mode_labels.get(mode, mode))
    
    axes[2].set_xlabel('SNR (dB)', fontsize=12)
    axes[2].set_ylabel('Rule Entropy (H_rule)', fontsize=12)
    axes[2].set_title('Rule Entropy vs SNR', fontsize=14, fontweight='bold')
    axes[2].legend(loc='best')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Lưu ảnh
    output_path = os.path.join(save_base_dir, 'all_snr_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Đã vẽ tổng hợp! Ảnh: {output_path}")
    
    # =========================================================
    # VẼ ẢNH RIÊNG CHO TỪNG METRIC
    # =========================================================
    
    # Ảnh 1: σA comparison
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for mode in modes:
        if data[mode]['snr']:
            ax1.plot(data[mode]['snr'], data[mode]['sigma_A'], 
                    marker='o', linewidth=2.5, markersize=10,
                    color=mode_colors.get(mode, 'gray'),
                    label=mode_labels.get(mode, mode))
    
    ax1.set_xlabel('SNR (dB)', fontsize=14)
    ax1.set_ylabel('Dispersion (σA)', fontsize=14)
    ax1.set_title('Amplitude Dispersion Comparison', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_base_dir, 'sigma_A_comparison.png'), dpi=300)
    
    # Ảnh 2: E_top20 comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for mode in modes:
        if data[mode]['snr']:
            ax2.plot(data[mode]['snr'], data[mode]['E_top20'], 
                    marker='s', linewidth=2.5, markersize=10,
                    color=mode_colors.get(mode, 'gray'),
                    label=mode_labels.get(mode, mode))
    
    ax2.set_xlabel('SNR (dB)', fontsize=14)
    ax2.set_ylabel('Energy Concentration (%)', fontsize=14)
    ax2.set_title('Energy Concentration Comparison', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_base_dir, 'E_top20_comparison.png'), dpi=300)
    
    # Ảnh 3: H_rule comparison
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for mode in modes:
        if data[mode]['snr']:
            ax3.plot(data[mode]['snr'], data[mode]['H_rule'], 
                    marker='^', linewidth=2.5, markersize=10,
                    color=mode_colors.get(mode, 'gray'),
                    label=mode_labels.get(mode, mode))
    
    ax3.set_xlabel('SNR (dB)', fontsize=14)
    ax3.set_ylabel('Rule Entropy (H_rule)', fontsize=14)
    ax3.set_title('Rule Entropy Comparison', fontsize=16, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_base_dir, 'H_rule_comparison.png'), dpi=300)
    
    print(f"✅ Đã lưu 3 ảnh riêng: sigma_A_comparison.png, E_top20_comparison.png, H_rule_comparison.png")
    
    plt.close('all')
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Run diagnose for all SNR values and plot comparison')
    parser.add_argument('--snr_values', type=str, default='1.0,4.0,7.0,10.0,13.0',
                       help='Comma-separated SNR values')
    parser.add_argument('--modes', type=str, default='baseline,linear,importance_only,snr_only,full',
                       help='Comma-separated modes to test')
    parser.add_argument('--save_dir', type=str, default='diag_all_snr',
                       help='Base directory to save results')
    parser.add_argument('--run_diagnose', action='store_true', default=True,
                       help='Run diagnose_controller.py for each SNR')
    parser.add_argument('--plot_only', action='store_true',
                       help='Only plot, skip running diagnose')
    
    # Checkpoint paths
    for mode in ['baseline', 'linear', 'importance_only', 'snr_only', 'full']:
        parser.add_argument(f'--{mode}_ckpt', type=str, 
                           default=DEFAULT_PATHS.get(mode, ''),
                           help=f'Checkpoint path for {mode}')
    
    args = parser.parse_args()
    
    # Parse SNR values
    snr_values = [float(s.strip()) for s in args.snr_values.split(',')]
    
    # Parse modes
    modes = [m.strip() for m in args.modes.split(',')]
    
    # Build checkpoints dict
    checkpoints = {}
    for mode in modes:
        ckpt = getattr(args, f'{mode}_ckpt', None)
        if ckpt:
            checkpoints[mode] = ckpt
    
    print(f"SNR values: {snr_values}")
    print(f"Modes: {modes}")
    print(f"Checkpoints: {checkpoints}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if not args.plot_only:
        # Chạy diagnose cho từng SNR
        for snr in snr_values:
            run_diagnose_for_snr(snr, modes, args.save_dir, checkpoints)
    
    # Vẽ so sánh
    data = plot_all_snr_comparison(args.save_dir, modes, snr_values)
    
    # In bảng tổng hợp
    print("\n" + "="*80)
    print("BẢNG TỔNG HỢP KẾT QUẢ")
    print("="*80)
    print(f"{'Mode':<20} {'SNR':<8} {'σA':<12} {'E_top20':<12} {'H_rule':<12}")
    print("-"*80)
    
    for mode in modes:
        for i, snr in enumerate(data[mode]['snr']):
            sigma_a = data[mode]['sigma_A'][i] if i < len(data[mode]['sigma_A']) else 'N/A'
            e_top = data[mode]['E_top20'][i] if i < len(data[mode]['E_top20']) else 'N/A'
            h_rule = data[mode]['H_rule'][i] if i < len(data[mode]['H_rule']) else 'N/A'
            
            sigma_str = f"{sigma_a:.4f}" if sigma_a != 'N/A' else 'N/A'
            e_str = f"{e_top*100:.2f}%" if e_top != 'N/A' else 'N/A'
            h_str = f"{h_rule:.4f}" if h_rule != 'N/A' else 'N/A'
            
            print(f"{mode:<20} {snr:<8.1f} {sigma_str:<12} {e_str:<12} {h_str:<12}")
        print("-"*80)


if __name__ == '__main__':
    main()
