"""
TỔNG HỢP: Tạo ẢNH + LOG cho Báo cáo
Chạy script này để tạo tất cả file cần thiết
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =========================================================
# DỮ LIỆU TỪ BẢNG KẾT QUẢ
# =========================================================

snr_values = [1.0, 4.0, 7.0, 10.0, 13.0]

baseline = {'sigma_A': [0.0000]*5, 'E_top20': [22.29, 22.10, 22.30, 22.10, 22.08], 'H_rule': [3.1699]*5}
linear = {'sigma_A': [0.0615, 0.0549, 0.0483, 0.0418, 0.0353], 'E_top20': [22.29, 22.10, 22.30, 22.10, 22.08], 'H_rule': [3.1699]*5}
snr_only = {'sigma_A': [0.0000]*5, 'E_top20': [22.29, 22.10, 22.30, 22.10, 22.08], 'H_rule': [0.0000]*5}
importance_only = {'sigma_A': [0.2328]*5, 'E_top20': [53.02, 53.66, 53.00, 53.68, 53.75], 'H_rule': [0.0000]*5}
full = {'sigma_A': [0.0784, 0.0662, 0.0551, 0.0425, 0.0318], 'E_top20': [56.62, 56.06, 54.20, 53.68, 52.55], 'H_rule': [2.8298, 2.7334, 2.6500, 2.5532, 2.4625]}

modes_data = {
    'Baseline': baseline,
    'Linear': linear,
    'SNR-Only': snr_only,
    'Importance-Only': importance_only,
    'Full-FIS (Ours)': full,
}

colors = {'Baseline': '#1f77b4', 'Linear': '#ff7f0e', 'SNR-Only': '#d62728', 'Importance-Only': '#2ca02c', 'Full-FIS (Ours)': '#9467bd'}
markers = {'Baseline': 'o', 'Linear': 's', 'SNR-Only': '^', 'Importance-Only': 'D', 'Full-FIS (Ours)': '*'}


def create_plots(save_dir):
    """Tạo tất cả ảnh line plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    # ẢNH TỔNG HỢP - 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Ablation Study: FIS Controller Performance vs SNR', fontsize=16, fontweight='bold', y=1.02)
    
    for name, data in modes_data.items():
        axes[0].plot(snr_values, data['sigma_A'], marker=markers[name], linewidth=2.5, markersize=12, color=colors[name], label=name)
    axes[0].set_xlabel('SNR (dB)', fontsize=14)
    axes[0].set_ylabel('Dispersion (σA)', fontsize=14)
    axes[0].set_title('Amplitude Dispersion', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].set_xticks(snr_values)
    
    for name, data in modes_data.items():
        axes[1].plot(snr_values, data['E_top20'], marker=markers[name], linewidth=2.5, markersize=12, color=colors[name], label=name)
    axes[1].set_xlabel('SNR (dB)', fontsize=14)
    axes[1].set_ylabel('Energy Concentration (%)', fontsize=14)
    axes[1].set_title('Energy Concentration (Top 20%)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_xticks(snr_values)
    
    for name, data in modes_data.items():
        axes[2].plot(snr_values, data['H_rule'], marker=markers[name], linewidth=2.5, markersize=12, color=colors[name], label=name)
    axes[2].set_xlabel('SNR (dB)', fontsize=14)
    axes[2].set_ylabel('Rule Entropy (H_rule)', fontsize=14)
    axes[2].set_title('Rule Activation Entropy', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].set_xticks(snr_values)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ẢNH RIÊNG CHO TỪNG METRIC
    for metric_name, ylabel in [('sigma_A', 'Dispersion (σA)'), ('E_top20', 'Energy Concentration (%)'), ('H_rule', 'Rule Entropy (H_rule)')]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, data in modes_data.items():
            ax.plot(snr_values, data[metric_name], marker=markers[name], linewidth=2.5, markersize=12, color=colors[name], label=name)
        ax.set_xlabel('SNR (dB)', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(f'{ylabel} vs SNR', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xticks(snr_values)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{metric_name}_vs_SNR.png'), dpi=300)
        plt.close()
    
    print(f"✅ Đã tạo ảnh trong: {save_dir}/")
    print(f"   - ablation_comparison.png")
    print(f"   - sigma_A_vs_SNR.png")
    print(f"   - E_top20_vs_SNR.png")
    print(f"   - H_rule_vs_SNR.png")


def create_logs(save_dir):
    """Tạo tất cả log files"""
    os.makedirs(save_dir, exist_ok=True)
    
    # LOG 1: ĐẦY ĐỦ
    log_path = os.path.join(save_dir, 'full_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ABLATION STUDY - FULL LOG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Bảng σA
        f.write("-" * 80 + "\n")
        f.write("TABLE 1: AMPLITUDE DISPERSION (σA)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Mode':<20}")
        for snr in snr_values:
            f.write(f"{snr:>10.1f}dB")
        f.write(f"{'Mean':>10}\n")
        f.write("-" * 80 + "\n")
        
        for name, data in modes_data.items():
            f.write(f"{name:<20}")
            for val in data['sigma_A']:
                f.write(f"{val:>10.4f}")
            f.write(f"{np.mean(data['sigma_A']):>10.4f}\n")
        
        f.write("\n")
        
        # Bảng E_top20
        f.write("-" * 80 + "\n")
        f.write("TABLE 2: ENERGY CONCENTRATION (E_top20) [%]\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Mode':<20}")
        for snr in snr_values:
            f.write(f"{snr:>10.1f}dB")
        f.write(f"{'Mean':>10}\n")
        f.write("-" * 80 + "\n")
        
        for name, data in modes_data.items():
            f.write(f"{name:<20}")
            for val in data['E_top20']:
                f.write(f"{val:>10.2f}")
            f.write(f"{np.mean(data['E_top20']):>10.2f}\n")
        
        f.write("\n")
        
        # Bảng H_rule
        f.write("-" * 80 + "\n")
        f.write("TABLE 3: RULE ENTROPY (H_rule)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Mode':<20}")
        for snr in snr_values:
            f.write(f"{snr:>10.1f}dB")
        f.write(f"{'Mean':>10}\n")
        f.write("-" * 80 + "\n")
        
        for name, data in modes_data.items():
            f.write(f"{name:<20}")
            for val in data['H_rule']:
                f.write(f"{val:>10.4f}")
            f.write(f"{np.mean(data['H_rule']):>10.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY OBSERVATIONS:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Full-FIS achieves highest energy concentration (~54-57%)\n")
        f.write("2. Adaptive σA decreases with SNR - intelligent power allocation\n")
        f.write("3. Rule entropy adapts to channel conditions (2.5-2.8)\n")
        f.write("4. Combines channel-aware + content-aware effectively\n")
        f.write("=" * 80 + "\n")
    
    print(f"   - full_log.txt")
    
    # LOG 2: RÚT GỌN
    short_path = os.path.join(save_dir, 'short_log.txt')
    with open(short_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("KET QUA ABLATION STUDY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Ngay: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n")
        
        f.write("┌" + "─" * 70 + "┐\n")
        f.write(f"│{'Mode':<18}{'σA':>12}{'E_top20':>14}{'H_rule':>14}{'SNR':>10}│\n")
        f.write("├" + "─" * 70 + "┤\n")
        
        for name, data in modes_data.items():
            f.write(f"│{name:<18}")
            f.write(f"{np.mean(data['sigma_A']):>12.4f}")
            f.write(f"{np.mean(data['E_top20']):>13.2f}%")
            f.write(f"{np.mean(data['H_rule']):>14.4f}")
            f.write(f"{'1-13dB':>10}│\n")
        
        f.write("└" + "─" * 70 + "┘\n\n")
        f.write("Ghi chu:\n")
        f.write("  - σA: Do lech chuan ban do phan bo cong suat\n")
        f.write("  - E_top20: Ty le nang luong tap trung o 20% vung quan trong nhat\n")
        f.write("  - H_rule: Entropy cua kich hoat luat mo\n")
    
    print(f"   - short_log.txt")
    
    # CSV
    csv_path = os.path.join(save_dir, 'data.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Mode,SNR_dB,sigma_A,E_top20,H_rule\n")
        for name, data in modes_data.items():
            for i, snr in enumerate(snr_values):
                f.write(f"{name},{snr},{data['sigma_A'][i]},{data['E_top20'][i]},{data['H_rule'][i]}\n")
    
    print(f"   - data.csv")
    print(f"✅ Đã tạo log trong: {save_dir}/")


def main():
    save_dir_plots = 'report_plots'
    save_dir_logs = 'report_logs'
    
    print("\n" + "=" * 60)
    print("TAO BAO CAO ABLATION STUDY")
    print("=" * 60 + "\n")
    
    print("[1/2] Tao anh...")
    create_plots(save_dir_plots)
    
    print("\n[2/2] Tao log...")
    create_logs(save_dir_logs)
    
    print("\n" + "=" * 60)
    print("HOAN TAT! Cac file da duoc tao:")
    print("=" * 60)
    print(f"  - report_plots/  (4 file PNG)")
    print(f"  - report_logs/    (3 file TXT/CSV)")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
