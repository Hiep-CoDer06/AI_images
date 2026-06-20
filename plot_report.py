"""
Vẽ LINE PLOTS cho báo cáo - Tất cả SNR trên cùng 1 ảnh
Dữ liệu được lấy trực tiếp từ bảng kết quả
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================================================
# DỮ LIỆU TỪ BẢNG KẾT QUẢ (điền vào đây)
# =========================================================

snr_values = [1.0, 4.0, 7.0, 10.0, 13.0]

# Baseline
baseline = {
    'sigma_A': [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    'E_top20': [22.29, 22.10, 22.30, 22.10, 22.08],
    'H_rule': [3.1699, 3.1699, 3.1699, 3.1699, 3.1699],
}

# Linear
linear = {
    'sigma_A': [0.0615, 0.0549, 0.0483, 0.0418, 0.0353],
    'E_top20': [22.29, 22.10, 22.30, 22.10, 22.08],
    'H_rule': [3.1699, 3.1699, 3.1699, 3.1699, 3.1699],
}

# SNR-Only
snr_only = {
    'sigma_A': [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    'E_top20': [22.29, 22.10, 22.30, 22.10, 22.08],
    'H_rule': [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
}

# Importance-Only
importance_only = {
    'sigma_A': [0.2328, 0.2328, 0.2328, 0.2328, 0.2328],
    'E_top20': [53.02, 53.66, 53.00, 53.68, 53.75],
    'H_rule': [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
}

# Full-FIS (Ours)
full = {
    'sigma_A': [0.0784, 0.0662, 0.0551, 0.0425, 0.0318],
    'E_top20': [56.62, 56.06, 54.20, 53.68, 52.55],
    'H_rule': [2.8298, 2.7334, 2.6500, 2.5532, 2.4625],
}

# =========================================================
# CẤU HÌNH VẼ
# =========================================================

modes = {
    'Baseline': baseline,
    'Linear': linear,
    'SNR-Only': snr_only,
    'Importance-Only': importance_only,
    'Full-FIS (Ours)': full,
}

colors = {
    'Baseline': '#1f77b4',           # blue
    'Linear': '#ff7f0e',              # orange
    'SNR-Only': '#d62728',            # red
    'Importance-Only': '#2ca02c',     # green
    'Full-FIS (Ours)': '#9467bd',     # purple
}

markers = {
    'Baseline': 'o',
    'Linear': 's',
    'SNR-Only': '^',
    'Importance-Only': 'D',
    'Full-FIS (Ours)': '*',
}

# =========================================================
# VẼ ẢNH
# =========================================================

def plot_comparison():
    """Vẽ tất cả 3 metrics trên cùng 1 ảnh lớn"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Ablation Study: FIS Controller Performance vs SNR', fontsize=16, fontweight='bold', y=1.02)
    
    # ----- Plot 1: Dispersion (σA) -----
    for name, data in modes.items():
        axes[0].plot(snr_values, data['sigma_A'], 
                    marker=markers[name], linewidth=2.5, markersize=12,
                    color=colors[name], label=name)
    
    axes[0].set_xlabel('SNR (dB)', fontsize=14)
    axes[0].set_ylabel('Dispersion (σA)', fontsize=14)
    axes[0].set_title('Amplitude Dispersion', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10, loc='best')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].set_xticks(snr_values)
    
    # ----- Plot 2: Energy Concentration (E_top20) -----
    for name, data in modes.items():
        axes[1].plot(snr_values, data['E_top20'], 
                    marker=markers[name], linewidth=2.5, markersize=12,
                    color=colors[name], label=name)
    
    axes[1].set_xlabel('SNR (dB)', fontsize=14)
    axes[1].set_ylabel('Energy Concentration (%)', fontsize=14)
    axes[1].set_title('Energy Concentration (Top 20%)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10, loc='best')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_xticks(snr_values)
    
    # ----- Plot 3: Rule Entropy (H_rule) -----
    for name, data in modes.items():
        axes[2].plot(snr_values, data['H_rule'], 
                    marker=markers[name], linewidth=2.5, markersize=12,
                    color=colors[name], label=name)
    
    axes[2].set_xlabel('SNR (dB)', fontsize=14)
    axes[2].set_ylabel('Rule Entropy (H_rule)', fontsize=14)
    axes[2].set_title('Rule Activation Entropy', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10, loc='best')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].set_xticks(snr_values)
    
    plt.tight_layout()
    
    # Lưu ảnh
    save_dir = 'report_plots'
    os.makedirs(save_dir, exist_ok=True)
    
    output_path = os.path.join(save_dir, 'ablation_study_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu: {output_path}")
    
    # Ảnh riêng cho từng metric
    plot_sigma_A(save_dir)
    plot_E_top20(save_dir)
    plot_H_rule(save_dir)
    
    plt.show()
    plt.close()


def plot_sigma_A(save_dir):
    """Vẽ riêng σA"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, data in modes.items():
        ax.plot(snr_values, data['sigma_A'], 
               marker=markers[name], linewidth=2.5, markersize=12,
               color=colors[name], label=name)
    
    ax.set_xlabel('SNR (dB)', fontsize=14)
    ax.set_ylabel('Dispersion (σA)', fontsize=14)
    ax.set_title('Amplitude Dispersion (σA) vs SNR', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(snr_values)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'sigma_A_vs_SNR.png')
    plt.savefig(path, dpi=300)
    print(f"✅ Đã lưu: {path}")
    plt.close()


def plot_E_top20(save_dir):
    """Vẽ riêng E_top20"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, data in modes.items():
        ax.plot(snr_values, data['E_top20'], 
               marker=markers[name], linewidth=2.5, markersize=12,
               color=colors[name], label=name)
    
    ax.set_xlabel('SNR (dB)', fontsize=14)
    ax.set_ylabel('Energy Concentration (%)', fontsize=14)
    ax.set_title('Energy Concentration (Top 20%) vs SNR', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(snr_values)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'E_top20_vs_SNR.png')
    plt.savefig(path, dpi=300)
    print(f"✅ Đã lưu: {path}")
    plt.close()


def plot_H_rule(save_dir):
    """Vẽ riêng H_rule"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, data in modes.items():
        ax.plot(snr_values, data['H_rule'], 
               marker=markers[name], linewidth=2.5, markersize=12,
               color=colors[name], label=name)
    
    ax.set_xlabel('SNR (dB)', fontsize=14)
    ax.set_ylabel('Rule Entropy (H_rule)', fontsize=14)
    ax.set_title('Rule Entropy vs SNR', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(snr_values)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'H_rule_vs_SNR.png')
    plt.savefig(path, dpi=300)
    print(f"✅ Đã lưu: {path}")
    plt.close()


if __name__ == '__main__':
    plot_comparison()
