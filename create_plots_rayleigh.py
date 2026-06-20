"""
Create ablation comparison plots for Rayleigh Equalized channel
"""
import os
import numpy as np
import matplotlib.pyplot as plt

snr_values = [1.0, 4.0, 7.0, 10.0, 13.0]

# Data extracted from diagnostics.json
sigma_A = {
    'Linear': [0.0711, 0.0636, 0.0562, 0.0487, 0.0413],
    'SNR-Only': [0.0, 0.0, 0.0, 0.0, 0.0],
    'Importance-Only': [0.0764, 0.0764, 0.0764, 0.0764, 0.0764],
    'Full-FIS (Ours)': [0.1092, 0.0904, 0.0816, 0.0616, 0.0499],
}

E_top20 = {
    'Linear': [51.15, 48.97, 47.93, 46.92, 46.52],
    'SNR-Only': [40.36, 40.36, 40.36, 40.36, 40.36],
    'Importance-Only': [48.47, 48.47, 48.47, 48.47, 48.47],
    'Full-FIS (Ours)': [57.42, 55.21, 53.86, 51.15, 50.06],
}

H_rule = {
    'Linear': [2.043, 2.043, 2.043, 2.043, 2.043],
    'SNR-Only': [0.544, 0.544, 0.544, 0.544, 0.544],
    'Importance-Only': [3.169, 3.169, 3.169, 3.169, 3.169],
    'Full-FIS (Ours)': [3.168, 3.167, 3.167, 3.165, 3.165],
}

colors = {
    'Linear': '#ff7f0e',
    'SNR-Only': '#d62728',
    'Importance-Only': '#2ca02c',
    'Full-FIS (Ours)': '#9467bd'
}
markers = {
    'Linear': 's',
    'SNR-Only': '^',
    'Importance-Only': 'D',
    'Full-FIS (Ours)': '*'
}

os.makedirs('report_plots_rayleigh', exist_ok=True)

# Combined plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Ablation Study: Rayleigh Equalized Channel', fontsize=16, fontweight='bold', y=1.02)

for name, vals in sigma_A.items():
    axes[0].plot(snr_values, vals, marker=markers[name], linewidth=2.5, markersize=12, color=colors[name], label=name)
axes[0].set_xlabel('SNR (dB)', fontsize=14)
axes[0].set_ylabel('Dispersion (σA)', fontsize=14)
axes[0].set_title('Amplitude Dispersion', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].set_xticks(snr_values)

for name, vals in E_top20.items():
    axes[1].plot(snr_values, vals, marker=markers[name], linewidth=2.5, markersize=12, color=colors[name], label=name)
axes[1].set_xlabel('SNR (dB)', fontsize=14)
axes[1].set_ylabel('Energy Concentration (%)', fontsize=14)
axes[1].set_title('Energy Concentration (Top 20%)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].set_xticks(snr_values)

for name, vals in H_rule.items():
    axes[2].plot(snr_values, vals, marker=markers[name], linewidth=2.5, markersize=12, color=colors[name], label=name)
axes[2].set_xlabel('SNR (dB)', fontsize=14)
axes[2].set_ylabel('Rule Entropy (H_rule)', fontsize=14)
axes[2].set_title('Rule Activation Entropy', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, linestyle='--', alpha=0.5)
axes[2].set_xticks(snr_values)

plt.tight_layout()
plt.savefig('report_plots_rayleigh/ablation_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: report_plots_rayleigh/ablation_comparison.png')

# Individual plots
for metric, ylabel, data in [
    ('sigma_A', 'Dispersion (σA)', sigma_A),
    ('E_top20', 'Energy Concentration (%)', E_top20),
    ('H_rule', 'Rule Entropy', H_rule),
]:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, vals in data.items():
        ax.plot(snr_values, vals, marker=markers[name], linewidth=2.5, markersize=12, color=colors[name], label=name)
    ax.set_xlabel('SNR (dB)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f'{ylabel} vs SNR - Rayleigh Equalized', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(snr_values)
    plt.tight_layout()
    plt.savefig(f'report_plots_rayleigh/{metric}_vs_SNR.png', dpi=300)
    plt.close()
    print(f'Saved: report_plots_rayleigh/{metric}_vs_SNR.png')

print('Done!')
