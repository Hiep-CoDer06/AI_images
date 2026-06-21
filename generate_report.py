"""
Generate reports from diagnostics.json files and metrics_log.txt
Loads data automatically from diag_* folders
"""
import os
import json
import re
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# SNR values to look for
SNR_VALUES = [1.0, 4.0, 7.0, 10.0, 13.0]

# Mode names for display
MODE_NAMES = {
    'baseline': 'Baseline',
    'linear': 'Linear',
    'snr_only': 'SNR-Only',
    'importance_only': 'Importance-Only',
    'full': 'Full-FIS (Ours)'
}

# Display order
MODES = ['baseline', 'linear', 'snr_only', 'importance_only', 'full']

# Colors for plots
COLORS = {
    'baseline': '#1f77b4',
    'linear': '#ff7f0e',
    'snr_only': '#d62728',
    'importance_only': '#2ca02c',
    'full': '#9467bd'
}


def load_all_diagnostics(base_path):
    """
    Load all diagnostics.json files from diag_* folders
    Returns: dict[folder_name] -> {snr, channel_type, data}
    """
    data = {}
    base = Path(base_path)
    diag_folders = sorted(base.glob('diag_*'))

    for folder in diag_folders:
        folder_name = folder.name
        diag_file = folder / 'diagnostics.json'

        if not diag_file.exists():
            continue

        with open(diag_file, 'r') as f:
            diag = json.load(f)

        # Parse folder name: diag_{channel}_snr{SNR}
        parts = folder_name.split('_')
        channel_type = None
        snr = None

        for i, part in enumerate(parts):
            if part == 'awgn':
                channel_type = 'awgn'
            elif part == 'noeq':
                channel_type = 'noeq'
            elif part == 'eq':
                channel_type = 'eq'
            elif part.startswith('snr'):
                try:
                    snr = float(part.replace('snr', ''))
                except:
                    pass

        if channel_type and snr:
            data[folder_name] = {
                'snr': snr,
                'channel': channel_type,
                'diagnostics': diag,
                'folder': str(folder)
            }

    return data


def load_metrics_from_log(folder, mode):
    """
    Load E_top20, sigma_A, H_rule from *metrics_log.txt
    """
    mode_file_map = {
        'baseline': 'baseline_metrics_log.txt',
        'linear': 'linear_metrics_log.txt',
        'snr_only': 'snr_only_metrics_log.txt',
        'importance_only': 'importance_only_metrics_log.txt',
        'full': 'full_metrics_log.txt'
    }

    filename = mode_file_map.get(mode)
    if not filename:
        return None

    filepath = Path(folder) / filename
    if not filepath.exists():
        return None

    with open(filepath, 'r') as f:
        content = f.read()

    result = {}

    # Parse sigma_A
    match = re.search(r'Dispersion \(sigma_A\)\s*:\s*([\d.]+)', content)
    if match:
        result['sigma_A'] = float(match.group(1))

    # [FIX BUG] Hỗ trợ cả "Energy" và "Energy Concentration"
    match = re.search(r'Energy(?: Concentration)? \(E_top20\)\s*:\s*([\d.]+)%', content)
    if match:
        result['E_top20'] = float(match.group(1))

    # Parse H_rule
    match = re.search(r'Rule Entropy \(H_rule\)\s*:\s*([\d.]+)', content)
    if match:
        result['H_rule'] = float(match.group(1))

    return result if result else None


def extract_mode_data(diag, folder):
    """
    Extract metrics from diagnostics.json and metrics_log.txt
    Returns: dict[mode] -> {sigma_A, E_top20, H_rule, corr_A_gamma_eff, corr_A_Importance}
    """
    result = {}
    modes_data = diag.get('modes', {})

    for mode in MODES:
        if mode not in modes_data:
            continue

        mode_data = modes_data[mode]
        corr_A_gamma_eff = mode_data.get('corr_A_gamma_eff', 0)
        corr_A_Importance = mode_data.get('corr_A_I', 0)

        metrics = load_metrics_from_log(folder, mode)

        if metrics:
            result[mode] = {
                'sigma_A': metrics.get('sigma_A', 0),
                'E_top20': metrics.get('E_top20', 20.0),  # [FIX BUG] 20% là vật lý tối thiểu
                'H_rule': metrics.get('H_rule', 0),
                'corr_A_gamma_eff': corr_A_gamma_eff if corr_A_gamma_eff is not None else 0,
                'corr_A_Importance': corr_A_Importance if corr_A_Importance is not None else 0
            }
        else:
            A_stats = mode_data.get('A_stats', {})
            sigma_A = A_stats.get('std', None)

            if sigma_A is not None:
                result[mode] = {
                    'sigma_A': sigma_A,
                    'E_top20': 20.0,  # [FIX BUG] Fallback chuẩn khoa học
                    'H_rule': 0,
                    'corr_A_gamma_eff': corr_A_gamma_eff if corr_A_gamma_eff is not None else 0,
                    'corr_A_Importance': corr_A_Importance if corr_A_Importance is not None else 0
                }

    return result


def filter_by_channel(all_data, channel_type):
    """Filter data by channel type"""
    filtered = {}
    for name, info in all_data.items():
        if info['channel'] == channel_type:
            filtered[info['snr']] = info
    return filtered


def create_plots(data_by_snr, save_dir, channel_name):
    """Create line plots for metrics vs SNR"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    metrics = ['sigma_A', 'E_top20', 'H_rule', 'corr_A_gamma_eff', 'corr_A_Importance']
    metric_labels = {
        'sigma_A': r'$\sigma_A$ (Amplitude Dispersion)',
        'E_top20': 'Energy Concentration E_top20 (%)',
        'H_rule': 'Rule Entropy H_rule',
        'corr_A_gamma_eff': r'Corr($A$, $\gamma_{eff}$)',
        'corr_A_Importance': r'Corr($A$, Importance)'
    }

    snrs = sorted(data_by_snr.keys())

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        for mode in MODES:
            values = []
            for snr in snrs:
                if mode in data_by_snr[snr]['diagnostics']:
                    values.append(data_by_snr[snr]['diagnostics'][mode].get(metric, 0))
                else:
                    values.append(None)

            valid_values = [(s, v) for s, v in zip(snrs, values) if v is not None]
            if valid_values:
                xs, ys = zip(*valid_values)
                ax.plot(xs, ys, 'o-', color=COLORS[mode], label=MODE_NAMES[mode],
                       linewidth=2, markersize=8)

        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
        ax.set_title(f'{metric_labels.get(metric, metric)} - {channel_name}', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xticks(snrs)

        plt.tight_layout()
        plt.savefig(str(Path(save_dir) / f'{metric}_vs_SNR.png'), dpi=300)
        plt.close()

    # Combined ablation comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        for mode in MODES:
            values = []
            for snr in snrs:
                if mode in data_by_snr[snr]['diagnostics']:
                    values.append(data_by_snr[snr]['diagnostics'][mode].get(metric, 0))
                else:
                    values.append(None)

            valid_values = [(s, v) for s, v in zip(snrs, values) if v is not None]
            if valid_values:
                xs, ys = zip(*valid_values)
                ax.plot(xs, ys, 'o-', color=COLORS[mode], label=MODE_NAMES[mode],
                       linewidth=2, markersize=6)

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_title(metric_labels.get(metric, metric))
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xticks(snrs)

    axes[5].axis('off')
    plt.tight_layout()
    plt.savefig(str(Path(save_dir) / 'ablation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Plots saved to: {save_dir}/")


def create_logs(data_by_snr, save_dir, channel_name):
    """Create log files"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    snrs = sorted(data_by_snr.keys())

    log_path = str(Path(save_dir) / 'full_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 85 + "\n")
        f.write(f"ABLATION STUDY - {channel_name.upper()}\n")
        f.write("=" * 85 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Available SNRs: {snrs}\n\n")

        modes_data = {mode: {m: [] for m in ['sigma_A', 'E_top20', 'H_rule', 'corr_A_gamma_eff', 'corr_A_Importance']} for mode in MODES}
        
        for mode in MODES:
            for snr in snrs:
                if mode in data_by_snr[snr]['diagnostics']:
                    d = data_by_snr[snr]['diagnostics'][mode]
                    for m in modes_data[mode]:
                        modes_data[mode][m].append(d.get(m, None))

        tables = [
            ('1', 'AMPLITUDE DISPERSION (σA)', 'sigma_A', '{:.4f}'),
            ('2', 'ENERGY CONCENTRATION (E_top20) [%]', 'E_top20', '{:.2f}'),
            ('3', 'RULE ENTROPY (H_rule)', 'H_rule', '{:.4f}'),
            ('4', 'CORRELATION A vs GAMMA_EFF (corr_A_gamma_eff)', 'corr_A_gamma_eff', '{:.4f}'),
            ('5', 'CORRELATION A vs IMPORTANCE (corr_A_Importance)', 'corr_A_Importance', '{:.4f}')
        ]

        for table_num, table_name, metric, fmt in tables:
            f.write("-" * 85 + "\n")
            f.write(f"TABLE {table_num}: {table_name}\n")
            f.write("-" * 85 + "\n")

            # Header
            f.write(f"{'Mode':<22}")
            for snr in snrs:
                f.write(f"{snr:>10.1f}dB")
            f.write(f"{'Mean':>12}\n")
            f.write("-" * 85 + "\n")

            # Data rows
            for mode in MODES:
                f.write(f"{MODE_NAMES[mode]:<22}")
                vals = modes_data[mode][metric]
                valid = [v for v in vals if v is not None]

                for v in vals:
                    if v is not None:
                        # [FIX BUG] Dùng đúng format truyền vào thay vì hardcode cắt chuỗi
                        formatted_val = fmt.format(float(v))
                        f.write(f"{formatted_val:>12}")
                    else:
                        f.write(f"{'N/A':>12}")

                if valid:
                    formatted_mean = fmt.format(np.mean(valid))
                    f.write(f"{formatted_mean:>12}\n")
                else:
                    f.write(f"{'N/A':>12}\n")
            f.write("\n")
        f.write("=" * 85 + "\n")

    print(f"  [OK] Full log: {log_path}")

    # CSV
    csv_path = str(Path(save_dir) / 'data.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Mode,SNR_dB,sigma_A,E_top20,H_rule,corr_A_gamma_eff,corr_A_Importance\n")
        for mode in MODES:
            for snr in snrs:
                if mode in data_by_snr[snr]['diagnostics']:
                    d = data_by_snr[snr]['diagnostics'][mode]
                    sa = f"{d.get('sigma_A', ''):.4f}" if d.get('sigma_A') is not None else ""
                    e = f"{d.get('E_top20', ''):.2f}" if d.get('E_top20') is not None else ""
                    h = f"{d.get('H_rule', ''):.4f}" if d.get('H_rule') is not None else ""
                    cg = f"{d.get('corr_A_gamma_eff', ''):.4f}" if d.get('corr_A_gamma_eff') is not None else ""
                    ci = f"{d.get('corr_A_Importance', ''):.4f}" if d.get('corr_A_Importance') is not None else ""
                    f.write(f"{MODE_NAMES[mode]},{snr},{sa},{e},{h},{cg},{ci}\n")

    print(f"  [OK] CSV: {csv_path}")


def main():
    print("=" * 60)
    print("GENERATING REPORTS FROM DIAGNOSTICS")
    print("=" * 60)

    base_path = os.path.dirname(os.path.abspath(__file__))

    print("\n[1/4] Loading diagnostics...")
    all_data = load_all_diagnostics(base_path)
    print(f"  Found {len(all_data)} diagnostic folders")

    channels = {
        'awgn': {'name': 'AWGN Channel', 'folders': []},
        'noeq': {'name': 'Rayleigh No-Equalizer', 'folders': []},
        'eq': {'name': 'Rayleigh Equalizer', 'folders': []}
    }

    for name, info in all_data.items():
        ch = info['channel']
        if ch in channels:
            channels[ch]['folders'].append(name)

    for ch_key, ch_info in channels.items():
        if not ch_info['folders']:
            continue

        print(f"\n{'='*60}")
        print(f"CHANNEL: {ch_info['name']}")
        print(f"{'='*60}")

        data_by_snr = filter_by_channel(all_data, ch_key)
        for snr, info in data_by_snr.items():
            info['diagnostics'] = extract_mode_data(info['diagnostics'], info['folder'])

        print(f"[2/4] Extracted data for {len(data_by_snr)} SNR levels")

        print("[3/4] Creating plots...")
        plot_dir = f'report_plots_{ch_key}' if ch_key != 'awgn' else 'report_plots'
        create_plots(data_by_snr, plot_dir, ch_info['name'])

        print("[4/4] Creating logs...")
        log_dir = f'report_logs_{ch_key}' if ch_key != 'awgn' else 'report_logs'
        create_logs(data_by_snr, log_dir, ch_info['name'])

    print("\n" + "=" * 60)
    print("ALL REPORTS GENERATED!")
    print("=" * 60)

if __name__ == '__main__':
    main()