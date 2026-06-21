"""
Ve bieu do tu paper_sims_results.json
Su dung matplotlib tao figure chat luong cao
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Cau hinh style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
})

# Duong dan den ket qua
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS = {
    "AWGN": "paper_sims_awgn/paper_sims_results.json",
    "Rayleigh (EQ)": "paper_sims_eq/paper_sims_results.json",
    "Rayleigh (No-EQ)": "paper_sims_noeq/paper_sims_results.json",
}

# Phuong phap va mau sac
METHODS = ["baseline", "linear", "importance_only", "full"]
METHODS_WITH_SNR_ONLY = ["baseline", "linear", "importance_only", "snr_only", "full"]
METHOD_LABELS = {
    "baseline": "Baseline",
    "linear": "Linear",
    "importance_only": "Importance Only",
    "snr_only": "SNR Only",
    "full": "Full FIS",
}
METHOD_COLORS = {
    "baseline": "#888888",       # Gray
    "linear": "#2196F3",        # Blue
    "importance_only": "#4CAF50", # Green
    "snr_only": "#FF9800",       # Orange
    "full": "#E91E63",           # Pink/Magenta
}
METHOD_MARKERS = {
    "baseline": "o",
    "linear": "s",
    "importance_only": "^",
    "snr_only": "D",
    "full": "*",
}
METHOD_LINE_STYLES = {
    "baseline": "--",
    "linear": "-",
    "importance_only": "-",
    "snr_only": "-.",
    "full": "-",
}


def load_results(path):
    """Doc ket qua tu JSON file"""
    full_path = os.path.join(RESULTS_DIR, path)
    if not os.path.exists(full_path):
        print(f"WARNING: File not found: {full_path}")
        return None
    with open(full_path, "r") as f:
        return json.load(f)


def extract_psnr(data, snr_db):
    """Trich xuat PSNR cho 1 SNR"""
    budget_key = "1.0"
    snr_key = str(float(snr_db))
    
    try:
        results = data["results"][budget_key][snr_key]
        return {m: results[m]["psnr"] for m in METHODS_WITH_SNR_ONLY if m in results}
    except KeyError:
        return None


def extract_ssim(data, snr_db):
    """Trich xuat SSIM cho 1 SNR"""
    budget_key = "1.0"
    snr_key = str(float(snr_db))
    
    try:
        results = data["results"][budget_key][snr_key]
        return {m: results[m]["ssim"] for m in METHODS_WITH_SNR_ONLY if m in results}
    except KeyError:
        return None


def get_available_snrs(data):
    """Lay danh sach SNR co san"""
    try:
        budget_key = "1.0"
        snrs = sorted([float(k) for k in data["results"][budget_key].keys()])
        return snrs
    except:
        return []


def plot_channel_comparison(ax, data_dict, metric="psnr", channel_names=None):
    """Ve bieu do so sanh giua cac kenh"""
    if channel_names is None:
        channel_names = list(data_dict.keys())
    
    x = np.arange(len(METHODS))
    width = 0.25
    
    # Lay SNR cuoi cung (thường la cao nhat)
    snrs = get_available_snrs(list(data_dict.values())[0])
    if snrs:
        snr_to_plot = snrs[-1]  # SNR cao nhat
    
    for i, ch_name in enumerate(channel_names):
        data = data_dict[ch_name]
        if data is None:
            continue
        
        vals = []
        for m in METHODS:
            if m == "snr_only":
                # SNR only chi can bang baseline, khong can tinh
                vals.append(0)
                continue
            
            if metric == "psnr":
                v = extract_psnr(data, snr_to_plot)
            else:
                v = extract_ssim(data, snr_to_plot)
            
            if v and m in v:
                vals.append(v[m])
            else:
                vals.append(0)
        
        # Tinh gain so voi baseline
        baseline_vals = vals[0]
        gains = [v - baseline_vals if i > 0 else 0 for i, v in enumerate(vals)]
        
        bars = ax.bar(x + i * width - width, gains, width, 
                     label=ch_name, alpha=0.8)
    
    ax.set_xlabel("Method")
    ax.set_ylabel(f"{metric.upper()} Gain vs Baseline (dB)")
    ax.set_title(f"{metric.upper()} Gain at SNR={snr_to_plot}dB")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=45, ha='right')
    ax.legend(loc='upper left')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)


def plot_psnr_by_snr(ax, data, channel_name, metric="psnr"):
    """Ve bieu do PSNR/SSIM theo SNR"""
    snrs = get_available_snrs(data)
    if not snrs:
        return
    
    for method in METHODS:
        vals = []
        for snr in snrs:
            if metric == "psnr":
                v = extract_psnr(data, snr)
            else:
                v = extract_ssim(data, snr)
            
            if v and method in v:
                vals.append(v[method])
            else:
                vals.append(np.nan)
        
        ax.plot(snrs, vals,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linestyle=METHOD_LINE_STYLES[method],
                label=METHOD_LABELS[method],
                linewidth=2,
                markersize=8)
    
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(f"{metric.upper()}")
    ax.set_title(f"{channel_name}")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set x ticks
    if snrs:
        ax.set_xticks(snrs)


def plot_all_channels(results_dict, output_dir, metric="psnr"):
    """Ve tat ca cac kenh tren cung 1 figure"""
    n_channels = len(results_dict)
    fig, axes = plt.subplots(1, n_channels, figsize=(6*n_channels, 5))
    
    if n_channels == 1:
        axes = [axes]
    
    for ax, (ch_name, data) in zip(axes, results_dict.items()):
        if data is None:
            ax.text(0.5, 0.5, f"No data for {ch_name}", 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        plot_psnr_by_snr(ax, data, ch_name, metric)
    
    plt.tight_layout()
    return fig


def create_summary_table(data_dict, metric="psnr"):
    """Tao bang tom tat ket qua"""
    lines = []
    lines.append("=" * 80)
    lines.append(f"SUMMARY TABLE: {metric.upper()}")
    lines.append("=" * 80)
    
    snrs = None
    for ch_name, data in data_dict.items():
        if data:
            snrs = get_available_snrs(data)
            break
    
    if not snrs:
        lines.append("No data available")
        return "\n".join(lines)
    
    # Header
    header = f"{'Channel':<20}"
    for m in METHODS_WITH_SNR_ONLY:
        header += f"{METHOD_LABELS[m]:>15}"
    lines.append(header)
    lines.append("-" * 80)
    
    for ch_name, data in data_dict.items():
        if data is None:
            continue
        
        row = f"{ch_name:<20}"
        for m in METHODS_WITH_SNR_ONLY:
            # Lay gia tri trung binh qua tat ca SNR
            vals = []
            for snr in snrs:
                if metric == "psnr":
                    v = extract_psnr(data, snr)
                else:
                    v = extract_ssim(data, snr)
                if v and m in v:
                    vals.append(v[m])
            
            if vals:
                mean_val = np.mean(vals)
                row += f"{mean_val:>15.3f}" if metric == "psnr" else f"{mean_val:>15.4f}"
            else:
                row += f"{'N/A':>15}"
        
        lines.append(row)
    
    lines.append("=" * 80)
    return "\n".join(lines)


def create_detailed_table(data_dict, snr_db, metric="psnr"):
    """Tao bang chi tiet cho 1 SNR"""
    lines = []
    lines.append("=" * 80)
    lines.append(f"DETAILED RESULTS: SNR={snr_db}dB, Budget=1.0")
    lines.append("=" * 80)
    
    header = f"{'Channel':<20}"
    for m in METHODS_WITH_SNR_ONLY:
        header += f"{METHOD_LABELS[m]:>15}"
    lines.append(header)
    header2 = f"{'':20}"
    for m in METHODS_WITH_SNR_ONLY:
        header2 += f"{'dB' if metric == 'psnr' else '':>15}"
    lines.append(header2)
    lines.append("-" * 80)
    
    for ch_name, data in data_dict.items():
        if data is None:
            continue
        
        row = f"{ch_name:<20}"
        for m in METHODS_WITH_SNR_ONLY:
            if metric == "psnr":
                v = extract_psnr(data, snr_db)
            else:
                v = extract_ssim(data, snr_db)
            
            if v and m in v:
                val = v[m]
                row += f"{val:>15.3f}" if metric == "psnr" else f"{val:>15.4f}"
            else:
                row += f"{'N/A':>15}"
        
        lines.append(row)
    
    # Gain vs baseline
    lines.append("-" * 80)
    lines.append(f"{'Gain vs Baseline':<20}")
    for ch_name, data in data_dict.items():
        if data is None:
            continue
        
        baseline_row = f"{ch_name:<20}"
        
        for m in METHODS_WITH_SNR_ONLY:
            if metric == "psnr":
                v = extract_psnr(data, snr_db)
            else:
                v = extract_ssim(data, snr_db)
            
            if v and m in v and "baseline" in v:
                val = v[m]
                baseline_row += f"{val:>15.3f}" if metric == "psnr" else f"{val:>15.4f}"
            else:
                baseline_row += f"{'N/A':>15}"
        
        lines.append(baseline_row)
        
        if metric == "psnr":
            gain_row = f"{'Full FIS Gain':<20}"
            for m in METHODS_WITH_SNR_ONLY:
                if metric == "psnr":
                    v = extract_psnr(data, snr_db)
                else:
                    v = extract_ssim(data, snr_db)
                
                if v and m in v and "baseline" in v:
                    gain = v[m] - v["baseline"]
                    if m == "full":
                        gain_row += f"+{gain:>14.3f}"
                    elif m == "baseline":
                        gain_row += f"{'0.000':>15}"
                    else:
                        gain_row += f"+{gain:>14.3f}"
                else:
                    gain_row += f"{'N/A':>15}"
            lines.append(gain_row)
        
        break  # Chi 1 kenh
    
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("GENERATING PLOTS FROM PAPER SIMS RESULTS")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load all results
    data_dict = {}
    for ch_name, path in RESULTS.items():
        print(f"Loading: {path}")
        data = load_results(path)
        if data:
            snrs = get_available_snrs(data)
            print(f"  -> {len(snrs)} SNR values: {snrs}")
        data_dict[ch_name] = data
    
    print()
    
    # Create output directory
    output_dir = "plots_fixed"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot PSNR by SNR for all channels
    print("Creating PSNR comparison plot...")
    fig_psnr = plot_all_channels(data_dict, output_dir, metric="psnr")
    fig_psnr.suptitle("PSNR vs SNR - Deep JSCC with FIS", fontsize=16, y=1.02)
    fig_psnr.savefig(os.path.join(output_dir, "psnr_comparison_all_channels.png"), 
                     bbox_inches='tight')
    plt.close(fig_psnr)
    
    # 2. Plot SSIM by SNR for all channels
    print("Creating SSIM comparison plot...")
    fig_ssim = plot_all_channels(data_dict, output_dir, metric="ssim")
    fig_ssim.suptitle("SSIM vs SNR - Deep JSCC with FIS", fontsize=16, y=1.02)
    fig_ssim.savefig(os.path.join(output_dir, "ssim_comparison_all_channels.png"), 
                    bbox_inches='tight')
    plt.close(fig_ssim)
    
    # 3. Create individual channel plots
    for ch_name, data in data_dict.items():
        if data is None:
            continue
        
        # PSNR plot
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_psnr_by_snr(ax, data, ch_name, metric="psnr")
        
        # Add gain annotation
        snrs = get_available_snrs(data)
        if snrs:
            snr_high = snrs[-1]  # Highest SNR
            v = extract_psnr(data, snr_high)
            if v and "full" in v and "baseline" in v:
                gain = v["full"] - v["baseline"]
                ax.annotate(f'Full FIS Gain: +{gain:.1f} dB', 
                           xy=(0.95, 0.95), xycoords='axes fraction',
                           fontsize=12, ha='right', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.savefig(os.path.join(output_dir, f"psnr_{ch_name.replace(' ', '_').lower()}.png"),
                   bbox_inches='tight')
        plt.close(fig)
        
        # SSIM plot
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_psnr_by_snr(ax, data, ch_name, metric="ssim")
        fig.savefig(os.path.join(output_dir, f"ssim_{ch_name.replace(' ', '_').lower()}.png"),
                   bbox_inches='tight')
        plt.close(fig)
    
    # 4. Gain comparison bar chart
    print("Creating gain comparison plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    snrs_to_compare = [1, 7, 13]  # Low, Mid, High SNR
    methods_to_compare = [m for m in METHODS if m not in ["snr_only", "baseline"]]  # linear, importance_only, full
    
    for idx, snr in enumerate(snrs_to_compare):
        ax = axes[idx]
        
        gains_dict = {}
        for ch_name, data in data_dict.items():
            if data is None:
                continue
            
            gains = []
            for m in methods_to_compare:
                v = extract_psnr(data, snr)
                if v and m in v and "baseline" in v:
                    gains.append(v[m] - v["baseline"])
                else:
                    gains.append(0)
            
            gains_dict[ch_name] = gains
        
        x = np.arange(len(methods_to_compare))
        width = 0.25
        
        for i, (ch_name, gains) in enumerate(gains_dict.items()):
            ax.bar(x + i * width - width, gains, width, label=ch_name, alpha=0.8)
        
        ax.set_xlabel("Method")
        ax.set_ylabel("PSNR Gain vs Baseline (dB)")
        ax.set_title(f"SNR = {snr} dB")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods_to_compare], rotation=45, ha='right')
        ax.legend(loc='upper left')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "psnr_gain_comparison.png"), bbox_inches='tight')
    plt.close(fig)
    
    # Print summary tables
    print("\n" + create_summary_table(data_dict, metric="psnr"))
    print("\n" + create_summary_table(data_dict, metric="ssim"))
    
    # Detailed results for key SNR values
    for snr in [1, 7, 13]:
        print("\n" + create_detailed_table(data_dict, snr, metric="psnr"))
    
    print("\n" + "=" * 70)
    print("PLOTS SAVED TO:", output_dir)
    print("=" * 70)
    
    # List saved files
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
