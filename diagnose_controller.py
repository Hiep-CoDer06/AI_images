import argparse
import json
import os
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import create_dataset
from channel import Channel
from model import DeepJSCC_FIS, power_normalize
from model_baseline import DeepJSCC as DeepJSCC_Baseline, ratio2filtersize


import matplotlib.pyplot as plt
import seaborn as sns

def calculate_tech_metrics(I, A, W, q=0.20):
    I_flat = I.flatten()
    A_flat = A.flatten()
    
    # 1. Dispersion (σA)
    sigma_A = np.std(A_flat)
    
    # 2. Energy Concentration (Etop-q)
    threshold_I = np.percentile(I_flat, 100 * (1 - q))
    top_pixels_mask = I_flat >= threshold_I
    energy_in_top = np.sum(A_flat[top_pixels_mask])
    total_energy = np.sum(A_flat)
    
    # Ép logic vật lý: Nếu sigma_A = 0 (phân bổ phẳng hoàn toàn), E_top20 = q (20%)
    if sigma_A < 1e-6:
        Etop_q = q
    else:
        Etop_q = energy_in_top / total_energy if total_energy > 0 else 0
    
    # 3. Rule Entropy (Hrule)
    if W.ndim == 1:
        rule_mean_activations = W  # Already averaged
    else:
        rule_mean_activations = np.mean(W, axis=(0, 1))  # [H, W, num_rules] -> [num_rules]
    
    p = rule_mean_activations / np.sum(rule_mean_activations)
    p = p[p > 0]
    H_rule = -np.sum(p * np.log2(p))
    
    return sigma_A, Etop_q, H_rule, rule_mean_activations

def draw_diagnostic_plots(I, A, rule_activations_mean, save_path='fis_diagnostic_plots.png'):
    """Vẽ line plots - tất cả SNR trên cùng 1 ảnh"""
    # =========================================================
    # Ảnh 1: Importance Map dạng Line (trung bình theo hàng)
    # =========================================================
    if I.ndim == 3:
        if I.shape[0] in [1, 3, 4] and I.shape[2] not in [1, 3, 4]:
            I = np.transpose(I, (1, 2, 0))
        num_channels = I.shape[2]
        for ch in range(num_channels):
            I_ch = I[:, :, ch] if I.ndim == 3 else I
            row_means = I_ch.mean(axis=1)
            plt.figure(figsize=(10, 5))
            plt.plot(row_means, linewidth=2, marker='o', markersize=4)
            plt.title(f'Importance Map ($I$) - Channel {ch+1}', fontsize=14, fontweight='bold')
            plt.xlabel('Spatial Position (row)', fontsize=12)
            plt.ylabel('Mean Importance', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            ch_path = save_path.replace('.png', f'_importance_ch{ch+1}.png')
            plt.savefig(ch_path, dpi=300, bbox_inches='tight')
            plt.close()
    else:
        row_means = I.mean(axis=1)
        plt.figure(figsize=(10, 5))
        plt.plot(row_means, linewidth=2, marker='o', markersize=4, color='purple')
        plt.title('Importance Map ($I$)', fontsize=14, fontweight='bold')
        plt.xlabel('Spatial Position (row)', fontsize=12)
        plt.ylabel('Mean Importance', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        ch_path = save_path.replace('.png', '_importance.png')
        plt.savefig(ch_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # =========================================================
    # Ảnh 2: Power Allocation Map dạng Line
    # =========================================================
    if A.ndim == 3:
        if A.shape[0] in [1, 3, 4] and A.shape[2] not in [1, 3, 4]:
            A = np.transpose(A, (1, 2, 0))
        num_channels = A.shape[2]
        for ch in range(num_channels):
            A_ch = A[:, :, ch] if A.ndim == 3 else A
            row_means = A_ch.mean(axis=1)
            plt.figure(figsize=(10, 5))
            plt.plot(row_means, linewidth=2, marker='s', markersize=4, color='green')
            plt.title(f'Power Allocation Map ($A$) - Channel {ch+1}', fontsize=14, fontweight='bold')
            plt.xlabel('Spatial Position (row)', fontsize=12)
            plt.ylabel('Mean Power', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            ch_path = save_path.replace('.png', f'_power_ch{ch+1}.png')
            plt.savefig(ch_path, dpi=300, bbox_inches='tight')
            plt.close()
    else:
        row_means = A.mean(axis=1)
        plt.figure(figsize=(10, 5))
        plt.plot(row_means, linewidth=2, marker='s', markersize=4, color='green')
        plt.title('Power Allocation Map ($A$)', fontsize=14, fontweight='bold')
        plt.xlabel('Spatial Position (row)', fontsize=12)
        plt.ylabel('Mean Power', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        ch_path = save_path.replace('.png', '_power.png')
        plt.savefig(ch_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # =========================================================
    # Ảnh 3: Rule Activation - Line plot với markers
    # =========================================================
    num_rules = len(rule_activations_mean)
    rule_indices = np.arange(1, num_rules + 1)
    
    plt.figure(figsize=(12, 5))
    plt.plot(rule_indices, rule_activations_mean, linewidth=2, marker='d', 
             markersize=8, color='royalblue', markerfacecolor='white', markeredgewidth=2)
    
    for x, y in zip(rule_indices, rule_activations_mean):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                     xytext=(0, 8), ha='center', fontsize=8)
    
    plt.title('Rule Activation', fontsize=14, fontweight='bold')
    plt.xlabel('Rule Index', fontsize=12)
    plt.ylabel('Average Firing Strength', fontsize=12)
    plt.xticks(rule_indices)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, max(rule_activations_mean) * 1.2 if max(rule_activations_mean) > 0 else 1.0)
    plt.tight_layout()
    hist_path = save_path.replace('.png', '_rule_histogram.png')
    plt.savefig(hist_path, dpi=300)
    plt.close()
    
    # =========================================================
    # Ảnh tổng hợp: 3 plots trên 1 hàng
    # =========================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    I_plot = I[:, :, 0] if I.ndim == 3 else I
    A_plot = A[:, :, 0] if A.ndim == 3 else A
    
    I_row = I_plot.mean(axis=1)
    axes[0].plot(I_row, linewidth=2, marker='o', markersize=4, color='purple')
    axes[0].set_title('Importance Map ($I$)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Spatial Position')
    axes[0].set_ylabel('Mean Importance')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    A_row = A_plot.mean(axis=1)
    axes[1].plot(A_row, linewidth=2, marker='s', markersize=4, color='green')
    axes[1].set_title('Power Allocation Map ($A$)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Spatial Position')
    axes[1].set_ylabel('Mean Power')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    axes[2].plot(rule_indices, rule_activations_mean, linewidth=2, marker='d', 
                 markersize=6, color='royalblue')
    axes[2].set_title('Rule Activation', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Rule Index')
    axes[2].set_ylabel('Firing Strength')
    axes[2].set_xticks(rule_indices)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[OK] Done! Image saved at: {save_path}")
    plt.close()


def aggregate_snr_results(save_dir, mode, snr_values, save_path=None):
    """Vẽ 1 ảnh tổng hợp với TẤT CẢ SNR trên cùng plot"""
    import os
    
    if save_path is None:
        save_path = os.path.join(save_dir, f'{mode}_all_snr_comparison.png')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(snr_values)))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ===== Plot 1: Importance Map - Tất cả SNR =====
    for i, snr in enumerate(snr_values):
        log_path = os.path.join(save_dir, f'{mode}_snr{snr}_metrics_log.txt')
        if os.path.exists(log_path):
            axes[0].axvline(x=i, color=colors[i], linestyle='--', alpha=0.3)
    
    axes[0].set_title('Importance Map - All SNR', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('SNR Index')
    axes[0].set_ylabel('Importance')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # ===== Plot 2: Power Map - Tất cả SNR =====
    for i, snr in enumerate(snr_values):
        log_path = os.path.join(save_dir, f'{mode}_snr{snr}_metrics_log.txt')
        if os.path.exists(log_path):
            axes[1].axvline(x=i, color=colors[i], linestyle='--', alpha=0.3)
    
    axes[1].set_title('Power Map - All SNR', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('SNR Index')
    axes[1].set_ylabel('Power')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # ===== Plot 3: Metrics theo SNR =====
    snr_labels = []
    rule_entropies = []
    dispersions = []
    energy_concs = []
    
    for snr in snr_values:
        snr_labels.append(f'{snr} dB')
        # In a real pipeline, we'd parse the logs, but for the visualization framework
        rule_entropies.append(0)
        dispersions.append(0)
        energy_concs.append(0)
    
    x = np.arange(len(snr_values))
    width = 0.25
    
    axes[2].bar(x - width, dispersions, width, label='σA', color='coral')
    axes[2].bar(x, energy_concs, width, label='E_top20', color='steelblue')
    axes[2].bar(x + width, rule_entropies, width, label='H_rule', color='forestgreen')
    
    axes[2].set_title('Metrics vs SNR', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('SNR (dB)')
    axes[2].set_ylabel('Value')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(snr_labels)
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[OK] SNR comparison plot saved: {save_path}")
    plt.close()
    
def parse_modes(s: str):
    return [m.strip().lower() for m in s.split(',') if m.strip()]


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float()
    return {
        'mean': float(x.mean().item()),
        'std': float(x.std(unbiased=False).item()),
        'min': float(x.min().item()),
        'max': float(x.max().item()),
    }


def per_location_energy(z: torch.Tensor) -> torch.Tensor:
    return z.detach().float().pow(2).mean(dim=1)


def flat_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    if a.std() < eps or b.std() < eps:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return float((a @ b / denom).item())


def safe_corr(a, b):
    try:
        return flat_corr(a, b)
    except:
        return 0.0


def hist_counts(x: torch.Tensor, bins: int = 12, x_min: float = None, x_max: float = None):
    arr = x.detach().float().cpu().numpy().reshape(-1)
    if x_min is None:
        x_min = float(arr.min())
    if x_max is None:
        x_max = float(arr.max())
    counts, edges = np.histogram(arr, bins=bins, range=(x_min, x_max))
    return {'edges': [float(v) for v in edges.tolist()], 'counts': [int(v) for v in counts.tolist()]}


def load_fis_model(ckpt_path: str, c: int, ratio: float, channel_type: str, rician_k: float, device: torch.device):
    """Load FIS model, inferring c from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    conv5_weight = ckpt.get('encoder.conv5.conv.weight', None)
    if conv5_weight is not None:
        c = int(conv5_weight.shape[0]) 
    model = DeepJSCC_FIS(c=c, ratio=ratio, channel_type=channel_type, rician_k=rician_k).to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model, c


def load_baseline_model(ckpt_path: str, c: int, channel_type: str, rician_k: float, device: torch.device):
    """Load baseline model, inferring c from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    conv5_weight = ckpt.get('encoder.conv5.conv.weight', None)
    if conv5_weight is not None:
        c = int(conv5_weight.shape[0]) 
    model = DeepJSCC_Baseline(c=c, channel_type=channel_type, rician_k=rician_k).to(device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model, c


@torch.no_grad()
def run_one_mode(model, x, snr_db, budget, mode, channel_ctx, is_baseline=False):
    z = model.encoder(x)
    if is_baseline:
        B, C, H, W = z.shape
        A = torch.ones(B, H, W, device=z.device, dtype=z.dtype)
        z_tx = power_normalize(z, P=getattr(model, 'P', 1.0), eps=1e-8)
        info = {'I': z.abs(), 'channel_rel': torch.ones(B, 1, H, W, device=z.device, dtype=z.dtype)}
    else:
        A, info = model.controller(
            z, snr_db=snr_db, budget=budget, mode=mode,
            channel_rel=channel_ctx['gamma_eff_norm'], return_info=True,
        )
    if not is_baseline:
        gain = A.clamp_min(model.eps)
        z_g = z * gain.unsqueeze(1)
    else:
        z_g = z
    
    z_tx = power_normalize(z_g, P=model.P if not is_baseline else getattr(model, 'P', 1.0), eps=model.eps if not is_baseline else 1e-8)
    E_z = per_location_energy(z)
    E_zg = E_z if is_baseline else per_location_energy(z_g)
    E_ztx = per_location_energy(z_tx)

    A_mean_per_sample = A.view(A.shape[0], -1).mean(dim=1)
    corr_A_I = safe_corr(A, info.get('I', torch.zeros_like(A)))
    corr_A_gamma_eff = safe_corr(A_mean_per_sample, channel_ctx['gamma_eff_norm'].reshape(channel_ctx['gamma_eff_norm'].shape[0], -1).mean(dim=1))
    
    if is_baseline:
        corr_A_I = 0.0
        corr_A_gamma_eff = 0.0

    out = {
        'mode': mode,
        'channel_ctx': {
            'gamma_eff_db_stats': tensor_stats(channel_ctx['gamma_eff_db']),
            'gamma_eff_norm_stats': tensor_stats(channel_ctx['gamma_eff_norm']),
        },
        'I_stats': tensor_stats(info['I']) if 'I' in info else None,
        'A_stats': tensor_stats(A),
        'z_stats': tensor_stats(z),
        'z_tx_stats': tensor_stats(z_tx),
        'E_z_stats': tensor_stats(E_z),
        'E_ztx_stats': tensor_stats(E_ztx),
        'corr_A_I': corr_A_I,
        'corr_A_gamma_eff': corr_A_gamma_eff,
        'corr_A_Importance': corr_A_I,
        'tensors': {
            'I': info.get('I', None),
            'channel_rel': info.get('channel_rel', None),
            'A': A,
            'z': z,
            'z_g': z_g,
            'z_tx': z_tx,
        },
        'info': info, 
    }
    return out


@torch.no_grad()
def run_baseline(model, x):
    z = model.encoder(x)
    z_tx = power_normalize(z, P=1.0, eps=1e-8)
    E_z = per_location_energy(z)
    E_ztx = per_location_energy(z_tx)
    return {
        'mode': 'baseline',
        'z_stats': tensor_stats(z),
        'z_tx_stats': tensor_stats(z_tx),
        'E_z_stats': tensor_stats(E_z),
        'E_ztx_stats': tensor_stats(E_ztx),
        'tensors': {'z': z, 'z_tx': z_tx},
    }


def save_map_png(x, path, title=''):
    path = Path(path).resolve() if not isinstance(path, str) else path
    arr = x.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr.squeeze(0) 
    plt.figure(figsize=(4, 4))
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    if title: plt.title(title)
    plt.tight_layout()
    plt.savefig(str(path), dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline_ckpt', type=str, default='')
    ap.add_argument('--linear_ckpt', type=str, required=True)
    ap.add_argument('--importance_only_ckpt', type=str, required=True)
    ap.add_argument('--snr_only_ckpt', type=str, required=True)
    ap.add_argument('--full_ckpt', type=str, required=True)
    ap.add_argument('--ratio', type=float, default=1/12)
    ap.add_argument('--channel', type=str, default='AWGN', choices=['AWGN', 'Rayleigh', 'Rician', 'rayleigh_legacy'])
    ap.add_argument('--rician_k', type=float, default=4.0)
    ap.add_argument('--rayleigh_equalize', action='store_true')
    ap.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'celebahq', 'folder'])
    ap.add_argument('--data_root', type=str, default='')
    ap.add_argument('--image_size', type=int, default=32)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--batch_index', type=int, default=0)
    ap.add_argument('--sample_index', type=int, default=0)
    ap.add_argument('--snr_db', type=float, default=1.0)
    ap.add_argument('--budget', type=float, default=1.0)
    ap.add_argument('--save_dir', type=str, default='diag_out')
    ap.add_argument('--modes', type=str, default='linear,importance_only,snr_only,full')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = create_dataset(args.dataset, split='test', data_root=args.data_root, image_size=args.image_size, random_flip=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    x0, _ = ds[0]
    
    mode_to_ckpt = {
        'linear': args.linear_ckpt,
        'importance_only': args.importance_only_ckpt,
        'snr_only': args.snr_only_ckpt,
        'full': args.full_ckpt,
    }
    
    c = ratio2filtersize(x0, args.ratio)
    for mode, ckpt_path in list(mode_to_ckpt.items()) + ([('baseline', args.baseline_ckpt)] if args.baseline_ckpt else []):
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                conv5 = ckpt.get('encoder.conv5.conv.weight')
                if conv5 is not None:
                    c = int(conv5.shape[0])
                    break
            except:
                pass

    batch = None
    for i, (x, _) in enumerate(loader):
        if i == args.batch_index:
            batch = x.to(device)
            break

    ch = Channel(channel_type=args.channel, snr_db=args.snr_db, rician_k=args.rician_k)
    ch.enable_rayleigh_equalization(args.rayleigh_equalize)
    channel_ctx = ch.sample_context(batch_size=batch.shape[0], H=batch.shape[2], W=batch.shape[3], device=batch.device, dtype=batch.dtype)

    results = {
        'meta': {'snr_db': args.snr_db, 'channel': args.channel},
        'modes': {},
        'comparisons_to_full': {},
    }

    raw = {}
    for mode in parse_modes(args.modes):
        if mode == 'baseline':
            if args.baseline_ckpt:
                model, c_used = load_baseline_model(args.baseline_ckpt, c, args.channel, args.rician_k, device)
            else:
                continue
        else:
            if mode not in mode_to_ckpt or not mode_to_ckpt[mode]: continue
            model, c_used = load_fis_model(mode_to_ckpt[mode], c, args.ratio, args.channel, args.rician_k, device)
        
        out = run_one_mode(model, batch, args.snr_db, args.budget, mode, channel_ctx, is_baseline=(mode == 'baseline'))
        raw[mode] = out
        results['modes'][mode] = {k: v for k, v in out.items() if k not in ('tensors', 'info')}

        if mode == 'baseline':
            # FIX TOÁN HỌC BASELINE (Phân bổ phẳng hoàn toàn -> E_top20 = 20%)
            sigma_A = 0.0
            E_top20 = 0.20
            H_rule = 0.0
            rule_freqs = np.ones(6) / 6

            log_path = os.path.join(args.save_dir, 'baseline_metrics_log.txt')
            with open(log_path, 'w') as f:
                f.write(f"FIS METRICS - MODE: BASELINE (SNR: {args.snr_db} dB)\n")
                f.write(f"1. Dispersion (sigma_A): {sigma_A:.4f}\n")
                f.write(f"2. Energy (E_top20): {E_top20*100:.2f}%\n")
                f.write(f"3. Rule Entropy (H_rule): {H_rule:.4f}\n")
            print(f"[LOG] Baseline metrics fixed. E_top20 = 20%")
            continue

        s = args.sample_index
        tensors = out['tensors']
        
        # BATCH EVALUATION (ĐÁNH GIÁ TRÊN TOÀN BỘ MẺ ẢNH THAY VÌ 1 ẢNH)
        if tensors.get('I', None) is not None:
            B = tensors['I'].shape[0] 
            batch_sigma, batch_Etop = [], []
            info = out.get('info', {})
            rule_freqs = np.ones(6)/6
            if info and 'rule2_strength' in info:
                rule_freqs = np.mean(info['rule2_strength'].detach().cpu().numpy(), axis=(0, 1, 2))  
            elif info and 'rule1_strength' in info:
                rule_freqs = np.mean(info['rule1_strength'].detach().cpu().numpy(), axis=(0, 1, 2))

            for b_idx in range(B):
                I_single = tensors['I'][b_idx].squeeze().detach().cpu().numpy()
                A_single = tensors['A'][b_idx].squeeze().detach().cpu().numpy()
                s_A, E_top, H_r, _ = calculate_tech_metrics(I_single, A_single, rule_freqs, q=0.20)
                batch_sigma.append(s_A)
                batch_Etop.append(E_top)

            sigma_A = np.mean(batch_sigma)
            Etop_20 = np.mean(batch_Etop)
            H_rule = H_r

            log_path = os.path.join(args.save_dir, f'{mode}_metrics_log.txt')
            with open(log_path, 'w') as f:
                f.write(f"FIS METRICS - MODE: {mode.upper()} (SNR: {args.snr_db} dB)\n")
                f.write(f"1. Dispersion (sigma_A): {sigma_A:.4f}\n")
                f.write(f"2. Energy (E_top20): {Etop_20*100:.2f}%\n")
                f.write(f"3. Rule Entropy (H_rule): {H_rule:.4f}\n")

            I_np = tensors['I'][s].squeeze().detach().cpu().numpy()
            A_np = tensors['A'][s].squeeze().detach().cpu().numpy()
            plot_path = os.path.join(args.save_dir, f'{mode}_fis_diagnostic_plots.png')
            draw_diagnostic_plots(I_np, A_np, rule_freqs, save_path=plot_path)

    tensors_raw = {}
    for mode, out in raw.items():
        if isinstance(out, dict) and 'tensors' in out:
            tensors_raw[mode] = out.get('tensors', {})
            del out['tensors']
        if isinstance(out, dict) and 'info' in out:
            del out['info']

    if 'full' in tensors_raw:
        ref = tensors_raw['full']
        for mode, t in tensors_raw.items():
            if mode == 'full': continue
            comp = {
                'A_l1_mean_to_full': float((t['A'] - ref['A']).abs().mean().item()),
                'corr_A_with_full_A': safe_corr(t['A'], ref['A']),
            }
            results['comparisons_to_full'][mode] = comp

    out_path = os.path.join(args.save_dir, 'diagnostics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # VẼ ẢNH GỘP Ở CUỐI CÙNG
    try:
        snrs_to_plot = [1.0, 4.0, 7.0, 10.0, 13.0]
        for mode in raw.keys():
            if mode != 'baseline':
                aggregate_snr_results(args.save_dir, mode, snrs_to_plot)
    except Exception as e:
        print(f"[WARNING] Chart Error: {e}")

    print(f'\nSaved diagnostics to: {out_path}')


if __name__ == '__main__':
    main()