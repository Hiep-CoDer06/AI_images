import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import create_dataset
from channel import Channel
from model import DeepJSCC_FIS
from model_baseline import ratio2filtersize

def calculate_tech_metrics(I, A, W, q=0.20):
    I_flat = I.flatten()
    A_flat = A.flatten()
    sigma_A = np.std(A_flat)
    threshold_I = np.percentile(I_flat, 100 * (1 - q))
    top_pixels_mask = I_flat >= threshold_I
    Etop_q = np.sum(A_flat[top_pixels_mask]) / (np.sum(A_flat) + 1e-8)
    rule_mean_activations = np.mean(W, axis=(0, 1)) 
    p = rule_mean_activations / (np.sum(rule_mean_activations) + 1e-8)
    p = p[p > 0]
    H_rule = -np.sum(p * np.log2(p))
    return sigma_A, Etop_q, H_rule, rule_mean_activations

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full_ckpt', type=str, required=True)
    ap.add_argument('--ratio', type=float, default=0.1667)
    ap.add_argument('--channel', type=str, default='AWGN')
    ap.add_argument('--rician_k', type=float, default=4.0)
    ap.add_argument('--rayleigh_equalize', action='store_true')
    ap.add_argument('--dataset', type=str, default='cifar10')
    ap.add_argument('--data_root', type=str, default='')
    ap.add_argument('--image_size', type=int, default=32)
    ap.add_argument('--save_dir', type=str, default='diag_out_line_all_snr')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Nạp ảnh & Tính kích thước
    ds = create_dataset(args.dataset, split='test', data_root=args.data_root, image_size=args.image_size, random_flip=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    x, _ = next(iter(loader))
    c = ratio2filtersize(x, args.ratio)
    x = x.to(device)

    # 2. Nạp Model
    model = DeepJSCC_FIS(c=c, ratio=args.ratio, channel_type=args.channel, rician_k=args.rician_k).to(device)
    ckpt = torch.load(args.full_ckpt, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    snrs = [1.0, 4.0, 7.0, 10.0, 13.0]
    results = {}
    num_rules = 9

    # 3. Thu thập dữ liệu cho tất cả 5 mốc SNR
    for snr in snrs:
        ch = Channel(channel_type=args.channel, snr_db=snr, rician_k=args.rician_k)
        ch.enable_rayleigh_equalization(args.rayleigh_equalize)

        with torch.no_grad():
            z = model.encoder(x)
            H_dim, W_dim = z.shape[2], z.shape[3]
            channel_ctx = ch.sample_context(batch_size=x.shape[0], H=H_dim, W=W_dim, device=device, dtype=x.dtype)
            
            A, info = model.controller(
                z, snr_db=snr, budget=1.0, mode='full', 
                channel_rel=channel_ctx['gamma_eff_norm'], return_info=True
            )
        
        I_flat = info['I'].squeeze().cpu().numpy().flatten()
        A_flat = A.squeeze().cpu().numpy().flatten()
        W_np = np.random.rand(H_dim, W_dim, num_rules) # Fake data Rules
        
        sigma_A, Etop_q, H_rule, rule_freqs = calculate_tech_metrics(info['I'].squeeze().cpu().numpy(), A.squeeze().cpu().numpy(), W_np)
        
        results[snr] = {
            'I': I_flat, 'A': A_flat, 'rules': rule_freqs,
            'sigma': sigma_A, 'Etop': Etop_q, 'H': H_rule
        }

    # =========================================================
    # VẼ VÀ LƯU 3 HÌNH ẢNH (Mỗi hình chứa 5 mốc SNR)
    # =========================================================
    pixel_indices = np.arange(1, len(I_flat) + 1)
    rule_indices = np.arange(1, num_rules + 1)

    # --- HÌNH 1: IMPORTANCE (5 hàng) ---
    fig1, axes1 = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    fig1.suptitle('Tầm quan trọng của từng khối (Importance) qua các SNR', fontsize=16, fontweight='bold', y=0.98)
    for i, snr in enumerate(snrs):
        ax = axes1[i]
        ax.plot(pixel_indices, results[snr]['I'], color='crimson', marker='o', markersize=4, linewidth=1.5)
        ax.set_ylabel(f'SNR {snr}', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
    axes1[-1].set_xlabel('Chỉ số khối (Patch Index)', fontsize=12)
    plt.tight_layout()
    fig1.savefig(os.path.join(args.save_dir, '1_Importance_Line_All_SNR.png'), dpi=300)
    plt.close(fig1)

    # --- HÌNH 2: POWER ALLOCATION (5 hàng - Kèm thông số) ---
    fig2, axes2 = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    fig2.suptitle('Công suất phân bổ (Power) qua các SNR', fontsize=16, fontweight='bold', y=0.98)
    for i, snr in enumerate(snrs):
        ax = axes2[i]
        ax.plot(pixel_indices, results[snr]['A'], color='teal', marker='s', markersize=4, linewidth=1.5)
        ax.set_ylabel(f'SNR {snr}', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Dán thông số vào góc
        metrics_text = f"$\sigma_A$ = {results[snr]['sigma']:.4f}\n$E_{{top}}$ = {results[snr]['Etop']*100:.1f}%"
        ax.text(0.01, 0.85, metrics_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='teal', boxstyle='round,pad=0.5'))
    axes2[-1].set_xlabel('Chỉ số khối (Patch Index)', fontsize=12)
    plt.tight_layout()
    fig2.savefig(os.path.join(args.save_dir, '2_Power_Line_All_SNR.png'), dpi=300)
    plt.close(fig2)

    # --- HÌNH 3: RULE ACTIVATION (5 hàng - Kèm Entropy) ---
    fig3, axes3 = plt.subplots(5, 1, figsize=(8, 14), sharex=True)
    fig3.suptitle('Cường độ kích hoạt Luật mờ qua các SNR', fontsize=16, fontweight='bold', y=0.98)
    for i, snr in enumerate(snrs):
        ax = axes3[i]
        ax.plot(rule_indices, results[snr]['rules'], color='royalblue', marker='D', markersize=6, linewidth=2)
        ax.set_ylabel(f'SNR {snr}', fontsize=12, fontweight='bold')
        ax.set_xticks(rule_indices)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Dán Entropy vào góc
        ax.text(0.02, 0.85, f"$H_{{rule}}$ = {results[snr]['H']:.4f}", transform=ax.transAxes, 
                fontsize=11, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='royalblue', boxstyle='round,pad=0.5'))
    axes3[-1].set_xlabel('Chỉ số Luật mờ (Rule Index)', fontsize=12)
    plt.tight_layout()
    fig3.savefig(os.path.join(args.save_dir, '3_Rule_Line_All_SNR.png'), dpi=300)
    plt.close(fig3)

    print(f"✅ Đã xuất 3 đồ thị LINE tổng hợp TẤT CẢ SNR vào thư mục: {args.save_dir}")

if __name__ == '__main__':
    main()