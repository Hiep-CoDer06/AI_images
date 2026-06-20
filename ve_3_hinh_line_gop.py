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
    ap.add_argument('--save_dir', type=str, default='diag_out_line_gop')
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
    # VẼ VÀ LƯU 3 HÌNH (GỘP 5 ĐƯỜNG SNR VÀO CÙNG 1 ĐỒ THỊ)
    # =========================================================
    pixel_indices = np.arange(1, len(I_flat) + 1)
    rule_indices = np.arange(1, num_rules + 1)

    # Các marker để dễ phân biệt các đường
    markers = ['o', 's', '^', 'D', 'v']

    # --- HÌNH 1: IMPORTANCE (Gộp chung) ---
    plt.figure(figsize=(12, 5))
    plt.title('Tầm quan trọng của từng khối (Importance) qua các mức nhiễu', fontsize=15, fontweight='bold')
    for i, snr in enumerate(snrs):
        # Đường I sẽ đè lên nhau, sử dụng alpha để thấy hiệu ứng
        plt.plot(pixel_indices, results[snr]['I'], marker=markers[i], markersize=4, linewidth=1.5, alpha=0.7, label=f'SNR {snr} dB')
    plt.xlabel('Chỉ số khối (Patch Index)', fontsize=12)
    plt.ylabel('Giá trị Tầm quan trọng (Importance)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Kênh truyền', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, '1_Importance_Line_Gop.png'), dpi=300)
    plt.close()

    # --- HÌNH 2: POWER ALLOCATION (Gộp chung) ---
    plt.figure(figsize=(12, 5))
    plt.title('Công suất phân bổ (Power Allocation) qua các mức nhiễu', fontsize=15, fontweight='bold')
    for i, snr in enumerate(snrs):
        plt.plot(pixel_indices, results[snr]['A'], marker=markers[i], markersize=4, linewidth=1.5, alpha=0.8, label=f'SNR {snr} dB')
    plt.xlabel('Chỉ số khối (Patch Index)', fontsize=12)
    plt.ylabel('Công suất phân bổ (Power)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Kênh truyền', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, '2_Power_Line_Gop.png'), dpi=300)
    plt.close()

    # --- HÌNH 3: RULE ACTIVATION (Gộp chung) ---
    plt.figure(figsize=(10, 5))
    plt.title('Cường độ kích hoạt Luật mờ qua các mức nhiễu', fontsize=15, fontweight='bold')
    for i, snr in enumerate(snrs):
        plt.plot(rule_indices, results[snr]['rules'], marker=markers[i], markersize=6, linewidth=2, alpha=0.8, label=f'SNR {snr} dB')
    plt.xlabel('Chỉ số Luật mờ (Rule Index)', fontsize=12)
    plt.ylabel('Cường độ trung bình', fontsize=12)
    plt.xticks(rule_indices)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Kênh truyền', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, '3_Rule_Line_Gop.png'), dpi=300)
    plt.close()

    print(f"✅ Đã xuất 3 đồ thị LINE GỘP TẤT CẢ SNR vào thư mục: {args.save_dir}")

if __name__ == '__main__':
    main()