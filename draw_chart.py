import matplotlib.pyplot as plt

# 1. Dữ liệu từ log kênh Rayleigh EQ
snrs = [1.0, 4.0, 7.0, 10.0, 13.0]
baseline = [17.752, 19.548, 21.170, 22.496, 23.470]
snr_only = [17.754, 19.550, 21.174, 22.500, 23.473]
linear = [19.373, 20.983, 22.492, 23.784, 24.787]
importance_only = [19.397, 20.988, 22.451, 23.694, 24.657]
full = [19.298, 20.923, 22.448, 23.765, 24.793]

# 2. Khởi tạo canvas
plt.figure(figsize=(10, 6))
plt.grid(True, linestyle='--', alpha=0.6, zorder=0)

# 3. Vẽ 5 đường chiến thuật
plt.plot(snrs, baseline, marker='o', linestyle='-', color='black', linewidth=2, label='Baseline', zorder=3)
plt.plot(snrs, snr_only, marker='x', linestyle='--', color='gray', linewidth=2, label='SNR Only', zorder=3)
plt.plot(snrs, linear, marker='s', linestyle='-', color='blue', linewidth=2, label='Linear', zorder=3)
plt.plot(snrs, importance_only, marker='^', linestyle='-', color='green', linewidth=2, label='Importance Only', zorder=3)
plt.plot(snrs, full, marker='D', linestyle='-', color='purple', linewidth=2.5, label='Full FIS (Channel-Aware)', zorder=4)

# 4. Trang trí đồ thị
plt.title('PSNR Performance over Equalized Rayleigh Fading Channel', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Channel SNR (dB)', fontsize=12, fontweight='bold')
plt.ylabel('Mean PSNR (dB)', fontsize=12, fontweight='bold')
plt.xticks(snrs)
plt.legend(loc='lower right', fontsize=11, framealpha=0.9)

# 5. Lưu ảnh
save_path = 'final_eq_chart.png'
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"✅ Đã vẽ xong đồ thị EQ! Ảnh được lưu tại: {save_path}")