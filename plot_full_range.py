import os
import json
import matplotlib.pyplot as plt

# 1. Cấu hình dải SNR FULL (Âm + Dương)
snrs = [-10, -7, -4, -1, 1, 4, 7, 10, 13]
snrs_str = " ".join(map(str, snrs))
methods = ["baseline", "linear", "importance_only", "snr_only", "full"]
labels = ["Baseline", "Linear", "Importance Only", "SNR Only", "Full FIS (Channel-Aware)"]
markers = ['o', 's', '^', 'x', '*']

# Lặp qua 2 kênh: AWGN và Rayleigh (EQ)
channels_config = [
    {"name": "AWGN", "tag": "", "eq_flag": ""},
    {"name": "Rayleigh", "tag": "eq_", "eq_flag": "--rayleigh_equalize"}
]

for ch in channels_config:
    c_name = ch["name"]
    tag = ch["tag"]
    eq_flag = ch["eq_flag"]
    
    print(f"\n{'█'*60}")
    print(f" 🚀 ĐANG XỬ LÝ KÊNH: {c_name.upper()} (DẢI -10 ĐẾN 13 dB)")
    print(f"{'█'*60}")

    # Khai báo đường dẫn tạ (Checkpoint)
    base_ckpt = f"exp_ctx/ckpts_{tag}baseline_{c_name}/baseline_best.pth"
    map_file = f".tmp_fis_map_{c_name}.json"
    
    fis_map = {
        "linear": f"exp_ctx/ckpts_{tag}linear_{c_name}_round2/fis_power_best.pth",
        "importance_only": f"exp_ctx/ckpts_{tag}importance_only_{c_name}_round2/fis_power_best.pth",
        "snr_only": f"exp_ctx/ckpts_{tag}snr_only_{c_name}_round2/fis_power_best.pth",
        "full": f"exp_ctx/ckpts_{tag}full_{c_name}_round2/fis_power_best.pth"
    }
    
    with open(map_file, "w") as f:
        json.dump(fis_map, f)

    save_dir = f"paper_sims_FULL_RANGE_{c_name}"

    # Chạy chấm điểm 5 model
    cmd = (
        f"python run_paper_sims.py "
        f"--baseline_ckpt {base_ckpt} "
        f"--fis_ckpt_map_json {map_file} "
        f"--channel {c_name} {eq_flag} "
        f"--ratio 0.1667 "
        f"--snrs {snrs_str} "
        f"--modes baseline,linear,importance_only,snr_only,full "
        f"--dataset cifar10 --image_size 32 --batch_size 128 "
        f"--save_dir {save_dir}"
    )
    os.system(cmd)
    if os.path.exists(map_file): os.remove(map_file)

    # Đọc dữ liệu và vẽ biểu đồ PSNR
    result_file = f"{save_dir}/paper_sims_results.json"
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
            
        ratio_key = list(data['results'].keys())[0]
        results = data['results'][ratio_key]

        plt.figure(figsize=(10, 6)) # Mở rộng bề ngang ảnh ra một chút vì dải SNR dài hơn
        
        for idx, m in enumerate(methods):
            y_psnr = []
            x_snrs = []
            for snr in snrs:
                json_key = f"{float(snr)}"
                if json_key in results and m in results[json_key]:
                    x_snrs.append(snr)
                    y_psnr.append(results[json_key][m]["psnr"])
            
            # Vẽ từng đường
            plt.plot(x_snrs, y_psnr, marker=markers[idx], linestyle='-', linewidth=2, markersize=7, label=labels[idx])

        # Trang trí biểu đồ chuẩn báo cáo
        plt.title(f"PSNR vs SNR under {c_name} Channel (Full Range)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("SNR (dB)", fontsize=12, fontweight='bold')
        plt.ylabel("PSNR (dB)", fontsize=12, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(snrs)
        plt.legend(fontsize=10)

        # Lưu ảnh
        img_name = f"Chart_5_Methods_FullRange_{c_name}.png"
        plt.savefig(img_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Đã vẽ xong biểu đồ: {img_name}")

    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ kênh {c_name}: {e}")

print("\n🎉 HOÀN THÀNH TẤT CẢ! 2 BIỂU ĐỒ TOÀN DẢI ĐÃ SẴN SÀNG!")