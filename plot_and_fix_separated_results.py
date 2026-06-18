import os
import json
import matplotlib.pyplot as plt

# ==============================================================================
# 🎯 [QUAN TRỌNG NHẤT] SỬA CẤU HÌNH ĐỂ AI HỌC CÁCH "SINH TỒN" VÀ THÍCH NGHI
# Chúng ta phải bắt AI học ở cả dải nhiễu âm khắc nghiệt để kích hoạt module SNR
# ==============================================================================
SNR_MIN = -10
SNR_MAX = 13
TRAIN_SNR_LIST = "-10 -7 -4 -1 1 4 7 10 13" # Phải có dải âm!
EVAL_SNR_LIST = "-10 -7 -4 -1 1 4 7 10 13"

# Cấu hình tối ưu mà bạn đã fix hôm qua
WARMSTART_EPOCHS = 20
FINETUNE_LR = 0.0001
RATIO = 0.1667
DATASET = "cifar10"
IMAGE_SIZE = 32
BUDGET = 1.0

print("\n" + "="*80)
print(" 🚀 [KỊCH BẢN CHỐT SỔ] ĐANG CẤU HÌNH LẠI DẢI NHIỄU VÀ FINETUNE MÔ HÌNH")
print("   - Đào tạo tại dải: -10 đến 13 dB (để AI học cách Channel-Aware)")
print("   - Khởi động controller: 20 epochs | Finetune LR: 0.0001")
print("="*80 + "\n")

# ==============================================================================
# 1. TỰ ĐỘNG CHẠY LẠI TRAIN CHO 4 MÔ HÌNH FIS CÒN THIẾU
# ==============================================================================
MODES = ["linear", "importance_only", "snr_only", "full"]
CHANNELS = ["AWGN", "Rayleigh"] # Chúng ta sẽ làm cho cả 2 kênh

TOTAL_JOBS = 4 * len(CHANNELS)
DONE = 0

for channel in CHANNELS:
    # 1a. Train Baseline trước (móng cố định cho Round 2)
    base_dir = f"exp_ctx/ckpts_baseline_{channel}"
    base_ckpt = f"{base_dir}/baseline_best.pth"
    # cờ Rayleigh EQ nếu cần
    eq_flag = "--rayleigh_equalize" if channel == "Rayleigh" else ""
    tag = "eq_" if channel == "Rayleigh" else ""

    if not os.path.exists(base_ckpt):
        print(f"\n🔰 Train Baseline {channel} (Dải mới)...")
        cmd_base = (
            f"python train_baseline.py \
            --dataset {DATASET} --image_size {IMAGE_SIZE} \
            --channel {channel} \
            --snr_min {SNR_MIN} --snr_max {SNR_MAX} \
            --eval_snr_list {EVAL_SNR_LIST} \
            --save_dir {base_dir} {eq_flag}"
        )
        os.system(cmd_base)

    # 1b. Train nốt 4 mô hình FIS của Round 2 với warm-start
    for mode in MODES:
        fis_dir = f"exp_ctx/ckpts_{tag}{mode}_{channel}_round2"
        fis_ckpt = f"{fis_dir}/fis_power_best.pth"
        if os.path.exists(fis_ckpt):
            print(f"⏭️ Bỏ qua {mode.upper()} {channel} vì đã có tạ!")
            continue 

        DONE += 1
        print(f"\n🔧 [{DONE}/{TOTAL_JOBS}] Đang train {mode.upper()} {channel} (Warm-start, Finetune)...")

        cmd_fis = (
            f"python train_fis_power.py \
            --dataset {DATASET} --image_size {IMAGE_SIZE} \
            --channel {channel} --mode {mode} --budget {BUDGET} \
            --snr_min {SNR_MIN} --snr_max {SNR_MAX} \
            --train_snr_list {TRAIN_SNR_LIST} --eval_snr_list {EVAL_SNR_LIST} \
            --save_dir {fis_dir} \
            --baseline_ckpt {base_ckpt} \
            --warmstart_controller_only_epochs {WARMSTART_EPOCHS} \
            --finetune_lr {FINETUNE_LR} \
            {eq_flag}"
        )
        os.system(cmd_fis)

print("\n" + "="*80)
print(" ✅ ĐÃ HOÀN THÀNH TRAIN ĐỦ 5 MÔ HÌNH! ĐANG CHUYỂN SANG VẼ BIỂU ĐỒ")
print("="*80 + "\n")


# ==============================================================================
# 2. TỰ ĐỘNG CHẤM ĐIỂM VÀ VẼ BIỂU ĐỒ 5 ĐƯỜNG TÁCH TOP TUYỆT ĐẸP
# ==============================================================================
plot_methods = ["baseline", "linear", "importance_only", "snr_only", "full"]
labels = ["Baseline", "Linear", "Importance Only", "SNR Only", "Full FIS (Channel-Aware)"]
markers = ['o', 's', '^', 'x', '*']

for channel in CHANNELS:
    tag = "eq_" if channel == "Rayleigh" else ""
    eq_flag = "--rayleigh_equalize" if channel == "Rayleigh" else ""
    
    print(f"\n--- 📊 Đang xử lý chấm điểm và vẽ biểu đồ kênh: {channel} ---")

    # 2a. Khai báo checkpoint
    base_ckpt = f"exp_ctx/ckpts_baseline_{channel}/baseline_best.pth"
    map_file = f".tmp_fis_map_{channel}.json"
    
    fis_map = {
        "linear": f"exp_ctx/ckpts_{tag}linear_{channel}_round2/fis_power_best.pth",
        "importance_only": f"exp_ctx/ckpts_{tag}importance_only_{channel}_round2/fis_power_best.pth",
        "snr_only": f"exp_ctx/ckpts_{tag}snr_only_{channel}_round2/fis_power_best.pth",
        "full": f"exp_ctx/ckpts_{tag}full_{channel}_round2/fis_power_best.pth"
    }
    
    with open(map_file, "w") as f:
        json.dump(fis_map, f)

    save_dir = f"paper_sims_FIXED_{channel}"

    # 2b. Chạy chấm điểm 5 model
    cmd_sim = (
        f"python run_paper_sims.py \
        --baseline_ckpt {base_ckpt} \
        --fis_ckpt_map_json {map_file} \
        --channel {channel} {eq_flag} \
        --ratio {RATIO} \
        --snrs {TRAIN_SNR_LIST} \
        --modes baseline,linear,importance_only,snr_only,full \
        --dataset {DATASET} --image_size {IMAGE_SIZE} \
        --batch_size 128 \
        --save_dir {save_dir}"
    )
    os.system(cmd_sim)
    if os.path.exists(map_file): os.remove(map_file)

    # 2c. Đọc dữ liệu và vẽ biểu đồ PSNR
    result_file = f"{save_dir}/paper_sims_results.json"
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
            
        ratio_key = list(data['results'].keys())[0]
        results = data['results'][ratio_key]

        plt.figure(figsize=(10, 6))
        
        # Sắp xếp lại SNR dB để vẽ đường
        snrs_list = list(results.keys())
        snrs_list = sorted([float(s) for s in snrs_list])

        for idx, m in enumerate(plot_methods):
            y_psnr = []
            x_snrs = []
            for snr in snrs_list:
                json_key = f"{float(snr)}"
                if json_key in results and m in results[json_key]:
                    x_snrs.append(snr)
                    y_psnr.append(results[json_key][m]["psnr"])
            
            # Vẽ từng đường
            plt.plot(x_snrs, y_psnr, marker=markers[idx], linestyle='-', linewidth=2, markersize=7, label=labels[idx])

        # Trang trí biểu đồ chuẩn báo cáo
        plt.title(f"PSNR vs SNR under {channel} Channel", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("SNR (dB)", fontsize=12, fontweight='bold')
        plt.ylabel("PSNR (dB)", fontsize=12, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(snrs_list)
        plt.legend(fontsize=10)

        # Lưu ảnh
        img_name = f"Chart_5_Methods_FIXED_{channel}.png"
        plt.savefig(img_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🎉 Đã vẽ xong biểu đồ chuẩn: {img_name}")

    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ kênh {channel}: {e}")

print("\n" + "="*80)
print(" ✅ HOÀN THÀNH TẤT CẢ! 2 TẤM ẢNH BỨT TÓP ĐÃ SẴN SÀNG!")
print("   - Mở ngay 2 file ảnh trong thư mục để tận hưởng thành quả nhé!")
print("="*80 + "\n")