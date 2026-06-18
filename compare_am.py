import os
import json

# Tạo file map JSON tạm thời để trỏ đường dẫn mô hình Full
map_file = ".tmp_fis_map_am.json"
with open(map_file, "w") as f:
    json.dump({
        "full": "exp_ctx/ckpts_full_AWGN_round2/fis_power_best.pth"
    }, f)

# Đường dẫn mô hình móng (Baseline)
base_ckpt = "exp_ctx/ckpts_baseline_AWGN/baseline_best.pth"

print("=== ĐÁNH GIÁ ĐIỂM PSNR: FULL vs BASELINE (DẢI ÂM) ===")

# Gọi script chấm điểm cho dải âm -10, -7, -4, -1
cmd = (
    f"python run_paper_sims.py "
    f"--baseline_ckpt {base_ckpt} "
    f"--fis_ckpt_map_json {map_file} "
    f"--channel AWGN "
    f"--ratio 0.1667 "
    f"--snrs -10 -7 -4 -1 "
    f"--modes baseline,full "
    f"--dataset cifar10 --image_size 32 --batch_size 128 "
    f"--save_dir paper_sims_awgn_am"
)

# Chạy lệnh
os.system(cmd)

# Dọn dẹp file rác
if os.path.exists(map_file):
    os.remove(map_file)
    
print("\n✅ ĐÃ CHẤM ĐIỂM XONG! Bạn hãy vào thư mục paper_sims_awgn_am để xem kết quả.")