import os
import json

# Khai báo gộp toàn bộ dải SNR cực độ và dải thông thường
snrs = [-10, -7, -4, -1, 1, 4, 7, 10, 13]
snrs_str = " ".join(map(str, snrs))

# Đường dẫn tạ (Checkpoint) - Kênh AWGN
channel = "AWGN"
base_ckpt = f"exp_ctx/ckpts_baseline_{channel}/baseline_best.pth"
full_ckpt = f"exp_ctx/ckpts_full_{channel}_round2/fis_power_best.pth"
save_dir = f"paper_sims_FULL_TABLE_{channel.lower()}"

print(f"████████████████████████████████████████████████████████")
print(f" 🚀 ĐANG CHẠY CHẤM ĐIỂM TOÀN DẢI {channel} (-10 đến 13 dB)")
print(f"████████████████████████████████████████████████████████")

# Tạo file cấu hình tạm
map_file = ".tmp_fis_map_full.json"
with open(map_file, "w") as f:
    json.dump({"full": full_ckpt}, f)

# Lệnh chạy chấm điểm toàn tập
cmd = (
    f"python run_paper_sims.py "
    f"--baseline_ckpt {base_ckpt} "
    f"--fis_ckpt_map_json {map_file} "
    f"--channel {channel} "
    f"--ratio 0.1667 "
    f"--snrs {snrs_str} "
    f"--modes baseline,full "
    f"--dataset cifar10 --image_size 32 --batch_size 128 "
    f"--save_dir {save_dir}"
)

# Kích hoạt chạy
os.system(cmd)

# Dọn dẹp
if os.path.exists(map_file): os.remove(map_file)
print(f"\n✅ ĐÃ HOÀN THÀNH! Kết quả đã được lưu tại thư mục: {save_dir}")