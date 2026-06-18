import json
import csv

# Trỏ vào file kết quả JSON bạn đã chạy
file_path = "paper_sims_FULL_TABLE_awgn/paper_sims_results.json" 
csv_output = "Ket_qua_Full_AWGN.csv"

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    # Lấy dữ liệu
    ratio_key = list(data['results'].keys())[0]
    results = data['results'][ratio_key]
    snrs = [-10.0, -7.0, -4.0, -1.0, 1.0, 4.0, 7.0, 10.0, 13.0]

    # Tạo và ghi vào file CSV
    with open(csv_output, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        
        # Ghi Header (Tiêu đề các cột)
        writer.writerow(["SNR (dB)", "Baseline Loss (MSE)", "Full FIS Loss (MSE)", 
                         "Baseline PSNR (dB)", "Full FIS PSNR (dB)", 
                         "Baseline SSIM", "Full FIS SSIM"])

        # Ghi từng dòng dữ liệu
        for snr in snrs:
            snr_str = str(snr)
            if snr_str in results:
                base_psnr = results[snr_str]["baseline"]["psnr"]
                full_psnr = results[snr_str]["full"]["psnr"]
                base_ssim = results[snr_str]["baseline"]["ssim"]
                full_ssim = results[snr_str]["full"]["ssim"]
                
                # Tính MSE (Loss)
                base_mse = 10 ** (-base_psnr / 10)
                full_mse = 10 ** (-full_psnr / 10)
                
                # Làm tròn và ghi vào file
                writer.writerow([
                    snr, 
                    round(base_mse, 5), round(full_mse, 5), 
                    round(base_psnr, 2), round(full_psnr, 2), 
                    round(base_ssim, 4), round(full_ssim, 4)
                ])

    print(f"\n✅ Đã tạo thành công file: {csv_output}")

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file {file_path}. Hãy kiểm tra lại.")