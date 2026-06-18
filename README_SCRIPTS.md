1. Nhóm Kịch bản Chính (Toàn vẹn hệ thống)

- run_all.sh: (Trùm cuối) Đây là file kịch bản tự động hóa từ A-Z. Nó làm nhiệm vụ: Train Baseline $\rightarrow$ Train ép xung FIS $\rightarrow$ Diagnose cơ bản $\rightarrow$ Chấm điểm PSNR dải dương (1 đến 13 dB) cho cả 3 kênh.
- Update mới nhất: Đã fix chuẩn thông số warmstart_controller_only_epochs 20, finetune_lr 0.0001 và trỏ đúng đường dẫn lưu tạ có đuôi \_round2.

-2. Nhóm Đo lường Năng lượng (Dải sóng khỏe: 1 đến 13 dB)
-get_sigma.py: File dùng để chạy lệnh Diagnose ngầm, ép mô hình "nhả" số liệu phân bổ năng lượng (để lấy thông số A_std / $\sigma_A$) tại các mốc SNR dương, lưu vào file JSON.

-draw_sigma.py: File vẽ đồ thị. Nó sẽ đọc file JSON sinh ra từ bước trên và vẽ thành biểu đồ đường cong $\sigma_A$ trực quan để nộp báo cáo.

- 3. Nhóm "Thử lửa" Môi trường Khắc nghiệt (Dải âm cực độ: -10 đến -1 dB)

-run_negative_snr.sh (hoặc run_am.py): File dùng để chạy Diagnose trích xuất số liệu $\sigma_A$ ở dải nhiễu âm cho kênh AWGN.

-Update mới nhất: Đã chèn cứng tham số --ratio 0.1667 để fix lỗi size mismatch do lệch tỷ lệ nén.

-compare_am.py: File "Trọng tài" kênh AWGN. Chạy module run_paper_sims.py để chấm điểm PSNR/SSIM, so sánh trực diện giữa bản Full và Baseline ở dải âm. (Chính là file chứng minh Full thắng đậm).

-run_rayleigh_am.py: File "2 trong 1" mới nhất dành cho kênh Rayleigh (No-EQ). Nó gộp cả 2 nhiệm vụ: vừa đo $\sigma_A$, vừa chấm điểm PSNR/SSIM ở dải âm cực độ chỉ trong một lần chạy.

- export_full_paper.py : full kết quả của cả âm and dương cho kênh AWGN

- print_table.py : in bảng kết quả kết quả âm và dương cho kênh AWGN

- plot_full_range.py : in ra kết quả của cả 2 kênh lẫn snrs dương và âm vẽ ra biểu đồ cả 5 model
- plot_and_fix_separated_results.py : train tham số tối ưu mô hình và xuất ảnh kết quả
