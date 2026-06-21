# Hướng Dẫn Chạy Báo Cáo Ablation Study

## 1. Tổng Quan Quy Trình

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│   INFERENCE     │ ──▶ │  DIAGNOSTIC         │ ──▶ │  GENERATE REPORT │
│  (test_jscc.py) │     │ (diagnose_controller)│     │(generate_report) │
└─────────────────┘     └─────────────────────┘     └──────────────────┘
```

## 2. Các Script Chính

| Script                   | Chức năng                      | Input              | Output                           |
| ------------------------ | ------------------------------ | ------------------ | -------------------------------- |
| `run_paper_sims.py`      | Chạy inference nhiều SNR/modes | Checkpoints `.pth` | Ảnh truyền + JSON                |
| `diagnose_controller.py` | Phân tích FIS controller       | Ảnh truyền         | `diagnostics.json` + plots       |
| `generate_report.py`     | Tạo báo cáo tổng hợp           | `diagnostics.json` | `report_logs/` + `report_plots/` |

## 3. Cấu Trúc Thư Mục

```
fullfis2/
├── ckpts_*/                    # Checkpoints (cần có file .pth)
│   └── fis_power_best.pth
├── exp_ctx/                    # Thống kê từ training
│   └── ckpts_*/
│       └── control_stats_best.json
├── diag_*/                     # Kết quả diagnostic
│   ├── diagnostics.json         # Dữ liệu phân tích
│   ├── *_metrics_log.txt       # Metrics log
│   └── *.png                    # Hình ảnh diagnostic
├── report_logs*/                # Báo cáo text
│   ├── full_log.txt
│   └── data.csv
└── report_plots*/               # Báo cáo hình ảnh
    ├── sigma_A_vs_SNR.png
    ├── E_top20_vs_SNR.png
    ├── H_rule_vs_SNR.png
    ├── corr_A_gamma_eff_vs_SNR.png
    ├── corr_A_Importance_vs_SNR.png
    └── ablation_comparison.png
```

## 4. Các Kênh Cần Chạy

| Kênh           | Folder Diagnostic                       | SNR cần chạy    |
| -------------- | --------------------------------------- | --------------- |
| AWGN           | `diag_awgn_snr{1,4,7,10,13}`            | 1, 4, 7, 10, 13 |
| Rayleigh No-EQ | `diag_noeq_snr{1,4,7,10,13}` + Baseline | 1, 4, 7, 10, 13 |
| Rayleigh EQ    | `diag_eq_snr{1,4,7,10,13}` Baseline     | 1, 4, 7, 10, 13 |

## 5. Bước 1: Chạy Inference (Tạo Ảnh Truyền)

### 5.1 AWGN Channel

```bash
python run_paper_sims.py \
    --baseline_ckpt exp_ctx/ckpts_baseline_AWGN/fis_power_best.pth \
    --fis_ckpt_root exp_ctx \
    --channel AWGN \
    --snrs 1 4 7 10 13 \
    --modes baseline,linear,importance_only,snr_only,full \
    --save_dir paper_sims_AWGN
```

### 5.2 Rayleigh No-Equalizer

```bash
python run_paper_sims.py \
    --baseline_ckpt exp_ctx/ckpts_baseline_Rayleigh/fis_power_best.pth \
    --fis_ckpt_root exp_ctx \
    --channel Rayleigh \
    --snrs 1 4 7 10 13 \
    --modes baseline,linear,importance_only,snr_only,full \
    --save_dir paper_sims_noeq
```

### 5.3 Rayleigh Equalizer

```bash
python run_paper_sims.py \
    --baseline_ckpt exp_ctx/ckpts_eq_baseline_Rayleigh/fis_power_best.pth \
    --fis_ckpt_root exp_ctx \
    --channel Rayleigh \
    --rayleigh_equalize \
    --snrs 1 4 7 10 13 \
    --modes baseline,linear,importance_only,snr_only,full \
    --save_dir paper_sims_eq
```

## 6. Bước 2: Chạy Diagnostic

### 6.1 AWGN Channel

```bash
# Chạy cho từng SNR
python diagnose_controller.py \
    --linear_ckpt exp_ctx/ckpts_linear_AWGN_round2/fis_power_best.pth \
    --importance_only_ckpt exp_ctx/ckpts_importance_only_AWGN_round2/fis_power_best.pth \
    --snr_only_ckpt exp_ctx/ckpts_snr_only_AWGN_round2/fis_power_best.pth \
    --full_ckpt exp_ctx/ckpts_full_AWGN_round2/fis_power_best.pth \
    --channel AWGN \
    --snr_db 1 \
    --ratio 0.1667 \
    --modes baseline,linear,importance_only,snr_only,full \
    --save_dir diag_awgn_snr1

# Lặp lại cho SNR 4, 7, 10, 13
```

### 6.2 Rayleigh No-EQ

```bash
python diagnose_controller.py \
    --baseline_ckpt exp_ctx/ckpts_noeq_baseline_Rayleigh/fis_power_best.pth \
    --linear_ckpt exp_ctx/ckpts_noeq_linear_Rayleigh_round2/fis_power_best.pth \
    --importance_only_ckpt exp_ctx/ckpts_noeq_importance_only_Rayleigh_round2/fis_power_best.pth \
    --snr_only_ckpt exp_ctx/ckpts_noeq_snr_only_Rayleigh_round2/fis_power_best.pth \
    --full_ckpt exp_ctx/ckpts_noeq_full_Rayleigh_round2/fis_power_best.pth \
    --channel Rayleigh \
    --snr_db 1 \
    --ratio 0.1667 \
    --modes baseline,linear,importance_only,snr_only,full \
    --save_dir diag_noeq_snr1
```

### 6.3 Rayleigh EQ

```bash
python diagnose_controller.py \
    --baseline_ckpt exp_ctx/ckpts_eq_baseline_Rayleigh/fis_power_best.pth \
    --linear_ckpt exp_ctx/ckpts_eq_linear_Rayleigh_round2/fis_power_best.pth \
    --importance_only_ckpt exp_ctx/ckpts_eq_importance_only_Rayleigh_round2/fis_power_best.pth \
    --snr_only_ckpt exp_ctx/ckpts_eq_snr_only_Rayleigh_round2/fis_power_best.pth \
    --full_ckpt exp_ctx/ckpts_eq_full_Rayleigh_round2/fis_power_best.pth \
    --channel Rayleigh \
    --rayleigh_equalize \
    --snr_db 1 \
    --ratio 0.1667 \
    --modes baseline,linear,importance_only,snr_only,full \
    --save_dir diag_eq_snr1
```

## 7. Bước 3: Tạo Báo Cáo

```bash
# Chạy script tạo báo cáo
python generate_report.py
```

### Kết quả tạo ra:

| Folder               | Nội dung                            |
| -------------------- | ----------------------------------- |
| `report_logs/`       | AWGN - Báo cáo text + CSV           |
| `report_logs_noeq/`  | Rayleigh No-EQ - Báo cáo text + CSV |
| `report_logs_eq/`    | Rayleigh EQ - Báo cáo text + CSV    |
| `report_plots/`      | AWGN - Đồ thị PNG                   |
| `report_plots_noeq/` | Rayleigh No-EQ - Đồ thị PNG         |
| `report_plots_eq/`   | Rayleigh EQ - Đồ thị PNG            |

## 8. Script Tự Động (Nếu Có)

```bash
# AWGN
python run_awgn_diag.py

# Rayleigh No-EQ
python run_noeq_diag.py

# Rayleigh EQ
python run_rayleigh_eq_diag.py
```

## 9. Các Metrics Trong Báo Cáo

| Metric                | Mô tả                        | Giá trị mong muốn cho Full-FIS      |
| --------------------- | ---------------------------- | ----------------------------------- |
| **σA** (sigma_A)      | Độ phân tán biên độ          | Cao (phân bổ công suất linh hoạt)   |
| **E_top20**           | Năng lượng tập trung top 20% | Cao (tập trung vào vùng quan trọng) |
| **H_rule**            | Entropy của luật FIS         | Thấp (ít luật được kích hoạt)       |
| **corr_A_gamma_eff**  | Tương quan A vs γ_eff        | Cao (nhận thức kênh)                |
| **corr_A_Importance** | Tương quan A vs Importance   | Cao (ưu tiên vùng quan trọng)       |

## 10. Tình Trạng Hiện Tại (2026-06-21)

### Đã có dữ liệu:

- ✅ `diag_awgn_snr1/` - 5 modes
- ✅ `diag_noeq_snr{1,4,7,10,13}/` - 4 modes (thiếu baseline)
- ✅ `diag_eq_snr{1,4,7,10,13}/` - 4 modes (thiếu baseline)

### Cần thêm:

- ❌ AWGN SNR 4, 7, 10, 13
- ❌ Baseline cho Rayleigh No-EQ
- ❌ Baseline cho Rayleigh EQ
- ❌ Checkpoints `.pth` (hiện chưa tìm thấy trong workspace)

## 11. Xem Kết Quả

```bash
# Mở file log
notepad report_logs/full_log.txt
notepad report_logs_noeq/full_log.txt
notepad report_logs_eq/full_log.txt

# Xem CSV
notepad report_logs/data.csv

# Xem plots
explorer report_plots/
explorer report_plots_noeq/
explorer report_plots_eq/
```

## 12. Lưu Ý Quan Trọng

1. **Đảm bảo công bằng (Fairness)**: Khi chạy Ablation Study, tất cả modes phải dùng chung:
   - Cùng ảnh đầu vào (batch)
   - Cùng điều kiện nhiễu (`_pending_fading`)
   - Cùng SNR

2. **Baseline**: Mode Baseline không có FIS, chỉ để so sánh PSNR/SSIM gốc

3. **SNR-Only**: Mode đặc biệt - dùng backbone của baseline nhưng không có FIS (A=1)
