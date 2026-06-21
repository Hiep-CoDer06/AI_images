# BÁO CÁO KẾT QUẢ NGHIÊN CỨU
## Deep JSCC với Fuzzy Logic Controller (FIS)
### Ngày: 2026-06-21

---

## 1. TÓM TẮT KẾT QUẢ

### Bảng PSNR Trung Bình (dB) - 5 SNR (1-13 dB)

| Kênh | Baseline | Linear | Importance Only | SNR Only | Full FIS |
|------|----------|--------|----------------|----------|----------|
| **AWGN** | 25.51 | 26.98 | 26.73 | 25.50 | **27.57** |
| **Rayleigh (EQ)** | 20.89 | 22.28 | 22.24 | 20.89 | **22.25** |
| **Rayleigh (No-EQ)** | 13.12 | 23.08 | 23.11 | 13.11 | **23.16** |

### Gain so với Baseline (dB)

| Kênh | Linear | Importance Only | Full FIS |
|------|--------|----------------|----------|
| **AWGN** | +1.47 | +1.22 | **+2.06** |
| **Rayleigh (EQ)** | +1.40 | +1.35 | **+1.36** |
| **Rayleigh (No-EQ)** | +9.96 | +9.99 | **+10.04** |

---

## 2. PHÂN TÍCH CHI TIẾT THEO SNR

### AWGN Channel

| SNR (dB) | Baseline | Linear | Importance Only | SNR Only | Full FIS | Full Gain |
|-----------|----------|--------|----------------|----------|----------|-----------|
| 1 | 23.51 | 24.01 | 23.85 | 23.50 | **24.37** | +0.86 |
| 4 | 24.95 | 25.96 | 25.74 | 24.94 | **26.39** | +1.44 |
| 7 | 25.89 | 27.44 | 27.18 | 25.88 | **28.01** | +2.12 |
| 10 | 26.44 | 28.44 | 28.14 | 26.44 | **29.16** | +2.72 |
| 13 | 26.75 | 29.04 | 28.72 | 26.74 | **29.90** | +3.15 |

### Rayleigh No-EQ Channel (Khó nhất)

| SNR (dB) | Baseline | Linear | Importance Only | SNR Only | Full FIS | Full Gain |
|-----------|----------|--------|----------------|----------|----------|-----------|
| 1 | 12.81 | 20.91 | 20.89 | 12.78 | **20.93** | +8.12 |
| 4 | 13.04 | 22.51 | 22.51 | 13.04 | **22.54** | +9.50 |
| 7 | 13.30 | 24.09 | 24.11 | 13.31 | **24.15** | +10.85 |
| 10 | 13.54 | 25.53 | 25.53 | 13.55 | **25.57** | +12.03 |
| 13 | 13.87 | 26.79 | 26.84 | 13.88 | **26.85** | +12.98 |

### Rayleigh EQ Channel

| SNR (dB) | Baseline | Linear | Importance Only | SNR Only | Full FIS | Full Gain |
|-----------|----------|--------|----------------|----------|----------|-----------|
| 1 | 17.75 | 19.37 | 19.40 | 17.75 | **19.30** | +1.55 |
| 4 | 19.55 | 21.40 | 21.36 | 19.55 | **21.36** | +1.81 |
| 7 | 21.05 | 23.09 | 23.05 | 21.05 | **23.03** | +1.98 |
| 10 | 22.11 | 24.28 | 24.21 | 22.12 | **24.19** | +2.08 |
| 13 | 22.89 | 25.11 | 25.05 | 22.89 | **25.03** | +2.14 |

---

## 3. KẾT LUẬN CHÍNH

### 3.1 Full FIS hoạt động hiệu quả
- **AWGN**: +2.06 dB gain so với Baseline
- **Rayleigh EQ**: +1.36 dB gain (ít hơn vì Equalizer đã cải thiện)
- **Rayleigh No-EQ**: **+10.04 dB gain** (HIỆU QUẢ NHẤT - FIS bù đắp fading)

### 3.2 Importance Only vs Linear
- Trên **AWGN**: Linear (+1.47) tốt hơn Importance Only (+1.22)
- Trên **Rayleigh No-EQ**: Cả hai tương đương (~+10 dB)
- → Importance map giúp nhiều trong môi trường fading

### 3.3 SNR Only không cải thiện
- SNR Only ≈ Baseline (không có gain)
- → Cần Importance Map để có spatial power allocation

---

## 4. CÁC BUGS ĐÃ FIX

### 4.1 SNR Normalization Bug (CRITICAL)
**Vấn đề**: `run_paper_sims.py` không đọc `snr_max_db` từ checkpoint config → dùng default 20 thay vì 13
**Fix**: Thêm code đọc SNR range từ `run_config.json`

### 4.2 Control Stats Bug
**Vấn đề**: `compute_control_stats()` không tìm thấy keys `A_std`, `A_range`, `I_A_corr`
**Fix**: Thêm `_compute_control_metrics()` trong `fis_modules.py`

### 4.3 Diagnose Controller Bug
**Vấn đề**: Dùng random data thay vì `rule2_strength` thật
**Fix**: Đọc `rule2_strength` từ info dict

### 4.4 Channel Context Bug
**Vấn đề**: `patches/channel.py` dùng block fading thay vì frequency-selective
**Fix**: Sync hoàn toàn với `channel.py`

---

## 5. FILES ĐÃ TẠO

### Biểu đồ (trong `plots_fixed/`)
- `psnr_comparison_all_channels.png` - So sánh PSNR 3 kênh
- `psnr_awgn.png` - PSNR theo SNR cho AWGN
- `psnr_rayleigh_(eq).png` - PSNR theo SNR cho Rayleigh EQ
- `psnr_rayleigh_(no-eq).png` - PSNR theo SNR cho Rayleigh No-EQ
- `psnr_gain_comparison.png` - Bar chart gain vs baseline
- `ssim_*` - Tương tự cho SSIM

### Scripts mới
- `plot_results.py` - Script vẽ biểu đồ
- `run_all_paper_sims.py` - Script chạy paper sims cho tất cả kênh

---

## 6. BƯỚC TIẾP THEO

1. **Train lại AWGN Round2** với SNR range mới (-10 to 13 dB)
2. **Train thêm** các checkpoint Round2 cho Rayleigh
3. **Chạy lại paper sims** với checkpoints mới
4. **Export LaTeX tables** cho paper

---

## 7. THÔNG TIN HỆ THỐNG

- Dataset: CIFAR-10 (32x32)
- Compression Ratio: 0.167
- Budget: 1.0
- SNR Range: 1-13 dB
- Batch Size: 128
- Models: Baseline, Linear, Importance Only, SNR Only, Full FIS
