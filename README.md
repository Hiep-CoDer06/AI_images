# Deep JSCC với Full FIS Controller (Round 2)

Implementation của Deep Joint Source-Channel Coding (JSCC) kết hợp bộ điều khiển Logic Mờ (FIS) 2 lớp cho việc truyền tải ảnh không dây.

## Kiến trúc

Kiến trúc Deep JSCC với FIS Controller:
- **Encoder**: CNN-based encoder cho ảnh đầu vào
- **Channel**: AWGN, Rayleigh No-EQ, Rayleigh EQ
- **FIS Controller**: Bộ điều khiển Fuzzy Inference System 2 lớp
  - Layer 1: Spatial Controller (điều khiển công suất theo vị trí)
  - Layer 2: Channel-Aware Controller (nhận biết trạng thái kênh truyền)

## Cấu trúc thư mục

```
fullfis2/
├── model.py           # Mô hình Deep JSCC (Encoder/Decoder)
├── channel.py         # Các loại kênh truyền (AWGN, Rayleigh)
├── fis_modules.py     # Module FIS Controller
├── utils.py           # Hàm utility
├── dataset.py         # Dataset loading (CIFAR-10)
├── diagnose_controller.py  # Diagnostic script
├── run_paper_sims.py  # Paper simulations
├── plot_results.py    # Vẽ biểu đồ kết quả
├── export_latex_and_plot.py # Export LaTeX tables
├── fix_rayleigh_only.py # Training script chính
└── run_all.sh         # Script chạy toàn bộ pipeline
```

## Cách sử dụng

### 1. Chạy toàn bộ pipeline

```bash
# Chạy tất cả 3 kênh
bash run_all.sh

# Chỉ train
bash run_all.sh --train-only

# Chỉ diagnose
bash run_all.sh --diag-only

# Chỉ paper sims
bash run_all.sh --sims-only
```

### 2. Chạy theo kênh cụ thể

```bash
# Chỉ AWGN
bash run_all.sh --awgn

# Chỉ Rayleigh No-Equalize
bash run_all.sh --noeq

# Chỉ Rayleigh Equalize
bash run_all.sh --eq

# Cả 2 Rayleigh
bash run_all.sh --rayleigh
```

### 3. Chế độ nhanh (test)

```bash
# Chỉ SNR: 1, 7, 13 dB
bash run_all.sh --fast
```

## Ablation Study

5 model variants được huấn luyện:
- **Baseline**: Không có FIS Controller
- **Linear**: FIS với hệ số tuyến tính
- **Importance**: FIS với độ quan trọng
- **SNR-Only**: FIS chỉ dùng SNR
- **Full-Ctx**: FIS đầy đủ (2 lớp)

## SNR Range

```
SNR: -10, -7, -4, -1, 1, 4, 7, 10, 13 dB
```

## Yêu cầu

- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib
- CIFAR-10 dataset (tự động download)

## License

MIT License
