# Ablation Study - Diagnostic & Report Pipeline

Huong dan chay diagnostic cho ngien cuu khoa hoc Deep JSCC voi bo dieu khien FIS.

## Cau truc Du An

```
fullfis2/
├── diagnose_controller.py    # Script chay diagnostic cho tung cau hinh
├── generate_report.py       # Script tao bao cao tu diagnostics
├── run_all_diag.py          # Script chay tat ca diagnostics (AWGN + Rayleigh)
├── exp_ctx/                 # Checkpoint models
│   ├── ckpts_linear_AWGN_round2/
│   ├── ckpts_importance_only_AWGN_round2/
│   ├── ckpts_snr_only_AWGN_round2/
│   ├── ckpts_full_AWGN_round2/
│   ├── ckpts_*_Rayleigh_round2/       # Rayleigh variants
│   └── ckpts_*_Rayleigh/              # Baseline models
├── diag_awgn_snr*/        # Diagnostic output folders (auto-created)
├── diag_noeq_snr*/        # Rayleigh No-EQ diagnostics
├── diag_eq_snr*/          # Rayleigh EQ diagnostics
├── report_plots/          # AWGN plots
├── report_logs/           # AWGN reports (txt + csv)
├── report_plots_noeq/     # Rayleigh No-EQ plots
├── report_logs_noeq/      # Rayleigh No-EQ reports
├── report_plots_eq/       # Rayleigh EQ plots
└── report_logs_eq/        # Rayleigh EQ reports
```

## Che Do Ablation

| Mode | Mo ta |
|------|-------|
| **Baseline** | Model khong co FIS controller |
| **Linear** | Chi su dung quan he tuyen tinh A = f(SNR) |
| **SNR-Only** | Chi su dung dau vao SNR |
| **Importance-Only** | Chi su dung dau vao Importance |
| **Full-FIS** | Tat ca dau vao (SNR + Importance + Channel) |

## Cach Chay

### 1. Chay Tat Ca Diagnostics (Khuyen nghi)

```bash
python run_all_diag.py
```

Script nay se tu dong:
- Chay AWGN: SNR = 1, 4, 7, 10, 13 dB
- Chay Rayleigh No-EQ: SNR = 1, 4, 7, 10, 13 dB
- Chay Rayleigh EQ: SNR = 1, 4, 7, 10, 13 dB
- Tao folder `diag_*/snr*/` voi diagnostics.json va metrics

**Luu y:** Lan chay tiep theo se skip cac folder da co san. Xoa folder `diag_*/snr*` neu muon chay lai.

### 2. Chay Diagnostic Don Le

```bash
# AWGN SNR = 10
python diagnose_controller.py \
    --channel AWGN \
    --snr_db 10 \
    --save_dir diag_awgn_snr10 \
    --ratio 0.1667 \
    --modes baseline,linear,importance_only,snr_only,full \
    --baseline_ckpt exp_ctx/ckpts_baseline_AWGN/baseline_best.pth \
    --linear_ckpt exp_ctx/ckpts_linear_AWGN_round2/fis_power_best.pth \
    --importance_only_ckpt exp_ctx/ckpts_importance_only_AWGN_round2/fis_power_best.pth \
    --snr_only_ckpt exp_ctx/ckpts_snr_only_AWGN_round2/fis_power_best.pth \
    --full_ckpt exp_ctx/ckpts_full_AWGN_round2/fis_power_best.pth

# Rayleigh No-EQ SNR = 7
python diagnose_controller.py \
    --channel Rayleigh \
    --snr_db 7 \
    --save_dir diag_noeq_snr7 \
    --ratio 0.1667 \
    --modes baseline,linear,importance_only,snr_only,full \
    --baseline_ckpt exp_ctx/ckpts_noeq_baseline_Rayleigh/baseline_best.pth \
    --linear_ckpt exp_ctx/ckpts_noeq_linear_Rayleigh_round2/fis_power_best.pth \
    --importance_only_ckpt exp_ctx/ckpts_noeq_importance_only_Rayleigh_round2/fis_power_best.pth \
    --snr_only_ckpt exp_ctx/ckpts_noeq_snr_only_Rayleigh_round2/fis_power_best.pth \
    --full_ckpt exp_ctx/ckpts_noeq_full_Rayleigh_round2/fis_power_best.pth

# Rayleigh EQ SNR = 10
python diagnose_controller.py \
    --channel Rayleigh \
    --snr_db 10 \
    --save_dir diag_eq_snr10 \
    --ratio 0.1667 \
    --rayleigh_equalize \
    --modes baseline,linear,importance_only,snr_only,full \
    --baseline_ckpt exp_ctx/ckpts_eq_baseline_Rayleigh/baseline_best.pth \
    --linear_ckpt exp_ctx/ckpts_eq_linear_Rayleigh_round2/fis_power_best.pth \
    --importance_only_ckpt exp_ctx/ckpts_eq_importance_only_Rayleigh_round2/fis_power_best.pth \
    --snr_only_ckpt exp_ctx/ckpts_eq_snr_only_Rayleigh_round2/fis_power_best.pth \
    --full_ckpt exp_ctx/ckpts_eq_full_Rayleigh_round2/fis_power_best.pth
```

### 3. Tao Bao Cao

```bash
python generate_report.py
```

Script nay tu dong:
- Doc tat ca folder `diag_*/snr*/`
- Tao plots vao `report_plots*/`
- Tao bao cao txt va csv vao `report_logs*/`

## Thu Tu Chay Duoc De Xuat

```bash
# Buoc 1: Chay tat ca diagnostics
python run_all_diag.py

# Buoc 2: Tao bao cao
python generate_report.py
```

## Cac Metrics Duoc Do

| Metric | Mo ta |
|--------|-------|
| **sigma_A** | Do phan tan bien do (Amplitude Dispersion) |
| **E_top20** | Nang luong tap trung trong top 20% pixel quan trong |
| **H_rule** | Entropy cua luat FIS |
| **corr_A_gamma_eff** | Tuong quan giua A va gamma_eff |
| **corr_A_Importance** | Tuong quan giua A va Importance |

## Xoa va Chay Lai

```bash
# Xoa tat ca diagnostic folders
python -c "import shutil, os; [shutil.rmtree(d) for d in ['diag_awgn_snr1','diag_awgn_snr4','diag_awgn_snr7','diag_awgn_snr10','diag_awgn_snr13','diag_noeq_snr1','diag_noeq_snr4','diag_noeq_snr7','diag_noeq_snr10','diag_noeq_snr13','diag_eq_snr1','diag_eq_snr4','diag_eq_snr7','diag_eq_snr10','diag_eq_snr13'] if os.path.exists(d)]"

# Xoa tat ca report folders
python -c "import shutil, os; [shutil.rmtree(d) if os.path.exists(d) else None for d in ['report_plots','report_logs','report_plots_noeq','report_logs_noeq','report_plots_eq','report_logs_eq']]"

# Chay lai tu dau
python run_all_diag.py
python generate_report.py
```

## Cac File Dau Ra

### Diagnostic Folders (diag_*/snr*/)
- `diagnostics.json` - Du lieu JSON day du
- `*_metrics_log.txt` - Chi so ky thuat cho tung mode
- `*_I_sample*.png` - Importance map
- `*_A_sample*.png` - Power allocation map
- `*_Ez*_sample*.png` - Energy maps
- `*_channelRel_sample*.png` - Channel reliability map
- `*_fis_diagnostic_plots.png` - Diagnostic plots

### Report Folders (report_*/)
- `full_log.txt` - Bao cao text day du
- `data.csv` - Du lieu CSV (co the import vao Excel)
- `sigma_A_vs_SNR.png` - Do phan tan theo SNR
- `E_top20_vs_SNR.png` - Nang luong tap trung theo SNR
- `H_rule_vs_SNR.png` - Entropy theo SNR
- `corr_A_gamma_eff_vs_SNR.png` - Tuong quan A vs gamma
- `corr_A_Importance_vs_SNR.png` - Tuong quan A vs Importance
- `ablation_comparison.png` - Tat ca metrics tren 1 plot

## Ghi Chu

- **Baseline**: Khong co FIS controller, nen cac chi so FIS (sigma_A, E_top20, H_rule) se la 0 hoac gia tri mac dinh
- **SNR-Only**: Co sigma_A = 0 vi khong co spatial variation (A = const)
- **Full-FIS**: Chi so tot nhat, thể hiện sự cân bằng giữa tất cả các yếu tố
