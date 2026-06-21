# Deep JSCC với Full FIS Controller (Round 2)

Implementation của Deep Joint Source-Channel Coding (JSCC) kết hợp bộ điều khiển Logic Mờ (FIS) 2 lớp cho việc truyền tải ảnh không dây.

# Deep JSCC

This implements training of Deep JSCC models for wireless image transmission as described in the paper [Deep Joint Source-Channel Coding for Wireless Image Transmission](https://ieeexplore.ieee.org/abstract/document/8723589) by Pytorch. And there has been a [Tensorflow and keras implementations ](https://github.com/irdanish11/DJSCC-for-Wireless-Image-Transmission).

This is my first time to use PyTorch and git to reproduce a paper, so there may be some mistakes. If you find any, please let me know. Thanks!

## Update-2024.06.04

- modify the `train.py` to omit most of the args in command line, you can just use `python train.py --dataset ${dataset_name}` to train the model.
- add tensorboard to record the results in exp.
- add the `visualization/` file to visualize the result.
- add bash file to run the code in parallel.

## Architecture

<div style="text-align: center;">
    <img src="./demo/arc.png" alt="Image 1" style="width: 500px; height: 250px;">
</div>

## Demo

The model trained on cifar10 which is 32\*32 but test on kodim which is 768\*512 (top); and the model trained on imagenet which is resized to 128\*128 but test on kodim which is 768\*512 (bottom).

<div style="display: flex;">
  <img src="./demo/cifar10_kodim08.png" alt="Image 1" style="flex: 1; max-width: 48%; height: auto;">
  <div style="width: 5px;"></div> 
  <img src="./demo/imagenet_kodim08.png" alt="Image 2" style="flex: 1; max-width: 48%; height: auto;">
</div>

## Installation

conda or other virtual environment is recommended.

```
git clone https://github.com/chunbaobao/Deep-JSCC-PyTorch.git
cd ./Deep-JSCC-PyTorch
pip install requirements.txt
```

## Usage

### Prepare Dataset

The cifar10 dataset can be downloaded automatically by torchvision. But the imagenet dataset should be downloaded manually from [ImageNet website](https://image-net.org/) and put in the right place, refer to [dataset.py](https://github.com/chunbaobao/Deep-JSCC-PyTorch/blob/main/dataset.py#L28). And run:

```
python dataset.py
```

### Training

The training command used to be very long, but now you can just use `python train.py --dataset ${dataset_name} --channel ${channel}` to train the model.
The default dataset is cifar10.
The parmeters can be modified in the `train.py` file. The default parameters are similar to the paper.

| Parameters     | CIFAR-10          | ImageNet          |
| -------------- | ----------------- | ----------------- |
| `batch_size`   | 64                | 32                |
| `init_lr`      | 1e-3              | 1e-4              |
| `weight_decay` | 5e-4              | 5e-4              |
| `snr_list`     | [19, 13, 7, 4, 1] | [19, 13, 7, 4, 1] |
| `ratio_list`   | [1/6, 1/12]       | [1/6, 1/12]       |
| `if_scheduler` | True              | False             |
| `step_size`    | 640               | N/A               |
| `gamma`        | 0.1               | 0.1               |

<!-- ALSO! The batch_size for cifar10 training in the paper is small causing the GPU utilization is low. So The bash script is provided to run the code in parallel for different snr and ratio for cifar10 dataset. (Example of two GPUs)
```
bash parallel_train_cifar.sh --channel ${channel}
``` -->

### Evaluation

The `eval.py` provides the evaluation of the trained model.

You may need modify slightly to evaluate the model for different snr_list and channel type in `main` function.

```
python eval.py
```

All training and evaluation results are saved in the `./out` directory by default. The `./out` directory may contain the structure as follows:

```
./out
├── checkpoint # trained models
│   ├── $DATASETNAME_$INNERCHANNEL_$SNR_$RATIO_$CHANNEL_TYPE_$TIMES_on_$DATE
│       ├── epoch_$num.pth
│       ├── ...
│   ├── CIFAR10_10_1.0_0.08_AWGN_13h21m37s_on_Jun_02_2024
│   ├── CIFAR10_20_7.0_0.17_Rayleigh_14h03m19s_on_Jun_03_2024
│   ├── ...
├── configs # training configurations
│   ├── $DATASETNAME_$INNERCHANNEL_$SNR_$RATIO_$CHANNEL_TYPE_$TIMES_on_$DATE
│   ├── $CIFAR10_10_4.0_0.08_AWGN_13h21m38s_on_Jun_02_2024.yaml
│   ├── ...
├── logs # training logs
│   ├── $DATASETNAME_$INNERCHANNEL_$SNR_$RATIO_$CHANNEL_TYPE_$TIMES_on_$DATE
│       ├── tensorboard logs
│   ├── ...
├── eval # evaluation results
│   ├── $DATASETNAME_$INNERCHANNEL_$SNR_$RATIO_$CHANNEL_TYPE_$TIMES_on_$DATE
│       ├── tensorboard logs
│   ├── ...
```

### Visualization

The `./visualization` directory contains the scripts for visualization of the training and evaluation results.

- `single_visualization.ipynb` is used to get demo of the model on single image like the demo above.
- `plot_visualization.ipynb` is used to get visualizations of the perfprmance of the model on different snr and ratio.

## Results

<div style="display: flex;">
  <img src="demo/cifar_0.08_AWGN.png" alt="Image 1" style="flex: 1; max-width: 48%; height: auto;">
  <div style="width: 0px;"></div> <!-- 为了让两个图像之间有一点间距 -->
  <img src="demo/cifar_0.17_AWGN.png" alt="Image 2" style="flex: 1; max-width: 48%; height: auto;">
</div>

<div style="display: flex;">
  <img src="demo/cifar_0.17_Rayleigh.png" alt="Image 1" style="flex: 1; max-width: 48%; height: auto;">
  <div style="width: 0px;"></div> <!-- 为了让两个图像之间有一点间距 -->
  <img src="demo/cifar_0.34_Rayleigh.png" alt="Image 2" style="flex: 1; max-width: 48%; height: auto;">
</div>

### TO-DO

- ~~Add visualization of the model~~
- ~~plot the results with different snr and ratio~~
- ~~add Rayleigh channel~~
- train on imagenet
- **Convert the real communication system to a complex communication system**

## Citation

If you find (part of) this code useful for your research, please consider citing

```
@misc{chunhang_Deep-JSCC,
  author = {chunhang},
  title = {a pytorch implementation of Deep JSCC},
  url ={https://github.com/chunbaobao/Deep-JSCC-PyTorch},
  year = {2023}
}



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
├── model.py # Mô hình Deep JSCC (Encoder/Decoder)
├── channel.py # Các loại kênh truyền (AWGN, Rayleigh)
├── fis_modules.py # Module FIS Controller
├── utils.py # Hàm utility
├── dataset.py # Dataset loading (CIFAR-10)
├── diagnose_controller.py # Diagnostic script
├── run_paper_sims.py # Paper simulations
├── plot_results.py # Vẽ biểu đồ kết quả
├── export_latex_and_plot.py # Export LaTeX tables
├── fix_rayleigh_only.py # Training script chính
└── run_all.sh # Script chạy toàn bộ pipeline

````

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
````

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
