"""Evaluate PSNR for AWGN: baseline + full_ctx (only models with available ckpts)."""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset import create_dataset
from model import DeepJSCC_FIS
from model_baseline import DeepJSCC, ratio2filtersize
from utils import get_psnr

CKPTS = {
    "baseline": "exp_ctx/ckpts_baseline_AWGN/baseline_best.pth",
    "full_ctx": "exp_ctx/ckpts_full_AWGN_round2/fis_power_best.pth",
}
FIS_MODES = {"full_ctx": "full"}
DEFAULT_SNR_LIST = [1.0, 4.0, 7.0, 10.0, 13.0]


@torch.no_grad()
def eval_baseline(model, loader, device, snr_db, channel, rician_k):
    model.eval()
    model.change_channel(channel_type=channel, snr=snr_db, rician_k=rician_k)
    psnr_sum, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        x_hat = model(x)
        psnr_sum += get_psnr(x_hat * 255.0, x * 255.0, max_val=255.0).mean().item()
        n += 1
    return psnr_sum / max(n, 1)


@torch.no_grad()
def eval_fis(model, loader, device, snr_db, budget, mode, channel, rician_k):
    model.eval()
    model.set_channel(channel_type=channel, snr=snr_db, rician_k=rician_k)
    psnr_sum, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        _, x_hat = model(x, snr=snr_db, budget=budget, mode=mode, return_info=False)
        psnr_sum += get_psnr(x_hat * 255.0, x * 255.0, max_val=255.0).mean().item()
        n += 1
    return psnr_sum / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=1 / 6)
    ap.add_argument("--channel", type=str, default="AWGN")
    ap.add_argument("--snr_list", type=float, nargs="+", default=DEFAULT_SNR_LIST)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--rician_k", type=float, default=4.0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--image_size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testset = create_dataset(
        "cifar10", split="test", data_root=args.data_root,
        image_size=args.image_size, random_flip=False,
    )
    loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    x0, _ = testset[0]
    c = ratio2filtersize(x0, args.ratio)

    models = {}
    for name, ckpt in CKPTS.items():
        if name == "baseline":
            m = DeepJSCC(c=c, channel_type=args.channel, rician_k=args.rician_k).to(device)
        else:
            m = DeepJSCC_FIS(c=c, ratio=args.ratio,
                             channel_type=args.channel, rician_k=args.rician_k).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
        m.eval()
        models[name] = m

    snr_list = [float(s) for s in args.snr_list]
    results = {n_: {} for n_ in CKPTS}

    print(f"Dataset: CIFAR-10 test | Channel: {args.channel} | SNR list: {snr_list}")
    print(f"Device: {device}\n")

    for snr_db in snr_list:
        print(f"=== SNR = {snr_db:g} dB ===")
        for name in CKPTS:
            if name == "baseline":
                psnr = eval_baseline(models[name], loader, device, snr_db,
                                     args.channel, args.rician_k)
            else:
                psnr = eval_fis(models[name], loader, device, snr_db, args.budget,
                                FIS_MODES[name], args.channel, args.rician_k)
            results[name][snr_db] = psnr
            print(f"  {name:18s}  PSNR = {psnr:.3f} dB")
        print()

    col_w = 10
    print("=== Summary PSNR table (dB) ===")
    header = f"{'Model':<18}" + "".join(
        f"{'SNR=' + str(int(s) if s == int(s) else s) + 'dB':>{col_w}}" for s in snr_list
    )
    print(header)
    print("-" * len(header))
    for name in CKPTS:
        row = f"{name:<18}"
        for snr in snr_list:
            row += f"{results[name][snr]:>{col_w}.3f}"
        print(row)


if __name__ == "__main__":
    main()
