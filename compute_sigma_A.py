"""
Compute A_std (std of power map A) at multiple SNR levels for a single FIS model.
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from channel import Channel
from dataset import create_dataset
from model import DeepJSCC_FIS, power_normalize
from model_baseline import ratio2filtersize

SNR_LIST_DEFAULT = [1.0, 4.0, 7.0, 10.0]


@torch.no_grad()
def compute_A_std(model, loader, device, snr_db, budget, mode, channel_ctx, rayleigh_equalize=False):
    model.eval()
    model.set_channel(
        channel_type="Rayleigh", snr=snr_db, rician_k=4.0,
        rayleigh_equalize=rayleigh_equalize,
    )

    A_vals = []
    for x, _ in loader:
        x = x.to(device)
        z = model.encoder(x)
        A, _ = model.controller(
            z,
            snr_db=snr_db,
            budget=budget,
            mode=mode,
            channel_rel=channel_ctx["gamma_eff_norm"],
            return_info=True,
        )
        A_vals.append(A.float())

    A_cat = torch.cat(A_vals, dim=0)
    return float(A_cat.std(unbiased=False).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default="exp_ctx/ckpts_eq_full_Rayleigh_round2/fis_power_best.pth")
    ap.add_argument("--ratio", type=float, default=1 / 6)
    ap.add_argument("--channel", type=str, default="Rayleigh")
    ap.add_argument("--rayleigh_equalize", action="store_true", default=True)
    ap.add_argument("--mode", type=str, default="full")
    ap.add_argument("--snr_list", type=float, nargs="+", default=SNR_LIST_DEFAULT)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_batches", type=int, default=8,
                    help="Number of batches for aggregation")
    ap.add_argument("--out_dir", type=str, default="diag_eq_full_round2")
    ap.add_argument("--plot", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Dataset ----
    testset = create_dataset(
        "cifar10", split="test", data_root="", image_size=32, random_flip=False,
    )
    loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    x0, _ = testset[0]
    c = ratio2filtersize(x0, args.ratio)

    # ---- Model ----
    model = DeepJSCC_FIS(
        c=c, ratio=args.ratio,
        channel_type=args.channel, rician_k=4.0,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=True)
    model.eval()
    if args.rayleigh_equalize:
        model.channel.enable_rayleigh_equalization(True)

    # ---- Compute A_std per SNR ----
    snr_list = [float(s) for s in args.snr_list]
    results = {}

    print(f"Checkpoint : {args.ckpt}")
    print(f"Rayleigh EQ: {args.rayleigh_equalize}")
    print(f"Mode       : {args.mode}")
    print(f"SNR list   : {snr_list}")
    print()

    for snr_db in snr_list:
        print(f"SNR = {snr_db:g} dB ...", end=" ", flush=True)

        ch = Channel(channel_type=args.channel, snr_db=snr_db, rician_k=4.0)
        ch.enable_rayleigh_equalization(args.rayleigh_equalize)
        channel_ctx = ch.sample_context(
            batch_size=args.batch_size, device=device, dtype=torch.float32,
        )

        a_std = compute_A_std(
            model, loader, device, snr_db, args.budget,
            args.mode, channel_ctx, rayleigh_equalize=args.rayleigh_equalize,
        )
        results[snr_db] = a_std
        print(f"A_std = {a_std:.4f}")

    # ---- Print summary table ----
    print()
    print("=" * 38)
    print(f"{'SNR (dB)':<12}{'A_std':>12}{'sigma_A':>12}")
    print("-" * 38)
    for snr_db in snr_list:
        print(f"{snr_db:<12.1f}{results[snr_db]:>12.4f}{results[snr_db]:>12.4f}")
    print("=" * 38)

    # ---- Plot ----
    if args.plot:
        snrs = list(results.keys())
        stds = [results[s] for s in snrs]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(snrs, stds, "o-", color="tab:blue", linewidth=2, markersize=8,
                label=r"$\sigma_A$")
        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel(r"$\sigma_A$  (Std of power map $A$)", fontsize=12)
        ax.set_title("EQ Full Round2 — Channel-Aware FIS", fontsize=13)
        ax.set_xticks(snrs)
        ax.grid(True, alpha=0.35)

        for x, y in zip(snrs, stds):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)

        fig.tight_layout()
        out_png = os.path.join(args.out_dir, "sigma_A_by_snr.png")
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        print(f"\nSaved plot: {out_png}")

    # ---- Save CSV ----
    csv_path = os.path.join(args.out_dir, "sigma_A_by_snr.csv")
    with open(csv_path, "w") as f:
        f.write("snr_db,a_std\n")
        for snr_db in snr_list:
            f.write(f"{snr_db},{results[snr_db]:.6f}\n")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
