import matplotlib.pyplot as plt

SNR_DB = [1, 4, 7, 10]

DATA = {
    "baseline": [20.402, 21.779, 22.874, 23.717],
    "linear": [20.442, 21.877, 22.793, 23.685],
    "importance_only": [20.439, 21.629, 22.689, 23.503],
    "snr_only": [20.421, 21.873, 23.056, 23.855],
    "full_ctx (round2)": [20.559, 21.990, 23.364, 24.319],
}

STYLES = {
    "baseline": {"marker": "o", "linestyle": "-"},
    "linear": {"marker": "s", "linestyle": "-"},
    "importance_only": {"marker": "^", "linestyle": "-"},
    "snr_only": {"marker": "D", "linestyle": "-"},
    "full_ctx (round2)": {"marker": "*", "linestyle": "-", "linewidth": 2.5, "markersize": 10},
}


def main():
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, psnr_values in DATA.items():
        style = STYLES[name]
        ax.plot(
            SNR_DB,
            psnr_values,
            label=name,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style.get("linewidth", 1.8),
            markersize=style.get("markersize", 7),
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR vs SNR (CIFAR-10, Rayleigh)")
    ax.set_xticks(SNR_DB)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig("psnr_chart.png", dpi=150)
    plt.close(fig)
    print("Saved: psnr_chart.png")


if __name__ == "__main__":
    main()
