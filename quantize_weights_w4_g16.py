import numpy as np
import os


GROUP_SIZE = 16


def find_best_mse_scale(group, num_steps=20):
    abs_w = np.abs(group)
    max_val = abs_w.max()

    if max_val < 1e-8:
        return 1e-8

    best_mse = float('inf')
    best_scale = max_val / 7.0

    for ratio in np.linspace(0.7, 1.0, num_steps):
        scale = max_val * ratio / 7.0

        w_q = np.round(group / scale).clip(-8, 7)
        w_deq = w_q * scale

        mse = np.mean((group - w_deq) ** 2)

        if mse < best_mse:
            best_mse = mse
            best_scale = scale

    return best_scale


def run_advanced_quantization():

    src_path = "weights/model_smoothed.npz"

    src_model = np.load(src_path)

    quantized_weights = {}
    weight_scales = {}
    stats = []

    print("=== SmoothQuant + G16-MSE 权重量化 ===")

    for key in src_model.files:

        data = src_model[key]

        is_linear_weight = 'weight' in key and any(
            x in key for x in [
                'attn.c_attn',
                'attn.c_proj',
                'mlp.c_fc',
                'mlp.c_proj'
            ]
        )

        if is_linear_weight:

            out_c, in_c = data.shape
            num_groups = in_c // GROUP_SIZE

            w_reshaped = data.reshape(out_c, num_groups, GROUP_SIZE)
            best_scales = np.zeros((out_c, num_groups), dtype=np.float32)

            print(f"优化层: {key:30s}", end='\r')

            for i in range(out_c):
                for j in range(num_groups):
                    best_scales[i, j] = find_best_mse_scale(
                        w_reshaped[i, j]
                    )

            w_q = np.round(
                w_reshaped / best_scales[:, :, None]
            ).clip(-8, 7).astype(np.int8)

            w_deq = (
                w_q.astype(np.float32)
                * best_scales[:, :, None]
            ).reshape(out_c, in_c)

            noise = data - w_deq
            mse = np.mean(noise ** 2)
            snr = 10 * np.log10(
                np.mean(data ** 2) / (mse + 1e-10)
            )

            stats.append((key, snr, mse))

            quantized_weights[key] = w_q.reshape(out_c, in_c)
            weight_scales[key] = best_scales

        else:
            quantized_weights[key] = data

    print("\nLayer SNR Summary")
    for k, snr, mse in stats:
        print(f"{k:40s} | {snr:6.2f} dB | {mse:.6f}")

    np.savez("weights/model_w4_sym_g16.npz", **quantized_weights)
    np.savez("weights/scales_w4_sym_g16.npz", **weight_scales)

    print("\n保存完成 G16 权重")


if __name__ == "__main__":
    run_advanced_quantization()
