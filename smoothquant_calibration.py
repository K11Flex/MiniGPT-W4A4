import torch
import torch.nn as nn
import numpy as np
import random
from src.model import MiniGPT, Config

# 配置参数
GROUP_SIZE = 8
CLIP_PERCENTILE = 99.5
NUM_CALIBRATION_SAMPLES = 64


def percentile_channel(x, percentile=99.9):
    return np.percentile(np.abs(x), percentile, axis=0)



def get_calibration_data(text, stoi, block_size, num_samples=64):
    data_list = []
    text_len = len(text)

    print(f"正在从全文（长度 {text_len}）中随机抽取 {num_samples} 个样本...")
    for _ in range(num_samples):
        # 随机选择起始位置
        start_idx = random.randint(0, text_len - block_size - 1)
        chunk = text[start_idx: start_idx + block_size]
        encoded = [stoi.get(c, 0) for c in chunk]
        data_list.append(torch.tensor(encoded, dtype=torch.long))

    return torch.stack(data_list)



# 收集真实激活样本
def collect_calibration_activations(model, data_samples):
    activations = {}

    def hook(name):
        def fn(module, input, output):
            # 获取输入激活值
            x = input[0].detach().cpu().numpy()
            x = x.reshape(-1, x.shape[-1])

            if name not in activations:
                activations[name] = x[:2048]  # 每层初始采样
            else:
                activations[name] = np.concatenate(
                    [activations[name], x[:2048]], axis=0
                )[:8192]

        return fn

    handles = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(hook(name)))

    model.eval()
    with torch.no_grad():
        # 遍历所有随机抽取的样本
        for i in range(data_samples.shape[0]):
            x = data_samples[i:i + 1]
            model(x)

    for h in handles:
        h.remove()

    return activations


# AWQ group quant模拟
def quantize_weight_awq(W, group_size):
    out_c, in_c = W.shape
    groups = in_c // group_size
    Wg = W.reshape(out_c, groups, group_size)
    scales = np.max(np.abs(Wg), axis=2, keepdims=True) / 7.0 + 1e-8
    Wq = np.round(Wg / scales).clip(-8, 7)
    Wdq = Wq * scales
    return Wdq.reshape(out_c, in_c)


# Activation quant模拟
def quantize_activation_awq(X, group_size):
    N, C = X.shape
    groups = C // group_size
    Xg = X.reshape(N, groups, group_size)
    thresh = np.percentile(np.abs(Xg), CLIP_PERCENTILE, axis=2, keepdims=True)
    scale = thresh / 7.0 + 1e-8
    Xq = np.round(Xg / scale).clip(-8, 7)
    Xdq = Xq * scale
    return Xdq.reshape(N, C)


#   Joint Alpha Search
def search_best_alpha_joint(name, W, X_sample):
    if "h.0.attn.c_attn" in name:
        alpha_candidates = np.linspace(0.8, 0.99, 20)
        #第一层强制提高alpha
    elif "h.0." in name:
        alpha_candidates = np.linspace(0.4, 0.95, 20)
    else:
        alpha_candidates = np.linspace(0.3, 0.95, 14)

    W_stat = percentile_channel(W, 99.9)
    X_stat = percentile_channel(X_sample, 99.9)

    best_alpha = 0.5
    best_scale = None
    best_mse = float('inf')

    ref_out = X_sample @ W.T

    for alpha in alpha_candidates:
        # 计算平滑尺度
        scale = (X_stat ** alpha) / (W_stat ** (1 - alpha) + 1e-5)
        scale = np.clip(scale, 1e-5, None)

        # 应用平滑
        W_s = W * scale
        X_s = X_sample / scale

        # 模拟量化损耗
        W_q = quantize_weight_awq(W_s, GROUP_SIZE)
        X_q = quantize_activation_awq(X_s, GROUP_SIZE)

        out = X_q @ W_q.T
        mse = np.mean((ref_out - out) ** 2)

        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_scale = scale

    return best_alpha, best_scale


# 主程序
if __name__ == "__main__":
    config = Config()
    MODEL_IN = "weights/fp32_model.pth"
    MODEL_OUT = "weights/model_smoothed.npz"
    VOCAB_PATH = "data/tinystories.txt"

    # 加载词表和模型
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        full_text = f.read()
    chars = sorted(list(set(full_text)))
    config.vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}

    model = MiniGPT(config)
    model.load_state_dict(torch.load(MODEL_IN, map_location='cpu'))

    print("构建随机校准数据集...")
    data_samples = get_calibration_data(full_text, stoi, config.block_size, num_samples=NUM_CALIBRATION_SAMPLES)

    print("收集校准激活...")
    act_samples = collect_calibration_activations(model, data_samples)

    new_weights = {}
    state_dict = model.state_dict()

    print("\n" + "=" * 70)
    print(f"| {'Layer':30s} | {'Alpha':7s} | {'Scale Mean':10s} |")
    print("-" * 70)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # lm_head 通常不进行平滑，直接保留 FP32 或单独量化
            if "lm_head" in name:
                new_weights[f"{name}.weight"] = module.weight.data.numpy()
                continue

            W = module.weight.data.numpy()
            X_sample = act_samples[name]

            # 调用带优化策略的搜索
            best_alpha, best_scale = search_best_alpha_joint(name, W, X_sample)

            # 保存平滑后的权重和对应的 Scale
            new_weights[f"{name}.weight"] = W * best_scale
            new_weights[f"{name}.smooth_scale"] = best_scale

            if module.bias is not None:
                new_weights[f"{name}.bias"] = module.bias.data.numpy()

            print(f"| {name:30s} | {best_alpha:7.3f} | {best_scale.mean():10.4f} |")

        # 处理非 Linear 层（如 LayerNorm, Embedding）
        elif hasattr(module, 'weight') and name != "":
            new_weights[f"{name}.weight"] = module.weight.data.numpy()
            if hasattr(module, 'bias') and module.bias is not None:
                new_weights[f"{name}.bias"] = module.bias.data.numpy()

    new_weights['transformer.wte.weight'] = state_dict['transformer.wte.weight'].numpy()
    new_weights['transformer.wpe.weight'] = state_dict['transformer.wpe.weight'].numpy()
    if 'transformer.ln_f.weight' in state_dict:
        new_weights['transformer.ln_f.weight'] = state_dict['transformer.ln_f.weight'].numpy()
        new_weights['transformer.ln_f.bias'] = state_dict['transformer.ln_f.bias'].numpy()

    print("-" * 70)
    np.savez(MODEL_OUT, **new_weights)
    print(f"\nSmoothQuant 已保存至 {MODEL_OUT}")