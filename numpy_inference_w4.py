import numpy as np
import os


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

def layer_norm(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b


def matmul_awq_g16(
        x,
        weight_q,
        s_w_group,
        smooth_scale,
        bias=None,
        group_size=16,
        clip_ratio=0.9998,
        name=""
):
    x = x / smooth_scale

    out_features, in_features = weight_q.shape
    num_groups = in_features // group_size

    x_group = x.reshape(-1, num_groups, group_size)
    abs_x = np.abs(x_group)

    thresh = np.percentile(
        abs_x,
        clip_ratio * 100,
        axis=2,
        keepdims=True
    )

    thresh = np.maximum(thresh, 1e-6)
    s_a = thresh / 7.0

    x_q = np.round(x_group / s_a).clip(-8, 7).astype(np.int8)
    x_deq = (x_q.astype(np.float32) * s_a).reshape(x.shape)

    w_q = weight_q.reshape(out_features, num_groups, group_size)

    w_fp32 = (
            w_q.astype(np.float32)
            * s_w_group[:, :, None]
    ).reshape(out_features, in_features)

    out = x_deq @ w_fp32.T

    if bias is not None:
        out += bias

    if name:
        ref = x @ w_fp32.T
        noise = ref - out

        snr = 10 * np.log10(
            np.mean(ref ** 2) /
            (np.mean(noise ** 2) + 1e-10)
        )

        print(f"{name:22s} | {snr:6.2f} dB")

    return out


class NumPyFinalGPT:
    def __init__(self, weights_path, scales_path, vocab_path, group_size=32, clip_ratio=0.9998):

        self.weights = np.load(weights_path)
        self.w_scales = np.load(scales_path)

        self.group_size = group_size
        self.clip_ratio = clip_ratio

        with open(vocab_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        self.n_embd = 160
        self.n_head = 8
        self.n_layer = 6
        self.head_size = self.n_embd // self.n_head

    def forward(self, idx, debug=False):
        b, t = idx.shape

        x = self.weights['transformer.wte.weight'][idx]
        pos = np.arange(0, t)[None, :]
        x = x + self.weights['transformer.wpe.weight'][pos]

        for i in range(self.n_layer):
            prefix = f"transformer.h.{i}."

            x_norm = layer_norm(
                x,
                self.weights[prefix + 'ln_1.weight'],
                self.weights[prefix + 'ln_1.bias']
            )

            qkv = matmul_awq_g16(
                x_norm,
                self.weights[prefix + 'attn.c_attn.weight'],
                self.w_scales[prefix + 'attn.c_attn.weight'],
                smooth_scale=self.weights[prefix + 'attn.c_attn.smooth_scale'],
                bias=self.weights[prefix + 'attn.c_attn.bias'],
                group_size=self.group_size,
                clip_ratio=self.clip_ratio,
                name=f"L{i}_Attn_QKV" if debug else ""
            )

            q, k, v = np.split(qkv, 3, axis=-1)

            q = q.reshape(b, t, self.n_head, self.head_size).transpose(0, 2, 1, 3)
            k = k.reshape(b, t, self.n_head, self.head_size).transpose(0, 2, 1, 3)
            v = v.reshape(b, t, self.n_head, self.head_size).transpose(0, 2, 1, 3)

            att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(self.head_size))
            mask = np.tril(np.ones((t, t))) == 0
            att[:, :, mask] = -1e10
            att = softmax(att)

            y = (att @ v).transpose(0, 2, 1, 3).reshape(b, t, self.n_embd)

            y = matmul_awq_g16(
                y,
                self.weights[prefix + 'attn.c_proj.weight'],
                self.w_scales[prefix + 'attn.c_proj.weight'],
                smooth_scale=self.weights[prefix + 'attn.c_proj.smooth_scale'],
                bias=self.weights[prefix + 'attn.c_proj.bias'],
                group_size=self.group_size,
                clip_ratio=self.clip_ratio,
                name=f"L{i}_Attn_Proj" if debug else ""
            )

            x = x + y

            x_norm = layer_norm(
                x,
                self.weights[prefix + 'ln_2.weight'],
                self.weights[prefix + 'ln_2.bias']
            )

            h = matmul_awq_g16(
                x_norm,
                self.weights[prefix + 'mlp.c_fc.weight'],
                self.w_scales[prefix + 'mlp.c_fc.weight'],
                smooth_scale=self.weights[prefix + 'mlp.c_fc.smooth_scale'],
                bias=self.weights[prefix + 'mlp.c_fc.bias'],
                group_size=self.group_size,
                clip_ratio=self.clip_ratio,
                name=f"L{i}_MLP_FC" if debug else ""
            )

            h = gelu(h)

            y = matmul_awq_g16(
                h,
                self.weights[prefix + 'mlp.c_proj.weight'],
                self.w_scales[prefix + 'mlp.c_proj.weight'],
                smooth_scale=self.weights[prefix + 'mlp.c_proj.smooth_scale'],
                bias=self.weights[prefix + 'mlp.c_proj.bias'],
                group_size=self.group_size,
                clip_ratio=self.clip_ratio,
                name=f"L{i}_MLP_Proj" if debug else ""
            )

            x = x + y

        x = layer_norm(
            x,
            self.weights['transformer.ln_f.weight'],
            self.weights['transformer.ln_f.bias']
        )

        logits = x @ self.weights['lm_head.weight'].T
        return logits

    def generate(self, prompt, max_new_tokens=100, temperature=0.08):

        idx = np.array([[self.stoi.get(c, 0) for c in prompt]])

        print("\n===== Layer SNR Report =====")
        idx_cond = idx[:, -256:]
        _ = self.forward(idx_cond, debug=True)

        print("\n===== PROMPT =====")
        print(prompt)

        print("\n===== GENERATED TEXT =====")
        generated_text = ""

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -256:]
            logits = self.forward(idx_cond, debug=False)

            logits = logits[:, -1, :] / (temperature + 1e-10)
            probs = softmax(logits)

            next_idx = np.random.choice(len(probs[0]), p=probs[0])
            idx = np.append(idx, [[next_idx]], axis=1)

            char = self.itos[next_idx]
            generated_text += char

        print(generated_text)
        print()


if __name__ == "__main__":

    WEIGHTS_FILE = "weights/model_w4_sym_g16.npz"
    SCALES_FILE = "weights/scales_w4_sym_g16.npz"
    VOCAB_FILE = "data/tinystories.txt"

    if os.path.exists(WEIGHTS_FILE) and os.path.exists(SCALES_FILE):

        engine = NumPyFinalGPT(
            weights_path=WEIGHTS_FILE,
            scales_path=SCALES_FILE,
            vocab_path=VOCAB_FILE,
            group_size=16,
            clip_ratio=0.9995
        )

        print("==== NumPy Quantized GPT ====")
        print("输入 exit 退出\n")

        while True:
            prompt = input("请输入 Prompt: ")

            if prompt.strip().lower() == "exit":
                break

            engine.generate(prompt, max_new_tokens=100, temperature=0.25)

    else:
        print("错误：未找到量化权重或 Scale 文件")
