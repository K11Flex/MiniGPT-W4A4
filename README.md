NumPy MiniGPT W4A4 量化推理示例

本项目演示了如何对 MiniGPT 模型进行 W4A4 量化（权重 4-bit、激活 4-bit）并使用 NumPy 做前向推理和文本生成。流程基于 SmoothQuant + AWQ 分组量化 (G16)。

🚀 功能

高精度权重量化（W4）+ 激活缩放（A4 模拟）。

NumPy 前向推理，支持文本生成。

每层矩阵乘法可输出 SNR（调试可选）。

支持用户输入 Prompt 生成连续文本。

📦 文件结构
.
├─ README.md
├─ data/
│  └─ tinystories.txt          # 用于构建词表
├─ weights/
│  ├─ fp32_model.pth           # 原始 FP32 MiniGPT 模型
│  ├─ model_smoothed.npz       # SmoothQuant 缩放后的权重（由 smoothquant_calibration.py 生成）
│  ├─ model_w4_sym_g16.npz     # W4 权重量化后的权重
│  └─ scales_w4_sym_g16.npz    # 每组权重的 scale
├─ smoothquant_calibration.py   # SmoothQuant 缩放与激活校准
├─ quantize_weights.py          # 权重量化（G16 + MSE 最小化）
└─ numpy_inference.py           # NumPy W4A4 推理 & 文本生成

⚡ 使用流程
1️⃣ 安装依赖
pip install numpy torch

2️⃣ SmoothQuant 权重缩放
python smoothquant_calibration.py


输入：weights/fp32_model.pth

输出：weights/model_smoothed.npz

功能：收集激活样本并生成 SmoothQuant 缩放后的权重

3️⃣ 权重量化 (W4)
python quantize_weights.py


输入：weights/model_smoothed.npz

输出：

weights/model_w4_sym_g16.npz

weights/scales_w4_sym_g16.npz

功能：按组 (G16) 对权重进行 4-bit 对称量化

4️⃣ NumPy 推理与文本生成
python numpy_inference.py


输入 Prompt，例如：

请输入 Prompt: Once upon a time


输出：

各层 SNR（调试模式）

Prompt

生成文本

⚙️ 配置说明

GROUP_SIZE: 权重量化分组大小，默认 16。

clip_ratio: 控制量化阈值，默认 0.9995。

temperature: 生成文本的采样温度，推荐 0.1~0.3。

max_new_tokens: 每次生成的最大 token 数。

⚡ 注意事项

NumPy 推理模拟 A4 激活，当前精度为 float32。

输入文本过短会导致 SmoothQuant 校准样本不足，可增加 text 长度。

mask 在 NumPy 推理中需保证广播正确，否则注意索引错误。

推荐先按顺序运行三个脚本，确保权重与 scales 文件存在。
