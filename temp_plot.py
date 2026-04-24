import numpy as np
import matplotlib.pyplot as plt
import os

# ===== 修改为你的结果路径 =====
result_dir = "results/Diffusion-Guidance-SubCrowding-DTLZ1-Exact-v0/Diffusion-Guidance-SubCrowding-seed0-DTLZ1-Exact-v0-ts-2026-3-8_13-30-3"

# ===== 加载数据 =====
# 所有生成解
res_y = np.load(os.path.join(result_dir, "res_y.npy"))
# Top 20 解
res_y_20 = np.load(os.path.join(result_dir, "res_y_20.npy"))

print("Generated solutions shape:", res_y.shape)
print("Top 20 solutions shape:", res_y_20.shape)

# ===== 绘图 =====
plt.figure(figsize=(6,6))

# 所有生成解
plt.scatter(res_y[:,0], res_y[:,1], s=20, alpha=0.5, label="Generated (256)")

# 严格 Top20 解
plt.scatter(res_y_20[:,0], res_y_20[:,1], s=80, marker="*", color="orange", label="Top 20")

plt.xlabel("Objective 1")
plt.ylabel("Objective 2")
plt.title("Generated Pareto Front with Top 20 Highlighted")
plt.legend()
plt.grid(True)

# ===== 保存图像 =====
save_path = os.path.join(result_dir, "pareto_top20_plot.png")
plt.savefig(save_path, dpi=300)
print("Figure saved to:", save_path)

plt.show()