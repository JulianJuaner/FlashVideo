txt_len = 256
T = 33
H = 34
W = 60

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import os

def visualize_attention(attention_matrix, q_pos, head_idx=0):
    from mpl_toolkits.mplot3d import Axes3D
    
    attn_scores = attention_matrix[head_idx, 0, :]
    text_scores = attn_scores[-txt_len:]
    image_scores = attn_scores[:-txt_len]
    image_scores = image_scores.reshape(T, H, W)
    
    # 计算文本和图像的attention总和及百分比
    total_attention = attn_scores.sum()
    text_attention = text_scores.sum()
    image_attention = image_scores.sum()
    text_percentage = (text_attention / total_attention) * 100
    image_percentage = (image_attention / total_attention) * 100
    
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.3)
    
    # Text attention visualization
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(text_scores.reshape(1, -1), aspect='auto', cmap='viridis')
    ax1.set_title(f'Text Attention Scores (Tokens 0-256)\n'
                 f'Text attention: {text_percentage:.2f}% of total attention')
    ax1.set_xticks(np.arange(0, txt_len, 32))
    ax1.set_yticks([])
    
    if q_pos < 0:
        rect = patches.Rectangle((-q_pos-0.5, -0.5), 1, 1, 
                               linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
    
    # 3D visualization using voxels
    ax2 = fig.add_subplot(gs[1], projection='3d')
    ax2.view_init(elev=20, azim=45)
    
    # 创建归一化后的attention scores
    norm = plt.Normalize(image_scores.min(), image_scores.max())
    normalized_scores = norm(image_scores)
    
    # 创建voxel网格
    voxels = np.zeros((W, T, H))
    colors = np.zeros((W, T, H, 4))
    
    # 计算显著voxel的softmax总和
    significant_sum = 0
    
    # 填充voxel数据
    for t in range(T):
        for h in range(H):
            for w in range(W):
                score = normalized_scores[t, h, w]
                raw_score = image_scores[t, h, w]
                if raw_score > 0.01:  # 显著性阈值
                    voxels[w, t, h] = True
                    color = plt.cm.RdYlBu_r(score)
                    colors[w, t, h] = (*color[:3], score)
                    significant_sum += raw_score
    
    # 计算显著voxel占总attention的百分比
    significant_percentage = (significant_sum / total_attention) * 100
    
    # 绘制voxels
    ax2.voxels(voxels, facecolors=colors, edgecolor='none')
    
    # 如果是查询位置所在的帧，用带描边的voxel标记
    if q_pos > 0:
        img_pos = q_pos
        frame_t = img_pos // (H * W)
        remainder = img_pos % (H * W)
        h = remainder // W
        w = remainder % W
        
        # 创建查询点的voxel
        query_voxel = np.zeros((W, T, H), dtype=bool)
        query_voxel[w, frame_t, h] = True
        
        # print score on the voxel
        print(f"score on the voxel: {image_scores[frame_t, h, w]}")
        # 绘制带红色描边的红色半透明voxel
        ax2.voxels(query_voxel, 
                   facecolors=np.array([1, 0, 0, 0.0]),
                  edgecolor='red',
                  linewidth=2)
    
    ax2.set_title(f'3D Voxel Visualization of Image Attention Scores\n'
                 f'Image attention: {image_percentage:.2f}% of total attention\n'
                 f'Displayed voxels: {significant_percentage:.2f}% of total attention')
    
    # 设置坐标轴范围和刻度
    ax2.set_xlim(0, W-1)
    ax2.set_ylim(0, T-1)
    ax2.set_zlim(0, H-1)
    
    ax2.set_xticks(np.arange(0, W, W//4))
    ax2.set_yticks(np.arange(0, T, max(1, T//4)))
    ax2.set_zticks(np.arange(0, H, H//4))
    
    ax2.set_xlabel('Width')
    ax2.set_ylabel('Time Steps')
    ax2.set_zlabel('Height')
    
    # 添加颜色条
    norm = plt.Normalize(image_scores.min(), image_scores.max())
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
    plt.colorbar(sm, ax=ax2)
    
    # 设置总标题
    if q_pos < txt_len:
        position_info = f"Text Token {q_pos}"
    else:
        img_pos = q_pos
        t = img_pos // (H * W)
        remainder = img_pos % (H * W)
        h = remainder // W
        w = remainder % W
        position_info = f"Image Position (t={t}, h={h}, w={w})"
    
    plt.suptitle(f'Attention Scores for Query Position {q_pos} ({position_info})\nHead {head_idx}', y=1.02)
    
    plt.tight_layout()
    
def get_attention_matrix(q, k, q_pos):
    # q, k: torch.tensor, shape: (num_heads, seq_len, d_k)
    # return: torch.tensor, shape: (num_heads, seq_len, seq_len)
    qk = torch.matmul(q[:, q_pos:q_pos+1, :], k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
    qk = torch.softmax(qk, dim=-1)
    return qk
        
layer_idx = 1
time_idx = 49
if layer_idx <= 20:
    layer_type = "double"
else:
    layer_type = "single"

root = "/dataset-vlm/yc/FinalProj/FlashVideo/results/results_attn_save"
# load the qk value.
q = os.path.join(root, f"q_{layer_type}_{layer_idx}_{time_idx}.pt")
k = os.path.join(root, f"k_{layer_type}_{layer_idx}_{time_idx}.pt")

q = torch.load(q).cuda()
k = torch.load(k).cuda()
q = q[0, :, :].permute(1, 0, 2)
k = k[0, :, :].permute(1, 0, 2)

print(q.shape, k.shape)
q_pos = 20
print(q.shape, k.shape)

qk = get_attention_matrix(q, k, q_pos).cpu().float()
visualize_attention(qk, q_pos, 4)

def plot_attention_distributions(attention_matrix, head_idx=0):
    """Plot histograms of attention score distributions for text and image regions.
    
    Args:
        attention_matrix: Attention matrix of shape (num_heads, 1, seq_len)
        head_idx: Index of attention head to analyze
    """
    attn_scores = attention_matrix[head_idx, 0, :]
    text_scores = attn_scores[-txt_len:].numpy()
    image_scores = attn_scores[:-txt_len].numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Text attention distribution
    ax1.hist(text_scores, bins=50, alpha=0.7, color='blue')
    ax1.set_title(f'Text Attention Distribution\nMean: {text_scores.mean():.6f}, Std: {text_scores.std():.6f}')
    ax1.set_xlabel('Attention Score')
    ax1.set_ylabel('Frequency')
    
    # Image attention distribution
    ax2.hist(image_scores, bins=50, alpha=0.7, color='red')
    ax2.set_title(f'Image Attention Distribution\nMean: {image_scores.mean():.6f}, Std: {image_scores.std():.6f}')
    ax2.set_xlabel('Attention Score')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Print raw statistics
    print("\nRaw Attention Statistics:")
    print(f"Text Attention:")
    print(f"  Mean: {text_scores.mean():.6f}")
    print(f"  Std:  {text_scores.std():.6f}")
    print(f"  Max:  {text_scores.max():.6f}")
    print(f"  Min:  {text_scores.min():.6f}")
    
    print(f"\nImage Attention:")
    print(f"  Mean: {image_scores.mean():.6f}")
    print(f"  Std:  {image_scores.std():.6f}")
    print(f"  Max:  {image_scores.max():.6f}")
    print(f"  Min:  {image_scores.min():.6f}")

# Add this after your existing visualization call
plot_attention_distributions(qk, head_idx=4)