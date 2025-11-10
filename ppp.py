# import torch
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from segment_anything import sam_model_registry, SamPredictor
# from PIL import Image

# def get_mask_image(image_path, mask_path):
#     # 加载模型
#     sam_checkpoint = "./sam_vit_h_4b8939.pth"
#     model_type = "vit_h"
#     sam = sam_model_registry[model_type](sam_checkpoint)
#     predictor = SamPredictor(sam)

#     # 加载图像
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     predictor.set_image(image)

#     # 加载 mask 图像（带透明背景）
#     mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
#     if mask.shape[2] == 4:
#         alpha_channel = mask[:, :, 3]
#         binary_mask = (alpha_channel > 0).astype(np.uint8)
#     else:
#         binary_mask = (mask > 0).astype(np.uint8)


#     # 获取 mask 区域内所有点坐标
#     mask_points = np.column_stack(np.where(binary_mask > 0))

#     # 计算 mask 区域的中心点
#     M = cv2.moments(binary_mask)
#     if M["m00"] != 0:
#         cx = int(M["m10"] / M["m00"])
#         cy = int(M["m01"] / M["m00"])
#     else:
#         cx, cy = binary_mask.shape[1] // 2, binary_mask.shape[0] // 2
#     # 找出所有掩码像素点的位置
#     mask_points = np.column_stack(np.where(binary_mask == 1))  # [[y1,x1], [y2,x2], ...]

#     # 计算每个点到质心的欧氏距离
#     distances = np.sqrt((mask_points[:,1] - cx)**2 + (mask_points[:,0] - cy)**2)

#     # 最大距离即为半径
#     radius = np.max(distances)

#     # 选取中心点附近的点（欧氏距离小于阈值），并确保在 mask 内
#     threshold = radius/3 #距离阈值，可根据需要调整
#     distances = np.sqrt((mask_points[:, 0] - cy)**2 + (mask_points[:, 1] - cx)**2)
#     near_center_indices = np.where(distances < threshold)[0]
#     valid_points = mask_points[near_center_indices]

#     #   如果附近点太少，则放宽阈值
#     if valid_points.shape[0] < 5:
#         threshold = radius
#         near_center_indices = np.where(distances < threshold)[0]
#         valid_points = mask_points[near_center_indices]

#     # 随机选取 5 个点（确保在 mask 区域内）
#     selected_indices = np.random.choice(valid_points.shape[0], 5, replace=False)
#     input_points = valid_points[selected_indices][:, ::-1]  # 转换为 (x, y)
#     input_labels = np.ones(5, dtype=np.int32)

#     # 预测 mask
#     masks, scores, logits = predictor.predict(
#         point_coords=input_points,
#         point_labels=input_labels,
#         multimask_output=True,
#     )

#     # 计算所有 mask 的重叠区域（多数投票）
#     mask_stack = np.stack(masks, axis=0)
#     majority_mask = (np.sum(mask_stack, axis=0) >= (len(masks) // 2)).astype(np.uint8)

#     # 创建一个新的透明图像
#     h, w = majority_mask.shape
#     transparent_output = np.zeros((h, w, 4), dtype=np.uint8)

#     # 将原图像中掩码区域的 RGB 值复制到新图像，并设置 alpha 为 255
#     transparent_output[:, :, :3] = mask[:, :, :3]
#     transparent_output[:, :, 3] = majority_mask * 255
#     # 保存结果图像
#     cv2.imwrite("./mid_output/masked_output_test.png", transparent_output)


# def crop_nontransparent_region(input_path, output_path):
#     # 打开图片并转换为 RGBA 模式
#     img = Image.open(input_path).convert("RGBA")
#     arr = np.array(img)

#     # 获取 alpha 通道
#     alpha = arr[:, :, 3]

#     # 找出非透明区域的边界
#     non_zero = np.argwhere(alpha > 0)
#     if non_zero.size == 0:
#         print("图片中没有非透明区域")
#         return

#     ymin, xmin = non_zero.min(axis=0)
#     ymax, xmax = non_zero.max(axis=0) + 1  # 包含最大值

#     # 裁剪图像
#     cropped = img.crop((xmin, ymin, xmax, ymax))
#     width, height = cropped.size
    
#     new_img = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))

#     # 计算居中位置
#     x_offset = (1024 - width) // 2
#     y_offset = (1024 - height) // 2

#     # 将原图粘贴到新图中间
#     new_img.paste(cropped, (x_offset, y_offset))

#     # 保存裁剪后的图像
#     new_img.save(output_path)
#     print(f"裁剪后的图像已保存为 {output_path}，尺寸为 {cropped.size}")


from PIL import Image
import numpy as np
import io
import torch
from segment_anything import sam_model_registry, SamPredictor
import random
import gc
import os

# 计算 alpha 区域的中心点和半径
def compute_center_radius(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    data = np.array(image)
    alpha_channel = data[:, :, 3]
    mask = alpha_channel <= 250
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return None, None, mask
    center_y, center_x = coords.mean(axis=0)
    distances = np.sqrt((coords[:, 0] - center_y)**2 + (coords[:, 1] - center_x)**2)
    radius = distances.max()
    return (center_x, center_y), radius, mask

# 在圆形区域内采样有效点
def sample_points(mask, center, radius, num_points=5):
    h, w = mask.shape
    cx, cy = center
    points = []
    attempts = 0
    while len(points) < num_points and attempts < 1000:
        angle = random.uniform(0, 2 * np.pi)
        r = random.uniform(0, 0.5 * radius)
        dx = r * np.cos(angle)
        dy = r * np.sin(angle)
        x, y = int(cx + dx), int(cy + dy)
        if 0 <= x < w and 0 <= y < h and mask[y, x]:
            points.append([x, y])
        attempts += 1
    return np.array(points)

# 多数投票合并掩码
def majority_vote(masks):
    stacked = np.stack(masks, axis=0)
    vote = np.sum(stacked, axis=0)
    return vote >= 3

# 应用掩码并设置背景透明
def apply_mask(image_bytes, final_mask):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    data = np.array(image)
    data[~final_mask] = [0, 0, 0, 0]
    result = Image.fromarray(data)
    output = io.BytesIO()
    result.save(output, format="PNG")
    output.seek(0)
    return output

# 本地调用函数
def run_segmentation(overlay_path, original_path, sam_checkpoint="sam_vit_h_4b8939.pth", model_type="vit_h"):
    with open(overlay_path, "rb") as f:
        overlay_bytes = f.read()
    with open(original_path, "rb") as f:
        original_bytes = f.read()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    center, radius, alpha_mask = compute_center_radius(overlay_bytes)
    if center is None:
        raise ValueError("overlay_image 中没有 alpha <= 250 的区域")

    points = sample_points(alpha_mask, center, radius)
    if len(points) < 3:
        raise ValueError("无法采样足够的有效点")

    orig_image = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    image_np = np.array(orig_image)
    predictor.set_image(image_np)

    masks = []
    for point in points:
        mask, _, _ = predictor.predict(
            point_coords=np.array([point]),
            point_labels=np.array([1]),
            multimask_output=False
        )
        masks.append(mask[0])

    final_mask = majority_vote(masks)
    result_image = apply_mask(original_bytes, final_mask)

    # 保存结果到本地
    with open("./mid_output/segmented_result.png", "wb") as f:
        f.write(result_image.read())
    print("分割结果已保存为 segmented_result.png")
    
    del predictor
    del sam
    gc.collect()
    torch.cuda.empty_cache()

 
