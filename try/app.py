# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import StreamingResponse
# from PIL import Image
# import numpy as np
# import io
# import torch
# from segment_anything import sam_model_registry, SamPredictor
# import random
# import uvicorn

# app = FastAPI()

# # 计算 alpha 区域的中心点和半径
# def compute_center_radius(image_bytes):
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
#     data = np.array(image)
#     alpha_channel = data[:, :, 3]
#     mask = alpha_channel <= 250
#     coords = np.column_stack(np.where(mask))
#     if coords.size == 0:
#         return None, None, mask
#     center_y, center_x = coords.mean(axis=0)
#     distances = np.sqrt((coords[:, 0] - center_y)**2 + (coords[:, 1] - center_x)**2)
#     radius = distances.max()
#     return (center_x, center_y), radius, mask

# # 在圆形区域内采样有效点
# def sample_points(mask, center, radius, num_points=5):
#     h, w = mask.shape
#     cx, cy = center
#     points = []
#     attempts = 0
#     while len(points) < num_points and attempts < 1000:
#         angle = random.uniform(0, 2 * np.pi)
#         r = random.uniform(0, 0.5 * radius)
#         dx = r * np.cos(angle)
#         dy = r * np.sin(angle)
#         x, y = int(cx + dx), int(cy + dy)
#         if 0 <= x < w and 0 <= y < h and mask[y, x]:
#             points.append([x, y])
#         attempts += 1
#     return np.array(points)

# # 多数投票合并掩码
# def majority_vote(masks):
#     stacked = np.stack(masks, axis=0)
#     vote = np.sum(stacked, axis=0)
#     return vote >= 3
# # 应用掩码并设置背景透明
# def apply_mask(image_bytes, final_mask):
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
#     data = np.array(image)
#     data[~final_mask] = [0, 0, 0, 0]
#     result = Image.fromarray(data)
#     output = io.BytesIO()
#     result.save(output, format="PNG")
#     output.seek(0)
#     return output

# @app.post("/segment/")
# async def segment(
#     overlay_image: UploadFile = File(...),
#     original_image: UploadFile = File(...)
# ):
#     overlay_bytes = await overlay_image.read()
#     original_bytes = await original_image.read()

#     # 加载 SAM 模型
#     sam_checkpoint = "sam_vit_h_4b8939.pth"
#     model_type = "vit_h"
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     sam = sam_model_registry[model_type](sam_checkpoint)
#     sam.to(device)
#     predictor = SamPredictor(sam)

#     # Step 1: 计算中心点和半径
#     center, radius, alpha_mask = compute_center_radius(overlay_bytes)
#     if center is None:
#         return {"error": "overlay_image 中没有 alpha <= 250 的区域"}

#     # Step 2: 采样点
#     points = sample_points(alpha_mask, center, radius)
#     if len(points) < 3:
#         return {"error": "无法采样足够的有效点"}

#     # Step 3: SAM 预测掩码
#     orig_image = Image.open(io.BytesIO(original_bytes)).convert("RGB")
#     image_np = np.array(orig_image)
#     predictor.set_image(image_np)
#     masks = []
#     for point in points:
#         mask, _, _ = predictor.predict(
#             point_coords=np.array([point]),
#             point_labels=np.array([1]),
#             multimask_output=False
#         )
#         masks.append(mask[0])

#     # Step 4: 多数投票合并掩码
#     final_mask = majority_vote(masks)

#     # Step 5: 应用掩码
#     result_image = apply_mask(original_bytes, final_mask)

#     return StreamingResponse(result_image, media_type="image/png")


    
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8899) 


    
from PIL import Image
import numpy as np
import io
import torch
from segment_anything import sam_model_registry, SamPredictor
import random
import time
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
def run_segmentation(overlay_path, original_path, sam_checkpoint="./sam_vit_h_4b8939.pth", model_type="vit_h"):
    with open(overlay_path, "rb") as f:
        overlay_bytes = f.read()
    with open(original_path, "rb") as f:
        original_bytes = f.read()
    
    time3 = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    time4 = time.time()
    print(f"SAM model loaded in {time4 - time3:.2f} seconds.")
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
    with open("segmented_result.png", "wb") as f:
        f.write(result_image.read())
    print("分割结果已保存为 segmented_result.png")
 
if __name__ == "__main__":
    time1 = time.time()
    run_segmentation("./screenshot.png","./mid_output.png")
    time2 = time.time()
    print(f"Segmentation completed in {time2 - time1:.2f} seconds.")