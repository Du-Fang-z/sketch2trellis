from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import requests
from PIL import Image, ImageFilter
import numpy as np
import io
import html
import re
import time
import os
import uvicorn
import json
from rembg import remove
from rembg.session_factory import new_session
import ppp
import predict_style

app = FastAPI()

def processing_image(image1):
    # 获取像素数据
    pixels = image1.load()

    # 遍历所有像素，将透明度小于255的像素设置为完全透明
    for y in range(image1.height):
        for x in range(image1.width):
            r, g, b, a = pixels[x, y]
            if a < 250:
                pixels[x, y] = (r, g, b, 0)
    return image1


# 将图像字节编码为 base64
def encode_image_to_base64(image_bytes):
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return "data:image/png;base64," + encoded

# 提取 alpha 通道作为透明图层
def extract_alpha_as_transparency(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    img = processing_image(img)
    img_np = np.array(img)
    alpha = img_np[:, :, 3]
    black_with_alpha = np.zeros_like(img_np)
    black_with_alpha[:, :, 3] = alpha
    result_img = Image.fromarray(black_with_alpha, mode="RGBA")
    buffered = io.BytesIO()
    result_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"

# 保存 base64 图像为文件
def save_base64_image(base64_str, output_path):
    base64_str = html.unescape(base64_str)
    if base64_str.startswith("data:image/png;base64,"):
        base64_str = base64_str.split(",")[1]
    base64_str = re.sub(r'[^A-Za-z0-9+/=]', '', base64_str)
    base64_str += '=' * (-len(base64_str) % 4)
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    img.save(output_path)

@app.post("/generate_ply/")
async def generate_ply(
    original_image: UploadFile = File(...),
    overlay_image: UploadFile = File(...),
    prompt_json: UploadFile = File(...)  # 接收 JSON 文件
):
    time1 = time.time()

    # 读取上传的图像
    original_bytes = await original_image.read()
    overlay_bytes = await overlay_image.read()

    
    # 读取并解析 JSON 文件
    prompt_data = await prompt_json.read()
    prompt_dict = json.loads(prompt_data)
    prompt = prompt_dict.get("prompt", "")

    best_style, best_color = predict_style.predict_style_and_color(original_bytes)


    # 构造请求体
    x = {
        "from_frontend": {
            "total_mask": extract_alpha_as_transparency(overlay_bytes),
            "original_image": encode_image_to_base64(original_bytes),
            "add_color_image": encode_image_to_base64(original_bytes),
            "add_edge_image": extract_alpha_as_transparency(overlay_bytes),
            "remove_edge_image": None
        },
        "from_backend": {
            "prompt": prompt + ', ' + best_style + ', ' + best_color
        }
    }

    print("Received prompt:", x["from_backend"]["prompt"])

    time2 = time.time()
    print(f"Style and color prediction time: {time2 - time1} seconds")


    # 请求图像生成接口
    response = requests.post(
        "http://localhost:7860/magic_quill/auto_add_brush",
        json=x
    )

    time3 = time.time()
    print(f"Image generation request time: {time3 - time2} seconds")

    if response.status_code != 200:
        return {"error": f"Request failed: {response.status_code} {response.text}"}

    print("Image generation response received")
    # 保存生成的图像
    result_base64 = response.json()["from_backend"]["generated_image"]
    # latent_base64 = response.json()["from_backend"]["latent_samples"]
    os.makedirs("mid_output", exist_ok=True)
    mid_output_path = "./mid_output/mid_output.png"
    save_base64_image(result_base64, mid_output_path)
    # mid_latent_path = "./mid_output/mid_latent.png"
    # save_base64_image(latent_base64, mid_latent_path)

    time4 = time.time()
    print(f"Image saving time: {time4 - time3} seconds")

    # 处理透明图层
    new_image = Image.open(mid_output_path)
    overlay_image_pil = Image.open(io.BytesIO(overlay_bytes)).convert("RGBA")
    overlay_image_pil = processing_image(overlay_image_pil)
    overlay_alpha = overlay_image_pil.split()[-1]
    overlay_alpha = overlay_alpha.filter(ImageFilter.MinFilter(13))


    for y in range(new_image.size[1]):
        for x in range(new_image.size[0]):
            alpha = overlay_alpha.getpixel((x, y))
            if alpha != 0:
                new_image.putpixel((x, y), (255, 255, 255, 255))

    mid_mask_path = "./mid_output/mid_mask.png"
    new_image.save(mid_mask_path)

    time5 = time.time()
    print(f"Transparency processing time: {time5 - time4} seconds") 

    with open("./mid_output/mid_mask.png", "rb") as input_file:
        input_data = input_file.read()
    session = new_session("u2net")
    output_data = remove(input_data, session=session)
    output_image = Image.open(io.BytesIO(output_data))
    output_image.save("./mid_output/mid_mask_no_bg.png")
    ppp.get_mask_image("./mid_output/mid_output.png","./mid_output/mid_mask_no_bg.png")
    ppp.crop_nontransparent_region("./mid_output/masked_output_test.png", "./mid_output/masked_output.png")

    # 请求 3D 模型生成接口
    time6 = time.time()
    print(f"Background removal time: {time6 - time5} seconds")


    url1 = "https://localhost:8000/sketch2trellis/picture23d/"
    with open("./mid_output/masked_output.png", 'rb') as f:
        files = {'file': ('mid_output.png', f, 'image/png')}
        mid_response = requests.post(url1, files=files)
    
    if mid_response.status_code == 200:
        glb_path = "./mid_output/sample.glb"
        with open(glb_path, "wb") as f:
            f.write(mid_response.content)
        time7 = time.time()
        print(f"3D model generation time: {time7 - time6} seconds")
        return FileResponse(glb_path, media_type='application/octet-stream', filename="sample.glb")
    else:
        return {"error": f"3D generation failed: {mid_response.status_code}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8899) 