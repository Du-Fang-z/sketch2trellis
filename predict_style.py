import torch
import clip
from PIL import Image
import numpy as np
from io import BytesIO

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# 风格和颜色词汇
style_words = ["medieval", "futuristic", "cyberpunk", "baroque", "gothic", "fantasy",
    "art deco", "noir", "steampunk", "renaissance", "surrealism", "minimalist",
    "brutalist", "modernist", "expressionist", "sci-fi fantasy", "oriental",
    "tribal", "vintage", "retro", "abstract", "pop art", "romantic", "Cozy Minimalist"]
color_words = ["light-colored", "dark-colored"]

def predict_style_and_color(image_bytes: bytes) -> tuple[str, str]:
    image = preprocess(Image.open(BytesIO(image_bytes))).unsqueeze(0).to(device)

    text_style = clip.tokenize(style_words).to(device)
    text_color = clip.tokenize(color_words).to(device)

    with torch.no_grad():
        logits_style, _ = model(image, text_style)
        probs_style = logits_style.softmax(dim=-1).cpu().numpy()
        best_style = style_words[np.argmax(probs_style)]

        logits_color, _ = model(image, text_color)
        probs_color = logits_color.softmax(dim=-1).cpu().numpy()
        best_color = color_words[np.argmax(probs_color)]

    return best_style, best_color

