from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
from io import BytesIO
import base64
from pyngrok import ngrok
import cv2
import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import copy


app = Flask(__name__)

# Add the following line to enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow requests from all sources


import torch

# Release memory before calling the model
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


# Loading models more suited to realistic styles
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting"
).to("cuda" if torch.cuda.is_available() else "cpu")

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def generate_architectural_sketch_white_bg(pil_image, blurIntensity, edgeThreshold):
    # Convert PIL image to OpenCV format (NumPy array)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (blurIntensity*2+1, blurIntensity*2+1), 0)

    # Step 4: Apply Canny Edge Detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Step 5: Invert edges to make it white-on-black
    inverted_edges = cv2.bitwise_not(edges)

    # Step 6: Add grid or texture (Optional)
    height, width = inverted_edges.shape
    grid = np.zeros_like(inverted_edges)
    grid_spacing = edgeThreshold
    for x in range(0, width, grid_spacing):
        cv2.line(grid, (x, 0), (x, height), color=128, thickness=1)
    for y in range(0, height, grid_spacing):
        cv2.line(grid, (0, y), (width, y), color=128, thickness=1)

    # Combine edges with the grid
    combined = cv2.addWeighted(inverted_edges, 0.8, grid, 0.2, 0)

    # Step 7: Invert again for white background
    white_bg_black_edges = cv2.bitwise_not(combined)

    # Step 8: Convert back to PIL image
    result = Image.fromarray(white_bg_black_edges)
    return result


def generate_watercolor_style(image, color_saturation=1.5, contrast=1.2):
    “"”
    Watercolor style conversion function
    Args.
        image (PIL.Image): input image
    Returns.
        PIL.Image: watercolor style image
    """
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Step 1: Apply bilateral filter
    smoothed = cv2.bilateralFilter(cv_image, d=15, sigmaColor=75, sigmaSpace=75)

    # Step 2: Extract edges
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY, blockSize=9, C=2
    )
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Step 3: Blend edges with smoothed image
    watercolor = cv2.addWeighted(smoothed, 0.98, edges_colored, 0.005, 0)

    # Step 4: Enhance color and contrast
    watercolor_image = Image.fromarray(cv2.cvtColor(watercolor, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Color(watercolor_image)
    watercolor_image = enhancer.enhance(color_saturation)  # Increase saturation
    enhancer = ImageEnhance.Contrast(watercolor_image)
    watercolor_image = enhancer.enhance(contrast)  # Increase contrast

    # Step 5: Add sharpening edges (optional)
    edges_gray = cv2.Canny(smoothed, threshold1=50, threshold2=150)
    edges_inverted = cv2.bitwise_not(edges_gray)
    edges_overlay = cv2.cvtColor(edges_inverted, cv2.COLOR_GRAY2BGR)
    final_result = cv2.addWeighted(
        np.array(watercolor_image), 0.9, edges_overlay, 0.3, 0
    )

    return Image.fromarray(final_result)




def generate_pixel_art_style(image, pixel_size=3, color_palette_size=16):
    """
    Pixel Style Conversion Functions
    Args.
        image (PIL.Image): input image
        pixel_size (int): pixel block size
        color_palette_size (int): number of colors in the palette
    Returns.
        PIL.Image: pixel style image
    """
    # 转换为 OpenCV 格式
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Step 1: Reduce resolution
    height, width = cv_image.shape[:2]
    small_image = cv2.resize(cv_image, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_NEAREST)

    # Step 2: Enlarge back to original size
    pixelated_image = cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)

    # Step 3: Reduce color palette
    quantized_image = Image.fromarray(cv2.cvtColor(pixelated_image, cv2.COLOR_BGR2RGB))
    quantized_image = quantized_image.convert("P", palette=Image.ADAPTIVE, colors=color_palette_size)
    final_image = quantized_image.convert("RGB")  # Convert back to RGB

    return final_image

def im_convert(tensor):
    """
    Converting PyTorch Tensor to PIL Image
    Args.
        tensor (torch.Tensor): input tensor
    Returns.
        PIL.Image: converted image
    """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)  # 维度从 (C, H, W) 转换为 (H, W, C)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # 反归一化
    image = image.clip(0, 1)  # 限制像素值在 [0, 1]
    image = (image * 255).astype("uint8")  # 转换为 uint8 格式
    return Image.fromarray(image)

# VGG 
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']  # Specific layers of VGG19
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

def apply_cyberpunk_style(content_image):
    """
    Image generation using a trained cyberpunk style model.
    Args.
        content_image (PIL.Image): input content image
    Returns.
        PIL.Image: image after cyberpunk style migration
    """
    # Loading VGG Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = VGG().to(device).eval()

    # Load the trained model
    checkpoint_path = "styled_model.pth"  
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vgg.load_state_dict(checkpoint["model_state"])

    # Converting content images to tensors
    content_img_tensor = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(content_image).unsqueeze(0).to(device)

    # Generating images using pre-trained models
    generated_image = checkpoint["generated_image"].to(device)

    # Returns the generated image (converted to PIL format)
    return im_convert(generated_image)



@app.route("/generate", methods=["POST"])
def generate():
    try:
        # Get uploaded images, masks and descriptions
        image_data = request.files["image"]
        mask_data = request.files["mask"]
        description = request.form["description"]
        num_images = int(request.form.get("num_images", 1))
        num_inference_steps = int(request.form.get("num_inference_steps", 75))
        guidance_scale = float(request.form.get("guidance_scale", 7.5))

        # Loading images and masks
        image = Image.open(image_data).convert("RGB")
        mask = Image.open(mask_data).convert("L")

        # Calling the model for image complementation
        result = pipe(
            prompt=description,
            image=image,
            mask_image=mask,
            num_images_per_prompt=num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        generated_images = result.images

        
        response_images = [image_to_base64(img) for img in generated_images]

        return jsonify({"images": response_images})

    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/style_transfer", methods=["POST"])
def style_transfer():
    try:
        data = request.get_json()
        image_data = data["image"]
        style = data["style"]
        intensity = float(data["intensity"])
        parameters = data.get("parameters", {})  

        # Decode Base64 image to a PIL Image
        image = Image.open(BytesIO(base64.b64decode(image_data)))

        # Apply the selected style
        if style == "watercolor":
            color_saturation = float(parameters.get("colorSaturation", 1.5))
            contrast = float(parameters.get("contrast", 1.2))
            styled_image = generate_watercolor_style(image, color_saturation, contrast)
        elif style == "sketch":
            blur_intensity = int(parameters.get("blurIntensity", 5))
            edge_threshold = int(parameters.get("edgeThreshold", 50))
            styled_image = generate_architectural_sketch_white_bg(image, blur_intensity, edge_threshold) 
        elif style == "pixel_art":
            pixel_size = int(parameters.get("pixelSize", 5))
            color_palette_size = int(parameters.get("colorPaletteSize", 16))
            styled_image = generate_pixel_art_style(image, pixel_size, color_palette_size)
        elif style == "cyberpunk":
            styled_image = apply_cyberpunk_style(image)
        else:
            raise ValueError(f"Unsupported style: {style}")

        # Convert result back to Base64
        buffered = BytesIO()
        styled_image.save(buffered, format="PNG")
        styled_image_base64 = base64.b64encode(buffered.getvalue()).decode()

        print("Generated styled image successfully.")

        return jsonify({"styled_image": styled_image_base64})

    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500




ngrok.set_auth_token("your_token")
public_url = ngrok.connect(5000)
print("Public URL:", public_url)

if __name__ == "__main__":
    app.run(port=5000)
