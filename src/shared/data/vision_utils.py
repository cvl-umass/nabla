import numpy as np
import cv2
import random
import math
from PIL import Image, ImageDraw

try:
    import bezier
except ImportError:
    bezier = None

def get_bbox_from_mask(mask):
    """Returns y1, y2, x1, x2"""
    if mask.sum() < 10:
        return 0, mask.shape[0], 0, mask.shape[1]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return y1, y2, x1, x2

def expand_image_mask(image, mask, ratio=1.4):
    h, w = image.shape[0], image.shape[1]
    H, W = int(h * ratio), int(w * ratio)
    h1 = int((H - h) // 2)
    h2 = H - h - h1
    w1 = int((W - w) // 2)
    w2 = W - w - w1
    
    # Pad image with 255 (white) and mask with 0
    image = np.pad(image, ((h1, h2), (w1, w2), (0, 0)), 'constant', constant_values=255)
    mask = np.pad(mask, ((h1, h2), (w1, w2)), 'constant', constant_values=0)
    return image, mask

def pad_to_square(image, pad_value=255, random_pad=False):
    H, W = image.shape[0], image.shape[1]
    if H == W:
        return image
    
    padd = abs(H - W)
    if random_pad:
        padd_1 = int(np.random.randint(0, padd))
    else:
        padd_1 = int(padd / 2)
    padd_2 = padd - padd_1
    
    if len(image.shape) == 2:
        pad_param = ((0, 0), (padd_1, padd_2)) if H > W else ((padd_1, padd_2), (0, 0))
    else:
        pad_param = ((0, 0), (padd_1, padd_2), (0, 0)) if H > W else ((padd_1, padd_2), (0, 0), (0, 0))
        
    return np.pad(image, pad_param, 'constant', constant_values=pad_value)

# --- RESTORED: Dynamic Ratio Logic from Original Code ---
def f(r, T=0.6, beta=0.1):
    return np.where(r < T, beta + (1 - beta) / T * r, 1)

def expand_bbox(mask, yyxx, ratio=[1.1, 1.2], min_crop=0):
    y1, y2, x1, x2 = yyxx
    H, W = mask.shape[0], mask.shape[1]

    if isinstance(ratio, (list, tuple)):
        # 1. Training Logic: Random expansion within range
        r = np.random.randint(ratio[0] * 10, ratio[1] * 10) / 10
    else:
        # 2. Inference Logic: Dynamic expansion based on area coverage
        # Note: The scalar input 'ratio' is intentionally ignored in calculation below,
        # matching the original callbacks.py behavior.
        if isinstance(ratio, (float, int)):
            r = ratio
        else:
            yyxx_area = (y2-y1+1) * (x2-x1+1)
            r1 = yyxx_area / (H * W)
            r2 = f(r1)
            r = math.sqrt(r2 / r1)
    
    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = r * (y2 - y1 + 1)
    w = r * (x2 - x1 + 1)
    
    h = max(h, min_crop)
    w = max(w, min_crop)
    
    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)
    
    return max(0, y1), min(H, y2), max(0, x1), min(W, x2)

def box2square(image, box):
    H, W = image.shape[0], image.shape[1]
    y1, y2, x1, x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h, w = y2 - y1, x2 - x1

    if h >= w:
        x1 = cx - h // 2
        x2 = cx + h // 2
    else:
        y1 = cy - w // 2
        y2 = cy + w // 2
    
    x1 = max(0, x1)
    x2 = min(W, x2)
    y1 = max(0, y1)
    y2 = min(H, y2)
    
    return y1, y2, x1, x2

def generate_probabilistic_mask(tar_image, tar_mask, disable_prob=False):
    """
    Generates a mask using Bezier curves, boxes, or dilation logic.
    """
    if disable_prob:
        if tar_mask.max() <= 1: 
            tar_mask = tar_mask.copy()
            tar_mask[tar_mask == 1] = 255
        kernel = np.ones((7, 7), np.uint8)
        return cv2.dilate(tar_mask, kernel, iterations=2)

    if bezier is None:
        kernel = np.ones((7, 7), np.uint8)
        return cv2.dilate(tar_mask, kernel, iterations=2)

    y1, y2, x1, x2 = get_bbox_from_mask(tar_mask)
    e_y1, e_y2, e_x1, e_x2 = expand_bbox(tar_mask, (y1, y2, x1, x2), ratio=[1.1, 1.2])

    prob = random.uniform(0, 1)
    prob_bezier = 0.5 
    prob_box = 1.0

    if prob <= prob_bezier:
        # BEZIER
        mask_img = Image.new('RGB', (tar_image.shape[1], tar_image.shape[0]), (0, 0, 0))
        top_nodes = np.asfortranarray([[x1, (x1+x2)/2, x2], [y1, e_y1, y1]])
        down_nodes = np.asfortranarray([[x2, (x1+x2)/2, x1], [y2, e_y2, y2]])
        left_nodes = np.asfortranarray([[x1, e_x1, x1], [y2, (y1+y2)/2, y1]])
        right_nodes = np.asfortranarray([[x2, e_x2, x2], [y1, (y1+y2)/2, y2]])
        
        curves = [
            bezier.Curve(top_nodes, degree=2),
            bezier.Curve(right_nodes, degree=2),
            bezier.Curve(down_nodes, degree=2),
            bezier.Curve(left_nodes, degree=2)
        ]
        
        pt_list = []
        random_width = 40
        for curve in curves:
            for i in range(1, 19):
                x_orig = curve.evaluate(i * 0.05)[0][0]
                y_orig = curve.evaluate(i * 0.05)[1][0]
                
                x = x_orig + random.randint(-random_width, random_width)
                y = y_orig + random.randint(-random_width, random_width)
                
                if not (x < x1 or x > x2): x = x_orig
                if not (y < y1 or y > y2): y = y_orig
                
                pt_list.append((x, y))

        draw = ImageDraw.Draw(mask_img)
        if len(pt_list) > 2:
            draw.polygon(pt_list, fill=(255, 255, 255))
        return np.array(mask_img)[:, :, 0]

    elif prob > prob_bezier and prob <= prob_box:
        # BOX
        mask = np.zeros_like(tar_mask, dtype=np.uint8)
        mask[e_y1:e_y2, e_x1:e_x2] = 255
        return mask

    else:
        # DILATION
        if tar_mask.max() <= 1: tar_mask[tar_mask == 1] = 255
        kernel = np.ones((7, 7), np.uint8)
        return cv2.dilate(tar_mask, kernel, iterations=2)

def unpad_image(image, target_h, target_w):
    """
    Crops the valid content from a squared (padded) image based on the target aspect ratio.
    """
    H, W = image.shape[:2] # This is the generated square size (e.g., 512x512)
    
    # Calculate what the dimensions of the valid content are inside the square
    if target_h > target_w:
        # Original was taller: Padding was added to width (Left/Right)
        # Valid Height is the full square Height
        valid_h = H
        # Valid Width scales based on aspect ratio
        valid_w = int(H * (target_w / target_h))
        
        # Calculate crop coordinates (centering)
        pad_total = W - valid_w
        pad_left = pad_total // 2
        crop = image[:, pad_left : pad_left + valid_w]
        
    else:
        # Original was wider (or equal): Padding was added to height (Top/Bottom)
        # Valid Width is the full square Width
        valid_w = W
        # Valid Height scales based on aspect ratio
        valid_h = int(W * (target_h / target_w))
        
        # Calculate crop coordinates (centering)
        pad_total = H - valid_h
        pad_top = pad_total // 2
        crop = image[pad_top : pad_top + valid_h, :]
        
    return crop