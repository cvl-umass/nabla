import argparse
import os
import yaml
import numpy as np
import torch
import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm
from diffusers.utils.logging import disable_progress_bar
import lpips
from transformers import AutoProcessor, SiglipModel, AutoImageProcessor, AutoModel
import torchvision

import sys
sys.path.append('src')

# Import Unified Components
from shared.data.nabirds_dataset import NABirdsDataset
from shared.data.inat_dataset import INatDataset
from shared.data.vision_utils import expand_image_mask, pad_to_square, get_bbox_from_mask, unpad_image

from omini.model_wrapper import OminiModelWrapper
from insert.model_wrapper import InsertAnythingWrapper


def render_caption_image(caption, width, height):
    """Creates an image with wrapped text for visualization."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (0, 0, 0)
    
    words = caption.split(' ')
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if text_width > width - 20:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    lines.append(current_line)
    
    y = 50
    for line in lines:
        (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = (width - text_width) // 2
        cv2.putText(img, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += text_height + 20
        
    return img

@torch.no_grad()
def eval_metrics(orig_image, gen_image, tar_mask, 
                 lpips_model, 
                 siglip_model, siglip_processor, 
                 dino_model, dino_processor, 
                 device):
    """Calculates metrics between the subject areas of two images."""
    h, w, _ = gen_image.shape
    
    # 1. Resize Images
    orig_resized = cv2.resize(orig_image, (w, h), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(tar_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    mask_float = mask_resized.astype(np.float32)
    if mask_float.max() > 1.0:
        mask_float = mask_float / 255.0
    
    # Binarize to ensure clean 0.0 or 1.0
    mask_float[mask_float > 0.5] = 1.0
    mask_float[mask_float <= 0.5] = 0.0
    
    mask_3d = np.repeat(mask_float[:, :, None], 3, axis=2)

    # 2. Create Masked Images (Safe Multiplication)
    masked_orig_np = (orig_resized.astype(np.float32) * mask_3d).astype(np.uint8)
    masked_gen_np = (gen_image.astype(np.float32) * mask_3d).astype(np.uint8)
    
    masked_orig_pil = Image.fromarray(masked_orig_np)
    masked_gen_pil = Image.fromarray(masked_gen_np)

    # 3. LPIPS
    orig_tensor = ((torchvision.transforms.functional.to_tensor(masked_orig_pil) * 2) - 1).unsqueeze(0).to(device)
    gen_tensor = ((torchvision.transforms.functional.to_tensor(masked_gen_pil) * 2) - 1).unsqueeze(0).to(device)
    lpips_loss = lpips_model(orig_tensor, gen_tensor).item()

    # 4. L2 (RMSE)
    # num_pixels is the count of valid pixels (sum of 1.0s)
    num_pixels = np.sum(mask_float)
    l2_loss = 0.0
    if num_pixels > 0:
        diff = orig_resized.astype(np.float32) - gen_image.astype(np.float32)
        squared_diff = (diff ** 2) * mask_3d
        # RMSE Calculation
        l2_loss = float(np.sqrt(np.sum(squared_diff) / (num_pixels * 3)))

    # 5. SigLIP
    siglip_inputs = siglip_processor(images=[masked_orig_pil, masked_gen_pil], padding="max_length", return_tensors="pt").to(device)
    siglip_out = siglip_model.get_image_features(**siglip_inputs)
    siglip_sim = torch.dot(torch.nn.functional.normalize(siglip_out[0], p=2, dim=0), 
                           torch.nn.functional.normalize(siglip_out[1], p=2, dim=0)).item()

    # 6. DINOv2
    dino_inputs = dino_processor(images=[masked_orig_pil, masked_gen_pil], return_tensors="pt").to(device)
    dino_out = dino_model(**dino_inputs).last_hidden_state
    dino_sim = torch.nn.functional.cosine_similarity(dino_out[0, 0], dino_out[1, 0], dim=0).item()
    
    return lpips_loss, l2_loss, siglip_sim, dino_sim

def infer_pair(model, ref_img, ref_mask, tar_img, tar_mask, tar_depth, tar_keypoints, tar_caption, config, seed, guidance_scale):
    """
    Constructs a batch dictionary manually to interface with the unified model.predict().
    """
    to_tensor = torchvision.transforms.ToTensor()
    cond_size = tuple(config['train']['dataset']['condition_size'])
    model_type = config.get('model_type', 'omini')

    # --- 1. Subject Condition (Reference) ---
    y1, y2, x1, x2 = get_bbox_from_mask(ref_mask)

    # Mask background to white (255)
    masked_ref = ref_img.copy()
    masked_ref = masked_ref[y1:y2, x1:x2]

    # Pad to square and resize
    masked_ref_padded = pad_to_square(masked_ref, pad_value=255, random_pad=False)
    subject_crop = np.array(Image.fromarray(masked_ref_padded.astype(np.uint8)).resize(cond_size, Image.LANCZOS))

    is_kontext = 'kontext' in config.get('flux_path', '').lower() or config.get('model_arch') == 'kontext'

    # Default delta for Schnell/Orig
    p_delta = torch.tensor([[0, -cond_size[0] // 16]])

    if is_kontext:
        p_delta = torch.tensor([[0, 0]])

    batch = {
        "description": tar_caption,
        "condition_0": to_tensor(Image.fromarray(subject_crop)).unsqueeze(0),
        "position_delta_0": p_delta,
        "condition_type_0": "default" if config['train'].get("single_adapter") else "subject"
    }

    # --- 2. Spatial Condition (Target) ---
    conditioning_mode = config['train'].get('conditioning_mode', 'fill')

    tar_mask_inf = tar_mask.copy()
    if tar_mask_inf.max() <= 1: tar_mask_inf = tar_mask_inf * 255

    if conditioning_mode == 'fill':
        if conditioning_mode == 'fill':
            batch['description'] = "A photo of a bird." 
        else:
            batch['description'] = tar_caption

        # Black out a 10% expanded bounding box
        bg_image = tar_img.copy()
        
        # EXACT SILHOUETTE BLACKOUT
        bg_image[tar_mask > 0] = [0, 0, 0]

        # Keep padding:
        bg_padded = pad_to_square(bg_image, pad_value=255, random_pad=False)
        cond_spatial = cv2.resize(bg_padded.astype(np.uint8), cond_size, interpolation=cv2.INTER_LINEAR)

    elif conditioning_mode == 'depth':
        depth_padded = pad_to_square(tar_depth, pad_value=0, random_pad=False)
        depth_resized = cv2.resize(depth_padded.astype(np.uint8), cond_size, interpolation=cv2.INTER_NEAREST)
        cond_spatial = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2RGB)

    elif conditioning_mode == 'keypoints':
        kp_padded = pad_to_square(tar_keypoints, pad_value=0, random_pad=False)
        
        cond_spatial = cv2.resize(kp_padded.astype(np.uint8), cond_size, interpolation=cv2.INTER_NEAREST)
    
    batch["condition_1"] = to_tensor(Image.fromarray(cond_spatial)).unsqueeze(0)
    batch["condition_type_1"] = "default" if config['train'].get("single_adapter") else conditioning_mode
    batch["position_delta_1"] = torch.tensor([[0, 0]])

    batch["ref"] = batch["condition_0"]
    batch["src"] = to_tensor(Image.fromarray(tar_img)).unsqueeze(0)
    batch["mask"] = to_tensor(Image.fromarray(tar_mask_inf.astype(np.uint8))).unsqueeze(0)
    
    # 3. Predict
    return np.array(model.predict(batch, seed=seed, guidance_scale=guidance_scale))

def create_vis_row(ref, gen, mask, depth, kp, caption, config):
    h, w, _ = ref.shape
    mask_rgb = np.repeat(mask[:, :, None], 3, axis=2) * 255
    items = [ref, cv2.resize(gen, (w, h)), mask_rgb]
    
    cond_mode = config['train'].get('conditioning_mode', 'fill')
    if cond_mode == 'depth':
        d_rgb = cv2.cvtColor(cv2.resize(depth, (w, h)), cv2.COLOR_GRAY2RGB)
        items.append(d_rgb)
    elif cond_mode == 'keypoints':
        items.append(cv2.resize(kp, (w, h)))
        
    items.append(render_caption_image(caption, w, h))
    return np.hstack(items)

def load_eval_data(dataset, idx, dataset_type):
    """
    Unified loader to fetch raw images/masks for evaluation.
    NABirds uses a custom get_raw_pair returning 10 items.
    iNat are adapted to return a standard dict.
    """
    if dataset_type == 'nabirds':
        # Returns tuple of 10 for bidirectional processing in main loop
        return dataset.get_raw_pair(idx)
    
    elif dataset_type == 'inat':
        # iNat dataset list is already flattened (contains both directions)
        item = dataset.pairs[idx]
        
        def load_components(path):
            if not os.path.exists(path): return None, None, None, None
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            
            mask_p = path.replace('images', 'mask').replace('.jpg', '.png')
            mask = cv2.imread(mask_p, 0) if os.path.exists(mask_p) else np.zeros(img.shape[:2], dtype=np.uint8)
            if mask is not None: mask = (mask > 128).astype(np.uint8)
            
            depth_p = path.replace('images', 'depth').replace('.jpg', '.png')
            depth = cv2.imread(depth_p, 0) if os.path.exists(depth_p) else np.zeros(img.shape[:2], dtype=np.uint8)
            
            kp = np.zeros(img.shape[:2], dtype=np.uint8)
            return img, mask, depth, kp

        r_img, r_mask, r_depth, r_kp = load_components(item['ref_path'])
        t_img, t_mask, t_depth, t_kp = load_components(item['tar_path'])

        return {
            'img1': r_img, 'mask1': r_mask, 'depth1': r_depth, 'kp1': r_kp,
            'img2': t_img, 'mask2': t_mask, 'depth2': t_depth, 'kp2': t_kp,
            'caption': item.get('tar_caption', "A photo of a bird."),
            'id': f"{item['obs_id']}_{item['direction']}"
        }

def main():
    disable_progress_bar()
    parser = argparse.ArgumentParser()
    # Paths & Setup
    parser.add_argument("--job_dir", type=str, required=True, help="Path to job directory")
    parser.add_argument("--ckpt_step", type=str, default=None, help="Specific checkpoint step")
    parser.add_argument("--dataset", type=str, default="nabirds", choices=["nabirds", "inat"])
    parser.add_argument("--split", type=str, default="test", help="NABirds split (val/test) or iNat dataset type (na/unseen)")
    parser.add_argument("--skip_existing", action="store_true")
    
    # NEW: Recompute Metrics Flag
    parser.add_argument("--recompute_metrics", action="store_true", help="If image exists, load it and recompute metrics instead of skipping.")
    
    # NEW: Baseline Flag
    parser.add_argument("--baseline", action="store_true", help="Use baseline pretrained weights (skip loading checkpoint).")
    
    # Distributed / Processing
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    args = parser.parse_args()
    if args.skip_existing:
        args.recompute_metrics = True

    # --- Distributed Setup ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        device_id = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device_id)
        print(f"[Rank {global_rank}] Running on GPU {local_rank}")
    else:
        device_id = torch.device("cpu")

    # 1. Load Config & Model
    config_path = os.path.join(args.job_dir, "config.yaml")
    with open(config_path, "r") as f: config = yaml.safe_load(f)

    print(f"[Rank {global_rank}] Loading Model...")
    if config.get("model_type") == "insert":
        model = InsertAnythingWrapper(config)
    else:
        model = OminiModelWrapper(config)

    # 2. Load Checkpoint or Baseline
    if args.baseline:
        print(f"[Rank {global_rank}] Mode: Baseline (Skipping Checkpoint Loading)")
        args.ckpt_step = "baseline"
    else:
        ckpt_dir = os.path.join(args.job_dir, "ckpt")
        if args.ckpt_step:
            if args.ckpt_step.isdigit():
                 ckpt_path = os.path.join(ckpt_dir, f"step_{args.ckpt_step}")
            else:
                 ckpt_path = os.path.join(ckpt_dir, args.ckpt_step)
        else:
            # Auto-detect latest "step_X"
            checkpoints = []
            if os.path.isdir(ckpt_dir):
                for d in os.listdir(ckpt_dir):
                    if d.startswith("step_") and d.split("_")[-1].isdigit():
                        checkpoints.append(int(d.split("_")[-1]))
            
            if not checkpoints: 
                raise FileNotFoundError(f"No 'step_X' checkpoints found in {ckpt_dir}")
                
            latest = max(checkpoints)
            ckpt_path = os.path.join(ckpt_dir, f"step_{latest}")
            args.ckpt_step = f"step_{latest}"
        
        print(f"[Rank {global_rank}] Loading checkpoint from: {ckpt_path}")
        model.load_lora(ckpt_path)

    model.to(device_id).eval()

    # 3. Load Metric Models (LPIPS, SigLIP, DINO)
    print(f"[Rank {global_rank}] Loading Metric Models...")
    
    # Ensure Rank 0 downloads everything first
    if torch.distributed.is_initialized():
        if global_rank == 0:
            print("[Rank 0] Pre-downloading metric models to cache...")
            lpips.LPIPS(net='vgg')
            AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
            SiglipModel.from_pretrained("google/siglip-base-patch16-224")
            AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            AutoModel.from_pretrained('facebook/dinov2-base')
        
        torch.distributed.barrier()

    # Now safe for all ranks to load from cache
    lpips_model = lpips.LPIPS(net='vgg').to(device_id).eval()
    siglip_proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-224", local_files_only=True)
    siglip_model = SiglipModel.from_pretrained("google/siglip-base-patch16-224", local_files_only=True).to(device_id).eval()
    dino_proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base', local_files_only=True)
    dino_model = AutoModel.from_pretrained('facebook/dinov2-base', local_files_only=True).to(device_id).eval()

    # 4. Setup Dataset
    print(f"[Rank {global_rank}] Initializing {args.dataset} dataset...")
    if args.dataset == 'nabirds':
        dataset = NABirdsDataset(
            root=config['train']['dataset']['root'],
            split=args.split,
            condition_size=tuple(config['train']['dataset']['condition_size']),
            target_size=tuple(config['train']['dataset']['target_size']),
            return_raw=False, 
            long_captions=config['train']['dataset'].get('long_captions', False)
        )
    elif args.dataset == 'inat':
        dataset = INatDataset(
            root=config['train']['dataset']['root'],
            split=args.split, 
            target_size=tuple(config['train']['dataset']['target_size']),
            condition_size=tuple(config['train']['dataset']['condition_size']),
            return_raw=True,
            long_captions=config['train']['dataset'].get('long_captions', False)
        )

    # 5. Determine Indices for this Rank
    total_len = len(dataset)
    start = args.start_index
    end = args.end_index if args.end_index is not None else total_len
    end = min(end, total_len)
    
    all_indices = list(range(start, end))
    my_indices = all_indices[global_rank::world_size]
    
    print(f"[Rank {global_rank}] Processing {len(my_indices)} items from {total_len} total.")

    # 6. Output Directories
    out_dir = os.path.join(args.job_dir, f"eval_{args.dataset}_{args.split}_{args.ckpt_step}")
    gen_dir = os.path.join(out_dir, "gen")
    
    if global_rank == 0:
        os.makedirs(gen_dir, exist_ok=True)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        
    results_path = os.path.join(out_dir, f'results_rank_{global_rank}.csv')
    
    # 7. Evaluation Loop
    # Check if CSV exists to handle header writing
    header_written = os.path.exists(results_path)
    
    for idx in tqdm(my_indices, desc=f"Rank {global_rank}", position=global_rank):
        
        # Load Data
        data = load_eval_data(dataset, idx, args.dataset)
        if not data: continue
        # --- Bidirectional Logic (Unified for NABirds and iNat) ---
        if args.dataset == 'nabirds':
            img1, mask1, depth1, kp1, cap1, img2, mask2, depth2, kp2, cap2 = data
            basename = f'{args.split}_pair_{idx:04d}'
        else:
            # Unpack iNat dict into variables
            img1, mask1, depth1, kp1 = data['img1'], data['mask1'], data['depth1'], data['kp1']
            img2, mask2, depth2, kp2 = data['img2'], data['mask2'], data['depth2'], data['kp2']
            cap1 = cap2 = data['caption']
            basename = data['id']

        grid_path = os.path.join(out_dir, f"{basename}.jpg")
        
        # --- Check Existing & Recompute Logic ---
        gen_1_in_2 = None
        gen_2_in_1 = None
        
        # Check 1->2
        p1 = os.path.join(gen_dir, f"{basename}_1_in_2.png")
        if os.path.exists(p1):
            if not args.skip_existing:
                # Case A: File exists, but user said DO NOT SKIP -> Force Regenerate
                gen_1_in_2 = None 
            elif args.recompute_metrics:
                # Case B: File exists, User skipped gen, wants metrics -> Load Image
                try:
                    gen_1_in_2 = np.array(Image.open(p1))
                except Exception:
                    gen_1_in_2 = None # Corrupt file, regenerate
            else:
                # Case C: File exists, User skipped gen, no metrics -> Skip entirely
                gen_1_in_2 = "skipped"
        
        # Check 2->1
        p2 = os.path.join(gen_dir, f"{basename}_2_in_1.png")
        if os.path.exists(p2):
            if not args.skip_existing:
                gen_2_in_1 = None
            elif args.recompute_metrics:
                try:
                    gen_2_in_1 = np.array(Image.open(p2))
                except Exception:
                    gen_2_in_1 = None
            else:
                gen_2_in_1 = "skipped"

        # Direction 1: 1 (Ref) -> 2 (Target)
        if gen_1_in_2 is None:
            gen_1_in_2 = infer_pair(model, img1, mask1, img2, mask2, depth2, kp2, cap2, config, args.seed, guidance_scale=args.guidance_scale)
            h2, w2 = img2.shape[:2]
            
            gen_1_in_2 = unpad_image(gen_1_in_2, h2, w2)
            gen_1_in_2 = cv2.resize(gen_1_in_2, (w2, h2), interpolation=cv2.INTER_AREA)
                
            Image.fromarray(gen_1_in_2).save(os.path.join(gen_dir, f"{basename}_1_in_2.png"))

        # Direction 2: 2 (Ref) -> 1 (Target)
        if gen_2_in_1 is None:
            gen_2_in_1 = infer_pair(model, img2, mask2, img1, mask1, depth1, kp1, cap1, config, args.seed, guidance_scale=args.guidance_scale)
            h1, w1 = img1.shape[:2]
            
            gen_2_in_1 = unpad_image(gen_2_in_1, h1, w1)
            gen_2_in_1 = cv2.resize(gen_2_in_1, (w1, h1), interpolation=cv2.INTER_AREA)
                
            Image.fromarray(gen_2_in_1).save(os.path.join(gen_dir, f"{basename}_2_in_1.png"))

        # Helper checks to avoid numpy ambiguity errors
        is_skipped_1 = isinstance(gen_1_in_2, str) and gen_1_in_2 == "skipped"
        is_skipped_2 = isinstance(gen_2_in_1, str) and gen_2_in_1 == "skipped"

        # If both were skipped, continue
        if is_skipped_1 and is_skipped_2:
            continue

        # Metrics
        # Handle cases where one direction might be skipped but other recomputed
        metrics_row = {'pair_id': basename}
        
        if not is_skipped_1:
            m1 = eval_metrics(img2, gen_1_in_2, mask2, lpips_model, siglip_model, siglip_proc, dino_model, dino_proc, device_id)
            metrics_row.update({'lpips_1_in_2': m1[0], 'l2_1_in_2': m1[1], 'siglip_1_in_2': m1[2], 'dino_1_in_2': m1[3]})
        
        if not is_skipped_2:
            m2 = eval_metrics(img1, gen_2_in_1, mask1, lpips_model, siglip_model, siglip_proc, dino_model, dino_proc, device_id)
            metrics_row.update({'lpips_2_in_1': m2[0], 'l2_2_in_1': m2[1], 'siglip_2_in_1': m2[2], 'dino_2_in_1': m2[3]})
        
        # --- Incremental Save ---
        pd.DataFrame([metrics_row]).to_csv(results_path, mode='a', header=not header_written, index=False)
        header_written = True

        # Save Grid (Only if freshly generated or recomputing BOTH)
        if not is_skipped_1 and not is_skipped_2:
            row1 = create_vis_row(img1, gen_2_in_1, mask1, depth1, kp1, cap1, config)
            row2 = create_vis_row(img2, gen_1_in_2, mask2, depth2, kp2, cap2, config)
            w1, w2 = row1.shape[1], row2.shape[1]
            if w1 > w2: row2 = np.hstack([row2, np.ones((row2.shape[0], w1-w2, 3), dtype=np.uint8)*255])
            elif w2 > w1: row1 = np.hstack([row1, np.ones((row1.shape[0], w2-w1, 3), dtype=np.uint8)*255])
            grid = np.vstack([row1, row2])
            Image.fromarray(grid).save(grid_path)

    print(f"[Rank {global_rank}] Finished. Results saved incrementally to {results_path}")

if __name__ == "__main__":
    main()