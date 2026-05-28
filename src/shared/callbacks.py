import os
import lightning as L
import wandb
import torch
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image

# Use shared vision utils
from shared.data.vision_utils import expand_image_mask, pad_to_square, get_bbox_from_mask, unpad_image

def render_caption_image(caption, width, height):
    if not isinstance(caption, str):
        return np.ones((height, width, 3), dtype=np.uint8) * 255
        
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, width / 1000.0)
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

class UnifiedLoggerCallback(L.Callback):
    def __init__(self, save_path, config, val_dataset=None, vis_indices=None, resume_step_offset=0):
        super().__init__()
        self.save_path = save_path
        self.config = config
        self.val_dataset = val_dataset
        self.test_indices = vis_indices if vis_indices is not None else []
        self.resume_step_offset = resume_step_offset
        self.print_every_n_steps = config['train'].get("print_every_n_steps", 10)
        self.save_interval = config['train'].get("save_interval", 1000)
        self.sample_interval = config['train'].get("sample_interval", 1000)
        self.use_wandb = config['train'].get("use_wandb", False)
        self.last_printed_step = -1
        self.last_saved_step = -1
        self.last_sampled_step = -1

        os.makedirs(os.path.join(save_path, "ckpt"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "samples"), exist_ok=True)
    
    def _get_effective_step(self, trainer):
        """Helper to calculate total steps including previous runs."""
        return trainer.global_step + self.resume_step_offset

    def _prepare_batch(self, ref_img, ref_mask, tar_img, tar_mask, tar_depth, tar_keypoints, tar_caption, device):
        """
        Constructs a unified batch dictionary for model.predict().
        """
        to_tensor = T.ToTensor()
        model_type = self.config.get('model_type', 'omini') 
        cond_size = tuple(self.config['train']['dataset']['condition_size'])
        
        # --- 1. Subject Condition (Reference) ---
        y1, y2, x1, x2 = get_bbox_from_mask(ref_mask)
        
        # Mask background to white (255)
        masked_ref = ref_img.copy()
        masked_ref = masked_ref[y1:y2, x1:x2]

        # Square & Resize
        masked_ref_padded = pad_to_square(masked_ref, pad_value=255, random_pad=False)
        subject_crop = cv2.resize(masked_ref_padded.astype(np.uint8), cond_size, interpolation=cv2.INTER_AREA)

        # --- Mask Processing ---
        tar_mask_inf = tar_mask.copy()
        if tar_mask_inf.max() <= 1: tar_mask_inf = tar_mask_inf * 255

        batch = {}
        
        # Condition 0 setup (Satisfies both Omini and Insert)
        batch["condition_0"] = to_tensor(Image.fromarray(subject_crop)).unsqueeze(0).to(device)
        
        # Delta setup
        is_kontext = 'kontext' in self.config.get('flux_path', '').lower() or self.config.get('model_arch') == 'kontext'
        batch["position_delta_0"] = torch.tensor([[0, 0]]).to(device) if is_kontext else torch.tensor([[0, -cond_size[0] // 16]]).to(device)
        
        batch["condition_type_0"] = "default" if self.config['train'].get("single_adapter") else "subject"
        
        conditioning_mode = self.config['train'].get('conditioning_mode', 'fill')
        batch["description"] = ["A photo of a bird."] if conditioning_mode == 'fill' else [tar_caption]

        # --- 2. Spatial Condition ---
        if conditioning_mode == 'fill':
            bg_image = tar_img.copy()
        
            # EXACT SILHOUETTE BLACKOUT
            bg_image[tar_mask > 0] = [0, 0, 0]

            # Keep padding:
            bg_padded = pad_to_square(bg_image, pad_value=255, random_pad=False)
            cond_spatial = Image.fromarray(cv2.resize(bg_padded.astype(np.uint8), cond_size, interpolation=cv2.INTER_AREA))

        elif conditioning_mode == 'depth':
            depth_padded = pad_to_square(tar_depth, pad_value=0, random_pad=False)
            depth_resized = cv2.resize(depth_padded.astype(np.uint8), cond_size, interpolation=cv2.INTER_NEAREST)
            cond_spatial = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2RGB)

        elif conditioning_mode == 'keypoints':
            kp_padded = pad_to_square(tar_keypoints, pad_value=0, random_pad=False)
            kp_resized = cv2.resize(kp_padded.astype(np.uint8), cond_size, interpolation=cv2.INTER_NEAREST)
            cond_spatial = Image.fromarray(kp_resized)

        batch["condition_1"] = to_tensor(cond_spatial).unsqueeze(0).to(device)
        batch["condition_type_1"] = "default" if self.config['train'].get("single_adapter") else conditioning_mode
        batch["position_delta_1"] = torch.tensor([[0, 0]]).to(device)
        
        # Satisfy specific pipeline kwargs
        batch["ref"] = batch["condition_0"]
        batch["src"] = to_tensor(Image.fromarray(tar_img)).unsqueeze(0).to(device)
        batch["mask"] = to_tensor(Image.fromarray(tar_mask_inf.astype(np.uint8))).unsqueeze(0).to(device)

        return batch

    def _generate_visualizations(self, trainer, pl_module):
        # 1. Remove the "rank != 0" return check
        if not self.val_dataset: return

        # 2. Split indices among GPUs
        total_indices = self.test_indices
        my_indices = [
            idx for i, idx in enumerate(total_indices) 
            if i % trainer.world_size == trainer.global_rank
        ]

        effective_step = self._get_effective_step(trainer)

        # 3. Iterate only over this GPU's assigned indices
        for i, idx in enumerate(my_indices):
            try:
                # 1. Get Raw Data (Full resolution, original AR)
                img1, mask1, depth1, kp1, cap1, img2, mask2, depth2, kp2, cap2 = self.val_dataset.get_raw_pair(idx)

                # --- Direction 1: 1 (Ref) -> 2 (Target) ---
                batch1 = self._prepare_batch(img1, mask1, img2, mask2, depth2, kp2, cap2, pl_module.device)
                
                with torch.no_grad():
                    pl_module.eval()
                    pred_pil1 = pl_module.predict(batch1, seed=42)
                    pl_module.train()
                
                # Resize Generated (Square) -> Original Target AR
                h2, w2 = img2.shape[:2]
                gen1_arr = np.array(pred_pil1)
                gen1_arr = unpad_image(gen1_arr, h2, w2)
                gen1_final = cv2.resize(gen1_arr, (w2, h2), interpolation=cv2.INTER_AREA)

                # --- Direction 2: 2 (Ref) -> 1 (Target) ---
                batch2 = self._prepare_batch(img2, mask2, img1, mask1, depth1, kp1, cap1, pl_module.device)
                
                with torch.no_grad():
                    pl_module.eval()
                    pred_pil2 = pl_module.predict(batch2, seed=42)
                    pl_module.train()
                    
                # Resize Generated (Square) -> Original Target AR
                h1, w1 = img1.shape[:2]
                gen2_arr = np.array(pred_pil2)
                gen2_arr = unpad_image(gen2_arr, h1, w1)
                gen2_final = cv2.resize(gen2_arr, (w1, h1), interpolation=cv2.INTER_AREA)

                # --- Create Grid Row Helper ---
                def create_row(ref, gen, mask, depth, kp, caption):
                    h, w, _ = ref.shape
                    mask_rgb = np.repeat(mask[:, :, None], 3, axis=2) * 255
                    
                    # Ensure mask is resized to match ref if needed (though get_raw_pair usually matches)
                    if mask_rgb.shape[:2] != (h, w):
                         mask_rgb = cv2.resize(mask_rgb, (w, h), interpolation=cv2.INTER_NEAREST)

                    items = [ref, gen, mask_rgb]
                    
                    cond_mode = self.config['train'].get('conditioning_mode', 'fill')
                    if cond_mode == 'depth':
                         d_rgb = cv2.cvtColor(cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2RGB)
                         items.append(d_rgb)
                    elif cond_mode == 'keypoints':
                         items.append(cv2.resize(kp, (w, h), interpolation=cv2.INTER_NEAREST))
                    
                    items.append(render_caption_image(caption, w, h))
                    return np.hstack(items)

                # Build Rows
                row1 = create_row(img1, gen2_final, mask1, depth1, kp1, cap1) # Target is 1
                row2 = create_row(img2, gen1_final, mask2, depth2, kp2, cap2) # Target is 2
                
                # Pad width to match
                w_r1, w_r2 = row1.shape[1], row2.shape[1]
                if w_r1 > w_r2: 
                    padding = np.ones((row2.shape[0], w_r1 - w_r2, 3), dtype=np.uint8) * 255
                    row2 = np.hstack([row2, padding])
                elif w_r2 > w_r1: 
                    padding = np.ones((row1.shape[0], w_r2 - w_r1, 3), dtype=np.uint8) * 255
                    row1 = np.hstack([row1, padding])

                grid_image = np.vstack([row1, row2])
                
                # Save
                base_name = f"step_{effective_step}_sample_{idx}"
                grid_path = os.path.join(self.save_path, "samples", f"{base_name}.jpg")
                Image.fromarray(grid_image).save(grid_path, quality=95)

            except Exception as e:
                print(f"[Rank {trainer.global_rank}] Error generating sample {idx}: {e}")

        # Optional: Wait for all to finish (prevents training from resuming while one GPU is slow)
        trainer.strategy.barrier()

    def on_train_start(self, trainer, pl_module):
        if self.use_wandb and wandb.run is None and trainer.global_rank == 0:
            wandb.init(project=self.config['train'].get("wandb_project", "bird_diffusion"), config=self.config)
        if self.val_dataset:
            self._generate_visualizations(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        effective_step = self._get_effective_step(trainer)
        # Logging
        if trainer.global_rank == 0:
            if effective_step % self.print_every_n_steps == 0 and effective_step != self.last_printed_step:
                loss = outputs["loss"].item()
                log_dict = {
                    "train/loss": loss,
                    "train/step": effective_step,
                    "train/epoch": trainer.current_epoch,
                    "train/lr": trainer.optimizers[0].param_groups[0]['lr']
                }
                if hasattr(pl_module, "log_loss"):
                    log_dict["train/avg_loss"] = pl_module.log_loss
                
                if self.use_wandb: wandb.log(log_dict)
                
                print(f"Step: {effective_step} | Epoch: {trainer.current_epoch} | Loss: {loss:.4f}", flush=True)
                self.last_printed_step = effective_step

            # Checkpointing
            if effective_step > 0 and effective_step % self.save_interval == 0 and effective_step != self.last_saved_step:
                ckpt_path = os.path.join(self.save_path, "ckpt", f"step_{effective_step}")
                print(f"Saving checkpoint to {ckpt_path}...")
                if hasattr(pl_module, "save_lora"):
                    pl_module.save_lora(ckpt_path)
                else:
                    trainer.save_checkpoint(os.path.join(ckpt_path, "lightning_model.ckpt"))
                self.last_saved_step = effective_step

        # Visualization
        if effective_step > 0 and effective_step % self.sample_interval == 0 and effective_step != self.last_sampled_step:
            self._generate_visualizations(trainer, pl_module)
            self.last_sampled_step = effective_step
