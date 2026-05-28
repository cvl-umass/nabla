import torch
import numpy as np
import cv2
import os
from PIL import Image
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
from peft import LoraConfig
from typing import Dict, Any

from shared.models.base_flux_trainer import SharedFluxTrainer
from insert.models.transformer import tranformer_forward
from insert.models.pipeline_tools import Flux_fill_encode_masks_images, prepare_text_input
from insert.models.image_project import image_output
from shared.data.vision_utils import expand_bbox, box2square, get_bbox_from_mask, pad_to_square

class InsertAnythingWrapper(SharedFluxTrainer):
    def __init__(self, config):
        super().__init__(
            optimizer_config=config['train']['optimizer'],
            model_config=config.get('model', {})
        )
        self.config = config
        dtype = getattr(torch, config.get("dtype", "bfloat16"))

        # 1. Load Pipelines
        self.flux_fill_pipe = FluxFillPipeline.from_pretrained(
            config['flux_fill_path'], 
            torch_dtype=dtype
        )
        self.flux_redux = FluxPriorReduxPipeline.from_pretrained(
            config['flux_redux_path'], 
            torch_dtype=dtype
        )
        self.flux_fill_pipe.vae.enable_tiling()
        
        # 2. Disable Internal Progress Bars
        self.flux_fill_pipe.set_progress_bar_config(disable=True)
        
        # 3. Freeze Components
        self.transformer = self.flux_fill_pipe.transformer
        self.flux_fill_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_fill_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_fill_pipe.vae.requires_grad_(False).eval()
        self.flux_redux.image_embedder.requires_grad_(False).eval()
        self.flux_redux.image_encoder.requires_grad_(False).eval()
        self.transformer.requires_grad_(False)
        self.transformer.train()

        # 4. Load Pretrained Weights OR Setup from Scratch
        pretrained_path = config.get('model', {}).get('insert_anything', {}).get('pretrained_lora_path')
        
        if pretrained_path:
            # If fine-tuning, load the weights (this automatically builds the adapter)
            self._load_pretrained_weights(pretrained_path)
        else:
            # If training from scratch, initialize a fresh adapter
            self.setup_lora(config)

        # 5. Enable Checkpointing
        if config['train'].get('gradient_checkpointing', False):
            self.transformer.enable_gradient_checkpointing()

        self.trainable_params = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))
        print(f"--- Registered {len(self.trainable_params)} trainable LoRA tensors ---")


    def setup_lora(self, config):
        lora_config = config['train']['lora_config']
        self.transformer.add_adapter(LoraConfig(**lora_config), adapter_name="default")

    def on_fit_start(self):
        print(f"--- Moving models to {self.device} ---")
        self.flux_fill_pipe.to(self.device)
        self.flux_redux.to(self.device)
    
    def _load_pretrained_weights(self, path):
        print(f"--- Loading Pretrained LoRA from: {path} ---")
        
        # Check if path is a file or directory to handle both cases
        if os.path.isfile(path):
            dir_name = os.path.dirname(path)
            weight_name = os.path.basename(path)
        else:
            dir_name = path
            weight_name = None # Diffusers will look for default names
            
        try:
            self.flux_fill_pipe.unload_lora_weights()
            
            # This maps the lora_A/B keys correctly to the 'default' adapter
            self.flux_fill_pipe.load_lora_weights(
                dir_name, 
                weight_name=weight_name, 
                adapter_name="default"
            )
            print(f"Successfully loaded LoRA weights from {path}")
        except Exception as e:
            print(f"Failed to load LoRA weights: {e}")

    # --- ABSTRACT METHODS ---
    def get_pipeline(self): return self.flux_fill_pipe
    def get_transformer(self): return self.transformer
    def get_target_image(self, batch): return batch["result"]

    def prepare_model_inputs(self, batch, x_t, t, x_0, img_ids) -> Dict[str, Any]:
        # 1. Redux Encoding (Reference)
        ref_images = batch["ref"] 
        prompt_embeds = []
        pooled_prompt_embeds = []
        
        for i in range(ref_images.shape[0]):
            img_tensor = ref_images[i].detach().cpu().permute(1, 2, 0).numpy()
            pil_image = Image.fromarray((img_tensor * 255).astype('uint8'))
            
            p_embed, pooled_embed = image_output(self.flux_redux, pil_image, self.device)
            
            if p_embed.ndim == 3: p_embed = p_embed.squeeze(0)
            if pooled_embed.ndim == 2: pooled_embed = pooled_embed.squeeze(0)
            prompt_embeds.append(p_embed)
            pooled_prompt_embeds.append(pooled_embed)

        prompt_embeds = torch.stack(prompt_embeds).to(self.device)
        pooled_prompt_embeds = torch.stack(pooled_prompt_embeds).to(self.device)

        # 2. Prepare Text Inputs
        prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
            self.flux_fill_pipe, 
            prompt_embeds=prompt_embeds, 
            pooled_prompt_embeds=pooled_prompt_embeds
        )

        # 3. Encode Conditions (Target + Mask)
        src_latents, mask_latents = Flux_fill_encode_masks_images(
            self.flux_fill_pipe, 
            batch["src"], 
            batch["mask"]
        )
        condition_latents = torch.cat((src_latents, mask_latents), dim=-1)
        
        # 4. Guidance
        guidance = None
        if getattr(self.transformer.config, "guidance_embeds", False):
            guidance = torch.ones((x_t.shape[0],), device=self.device, dtype=x_t.dtype)

        hidden_states = torch.cat((x_t, condition_latents), dim=2)

        return {
            "transformer": self.transformer,
            "model_config": self.model_config,
            "hidden_states": hidden_states,
            "timestep": t,
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": img_ids,
            "joint_attention_kwargs": None,
            "return_dict": False
        }

    def forward_transformer(self, **kwargs):
        return tranformer_forward(**kwargs)[0]
    
    def save_lora(self, path: str):
        from diffusers import FluxFillPipeline
        from peft import get_peft_model_state_dict
        import os
        os.makedirs(path, exist_ok=True)
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
    
    def load_lora(self, path: str):
        print(f"--- Loading LoRA from {path} ---")
        self.flux_fill_pipe.unload_lora_weights()
        self.flux_fill_pipe.load_lora_weights(path, adapter_name="default")

    @torch.no_grad()
    def predict(self, batch, seed=42, height=None, width=None, guidance_scale=30.0):
        # 1. Pipeline Setup
        if self.flux_fill_pipe.device != self.device: 
            self.flux_fill_pipe.to(self.device)
        if self.flux_redux.device != self.device: 
            self.flux_redux.to(self.device)

        if height is None or width is None:
            default_h, default_w = self.config['train']['dataset'].get('target_size', [1024, 1024])
            height = height or default_h
            width = width or default_w

        def to_np(t):
            if t.ndim == 4: t = t[0]
            if t.shape[0] == 1: t = t.repeat(3, 1, 1)
            return np.round(t.permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)

        # 2. Unpack Batch
        ref_np = to_np(batch["ref"])
        src_np = to_np(batch["src"]) 
        mask_np = to_np(batch["mask"])
        
        # Use 0/1 mask for bbox logic (Consistent with Original)
        mask_bool = (mask_np[:, :, 0] > 127).astype(np.uint8)

        # 3. Smart Crop Logic
        kernel = np.ones((7, 7), np.uint8)
        mask_dilated = cv2.dilate(mask_bool, kernel, iterations=2) # Dilate 0/1 mask
        
        tar_box_yyxx = get_bbox_from_mask(mask_dilated)
        tar_box_yyxx_expanded = expand_bbox(mask_dilated, tar_box_yyxx, ratio=1.2)
        tar_box_yyxx_crop = expand_bbox(src_np, tar_box_yyxx_expanded, ratio=2.0)
        # tar_box_yyxx_expanded = expand_bbox(mask_dilated, tar_box_yyxx, ratio=None)
        # tar_box_yyxx_crop = expand_bbox(src_np, tar_box_yyxx_expanded, ratio=None)
        y1, y2, x1, x2 = box2square(src_np, tar_box_yyxx_crop)

        src_crop = src_np[y1:y2, x1:x2]
        mask_crop = mask_np[y1:y2, x1:x2]
        
        src_crop_padded = pad_to_square(src_crop, pad_value=255)
        mask_crop_padded = pad_to_square(mask_crop, pad_value=0)
        
        src_crop_resized = cv2.resize(src_crop_padded, (width, height))
        src_crop_pil = Image.fromarray(src_crop_resized)

        mask_crop_resized = cv2.resize(mask_crop_padded, (width, height))
        mask_crop_resized = (mask_crop_resized > 127).astype(np.uint8) * 255
        mask_crop_pil = Image.fromarray(mask_crop_resized)

        ref_resized = cv2.resize(ref_np, (width, height))
        ref_pil = Image.fromarray(ref_resized)
        
        # 4. CONSTRUCT DIPTYCH
        diptych_img = np.concatenate([np.array(ref_pil), np.array(src_crop_pil)], axis=1)
        
        mask_black = np.zeros_like(np.array(ref_pil))
        diptych_mask = np.concatenate([mask_black, np.array(mask_crop_pil)], axis=1)

        # 5. Run Inference
        prompt_embeds, pooled_prompt_embeds = image_output(self.flux_redux, ref_pil, self.device)
        
        gen_output = self.flux_fill_pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            image=Image.fromarray(diptych_img),
            mask_image=Image.fromarray(diptych_mask),
            height=height,
            width=width * 2,
            guidance_scale=guidance_scale,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            output_type="pil"
        ).images[0]

        # 6. Extract Result
        gen_w, gen_h = gen_output.size
        gen_target = gen_output.crop((gen_w // 2, 0, gen_w, gen_h))
        
        # Resize back using cv2 (Linear)
        h_pad, w_pad = src_crop_padded.shape[:2]
        gen_target_np = cv2.resize(np.array(gen_target), (w_pad, h_pad))
        
        # Unpad
        h_crop, w_crop = src_crop.shape[:2]
        if h_crop == w_crop:
            gen_unpadded = gen_target_np
        else:
            padd = abs(h_crop - w_crop)
            pad_1 = int(padd / 2)
            if h_crop > w_crop:
                gen_unpadded = gen_target_np[:, pad_1 : pad_1 + w_crop]
            else:
                gen_unpadded = gen_target_np[pad_1 : pad_1 + h_crop, :]

        final_image = src_np.copy()
        final_image[y1:y2, x1:x2] = gen_unpadded

        return Image.fromarray(final_image)

    @torch.no_grad()
    def log_samples(self, batch, save_path):
        mini_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor): mini_batch[k] = v[0:1] 
            elif isinstance(v, list): mini_batch[k] = [v[0]]
            else: mini_batch[k] = v

        was_training = self.transformer.training
        self.transformer.eval()
        try:
            # Dynamically use the target size from config
            h, w = self.config['train']['dataset']['target_size']
            generated_image = self.predict(mini_batch, seed=42, height=h, width=w)
            generated_image.save(save_path)
            print(f"Saved sample to {save_path}")
        except Exception as e:
            print(f"Failed to generate sample: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if was_training: self.transformer.train()