import torch
import os
import numpy as np
from PIL import Image
from peft import LoraConfig
from diffusers import FluxPipeline
from diffusers.pipelines import FluxKontextPipeline
from typing import Dict, Any

from shared.models.base_flux_trainer import SharedFluxTrainer
from shared.pipeline_utils import encode_images
from omini.pipeline.flux_omini_orig import transformer_forward as forward_orig
from omini.pipeline.flux_omini_olin import transformer_forward as forward_olin

class OminiModelWrapper(SharedFluxTrainer):
    def __init__(self, config):
        super().__init__(
            optimizer_config=config['train']['optimizer'],
            model_config=config.get('model', {})
        )
        self.config = config
        dtype = getattr(torch, config.get("dtype", "bfloat16"))
        
        self.is_kontext = 'kontext' in config['flux_path'].lower() or config.get('model_arch') == 'kontext'

        if self.is_kontext:
            print(f"--- Using Omini Kontext (Olin) Architecture ---")
            self.flux_pipe = FluxKontextPipeline.from_pretrained(
                config['flux_path'], 
                torch_dtype=dtype
            )
            self.forward_fn = forward_olin
        else:
            print(f"--- Using Omini Original (Schnell) Architecture ---")
            self.flux_pipe = FluxPipeline.from_pretrained(
                config['flux_path'], 
                torch_dtype=dtype
            )
            self.forward_fn = forward_orig
        
        # Disable Progress Bar
        self.flux_pipe.set_progress_bar_config(disable=True)
        
        self.transformer = self.flux_pipe.transformer
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()
        self.transformer.requires_grad_(False)
        self.transformer.train()
        
        self.adapter_names = [None, None] 
        if config['train'].get("single_adapter", False):
             self.adapter_names.append("default")
        else:
            cond_mode = config['train'].get('conditioning_mode', 'fill')
            self.adapter_names.append("subject")
            if cond_mode in ['fill', 'depth', 'keypoints']:
                self.adapter_names.append(cond_mode)

        self.setup_lora(config)

    def on_fit_start(self):
        print(f"--- Moving models to {self.device} ---")
        self.flux_pipe.to(self.device)
    
    def setup_lora(self, config):
        lora_config = config['train']['lora_config']
        active_adapters = [a for a in self.adapter_names if a is not None]
        
        for name in active_adapters:
            if name == "subject":
                print("Loading pretrained Omini 'subject' adapter...")
                self.flux_pipe.load_lora_weights("Yuanshi/OminiControl", weight_name="experimental/subject.safetensors", adapter_name="subject", cache_dir="ckpts")
            elif name == "fill":
                print("Loading pretrained Omini 'fill' adapter...")
                self.flux_pipe.load_lora_weights("Yuanshi/OminiControl", weight_name="experimental/fill.safetensors", adapter_name="fill", cache_dir="ckpts")
            elif name == "depth":
                print("Loading pretrained Omini 'depth' adapter...")
                self.flux_pipe.load_lora_weights("Yuanshi/OminiControl", weight_name="experimental/depth.safetensors", adapter_name="depth", cache_dir="ckpts")
            else:
                # Initialize new adapter for anything else (e.g., 'default' or custom)
                print(f"Initializing new adapter '{name}'...")
                self.transformer.add_adapter(LoraConfig(**lora_config), adapter_name=name)

        if active_adapters:
            self.flux_pipe.set_adapters(active_adapters)

        if config['train'].get('gradient_checkpointing', False):
            self.transformer.enable_gradient_checkpointing()
        
        self.trainable_params = list(filter(lambda p: p.requires_grad, self.transformer.parameters()))

    # --- IMPLEMENTING ABSTRACT METHODS ---
    def get_pipeline(self): return self.flux_pipe
    def get_transformer(self): return self.transformer
    def get_target_image(self, batch): return batch["image"]

    def prepare_model_inputs(self, batch, x_t, t, x_0, img_ids) -> Dict[str, Any]:
        conditions = []
        position_deltas = []
        position_scales = []
        latent_masks = []
        
        for i in range(100):
            if f"condition_{i}" not in batch: break
            conditions.append(batch[f"condition_{i}"])
            position_deltas.append(batch.get(f"position_delta_{i}", [[0, 0]]))
            position_scales.append(batch.get(f"position_scale_{i}", [1.0])[0])
            latent_masks.append(batch.get(f"condition_latent_mask_{i}", None))

        condition_latents, condition_ids = [], []
        for i, (cond, p_delta, p_scale, latent_mask) in enumerate(zip(conditions, position_deltas, position_scales, latent_masks)):
            c_latents, c_ids = encode_images(self.flux_pipe, cond)

            if self.is_kontext and i == 0:
                c_ids[..., 0] = 1
            
            if isinstance(p_delta, torch.Tensor): p_delta = p_delta.to(self.device)
            if isinstance(p_scale, torch.Tensor): p_scale = p_scale.to(self.device)
            if latent_mask is not None and isinstance(latent_mask, torch.Tensor):
                latent_mask = latent_mask.to(self.device)
    
            # 2. ONLY apply position delta if NOT Kontext (or if you are sure you want it enabled now)
            if not self.is_kontext: 
                c_ids[:, 1] += p_delta[0][0]
                c_ids[:, 2] += p_delta[0][1]
            
            if p_scale != 1.0:
                scale_bias = (p_scale - 1.0) / 2
                c_ids[:, 1:] *= p_scale
                c_ids[:, 1:] += scale_bias

            if latent_mask is not None:
                c_latents = c_latents[latent_mask]
                c_ids = c_ids[latent_mask[0]]
                
            condition_latents.append(c_latents)
            condition_ids.append(c_ids)
            
        prompt_embeds, pooled_prompt_embeds, text_ids = self.flux_pipe.encode_prompt(
            prompt=batch["description"],
            prompt_2=None,
            device=self.device
        )
        
        branch_n = 2 + len(conditions)
        group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool, device=self.device)
        group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(conditions), device=self.device))
        if self.model_config.get("independent_condition", False):
            group_mask[2:, :2] = False
        
        current_adapters = self.adapter_names[:]
        if len(current_adapters) < branch_n:
            current_adapters.extend([current_adapters[-1]] * (branch_n - len(current_adapters)))
        current_adapters = current_adapters[:branch_n]

        guidances_list = [None] * branch_n
        if getattr(self.transformer.config, "guidance_embeds", False):
             g = torch.ones((x_t.shape[0],), device=self.device, dtype=x_t.dtype)
             guidances_list = [g] * branch_n

        return {
            "transformer": self.transformer,
            "image_features": [x_t, *condition_latents],
            "text_features": [prompt_embeds],
            "img_ids": [img_ids, *condition_ids],
            "txt_ids": [text_ids],
            "timesteps": [t, t] + [torch.zeros_like(t)] * len(conditions),
            "pooled_projections": [pooled_prompt_embeds] * branch_n,
            "guidances": guidances_list,
            "adapters": current_adapters,
            "group_mask": group_mask,
            "return_dict": False
        }

    def forward_transformer(self, **kwargs):
        return self.forward_fn(**kwargs)[0]
    
    def save_lora(self, path: str):
        from diffusers import FluxPipeline
        from peft import get_peft_model_state_dict
        import os
        os.makedirs(path, exist_ok=True)
        unique_adapters = set([a for a in self.adapter_names if a is not None])
        for adapter_name in unique_adapters:
            print(f"Saving adapter: {adapter_name}")
            FluxPipeline.save_lora_weights(
                save_directory=path,
                weight_name=f"{adapter_name}.safetensors",
                transformer_lora_layers=get_peft_model_state_dict(
                    self.transformer, adapter_name=adapter_name
                ),
                safe_serialization=True,
            )
    
    def load_lora(self, path: str):
        print(f"--- Loading Omini LoRA adapters from {path} ---")
        self.flux_pipe.unload_lora_weights()
        unique_adapters = set([a for a in self.adapter_names if a is not None])
        loaded_adapters = []
        for adapter_name in unique_adapters:
            weight_name = f"{adapter_name}.safetensors"
            file_path = os.path.join(path, weight_name)
            if os.path.exists(file_path):
                print(f"Loading adapter: {adapter_name}")
                self.flux_pipe.load_lora_weights(
                    path, weight_name=weight_name, adapter_name=adapter_name
                )
                loaded_adapters.append(adapter_name)
        
        if loaded_adapters:
            self.flux_pipe.set_adapters(loaded_adapters)

    @torch.no_grad()
    def predict(self, batch, seed=42, height=None, width=None, guidance_scale=3.5):
        if self.flux_pipe.device != self.device:
            self.flux_pipe.to(self.device)
        
        if height is None or width is None:
            default_h, default_w = self.config['train']['dataset'].get('target_size', [1024, 1024])
            height = height or default_h
            width = width or default_w
        
        if self.is_kontext:
            from omini.pipeline.flux_omini_olin import generate, Condition
        else:
            from omini.pipeline.flux_omini_orig import generate, Condition
        
        def tensor_to_pil(tensor):
            if tensor.ndim == 4: tensor = tensor[0]
            if tensor.shape[0] == 1: tensor = tensor.repeat(3, 1, 1)
            tensor = tensor.permute(1, 2, 0).cpu().float()
            arr = (tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(arr)

        conditions = []
        if 'condition_0' in batch:
            cond_img = tensor_to_pil(batch['condition_0'])
            
            # --- FIX: Convert Tensor to List to prevent Omini crash ---
            p_delta = batch.get('position_delta_0', [[0, -16]])
            if isinstance(p_delta, torch.Tensor):
                p_delta = p_delta.tolist()
            if isinstance(p_delta[0], list): 
                p_delta = p_delta[0] 
            
            cond_type = batch.get('condition_type_0', 'subject')
            c = Condition(condition=cond_img, adapter_setting=cond_type, position_delta=p_delta, is_spatial=False)
            conditions.append(c)

        if 'condition_1' in batch:
            cond_img = tensor_to_pil(batch['condition_1'])
            cond_type = batch.get('condition_type_1', 'fill')
            c = Condition(condition=cond_img, adapter_setting=cond_type, position_delta=[0,0], is_spatial=True)
            conditions.append(c)

        images = generate(
            self.flux_pipe,
            prompt=batch.get("description", "A photo of a bird."),
            conditions=conditions,
            num_inference_steps=28,
            guidance_scale=guidance_scale or 3.5,
            height=height,
            width=width,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            output_type="pil"
        ).images

        return images[0]

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