import lightning as L
import torch
import torch.nn.functional as F
import prodigyopt
from typing import Dict, Any

# Import the shared utility
from shared.pipeline_utils import encode_images

class SharedFluxTrainer(L.LightningModule):
    def __init__(self, optimizer_config: dict = None, model_config: dict = None):
        super().__init__()
        self.optimizer_config = optimizer_config or {}
        self.model_config = model_config or {}
        self.trainable_params = [] # Subclasses must populate this
        
        # Track logging metrics
        self.log_loss = 0.0
        self.last_t = 0.0

    # --- 🏗️ ABSTRACT METHODS (Subclasses MUST Implement) ---

    def get_pipeline(self):
        """Return the underlying FluxPipeline (for VAE/Tokenizer access)."""
        raise NotImplementedError

    def get_transformer(self):
        """Return the Transformer model to be trained."""
        raise NotImplementedError

    def get_target_image(self, batch) -> torch.Tensor:
        """Extract the Ground Truth target image from the batch."""
        raise NotImplementedError

    def prepare_model_inputs(self, batch, x_t, t, x_0, img_ids) -> Dict[str, Any]:
        """
        Prepare the specific kwargs for the transformer forward pass.
        
        Args:
            batch: The raw batch from the dataloader.
            x_t: Noisy latents (at time t).
            t: Timestep tensor.
            x_0: Original clean latents (for reference/conditioning if needed).
            img_ids: Positional IDs for the image tokens.
            
        Returns:
            Dict of arguments to pass to `forward_transformer`.
        """
        raise NotImplementedError

    def forward_transformer(self, **kwargs):
        """
        Call the specific transformer forward wrapper.
        Return the prediction tensor (noise or velocity).
        """
        raise NotImplementedError

    # --- 🧪 EXPERIMENTAL HOOKS (Override for Research) ---

    def manipulate_latents(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Hook to modify clean latents BEFORE noise is added.
        Useful for Latent Regularization or editing experiments.
        """
        return x_0

    def compute_custom_loss(self, pred, target, x_0, x_t, t) -> torch.Tensor:
        """
        Hook to add auxiliary losses (e.g., preservation loss, geometric alignment).
        Default is 0.
        """
        return torch.tensor(0.0, device=self.device, dtype=self.dtype)

    # --- 🔄 UNIFIED TRAINING LOOP ---

    def training_step(self, batch, batch_idx):
        pipeline = self.get_pipeline()
        
        # 1. Get Target Image
        img = self.get_target_image(batch)
        if isinstance(img, torch.Tensor):
            img = img.to(self.device)

        # START NO_GRAD HERE
        with torch.no_grad():
            # 2. Encode to Latents
            x_0, img_ids = encode_images(pipeline, img)
            x_0 = self.manipulate_latents(x_0)

            # 3. Flow Matching Noise Schedule
            x_1 = torch.randn_like(x_0)
            t = torch.sigmoid(torch.randn((x_0.shape[0],), device=self.device))
            t_expanded = t.view(-1, 1, 1)
            x_t = ((1 - t_expanded) * x_0 + t_expanded * x_1).to(self.dtype)

            # 4. Prepare Conditions
            # Safely call this here without generating gradients
            forward_kwargs = self.prepare_model_inputs(batch, x_t, t, x_0, img_ids)

        # 5. Transformer Forward Pass (Gradients ON)
        pred = self.forward_transformer(**forward_kwargs)

        # 6. Loss Calculation
        target = x_1 - x_0 # Flow matching target: velocity (v = x_1 - x_0)
        
        # Standard MSE Loss
        diffusion_loss = F.mse_loss(pred, target, reduction="mean")
        
        # [HOOK] Custom Loss
        custom_loss = self.compute_custom_loss(pred, target, x_0, x_t, t)
        
        total_loss = diffusion_loss + custom_loss

        # 7. Logging
        self.log("train_loss", diffusion_loss, prog_bar=True)
        if custom_loss.item() != 0:
            self.log("custom_loss", custom_loss)
        
        # Update internal metric for progress bars
        self.last_t = t.mean().item()
        self.log_loss = (
            total_loss.item() 
            if self.log_loss == 0 
            else self.log_loss * 0.95 + total_loss.item() * 0.05
        )

        return total_loss

    def configure_optimizers(self):
        """
        Shared optimizer configuration. 
        Relies on 'self.optimizer_config' and 'self.trainable_params'.
        """
        opt_config = self.optimizer_config
        
        # Ensure parameters are set
        if not self.trainable_params:
            print("Warning: No trainable parameters found. Did you forget to set self.trainable_params in init?")
            return None

        # Unfreeze
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Select Optimizer
        opt_type = opt_config.get("type", "AdamW")
        opt_params = opt_config.get("params", {"lr": 1e-4})

        if opt_type == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_params)
        elif opt_type == "Prodigy":
            optimizer = prodigyopt.Prodigy(self.trainable_params, **opt_params)
        elif opt_type == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_params)
        else:
            raise NotImplementedError(f"Optimizer {opt_type} not implemented.")
            
        return optimizer