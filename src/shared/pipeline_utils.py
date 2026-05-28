import torch

def encode_images(pipeline, images: torch.Tensor):
    """
    Encodes the images into tokens and ids for FLUX pipeline.
    Common logic used by both Omini and Insert Anything.
    """
    # 1. Preprocess (Range [-1, 1])
    # Note: different pipeline versions might handle this differently, 
    # but generally image_processor.preprocess expects [0, 1] or [0, 255] inputs.
    # We assume the Dataset returns tensors in [0, 1].
    if hasattr(pipeline.image_processor, "preprocess"):
        images = pipeline.image_processor.preprocess(images)
    
    # 2. VAE Encoding
    images = images.to(pipeline.vae.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    
    # 3. Scaling & Shifting (Flux specific)
    if hasattr(pipeline.vae.config, "shift_factor"):
        images = (images - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
        
    # 4. Pack Latents (Flux specific)
    # Using private method _pack_latents from the pipeline
    images_tokens = pipeline._pack_latents(images, *images.shape)
    
    # 5. Prepare IDs
    # Using private method _prepare_latent_image_ids
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    
    # Robustness check for ID packing mismatch (borrowed from Omini)
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
        
    return images_tokens, images_ids