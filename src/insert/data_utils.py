import torch
from torch.utils.data import Dataset

class InsertBatchWrapper(Dataset):
    """
    Wraps the shared NABirds dataset to match Insert Anything input format.
    Expected output keys: 'result' (target), 'src' (masked_input), 'mask', 'ref' (subject).
    """
    def __init__(self, base_dataset, config):
        self.base_dataset = base_dataset
        self.config = config

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        # Extract tensors (assume they are already tensors from NABirdsDataset)
        # Dimensions: [C, H, W]
        ref_img = item['condition_subject']
        target_img = item['image']
        src_img = item['condition_fill']
        mask = item['condition_mask']
        
        # --- Create Diptychs (Concatenate along Width / Dimension 2) ---
        
        # 1. Result: [Ref, Target]
        diptych_result = torch.cat([ref_img, target_img], dim=2)
        
        # 2. Source: [Ref, Masked_Target]
        diptych_src = torch.cat([ref_img, src_img], dim=2)
        
        # 3. Mask: [Black_Mask, Target_Mask]
        # Create a black mask (zeros) for the reference side so the model keeps it unchanged
        mask_black = torch.zeros_like(mask)
        diptych_mask = torch.cat([mask_black, mask], dim=2)

        return {
            "result": diptych_result,    # Now 2x width
            "ref": ref_img,              # Keeps single width (for Redux encoder)
            "src": diptych_src,          # Now 2x width
            "mask": diptych_mask         # Now 2x width
        }
