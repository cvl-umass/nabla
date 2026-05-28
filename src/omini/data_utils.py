import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T

class OminiBatchWrapper(Dataset):
    """
    Wraps the shared NABirds dataset to match OminiControl input format.
    """
    def __init__(self, base_dataset, config):
        self.base_dataset = base_dataset
        self.config = config
        self.conditioning_mode = config['train'].get('conditioning_mode', 'fill')
        self.single_adapter = config['train'].get("single_adapter", False)
        self.condition_size = config['train']['dataset']['condition_size']
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        batch = { "image": item['image'] }
        
        # Condition 0: Subject (Used in almost all modes)
        batch['condition_0'] = item['condition_subject']
        batch['condition_type_0'] = "default" if self.single_adapter else 'subject'
        batch['position_delta_0'] = np.array([0, -self.condition_size[0] // 16])

        next_cond_idx = 1
        
        if self.conditioning_mode == 'fill':
            batch['description'] = "A photo of a bird."
            batch[f'condition_{next_cond_idx}'] = item['condition_fill']
            batch[f'condition_type_{next_cond_idx}'] = "default" if self.single_adapter else 'fill'
            batch[f'position_delta_{next_cond_idx}'] = np.array([0, 0])
            
        elif self.conditioning_mode == 'keypoints':
            batch['description'] = item['caption']
            batch[f'condition_{next_cond_idx}'] = item['condition_keypoints']
            batch[f'condition_type_{next_cond_idx}'] = "default" if self.single_adapter else 'keypoints'
            batch[f'position_delta_{next_cond_idx}'] = np.array([0, 0])
            
        else: # depth
            batch['description'] = item['caption']
            batch[f'condition_{next_cond_idx}'] = item['condition_pose']
            batch[f'condition_type_{next_cond_idx}'] = "default" if self.single_adapter else 'depth'
            batch[f'position_delta_{next_cond_idx}'] = np.array([0, 0])
                
        return batch