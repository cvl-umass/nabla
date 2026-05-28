import os
import json
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# Import shared vision utils for consistency
from shared.data.vision_utils import get_bbox_from_mask, expand_image_mask, pad_to_square

class INatDataset(Dataset):
    def __init__(self, root, split='na', target_size=(512, 512), condition_size=(256, 256), return_raw=False, long_captions=False):
        """
        Args:
            root: Base path containing 'na' and 'unseen' folders.
            split: 'na' or 'unseen'.
        """
        self.split_root = os.path.join(root, split)
        self.target_size = target_size
        self.condition_size = condition_size
        self.return_raw = return_raw
        self.to_tensor = T.ToTensor()
        self.long_captions = long_captions
        
        self.pairs = self._load_data()

    def _load_data(self):
        csv_path = os.path.join(self.split_root, "final_pairs.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at {csv_path}")
            
        df = pd.read_csv(csv_path)
        
        # Load Captions
        cap_file = 'captions_long.json' if self.long_captions else 'captions_short.json'
        cap_path = os.path.join(self.split_root, cap_file)
        captions = {}
        if os.path.exists(cap_path):
            with open(cap_path, 'r') as f:
                captions = json.load(f)
        
        data_list = []
        # We flatten bidirectional pairs here: 
        # Row 0 -> Item (1->2), Item (2->1)
        
        for _, row in df.iterrows():
            obs_dir = row['relative_path']
            
            # Paths
            p1 = os.path.join(self.split_root, obs_dir, row['image_1'])
            p2 = os.path.join(self.split_root, obs_dir, row['image_2'])
            
            # Check existence
            if not (os.path.exists(p1) and os.path.exists(p2)):
                continue
                
            # Caption Keys
            k1 = f"{row['observation_id']}_{os.path.splitext(row['image_1'])[0]}"
            k2 = f"{row['observation_id']}_{os.path.splitext(row['image_2'])[0]}"
            
            c1 = captions.get(k1, f"a photo of a {row.get('common_name', 'bird')}")
            c2 = captions.get(k2, f"a photo of a {row.get('common_name', 'bird')}")

            # Append forward direction (1 is ref, 2 is target)
            data_list.append({
                'ref_path': p1, 'tar_path': p2,
                'ref_caption': c1, 'tar_caption': c2,
                'obs_id': row['observation_id'],
                'direction': '1_in_2' 
            })
            
            # Append reverse direction (2 is ref, 1 is target)
            data_list.append({
                'ref_path': p2, 'tar_path': p1,
                'ref_caption': c2, 'tar_caption': c1,
                'obs_id': row['observation_id'],
                'direction': '2_in_1'
            })
            
        return data_list

    def __len__(self):
        return len(self.pairs)

    def _process_image(self, path, is_mask=False, is_depth=False):
        if is_mask:
            # Mask path construction logic based on eval_inat.py
            # image.jpg -> ../mask/image.png
            # We assume 'path' is the image path
            mask_path = path.replace('images', 'mask').replace('.jpg', '.png')
            if not os.path.exists(mask_path): return np.zeros(self.target_size, dtype=np.uint8) # Fallback
            img = cv2.imread(mask_path, 0)
            return (img > 128).astype(np.uint8)
        
        if is_depth:
            depth_path = path.replace('images', 'depth').replace('.jpg', '.png')
            if not os.path.exists(depth_path): return np.zeros(self.target_size, dtype=np.uint8)
            img = cv2.imread(depth_path, 0)
            return img

        # Image
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        item = self.pairs[idx]
        
        # Load Raw
        ref_img = self._process_image(item['ref_path'])
        ref_mask = self._process_image(item['ref_path'], is_mask=True)
        
        tar_img = self._process_image(item['tar_path'])
        tar_mask = self._process_image(item['tar_path'], is_mask=True)
        tar_depth = self._process_image(item['tar_path'], is_depth=True)
        
        # Prepare Target (Resize)
        target_image = Image.fromarray(tar_img).resize(self.target_size, Image.LANCZOS)
        
        # Prepare Condition Subject (Crop from Ref)
        y1, y2, x1, x2 = get_bbox_from_mask(ref_mask)
        if x2 > x1 and y2 > y1:
            crop = ref_img[y1:y2, x1:x2]
            condition_subject = Image.fromarray(crop).resize(self.condition_size, Image.LANCZOS)
        else:
            condition_subject = Image.new("RGB", self.condition_size) # Fallback

        # Prepare Condition Spatial (Fill/Depth/Mask from Target)
        # 1. Fill
        fill_np = tar_img.copy()
        fill_np[tar_mask > 0] = [0,0,0]
        condition_fill = Image.fromarray(fill_np).resize(self.condition_size, Image.LANCZOS)
        
        # 2. Mask
        mask_resized = cv2.resize(tar_mask, self.condition_size, interpolation=cv2.INTER_NEAREST)
        condition_mask = Image.fromarray(mask_resized)
        
        # 3. Depth
        depth_resized = cv2.resize(tar_depth, self.condition_size, interpolation=cv2.INTER_NEAREST)
        condition_depth = Image.fromarray(cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2RGB))
        
        caption = item['tar_caption']

        if self.return_raw:
            return {
                'image': target_image,
                'caption': caption,
                'condition_subject': condition_subject,
                'condition_fill': condition_fill,
                'condition_depth': condition_depth,
                'condition_mask': condition_mask,
                # Metadata for saving
                'filename_id': f"{item['obs_id']}_{item['direction']}" 
            }
        else:
            # Tensor return
            return {
                'image': self.to_tensor(target_image),
                'caption': caption,
                'condition_subject': self.to_tensor(condition_subject),
                'condition_fill': self.to_tensor(condition_fill),
                'condition_depth': self.to_tensor(condition_depth),
                'condition_mask': self.to_tensor(condition_mask)
            }