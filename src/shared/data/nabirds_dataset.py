import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from seaborn import color_palette
import cv2
import torchvision.transforms as T
from collections import defaultdict

from shared.data.vision_utils import (
    get_bbox_from_mask, expand_image_mask, pad_to_square, 
    generate_probabilistic_mask
)

class NABirdsDataset(Dataset):
    """
    NABirds dataset for compositional reposing (Subject, Background, Pose).
    """
    def __init__(self, root, split='train', condition_size=(256, 256), target_size=(512, 512), 
                 return_raw=False, part_to_color_map=None, edge_to_color_map=None, 
                 long_captions=False, skeleton_sparse=True, dot_size=15, 
                 line_thickness=10, augment=False,
                 expand_crop=False, 
                 probabilistic_mask=False,
                 mask_background=True,
                 fixed_prompt="A photo of a bird."): # <-- Added fixed prompt
        
        self.root = root
        self.split = split.lower()
        self.condition_size = condition_size
        self.target_size = target_size
        self.return_raw = return_raw
        self.to_tensor = T.ToTensor()
        self.augment = augment
        
        self.expand_crop = expand_crop
        self.probabilistic_mask = probabilistic_mask
        self.mask_background = mask_background
        self.fixed_prompt = fixed_prompt

        self.part_to_color_map = part_to_color_map
        self.edge_to_color_map = edge_to_color_map
        self.long_captions = long_captions
        self.dot_size = dot_size
        self.line_thickness = line_thickness
        self.skeleton_sparse = skeleton_sparse

        self.skeleton = [
            ('crown', 'bill'), ('crown', 'nape'), ('crown', 'left eye'), ('crown', 'right eye'),
            ('bill', 'left eye'), ('bill', 'right eye'), ('bill', 'breast'),
            ('nape', 'left eye'), ('nape', 'right eye'), ('nape', 'back'),
            ('nape', 'left wing'), ('nape', 'right wing'),
            ('back', 'left wing'), ('back', 'right wing'),
            ('tail', 'back'), ('tail', 'right wing'), ('tail', 'left wing'),
            ('breast', 'left eye'), ('breast', 'right eye'), ('breast', 'belly'),
            ('belly', 'tail'), ('breast', 'right wing'), ('breast', 'left wing'),
            ('belly', 'right wing'), ('belly', 'left wing')
        ] if not skeleton_sparse else [ 
            ('bill', 'crown'), ('crown', 'nape'), ('left eye', 'bill'),
            ('right eye', 'bill'), ('belly', 'breast'), ('breast', 'bill'), 
            ('back', 'nape'), ('tail', 'back'), ('left wing', 'back'),
            ('right wing', 'back'),
        ]

        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Split must be one of 'train', 'val', or 'test', but got {self.split}")

        self._load_metadata()
        self._load_handpicked_pairs()

        if self.part_to_color_map is None:
            cmap = color_palette('husl', len(self.part_id_to_name))
            self.part_to_color_map = {v:tuple(int(255 * x) for x in cmap[i]) for i, (k, v) in enumerate(self.part_id_to_name.items())}
        
        if self.edge_to_color_map is None:
            cmap = color_palette('husl', len(self.skeleton))
            self.edge_to_color_map = {i:tuple(int(255 * x) for x in cmap[i]) for i in range(len(self.skeleton))}

        if self.split == 'train':
            train_ids = {i for i, is_train in enumerate(self.train_test_info) if is_train}
            self.class_frames = self._group_by_class(train_ids)
            self._create_train_pairs()
        elif self.split == 'test':
            self.pairs = self.handpicked_pairs
        elif self.split == 'val':
            all_test_ids = {i for i, is_train in enumerate(self.train_test_info) if not is_train}
            val_ids = all_test_ids - self.handpicked_ids
            self.val_frames = sorted(list(val_ids))
            self.val_class_frames = self._group_by_class(val_ids)

    def _load_metadata(self):
        self.image_paths = []
        self.idx_to_uuid = {} 
        with open(os.path.join(self.root, 'images.txt')) as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) == 2:
                    uuid, path = parts
                    self.image_paths.append(os.path.join(self.root, 'images', path))
                    self.idx_to_uuid[idx] = uuid
        self.image_path_to_id = {
            os.path.relpath(p, os.path.join(self.root, 'images')).replace('\\', '/'): i 
            for i, p in enumerate(self.image_paths)
        }
        with open(os.path.join(self.root, 'train_test_split.txt')) as f:
            self.train_test_info = [bool(int(line.strip().split(' ')[1])) for line in f]
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            self.image_class_labels = [int(line.strip().split(' ')[1]) for line in f]
        with open(os.path.join(self.root, 'bounding_boxes.txt')) as f:
            self.bounding_boxes = [list(map(float, line.strip().split(' ')[1:])) for line in f]
        with open(os.path.join(self.root, 'sizes.txt')) as f: 
            image_sizes = {l.split()[0]: (float(l.split()[1]), float(l.split()[2])) for l in f}
        
        # We no longer strictly need captions.json since we hardcode the prompt, 
        # but leaving it for backward compatibility
        captions_json_path = os.path.join(self.root, 'captions_long.json' if self.long_captions else 'captions.json')
        self.captions = {}
        if os.path.exists(captions_json_path):
            with open(captions_json_path, 'r') as f:
                uuid_to_caption = json.load(f)
            for idx, uuid in self.idx_to_uuid.items():
                if uuid in uuid_to_caption:
                    self.captions[idx] = uuid_to_caption[uuid]

        self.part_id_to_name = {int(l.split()[0]): ' '.join(l.split()[1:]) for l in open(os.path.join(self.root, 'parts', 'parts.txt'))}
        uuid_to_keypoints = defaultdict(list)
        with open(os.path.join(self.root, 'parts', 'part_locs.txt')) as f:
            for line in f:
                img_uuid, part_id, x, y, visible = line.strip().split()
                uuid_to_keypoints[img_uuid].append((int(part_id), float(x), float(y), int(visible)))
        self.keypoints = defaultdict(list)
        self.image_sizes = defaultdict(list)
        for idx, uuid in self.idx_to_uuid.items():
            if uuid in uuid_to_keypoints:
                self.keypoints[idx] = uuid_to_keypoints[uuid]
                self.image_sizes[idx] = image_sizes[uuid]

    def _load_handpicked_pairs(self):
        self.handpicked_pairs = []
        self.handpicked_ids = set()
        pairs_path = os.path.join(self.root, 'pairs.csv')
        if not os.path.exists(pairs_path): return
        df = pd.read_csv(pairs_path)
        handpicked_ids_list = []
        for _, row in df.iterrows():
            id1 = self.image_path_to_id.get(row['image_1'].replace('./images/', ''))
            id2 = self.image_path_to_id.get(row['image_2'].replace('./images/', ''))
            if id1 is not None and id2 is not None:
                self.handpicked_pairs.append((id1, id2))
                handpicked_ids_list.extend([id1, id2])
        self.handpicked_ids = set(handpicked_ids_list)

    def _group_by_class(self, id_set):
        class_frames = defaultdict(list)
        for img_id in id_set:
            class_id = self.image_class_labels[img_id]
            class_frames[class_id].append(img_id)
        return class_frames

    def _create_train_pairs(self):
        self.pairs = []
        for class_id in self.class_frames:
            image_ids = self.class_frames[class_id]
            if len(image_ids) > 1:
                for i in range(len(image_ids)):
                    for j in range(i + 1, len(image_ids)):
                        self.pairs.append((image_ids[i], image_ids[j]))
                        self.pairs.append((image_ids[j], image_ids[i]))

    def __len__(self):
        if self.split in ['train', 'test']: return len(self.pairs)
        elif self.split == 'val': return len(self.val_frames)
        return 0

    def _load_image_data(self, image_id):
        path = self.image_paths[image_id]
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(path.replace('images', 'masks').replace('jpg', 'png'), 0) > 128).astype(np.uint8)
        depth = cv2.imread(path.replace('images', 'depth').replace('jpg', 'png'), 0)
        
        caption = self.captions.get(image_id, "A photo of a bird.")
        
        keypoints = self._generate_keypoints_image(image_id, dot_size=self.dot_size, line_thickness=self.line_thickness, draw_skeleton=self.line_thickness > 0)
        if mask.shape[0] != image.shape[0] and mask.shape[0] == image.shape[1] and mask.shape[1] == image.shape[0]:
            mask = np.swapaxes(mask, 0, 1)
        if depth.shape[0] != image.shape[0] and depth.shape[0] == image.shape[1] and depth.shape[1] == image.shape[0]:
            depth = np.swapaxes(depth, 0, 1)
        return image, mask, depth, keypoints, caption

    def _generate_keypoints_image(self, image_id, dot_size=15, bg_color=(0, 0, 0), line_thickness=2, draw_skeleton=True):
        keypoints_raw = self.keypoints.get(image_id, [])
        orig_w, orig_h = self.image_sizes.get(image_id, self.condition_size)
        keypoint_img = Image.new('RGB', (int(orig_w), int(orig_h)), bg_color)
        draw = ImageDraw.Draw(keypoint_img)
        
        scaled_keypoints = {}
        for part_id, x, y, is_visible in keypoints_raw:
            part_name = self.part_id_to_name.get(part_id)
            if part_name:
                scaled_keypoints[part_name] = {'coords': (x, y), 'visible': is_visible}
        if draw_skeleton and hasattr(self, 'skeleton'):
            for i, (part1_name, part2_name) in enumerate(self.skeleton):
                part1_data = scaled_keypoints.get(part1_name)
                part2_data = scaled_keypoints.get(part2_name)
                if part1_data and part2_data and part1_data['visible'] and part2_data['visible']:
                    edge_color = self.edge_to_color_map[i]
                    draw.line([part1_data['coords'], part2_data['coords']], fill=edge_color, width=line_thickness)
        radius = dot_size / 2
        for part_name, data in scaled_keypoints.items():
            if data['visible']:
                scaled_x, scaled_y = data['coords']
                color = self.part_to_color_map.get(part_name, (255, 255, 255)) 
                box = (scaled_x - radius, scaled_y - radius, scaled_x + radius, scaled_y + radius)
                draw.ellipse(box, fill=color)
        keypoint_np = np.array(keypoint_img)
        keypoint_padded = pad_to_square(keypoint_np, pad_value=0, random_pad=False)
        keypoint_resized = cv2.resize(keypoint_padded.astype(np.uint8), self.condition_size, interpolation=cv2.INTER_NEAREST)
        
        return Image.fromarray(keypoint_resized)

    def get_raw_pair(self, idx):
        if self.split == 'train': raise NotImplementedError
        if self.split == 'test': img1_id, img2_id = self.pairs[idx]
        elif self.split == 'val':
            img1_id = self.val_frames[idx]
            class_id = self.image_class_labels[img1_id]
            candidates = self.val_class_frames[class_id]
            if len(candidates) < 2: img2_id = img1_id
            else:
                rand_state = np.random.RandomState(idx)
                img2_id = rand_state.choice([i for i in candidates if i != img1_id])
        img1, mask1, depth1, keypoints1, caption1 = self._load_image_data(img1_id)
        img2, mask2, depth2, keypoints2, caption2 = self._load_image_data(img2_id)
        return img1, mask1, depth1, np.array(keypoints1.resize(depth1.shape, Image.LANCZOS)), caption1, img2, mask2, depth2, np.array(keypoints2.resize(depth2.shape, Image.LANCZOS)), caption2

    def __getitem__(self, idx):
        if self.split == 'val':
            img1_id = self.val_frames[idx]
            class_id = self.image_class_labels[img1_id]
            candidates = self.val_class_frames[class_id]
            if len(candidates) < 2: img2_id = img1_id
            else:
                rand_state = np.random.RandomState(idx)
                img2_id = rand_state.choice([i for i in candidates if i != img1_id])
        else:
            img1_id, img2_id = self.pairs[idx]

        img1, mask1, depth1, keypoints1, caption1 = self._load_image_data(img1_id)
        img2, mask2, depth2, keypoints2, caption2 = self._load_image_data(img2_id)
        caption2 = self.fixed_prompt

        # --- 1. TARGET IMAGE ---
        if self.augment:
            target_image_np = pad_to_square(img2, pad_value=255, random_pad=False)
            target_image_np = cv2.resize(target_image_np.astype(np.uint8), self.target_size)
            target_image = Image.fromarray(target_image_np)
        else:
            target_image_np = pad_to_square(img2, pad_value=255, random_pad=False)
            target_image = Image.fromarray(target_image_np).resize(self.target_size, Image.LANCZOS)

        # --- 2. CONDITION 1: SUBJECT (REF) ---
        x, y, w, h = cv2.boundingRect(mask1)
        if w==0 or h==0: 
            subject_crop = img1
        else:
            subject_crop = img1[y:y+h, x:x+w]

        y1, y2, x1, x2 = get_bbox_from_mask(mask1)
        masked_ref = img1.copy()
        masked_ref = masked_ref[y1:y2, x1:x2]
        mask_ref_crop = mask1[y1:y2, x1:x2]

        if self.expand_crop:
            ratio = np.random.randint(11, 15) / 10.0
            masked_ref, _ = expand_image_mask(masked_ref, mask_ref_crop, ratio=ratio)
        
        masked_ref_padded = pad_to_square(masked_ref, pad_value=255, random_pad=False)
        condition_subject = Image.fromarray(masked_ref_padded.astype(np.uint8)).resize(self.condition_size, Image.LANCZOS)

        # --- 3. CONDITION 2: POSE MASK ---
        depth_padded = pad_to_square(depth2, pad_value=0, random_pad=False)
        depth_resized = cv2.resize(depth_padded.astype(np.uint8), self.condition_size, interpolation=cv2.INTER_NEAREST)
        depth_rgb = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2RGB)
        condition_pose = Image.fromarray(depth_rgb)

        # --- 4. CONDITION 3: BACKGROUND (Bounding Box Blackout) ---
        bg_image = img2.copy()
        
        # EXACT SILHOUETTE BLACKOUT
        bg_image[mask2 > 0] = [0, 0, 0]

        # Keep padding:
        bg_padded = pad_to_square(bg_image, pad_value=255, random_pad=False)
        bg_resized = cv2.resize(bg_padded.astype(np.uint8), self.condition_size)
        condition_bg = Image.fromarray(bg_resized)

        # Kept for backward compatibility with older pipeline scripts
        condition_fill = condition_bg 
        condition_mask = condition_pose 

        if self.return_raw:
            return {
                'image': target_image,
                'caption': caption2, 
                'condition_subject': condition_subject,
                'condition_keypoints': keypoints2,
                'condition_bg': condition_bg,
                'condition_pose': condition_pose,
                'condition_fill': condition_fill, # legacy
                'condition_mask': condition_mask  # legacy
            }
        else:
            return {
                'image': self.to_tensor(target_image),
                'caption': caption2,
                'condition_subject': self.to_tensor(condition_subject),
                'condition_bg': self.to_tensor(condition_bg),
                'condition_pose': self.to_tensor(condition_pose),
                'condition_fill': self.to_tensor(condition_fill), # legacy
                'condition_mask': self.to_tensor(condition_mask),  # legacy
                'condition_keypoints': self.to_tensor(keypoints2)
            }