from torch.utils.data import DataLoader, IterableDataset
import torch
import torch.nn.functional as F

import os
import random

class CoDDataset(IterableDataset):
    def __init__(self, window_length = 120, root = "/mnt/nas/YoutubeCoD/"):
        super().__init__()

        self.window = window_length
        self.paths = []
        print(f"Loading CoD data from {root}")
        for root_dir in os.listdir(root):
            print(f"Loading {root_dir}")
            splits_dir = os.path.join(root, root_dir, "splits")
            print(f"Splits dir: {splits_dir}")
            if not os.path.isdir(splits_dir):
                print(f"Splits dir does not exist: {splits_dir}")
                continue
            print(f"Splits dir exists: {splits_dir}")
            # Get all files in splits dir
            files = os.listdir(splits_dir)
            # Filter to just the base files (without _mouse or _buttons)
            base_files = [f for f in files if f.endswith("_rgb.pt")]
            print(f"Base files: {len(base_files)}")
            
            for base_file in base_files:
                base_path = os.path.join(splits_dir, base_file)
                # Get just the numeric prefix from the filename
                if os.path.exists(base_path):
                    self.paths.append(base_path)
            print(f"Total paths: {len(self.paths)}")
    
    def get_item(self):
        vid_path = random.choice(self.paths)
        # Load tensors with memory mapping
        print(f"Loading video from {vid_path}")
        vid = torch.load(vid_path, map_location='cpu', mmap=True)

        # Get minimum length
        min_len = len(vid)

        # Get random starting point that allows for full window
        max_start = min_len - self.window
        window_start = random.randint(0, max_start)
        
        # Extract window slices
        vid_slice = vid[window_start:window_start+self.window]

        return vid_slice # [n,c,h,w]

    def __iter__(self):
        while True:
            yield self.get_item()

def collate_fn(x):
    # x is list of triples
    vids = torch.stack(x)      # [b,n,c,h,w]
    return vids

def get_loader(batch_size, **data_kwargs):
    """
    Creates a DataLoader for the CoDDataset with the specified batch size
    
    Args:
        batch_size: Number of samples per batch
        **dataloader_kwargs: Additional arguments to pass to DataLoader
        
    Returns:
        DataLoader instance
    """
    dataset = CoDDataset(**data_kwargs)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    return loader

if __name__ == "__main__":
    import time
    loader = get_loader(32)

    start = time.time()
    batch = next(iter(loader))
    end = time.time()
    
    x = batch
    print(f"Time to load batch: {end-start:.2f}s")
    print(f"Video shape: {x.shape}")