import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class FlickrAlignDataset(Dataset):
    """Loads image-text pair from Flickr30k."""

    def __init__(self, df, text_embeddings, images_dir):
        self.df = df.reset_index(drop=True)
        self.text_embeddings = text_embeddings
        self.images_dir = images_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image"])
        try:
            image = Image.open(img_path).convert("RGB")
            return {
                "image": image,
                "text_embedding": torch.tensor(
                    self.text_embeddings[idx], dtype=torch.float32
                ),
            }
        except Exception:
            return None
