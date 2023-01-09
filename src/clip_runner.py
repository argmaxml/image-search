import json
import tempfile
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch
from smart_open import open
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from tqdm.auto import tqdm


class ClipRemoteImageDataset(Dataset):
    def __init__(self, images, transform=None, download_path=Path(".") / "images"):
        self.images = images
        self.transform = transform
        self.download_path = download_path
        self.download_path.mkdir(exist_ok=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        with open(self.images[idx], 'rb') as remote_path:
            local_path = tempfile.NamedTemporaryFile('wb')
            local_path.write(remote_path.read())
            image = read_image(local_path.name)
            local_path.close()
            return self.transform(image)


def encode_images(images, batch_size=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    img_dl = DataLoader(
        dataset=ClipRemoteImageDataset(images, transforms.Resize((224, 224))),
        batch_size=batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        return np.vstack([model.encode_image(samples.to(device)).cpu().numpy()
                          for samples in tqdm(img_dl)])


if __name__ == "__main__":
    n_items = 20
    __dir__ = Path(__file__).absolute().parent.parent
    df = pd.read_parquet(__dir__ / "data" / "product_images.parquet")
    embeddings = encode_images(df['primary_image'].iloc[:n_items])
    np.save(__dir__ / 'data/clip_emb.npy', embeddings)
    with open(__dir__ / 'data/clip_ids.json', 'w') as f:
        json.dump(list(df['asin'].iloc[:n_items]), f)
