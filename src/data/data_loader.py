import logging
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRIDataset(Dataset):
    def __init__(self, data_dir=None, modalities=["flair", "t1", "t1ce", "t2"], transforms=None):
        """

        Args:
            data_dir (str): split directory containing images/ and masks/ subfolders
            cases (list(tuple(str))): optional list of tuples (flair_path, t1_path, t1ce_path, t2_path, seg_path)
            modalities (list(str)): ordered list of modality names (matches filenames saved by preprocessor)
            transforms (tuple(func, prob)): optional list of transforms for data augmentation
        """

        self.modalities = modalities
        self.transforms = transforms

        if data_dir is None:
            raise ValueError("Either data_dir or cases must be provided")
        self.cases = self._build_dataset(Path(data_dir))

    def __len__(self):
        return len(self.cases)

    def _load_volume(self, path, dtype):
        # Load volume (expected .npy format)
        return np.load(path).astype(dtype, copy=False)

    def _build_dataset(self, data_dir):
        '''
        Discover cases from the given data directory. Expects cases to exist in `data_dir`/images or masks/.
        '''
        images_dir = data_dir / "images"
        masks_dir = data_dir / "masks"

        if not images_dir.exists():
            raise FileNotFoundError(f"Missing images directory: {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Missing masks directory: {masks_dir}")

        cases = []
        for patient_dir in sorted(d for d in images_dir.iterdir() if d.is_dir()):
            patient_id = patient_dir.name
            mod_paths = [patient_dir / f"{mod}.npy" for mod in self.modalities]
            mask_path = masks_dir / patient_id / "mask.npy"

            if not mask_path.exists() or not all(p.exists() for p in mod_paths):
                logger.warning(
                    "Skipping %s due to missing modality or mask", patient_id)
                continue

            cases.append(tuple([str(p) for p in mod_paths] + [str(mask_path)]))

        return cases

    def _to_dhw(self, vol):
        """
        Convert BraTS array from (H, W, D) to (D, H, W) so D is channel dim for Conv2d.
        """
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
        return np.transpose(vol, (2, 0, 1))

    def __getitem__(self, idx):
        case = self.cases[idx]
        mod_paths = case[:-1]
        seg_path = case[-1]

        # TVTensors are essentially regular Tensors with subclasses for diverse image tasks
        # Required when applying torchvision v2 transformations
        modalities = tuple(tv_tensors.Image(
                self._to_dhw(self._load_volume(path, np.float32)), dtype=torch.float32)
        for path in mod_paths)

        mask = tv_tensors.Mask(
            self._to_dhw(self._load_volume(seg_path, np.int64)), dtype=torch.long)

        if self.transforms:
            for transform, prob in self.transforms:
                if random.random() <= prob:
                    modalities, mask = transform([modalities, mask])

        return modalities, mask


def collate_modalities(batch):
    """
    Collate a batch of (modalities_tuple, mask) into
    (modalities_tuple_batch, mask_batch).
    """
    modalities, mask = zip(*batch)

    modality_batches = tuple(torch.stack(mod_list, dim=0)
                             for mod_list in zip(*modalities))
    mask_batch = torch.stack(mask, dim=0)
    return modality_batches, mask_batch


if __name__ == "__main__":
    split_dir = Path("src/data/processed/train")

    dataset = MRIDataset(data_dir=split_dir)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        collate_fn=collate_modalities
    )

    # Iterate
    for modalities, masks in loader:
        for name, tensor in zip(dataset.modalities, modalities):
            print(f"{name} batch shape:", tensor.shape)  # (B, 1, D, H, W)
        print("Mask batch shape:", masks.shape)     # (B, 1, D, H, W)
        break
