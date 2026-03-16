import logging
from pathlib import Path
import random

import numpy as np
import torch
import torchio as tio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRIDataset(tio.SubjectsDataset):
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
        case: list[str] = self.cases[idx]

        # The modality type (t1, mask, etc.) should be the name of the file
        mod_map = {}
        for scan_path in case:
            scan_type = scan_path.split('/')[-1].removesuffix('.npy') 
            mod_map[scan_type] = scan_path

        # ScalarImages expect 4D data, so we add channel dimension in front
        subject = tio.Subject(
            flair=tio.ScalarImage(tensor=torch.from_numpy(
                self._load_volume(mod_map['flair'], np.float32)).unsqueeze(0)),
            t1=tio.ScalarImage(tensor=torch.from_numpy(
                self._load_volume(mod_map['t1'], np.float32)).unsqueeze(0)),
            t2=tio.ScalarImage(tensor=torch.from_numpy(
                self._load_volume(mod_map['t1ce'], np.float32)).unsqueeze(0)),
            t1ce=tio.ScalarImage(tensor=torch.from_numpy(
                self._load_volume(mod_map['t2'], np.float32)).unsqueeze(0)),
            mask=tio.LabelMap(tensor=torch.from_numpy(
                self._load_volume(mod_map['mask'], np.int64)).unsqueeze(0))
        )

        if self.transforms:
            for transform, prob in self.transforms:
                if random.random() <= prob:
                    subject = transform(subject)

        # Return images as Tensors with depth as first dimension
        # and remove the channel dimension from the transforms
        flair = self._to_dhw(subject['flair'][tio.DATA].squeeze(0))
        t1 = self._to_dhw(subject['t1'][tio.DATA].squeeze(0))
        t1ce = self._to_dhw(subject['t1ce'][tio.DATA].squeeze(0))
        t2 = self._to_dhw(subject['t2'][tio.DATA].squeeze(0))
        mask = self._to_dhw(subject['mask'][tio.DATA].squeeze(0))

        return tuple(flair, t1, t1ce, t2), mask


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

    tf_prob = 1
    transforms = [
        (tio.RandomElasticDeformation(num_control_points=7), tf_prob),
        (tio.RandomFlip(axes=0), tf_prob),
        (tio.RandomFlip(axes=1), tf_prob),
        (tio.RandomAffine(degrees=20), tf_prob),
        (tio.RandomBiasField(), tf_prob),
    ]

    dataset = MRIDataset(data_dir=split_dir, transforms=transforms)

    print(len(dataset))

    loader = tio.SubjectsLoader(
        dataset,
        collate_fn=collate_modalities,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )

    # Iterate
    for modalities, masks in loader:
        for name, tensor in zip(dataset.modalities, modalities):
            print(f"{name} batch shape:", tensor.shape)  # (B, 1, D, H, W)
        print("Mask batch shape:", masks.shape)     # (B, 1, D, H, W)
        break
