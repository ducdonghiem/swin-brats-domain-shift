import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RandomCrop(object):
    def __int__(self, crop_size=128):
        '''
        Applies a crop transformation on a set of scans of size `crop_size`. 
        The starting index of a crop dimension D is within D - `crop_size`

        Args:
            crop_size (int): The size of the cropped sample. Default: 128
        '''
        self.crop_size = crop_size

    def __call__(self, modalities, label):
        '''
        Introduce a random crop of the sample scans and label with size `self.crop_size`.

        Args:
            modalities (list): A list of tensors containing the scans
            label (tensor): A tensor representing the segmentation label
        '''

        shape = label.shape
        start_range = shape[0] - self.crop_size # assumes H and W dimensions are the same
        start = torch.randint(low=0, high=start_range, size=(2,))

        # Crop the modalities
        mod_cropped = []
        for mod in modalities:
            mod_cropped.append(
                mod[:, start[0]:start[0]+self.crop_size, start[1]:start[1]+self.crop_size]
            )

        # Crop the label
        lab_cropped = label[:, start[0]:start[0]+self.crop_size, start[1]:start[1]+self.crop_size]

        return mod_cropped, lab_cropped

class CenterCrop(RandomCrop):
    def __call__(self, modalities, label):
        '''
        Crop the scans at the center of the volumes instead of randomly
        '''

        shape = label.shape
        start = (shape[0] - self.crop_size) // 2 # assumes H and W dimensions are the same

        mod_cropped = []
        for mod in modalities:
            mod_cropped.append(
                mod[:, start:start+self.crop_size, start:start+self.crop_size]
            )

        lab_cropped = label[:, start:start+self.crop_size, start:start+self.crop_size]

        return mod_cropped, lab_cropped

class MRIDataset(Dataset):
    def __init__(self, data_dir=None, cases=None, modalities=["flair", "t1", "t1ce", "t2"], transform=None):
        """

        Args:
            data_dir (str): split directory containing images/ and masks/ subfolders
            cases: optional list of tuples (flair_path, t1_path, t1ce_path, t2_path, seg_path)
            modalities ([str]): ordered list of modality names (matches filenames saved by preprocessor)
            transform (object): optional augmentation / preprocessing function
        """

        self.modalities = modalities
        self.transform = transform

        if cases is None:
            if data_dir is None:
                raise ValueError("Either data_dir or cases must be provided")
            self.cases = self._discover_cases(Path(data_dir))
        else:
            self.cases = self._validate_cases(cases)

    def __len__(self):
        return len(self.cases)

    def _load_volume(self, path, dtype):
        # Load volume (expected .npy format)
        return np.load(path).astype(dtype, copy=False)

    def _discover_cases(self, data_dir):
        '''
        Discover cases from the given data directory. Expects cases to exist in `data_dir`/images or masks/.

        Args:
            data_dir (Path): Path object to data directory
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

    def _validate_cases(self, cases):
        '''
        Validate the provided cases list. Each case should be a tuple of paths matching the `modalities` order, followed by the segmentation path.
        '''
        if len(cases) == 0:
            return []

        expected_len = len(self.modalities) + 1
        for case in cases:
            if not isinstance(case, (tuple, list)) or len(case) != expected_len:
                raise ValueError(
                    "cases must be a list of tuples matching the modalities order"
                )

        return cases

    def _to_dhw(self, vol):
        """
        Convert BraTS array from (H, W, D) to (D, H, W) so D is channel dim for Conv2d in projection block.
        """

        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
        return np.transpose(vol, (2, 0, 1))

    def __getitem__(self, idx):
        '''
        Retrieve scans and segmentation label for a case and apply transformations.

        Args:
            idx (int): Index of the case tuple
        '''
        case = self.cases[idx]
        mod_paths = case[:-1]
        seg_path = case[-1]

        modalities = tuple(
            torch.tensor(self._to_dhw(self._load_volume(path, np.float32)),
                         dtype=torch.float32)
            for path in mod_paths
        )

        label = torch.tensor(self._to_dhw(self._load_volume(
            seg_path, np.int64)), dtype=torch.long)

        if self.transform:
            modalities, label = self.transform(modalities, label)

        return modalities, label


def collate_modalities(batch):
    """
    Collate a batch of (modalities_tuple, label) into
    (modalities_tuple_batch, label_batch).
    """
    modalities, label = zip(*batch)

    modality_batches = tuple(torch.stack(mod_list, dim=0)
                             for mod_list in zip(*modalities))
    label_batch = torch.stack(label, dim=0)
    return modality_batches, label_batch


if __name__ == "__main__":
    # Example usage
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
