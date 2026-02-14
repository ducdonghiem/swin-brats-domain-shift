import torch
from torch.utils.data import Dataset
import numpy as np


class MRIDataset(Dataset):
    def __init__(self, cases, transform=None):
        """
        cases: list of tuples (flair_path, t1_path, t1ce_path, t2_path, seg_path)
        transform: optional augmentation / preprocessing function
        """

        self.cases = cases
        self.transform = transform

    def __len__(self):
        return len(self.cases)

    def _load_volume(self, path, dtype):
        # Load volume (expected .npy format)
        return np.load(path).astype(dtype, copy=False)

    def _to_dhw(self, vol):
        """
        Convert BraTS array from (H, W, D) to (D, H, W) so D is channel dim for Conv2d.
        """
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {vol.shape}")
        return np.transpose(vol, (2, 0, 1))

    def __getitem__(self, idx):
        flair_path, t1_path, t1ce_path, t2_path, seg_path = self.cases[idx]

        # Treat depth as channel dim -> (D, H, W)
        flair = torch.tensor(self._to_dhw(self._load_volume(flair_path, np.float32)), dtype=torch.float32)
        t1 = torch.tensor(self._to_dhw(self._load_volume(t1_path, np.float32)), dtype=torch.float32)
        t1ce = torch.tensor(self._to_dhw(self._load_volume(t1ce_path, np.float32)), dtype=torch.float32)
        t2 = torch.tensor(self._to_dhw(self._load_volume(t2_path, np.float32)), dtype=torch.float32)

        # Create segmentation tensor
        label = torch.tensor(self._to_dhw(self._load_volume(seg_path, np.int64)), dtype=torch.long)

        if self.transform:
            (flair, t1, t1ce, t2), label = self.transform((flair, t1, t1ce, t2), label)

        return (flair, t1, t1ce, t2), label


def collate_modalities(batch):
    """
    Collate a batch of ((flair, t1, t1ce, t2), label) into
    ((flair_batch, t1_batch, t1ce_batch, t2_batch), label_batch)
    to identify scans as single case.
    """
    modalities, labels = zip(*batch)

    flair_list, t1_list, t1ce_list, t2_list = zip(*modalities)

    flair_batch = torch.stack(flair_list, dim=0)
    t1_batch = torch.stack(t1_list, dim=0)
    t1ce_batch = torch.stack(t1ce_list, dim=0)
    t2_batch = torch.stack(t2_list, dim=0)

    label_batch = torch.stack(labels, dim=0)
    return (flair_batch, t1_batch, t1ce_batch, t2_batch), label_batch

# if __name__ == "__main__":
#     # Example usage
#     base_dir = Path("src/data/raw/BraTS2021_00495")
#     cases = [
#         (
#             str(base_dir / "BraTS2021_00495_flair.nii.gz"),
#             str(base_dir / "BraTS2021_00495_t1.nii.gz"),
#             str(base_dir / "BraTS2021_00495_t1ce.nii.gz"),
#             str(base_dir / "BraTS2021_00495_t2.nii.gz"),
#             str(base_dir / "BraTS2021_00495_seg.nii.gz"),
#         )
#     ]

#     dataset = MRIDataset(cases)

#     loader = DataLoader(
#         dataset,
#         batch_size=2,
#         shuffle=True,
#         num_workers=2,
#         pin_memory=False,
#         collate_fn=collate_modalities
#     )

#     # Iterate
#     for (flair, t1, t1ce, t2), masks in loader:
#         print("Flair batch shape:", flair.shape)    # (B, D, H, W)
#         print("T1 batch shape:", t1.shape)          # (B, D, H, W)
#         print("T1ce batch shape:", t1ce.shape)      # (B, D, H, W)
#         print("T2 batch shape:", t2.shape)          # (B, D, H, W)
#         print("Mask batch shape:", masks.shape)     # (B, D, H, W)
#         break
