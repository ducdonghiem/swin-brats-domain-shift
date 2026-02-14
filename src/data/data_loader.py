import torch
from torch.utils.data import Dataset
import numpy as np


class MRIDataset(Dataset):
    def __init__(self, cases, transform=None):
        """
        cases: list of tuples (t1_path, t1ce_path, t2_path, flair_path, seg_path)
        transform: optional augmentation / preprocessing function
        """

        self.cases = cases
        self.transform = transform

    def __len__(self):
        return len(self.cases)

    def _load_volume(self, path, dtype):
        # Load volume (expected .npy format)
        return np.load(path).astype(dtype, copy=False)

    def __getitem__(self, idx):
        t1_path, t1ce_path, t2_path, flair_path, seg_path = self.cases[idx]

        # Each modality tensor shape: (155, 240, 240)
        t1 = torch.tensor(self._load_volume(t1_path, np.float32), dtype=torch.float32)
        t1ce = torch.tensor(self._load_volume(t1ce_path, np.float32), dtype=torch.float32)
        t2 = torch.tensor(self._load_volume(t2_path, np.float32), dtype=torch.float32)
        flair = torch.tensor(self._load_volume(flair_path, np.float32), dtype=torch.float32)

        # Create segmentation tensor
        label = None
        label = torch.tensor(self._load_volume(seg_path, np.int64), dtype=torch.long)

        if self.transform:
            (t1, t1ce, t2, flair), label = self.transform((t1, t1ce, t2, flair), label)

        return (t1, t1ce, t2, flair), label


def collate_modalities(batch):
    """
    Collate a batch of ((t1, t1ce, t2, flair), label) into
    ((t1_batch, t1ce_batch, t2_batch, flair_batch), label_batch)
    to identify scans as single case.
    """
    modalities, labels = zip(*batch)

    t1_list, t1ce_list, t2_list, flair_list = zip(*modalities)

    t1_batch = torch.stack(t1_list, dim=0)
    t1ce_batch = torch.stack(t1ce_list, dim=0)
    t2_batch = torch.stack(t2_list, dim=0)
    flair_batch = torch.stack(flair_list, dim=0)

    label_batch = torch.stack(labels, dim=0)
    return (t1_batch, t1ce_batch, t2_batch, flair_batch), label_batch

# if __name__ == "__main__":
#     # Example usage
#     base_dir = Path("src/data/raw/BraTS2021_00495")
#     cases = [
#         (
#             str(base_dir / "BraTS2021_00495_t1.nii.gz"),
#             str(base_dir / "BraTS2021_00495_t1ce.nii.gz"),
#             str(base_dir / "BraTS2021_00495_t2.nii.gz"),
#             str(base_dir / "BraTS2021_00495_flair.nii.gz"),
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
#     for (t1, t1ce, t2, flair), masks in loader:
#         print("T1 batch shape:", t1.shape)          # (B, D, H, W)
#         print("T1ce batch shape:", t1ce.shape)      # (B, D, H, W)
#         print("T2 batch shape:", t2.shape)          # (B, D, H, W)
#         print("Flair batch shape:", flair.shape)    # (B, D, H, W)
#         print("Mask batch shape:", masks.shape)     # (B, D, H, W)
#         break
