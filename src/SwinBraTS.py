import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# folder = "/home/student/dod2/Downloads/BraTS2021_00495"
folder = "/home/student/dod2/Downloads/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_100"

# list all nii.gz files
files = sorted([f for f in os.listdir(folder) if f.endswith(".nii")]) # .nii.gz

print("Found files:", files)

arrays = {}

for f in files:
    path = os.path.join(folder, f)
    img = nib.load(path)
    data = img.get_fdata()          # convert to numpy float64 array
    arrays[f] = data
    print(f"{f}: shape = {data.shape}, dtype = {data.dtype}")

# -------------------------
# Visualization (simple)
# -------------------------

# pick one modality to visualize (e.g., FLAIR)
flair_name = [f for f in files if "flair" in f.lower()][0]
flair = arrays[flair_name]

# print(np.unique(flair))
# print the range of the FLAIR values
print(f"FLAIR value range: {flair.min()} to {flair.max()}")
# print the mean and standard deviation of the FLAIR values
print(f"FLAIR mean: {flair.mean():.2f}, std: {flair.std():.2f}")
# count the number of voxels for each label
# print("Label counts:", {label: np.sum(flair == label) for label in np.unique(flair)})


# choose a slice index (middle of the volume)
slice_idx = flair.shape[2] // 2

plt.figure(figsize=(6,6))
plt.imshow(flair[:, :, slice_idx], cmap="gray")
plt.title(f"{flair_name} — slice {slice_idx}")
plt.axis("off")
plt.show()
