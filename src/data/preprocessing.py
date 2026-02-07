import os
import numpy as np
import nibabel as nib
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BraTSPreprocessor:
    """Preprocesses BraTS 2021 dataset for Swin Transformer segmentation"""
    
    def __init__(self, data_dir, output_dir, modalities=None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.modalities = modalities or ['flair', 't1', 't1ce', 't2']
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.test_dir = self.output_dir / 'test'
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(exist_ok=True)
            (dir_path / 'images').mkdir(exist_ok=True)
            (dir_path / 'masks').mkdir(exist_ok=True)
    
    def load_nifti_volume(self, filepath):
        """Load a .nii.gz file and return as numpy array"""
        img = nib.load(filepath)
        return np.asarray(img.dataobj)
    
    def normalize_volume(self, volume):
        """
        Normalize volume to [0, 1] range
        Uses percentile-based normalization to handle outliers
        """
        # Get non-zero voxels (background is 0)
        non_zero = volume[volume > 0]
        
        if len(non_zero) == 0:
            return volume
        
        # Use percentile normalization
        p_low = np.percentile(non_zero, 0.5)
        p_high = np.percentile(non_zero, 99.5)
        
        # Clip and normalize
        volume = np.clip(volume, p_low, p_high)
        volume = (volume - p_low) / (p_high - p_low + 1e-6)
        
        return volume
    
    def process_patient(self, patient_dir):
        """
        Process a single patient directory
        
        Returns:
            tuple: (image_4ch, mask)
        """
        patient_id = patient_dir.name
        logger.info(f"Processing {patient_id}...")
        
        try:
            # Load all modalities
            modalities_dict = {}
            for mod in self.modalities:
                filepath = patient_dir / f"{patient_id}_{mod}.nii.gz"
                if not filepath.exists():
                    logger.warning(f"Missing {mod} for {patient_id}")
                    return None
                
                volume = self.load_nifti_volume(filepath)
                volume = self.normalize_volume(volume)
                modalities_dict[mod] = volume
            
            # Load segmentation
            seg_filepath = patient_dir / f"{patient_id}_seg.nii.gz"
            if not seg_filepath.exists():
                logger.warning(f"Missing segmentation for {patient_id}")
                return None
            
            segmentation = self.load_nifti_volume(seg_filepath)
            image_4ch = np.stack(
                [modalities_dict[mod] for mod in self.modalities],
                axis=3
            )  # Shape: (H, W, D, 4)
            
            return image_4ch, segmentation
        
        except Exception as e:
            logger.error(f"Error processing {patient_id}: {str(e)}")
            return None

    def save_volume(self, image_4ch, mask, patient_id, output_subdir):
        """Save a single 3D volume as .npy files"""
        image_dir = output_subdir / 'images'
        mask_dir = output_subdir / 'masks'
        
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        image_filename = str(image_dir / f"{patient_id}.npy")
        mask_filename = str(mask_dir / f"{patient_id}.npy")
        
        np.save(image_filename, image_4ch.astype(np.float32))
        np.save(mask_filename, mask.astype(np.uint8))
    
    def run(self, train_split=0.75, val_split=0.15, test_split=0.10):
        """
        Main preprocessing pipeline
        
        Args:
            train_split (float): Proportion for training (default 0.75)
            val_split (float): Proportion for validation (default 0.15)
            test_split (float): Proportion for testing (default 0.10)
        """
        logger.info("=" * 60)
        logger.info("Starting BraTS 2021 Preprocessing")
        logger.info("=" * 60)
        
        # Get all patient directories
        patient_dirs = sorted([
            d for d in self.data_dir.iterdir() 
            if d.is_dir() and d.name.startswith('BraTS')
        ])
        
        logger.info(f"Found {len(patient_dirs)} patients")
        
        # First, split patients into train/val/test WITHOUT loading all data
        patient_ids = [d.name for d in patient_dirs]
        
        # First split: 75% train, 25% temp
        train_ids, temp_ids = train_test_split(
            patient_ids, 
            test_size=(val_split + test_split),
            random_state=42
        )
        
        # Second split: Split temp into val/test
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=test_split / (val_split + test_split),
            random_state=42
        )
        
        logger.info(f"Train patients: {len(train_ids)}")
        logger.info(f"Val patients: {len(val_ids)}")
        logger.info(f"Test patients: {len(test_ids)}")
        
        # Convert to sets for O(1) lookup
        train_ids_set = set(train_ids)
        val_ids_set = set(val_ids)
        test_ids_set = set(test_ids)
        
        # Save volumes to appropriate directories
        total_volumes = {'train': 0, 'val': 0, 'test': 0}
        successfully_processed = 0
        
        for patient_dir in tqdm(patient_dirs, desc="Processing and saving patients"):
            patient_id = patient_dir.name
            
            # Process patient (loads data into memory)
            processed = self.process_patient(patient_dir)
            if processed is None:
                continue
            image_4ch, mask = processed
            
            successfully_processed += 1
            
            # Determine which split this patient belongs to
            if patient_id in train_ids_set:
                output_dir = self.train_dir
                split = 'train'
            elif patient_id in val_ids_set:
                output_dir = self.val_dir
                split = 'val'
            else:
                output_dir = self.test_dir
                split = 'test'
            
            # Save volume immediately
            self.save_volume(image_4ch, mask, patient_id, output_dir)
            total_volumes[split] += 1
            
            # Free memory (important!)
            del image_4ch, mask
        
        logger.info("=" * 60)
        logger.info("Preprocessing Complete!")
        logger.info(f"Train volumes: {total_volumes['train']}")
        logger.info(f"Val volumes: {total_volumes['val']}")
        logger.info(f"Test volumes: {total_volumes['test']}")
        logger.info(f"Total volumes: {sum(total_volumes.values())}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)
        
        return total_volumes


def main():
    """Example usage"""
    parser = argparse.ArgumentParser(description="Preprocess BraTS 2021 dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to BraTS2021_Training_Data folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path where processed data will be saved"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.75,
        help="Proportion of patients for training (default: 0.75)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Proportion of patients for validation (default: 0.15)"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.10,
        help="Proportion of patients for testing (default: 0.10)"
    )
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = BraTSPreprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        modalities=['flair', 't1', 't1ce', 't2']
    )
    
    # Run preprocessing
    preprocessor.run(
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split
    )


if __name__ == "__main__":
    main()
