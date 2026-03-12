"""Simple viewer for 3D `.npy` volumes saved by the BraTS preprocessor.

Features:
- Open a 3D multi-modal volume saved as (H, W, D, C) and its mask (H, W, D).
- Browse slices with left/right arrow keys.
- Choose view axis: axial (default), coronal, sagittal.
- Optional: open in napari if installed.

Usage examples:
    python src/data/visualize_npy.py --image data/processed/brats/train/images/BraTS2021_0000.npy
    python src/data/visualize_npy.py --image ... --mask ... --modality 2 --axis coronal
    python src/data/visualize_npy.py --image ... --use-napari

"""
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def load_volume(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    if os.path.isdir(path):
        raise FileNotFoundError(f"Expected a .npy file but got a directory: {path}\n" \
                                f"If this is a patient folder, pass the folder as --image and use --modality-name or --modality to pick a modality file.")
    vol = np.load(path)
    return vol


def get_slice(volume, axis, idx, modality=None):
    if volume.ndim == 4:
        H, W, D, C = volume.shape
    elif volume.ndim == 3:
        H, W, D = volume.shape
    else:
        raise ValueError(f"Unsupported volume shape: {volume.shape}")

    if axis == 'axial':
        sl = slice(None), slice(None), idx
    elif axis == 'coronal':
        sl = slice(None), idx, slice(None)
    elif axis == 'sagittal':
        sl = idx, slice(None), slice(None)
    else:
        raise ValueError('axis must be axial/coronal/sagittal')

    if volume.ndim == 4:
        if modality is None:
            modality = 0
        img2d = volume[sl + (modality,)]
    else:
        img2d = volume[sl]

    return np.rot90(img2d)


def get_max_index(volume, axis):
    if axis == 'axial':
        return volume.shape[2] - 1
    if axis == 'coronal':
        return volume.shape[1] - 1
    if axis == 'sagittal':
        return volume.shape[0] - 1


def matplotlib_viewer(image_path, mask_path=None, modality=0, axis='axial'):
    img = load_volume(image_path)
    mask = load_volume(mask_path) if mask_path else None

    idx = get_max_index(img, axis) // 2

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    def render():
        ax.clear()
        try:
            im = get_slice(img, axis, idx, modality=modality)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha='center')
            fig.canvas.draw_idle()
            return

        ax.imshow(im, cmap='gray')
        if mask is not None:
            m = get_slice(mask, axis, idx, modality=None)
            ax.imshow(m, cmap='jet', alpha=0.4, vmin=0, vmax=max(1, m.max()))

        ax.set_title(f"{image_path} | axis={axis} | slice={idx} | mod={modality}")
        ax.axis('off')
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal idx
        if event.key in ['right', 'd']:
            idx = min(idx + 1, get_max_index(img, axis))
            render()
        elif event.key in ['left', 'a']:
            idx = max(idx - 1, 0)
            render()
        elif event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    render()
    print('Use left/right arrows or a/d to navigate slices. Press q to quit.')
    plt.show()


def napari_viewer(image_path, mask_path=None):
    try:
        import napari
    except Exception as e:
        print('Napari is not available:', e)
        return False

    img = load_volume(image_path)
    mask = load_volume(mask_path) if mask_path else None

    if img.ndim == 4:
        img_n = np.transpose(img, (2, 0, 1, 3))
        channel_axis = 3
    else:
        img_n = np.transpose(img, (2, 0, 1))
        channel_axis = None

    viewer = napari.Viewer()
    viewer.add_image(img_n, name='image', channel_axis=channel_axis)
    if mask is not None:
        mask_n = np.transpose(mask, (2, 0, 1))
        viewer.add_labels(mask_n, name='labels')
    napari.run()
    return True


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D .npy volumes')
    parser.add_argument('--image', required=True, help='Path to image .npy file')
    parser.add_argument('--mask', required=False, help='Path to mask .npy file')
    parser.add_argument('--modality', type=int, default=0, help='Modality index (0..3)')
    parser.add_argument('--modality-name', type=str, default=None, help='Modality name (flair,t1,t1ce,t2)')
    parser.add_argument('--axis', choices=['axial', 'coronal', 'sagittal'], default='axial')
    parser.add_argument('--use-napari', action='store_true', help='Open in napari if available')

    args = parser.parse_args()

    image_path = args.image
    mask_path = args.mask

    if os.path.isdir(args.image):
        mapping = ['flair', 't1', 't1ce', 't2']
        if args.modality_name:
            mod_name = args.modality_name
        else:
            idx = int(args.modality)
            mod_name = mapping[idx]

        image_path = os.path.join(args.image, f"{mod_name}.npy")

    if mask_path and os.path.isdir(mask_path):
        mask_path = os.path.join(mask_path, 'mask.npy')

    if args.use_napari:
        ok = napari_viewer(image_path, mask_path)
        if ok:
            return

    matplotlib_viewer(image_path, mask_path, modality=args.modality, axis=args.axis)


if __name__ == '__main__':
    main()
