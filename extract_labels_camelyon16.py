# nohup python code/extract_labels_camelyon16.py --csv_path /data/datasets/CAMELYON16/original/wsi_labels.csv --masks_dir /data/datasets/CAMELYON16/original/masks --coords_dir /home/fran/data/datasets/CAMELYON16/trident_processed/20x_512px_0px_overlap/patches/ --save_dir /home/fran/data/datasets/CAMELYON16/trident_processed/20x_512px_0px_overlap/patch_labels/ --patch_size 1024 --patch_level 0 > extract_labels_camelyon16.log 2>&1 &

import argparse
import os

import pandas as pd
import numpy as np

import h5py
import zarr
import tifffile

from tqdm import tqdm


def extract_labels(mask, coords, patch_size, wsi_label):

    if wsi_label==0:
        labels_array = np.zeros(len(coords))
        print(f'{wsi_name} is normal')
    else:
        threshold = 0.5
        repeat = True
        while repeat:
            labels = []
            pbar = tqdm(total=len(coords))
            pbar.set_description(f'{wsi_name}')
            for coord in coords:
                x, y = coord

                # 0: background, 1: tissue (normal), 2: tumor
                # Remember that the mask is transposed
                patch = mask[y:y+patch_size, x:x+patch_size]
                patch_max = np.max(patch)
                patch_min = np.min(patch)
                if patch_max > 2 or patch_min < 0:
                    raise ValueError('Error in the patch with coordinates: ', coord)
                
                patch = np.where(patch == 2, 1, 0)
                
                num_c = np.sum(patch)
                num_pixels = patch_size*patch_size

                if num_c/num_pixels > threshold:
                    label = 1
                else:
                    label = 0
                
                labels.append(label)
                pbar.update(1)
            pbar.close()
            labels_array = np.array(labels)
            num_tumor = np.sum(labels_array)
            repeat = num_tumor == 0
            if repeat:
                print(f'{wsi_name} is tumor but has no tumor patches. Repeating the process with a new threshold.')
                threshold = threshold / 2.0
    
    return labels_array

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', default='/data/datasets/CAMELYON16/original/wsi_labels.csv', type=str, help="CSV with WSI labels")
parser.add_argument('--masks_dir', default='/data/datasets/CAMELYON16/original/masks', type=str, help=".tif dir")
parser.add_argument('--coords_dir', default='/data/datasets/CAMELYON16/patches_512_preset/coords/', type=str, help="Patches coordinates dir")
parser.add_argument('--save_dir', default='/data/datasets/CAMELYON16/patches_512_preset/patch_labels/', type=str, help="Save dir")
parser.add_argument('--patch_size', default=1024, type=int, help="Patch size")
parser.add_argument('--patch_level', default=0, type=int, help="Patch level")

args = parser.parse_args()

print('Arguments:')
for arg in vars(args):
    print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

wsi_names = os.listdir(args.coords_dir)
wsi_names = [ wsi_name.split('_patches')[0] for wsi_name in wsi_names ]
wsi_names = sorted(wsi_names)

df_wsi_labels = pd.read_csv(args.csv_path)
df_wsi_labels['wsi_name'] = df_wsi_labels['wsi_name'].apply(lambda x: x.split('.')[0])
wsi_labels = [ int(df_wsi_labels[df_wsi_labels['wsi_name'] == wsi_name]['wsi_label'].values[0]) for wsi_name in wsi_names ]

patch_size = args.patch_size
patch_level = args.patch_level

for wsi_name, wsi_label in zip(wsi_names, wsi_labels):

    save_labels_path = os.path.join(args.save_dir, wsi_name)

    if os.path.exists(save_labels_path + '.h5'):
        print(f'{wsi_name} already exists. Skipping...')
        continue
    
    wsi_path = os.path.join(args.coords_dir, wsi_name + '_patches.h5')
    f = h5py.File(wsi_path, 'r')
    wsi_file = f['coords']

    coords = np.array(wsi_file) # (n_patches, 2)
    coords = coords.astype(int)

    mask_path = os.path.join(args.masks_dir, wsi_name + '_mask.tif')
    store = tifffile.imread(mask_path, aszarr=True)
    z = zarr.open(store, mode='r')
    mask = z[str(patch_level)]

    labels = extract_labels(mask, coords, patch_size, wsi_label)

    # Save the features
    
    # np.save(save_labels_path, labels)
    with h5py.File(save_labels_path + '.h5', 'w') as f:
        f.create_dataset('patch_labels', data=labels, compression='gzip')


