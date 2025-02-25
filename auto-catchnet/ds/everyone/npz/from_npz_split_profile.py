import argparse
import glob
import os

import numpy as np

from ds.everyone.model.item_index import ItemIndex
from ds.everyone.profile.profiles import get_profile_ids


def load_npz_from(data_path, block_id):
    npz_path = os.path.join(data_path, "item-{:05d}.npz".format(block_id))
    return load_npz(npz_path)


def load_npz(npz_path):
    npz = np.load(npz_path)
    images = npz['images']
    indexes = npz['metas']
    return ItemIndex.from_npz(indexes, images)


def convert(npz_path, out_path):
    print("*** convert ***")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    profile_ids = {} 
    blocks = glob.glob(os.path.join(npz_path, "item-*.npz"))
    
    # count profiles
    for i, block_path in enumerate(blocks):
        item_indexes = load_npz(block_path)

        for item in item_indexes:
            if item.pid in profile_ids:
                continue
            profile_ids[item.pid] = 1
    profile_ids = profile_ids.keys()
                    
    # slow but simple way
    for profile_id in profile_ids:
        items_per_so = {}

        print("*** handle profile {} ***".format(profile_id))
        for i, block_path in enumerate(blocks):
            item_indexes = load_npz(block_path)

            for item in item_indexes:
                if profile_id != item.pid:
                    continue
                if item.orientation not in items_per_so:
                    items = [item]
                    items_per_so[item.orientation] = items
                else:
                    items_per_so[item.orientation].append(item)

        print("*** writing profile {} ***".format(profile_id))
        for so, items in items_per_so.items():
            ItemIndex.to_profile_npz(out_path, profile_id, so, items)
            print("*** writing complete so {} ***".format(so))

    print("*** complete ***")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()

    convert(args.npz_path, args.out_path)
