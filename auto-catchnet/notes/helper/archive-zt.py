
import os, glob
import zipfile
import random
from random import shuffle
from itertools import chain

def grep_recur(base_path, pattern="*.*"):
    sub_greps = list(chain(*[grep_recur(dp, pattern) for dp in grep_dirs(base_path)]))
    return grep_files(base_path, pattern) + sub_greps

def grep_files(base_path, pattern="*.*"):
    return glob.glob("{}/{}".format(base_path, pattern))

def grep_dirs(base_path):
    file_paths = [os.path.join(base_path, name) for name in os.listdir(base_path)]
    return [p for p in file_paths if os.path.isdir(p)]


anno_path = "/ds/processed/annotation/"
frame_path = "/ds/processed/vc-one-frame/"

print("*** load jsons ***")
json_paths = grep_recur(anno_path, pattern="*.json")


random.seed(860515)
shuffle(json_paths)

full_json_paths = json_paths
print(len(full_json_paths))
print(full_json_paths[0])

def package_pair(zip_dir_path, js_paths, pack_size=512):
    json_blocks = [js_paths[i:i+pack_size] for i in (range(0, len(js_paths), pack_size))]
   
    for block_idx, js_block in enumerate(json_blocks):
        zip_file_path = "{}/frames-{:05d}.zip".format(zip_dir_path, block_idx)
        package = zipfile.ZipFile(zip_file_path, 'w')

        for p in js_block:
            arcname = p.replace("/ds/processed/annotation/", "")
            arcname = arcname.replace("/", "-")
            package.write(p, arcname=arcname, compress_type=zipfile.ZIP_DEFLATED)

            frame_path = p.replace("/ds/processed/annotation/", "/ds/processed/vc-one-frame/")
            frame_path = frame_path.replace("json", "jpg")
            arcname = frame_path.replace("/ds/processed/vc-one-frame/", "")
            arcname = arcname.replace("/", "-")
            package.write(frame_path, arcname=arcname, compress_type=zipfile.ZIP_DEFLATED)

        print("*** complete package [{}] ***".format(block_idx))

print("*** run package ***")
package_pair("/ds/bench", full_json_paths)
