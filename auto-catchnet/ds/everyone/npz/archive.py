from ds.everyone.model.indexer import EveryoneIndexer
from ds.everyone.model.item_index import *


class EveryoneArchive:
    def __init__(self, out_dir, indexer: EveryoneIndexer):
        self.out_dir = out_dir
        self.random_seed = "860515"
        self.indexer = indexer
        self.min_batch_size = 64
        self.record_size = self.min_batch_size * 9

    def archive(self):
        for profile_id in self.indexer.profile_ids:
            self.archive_profile(profile_id, self.indexer.get_index(profile_id))

    def archive_profile(self, profile_id, item_indexes):
        count = len(item_indexes)
        if count < 100:
            return

        size = self.record_size
        print(">>> profile [{}] count: {}".format(profile_id, count))

        item_per_so = {}
        for ii in item_indexes:
            if ii.orientation not in item_per_so:
                item_per_so[ii.orientation] = []
            item_per_so[ii.orientation].append(ii)

        for so, items in item_per_so.items():
            count = len(items)
            for idx, pos in enumerate(range(0, count, size)):
                item_block = items[pos:min(pos+size, count)]
                [x.load_images() for x in item_block]
                ItemIndex.to_profile_npz(self.out_dir, profile_id, so, idx, item_block)
                [x.clear() for x in item_block]

        print(">>> complete profile [{}]".format(profile_id))

