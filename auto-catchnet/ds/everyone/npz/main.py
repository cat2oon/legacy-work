import argparse

from ds.everyone.npz.archive import EveryoneArchive
from ds.everyone.model.indexer import EveryoneIndexer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    ei = EveryoneIndexer(args.dataset)
    ei.index()

    ea = EveryoneArchive(args.outdir, ei)
    ea.archive()
