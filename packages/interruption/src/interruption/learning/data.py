import _pickle as cPickle
import glob
import gzip
import os

import torch.utils.data as torch_data

COMPRESS_LEVEL = 2


def write_compressed_pickle(pickle_filepath, datum, compresslevel=COMPRESS_LEVEL):
    """Write a datum to file as a compressed pickle."""
    with gzip.GzipFile(pickle_filepath, 'wb', compresslevel) as f:
        cPickle.dump(datum, f, protocol=-1)


def load_compressed_pickle(pickle_filepath):
    """Load a datum to file as a compressed pickle."""
    with gzip.GzipFile(pickle_filepath, 'rb') as pfile:
        return cPickle.load(pfile)


class CSVPickleDataset(torch_data.Dataset):
    def __init__(self, csv_filename, preprocess_function=None):
        if not isinstance(csv_filename, list):
            csv_filename = [csv_filename]

        self.preprocess_function = preprocess_function
        self._pickle_paths = []

        csvs = []
        for csvf in csv_filename:
            if '*' in csvf:
                csvs += glob.glob(csvf)
            else:
                csvs.append(csvf)

        for csvf in csvs:
            csv_file_directory = os.path.split(csvf)[0]
            with open(csvf, 'r') as fdata:
                self._pickle_paths += [
                    os.path.join(csv_file_directory, line.rstrip('\n'))
                    for line in fdata]

    def __getitem__(self, index):
        datum = load_compressed_pickle(self._pickle_paths[index])

        # Preprocess the data
        if self.preprocess_function:
            datum = self.preprocess_function(datum)

        return datum

    def __len__(self):
        return len(self._pickle_paths)
