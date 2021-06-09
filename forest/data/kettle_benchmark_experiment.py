"""Data class, holding information about dataloaders and poison ids."""

import pickle

from .kettle_base import _Kettle
from .datasets import Subset



class KettleBenchmark(_Kettle):
    """Generate parameters for an experiment as specific in the data poisoning benchmark of

    https://github.com/aks2203/poisoning-benchmark

    """

    def prepare_experiment(self):
        """Choose sources from some label which will be poisoned toward some other chosen label, by modifying some
        subset of the training data within some bounds."""
        with open(self.args.benchmark, 'rb') as handle:
            setup_dict = pickle.load(handle)
        self.benchmark_construction(setup_dict[self.args.benchmark_idx])


    def benchmark_construction(self, setup_dict):
        """Construct according to the benchmark."""
        source_class, poison_class = setup_dict['source class'], setup_dict['base class']

        budget = len(setup_dict['base indices']) / len(self.trainset)
        self.poison_setup = dict(poison_budget=budget,
                                 source_num=self.args.sources, poison_class=poison_class, source_class=source_class,
                                 target_class=[poison_class])
        self.init_seed = self.args.poisonkey
        self.poisonset, self.sourceset, self.validset = self._choose_poisons_benchmark(setup_dict)

    def _choose_poisons_benchmark(self, setup_dict):
        # poisons
        class_ids = setup_dict['base indices']
        poison_num = len(class_ids)
        self.poison_ids = class_ids

        # the source
        self.source_ids = [setup_dict['source index']]
        # self.source_ids = setup_dict['source index']

        sourceset = Subset(self.validset, indices=self.source_ids)
        valid_indices = []
        for index in range(len(self.validset)):
            _, idx = self.validset.get_target(index)
            if idx not in self.source_ids:
                valid_indices.append(idx)
        validset = Subset(self.validset, indices=valid_indices)
        poisonset = Subset(self.trainset, indices=self.poison_ids)

        # Construct lookup table
        self.poison_lookup = dict(zip(self.poison_ids, range(poison_num)))

        return poisonset, sourceset, validset
