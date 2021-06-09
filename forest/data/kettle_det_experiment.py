"""Data class, holding information about dataloaders and poison ids."""

import numpy as np
from .kettle_base import _Kettle
from .datasets import Subset


class KettleDeterministic(_Kettle):
    """Generate parameters for an experiment based on a fixed triplet a-b-c given via --poisonkey.

    This construction replicates the experiment definitions for MetaPoison.

    The triplet key, e.g. 5-3-1 denotes in order:
    source_class - poison_class - source_id
    """

    def prepare_experiment(self):
        """Choose sources from some label which will be poisoned toward some other chosen label, by modifying some
        subset of the training data within some bounds."""
        self.deterministic_construction()

    def deterministic_construction(self):
        """Construct according to the triplet input key.

        Poisons are always the first n occurences of the given class.
        [This is the same setup as in metapoison]
        """
        if self.args.threatmodel != 'single-class':
            raise NotImplementedError()

        split = self.args.poisonkey.split('-')
        if len(split) != 3:
            raise ValueError('Invalid poison triplet supplied.')
        else:
            source_class, poison_class, source_id = [int(s) for s in split]
        self.init_seed = self.args.poisonkey
        print(f'Initializing Poison data (chosen images, examples, sources, labels) as {self.args.poisonkey}')

        self.poison_setup = dict(poison_budget=self.args.budget,
                                 source_num=self.args.sources, poison_class=poison_class, source_class=source_class,
                                 target_class=[poison_class])
        self.poisonset, self.sourceset, self.validset = self._choose_poisons_deterministic(source_id)

    def _choose_poisons_deterministic(self, source_id):
        # poisons
        class_ids = []
        for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
            source, idx = self.trainset.get_target(index)
            if source == self.poison_setup['poison_class']:
                class_ids.append(idx)

        poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
        if len(class_ids) < poison_num:
            warnings.warn(f'Training set is too small for requested poison budget.')
            poison_num = len(class_ids)
        self.poison_ids = class_ids[:poison_num]

        # the source
        # class_ids = []
        # for index in range(len(self.validset)):  # we actually iterate this way not to iterate over the images
        #     source, idx = self.validset.get_target(index)
        #     if source == self.poison_setup['source_class']:
        #         class_ids.append(idx)
        # self.source_ids = [class_ids[source_id]]
        # Disable for now for benchmark sanity check. This is a breaking change.
        self.source_ids = [source_id]

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
        dict(zip(self.poison_ids, range(poison_num)))
        return poisonset, sourceset, validset
