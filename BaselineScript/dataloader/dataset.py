import csv
import os.path
import pprint
from collections import defaultdict
from itertools import combinations, chain
from typing import List, Dict

import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class CALFWDataset(Dataset):
    def __init__(self, root, filelist_path, transform=None):
        self.root = root
        self.filenames = []
        self.identities = []
        self.name_to_id = {}
        self.transform = transform

        with open(os.path.join(self.root, filelist_path),
                  'r', encoding='utf-8', newline='') as f:
            for filename, id_name in csv.reader(f):
                id = self.name_to_id.setdefault(id_name, len(self.name_to_id))
                self.filenames.append(filename)
                self.identities.append(id)

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.root, 'aligned images', self.filenames[item]))
        label = self.identities[item]

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.filenames)

    @property
    def num_class(self):
        return len(self.name_to_id)


class PairCALFWDataset(Dataset):
    def __init__(self, root, filelist_path, transform=None):
        self.root = root
        self.filenames: Dict[str, List[str]] = defaultdict(list)
        self.transform = transform

        with open(os.path.join(self.root, filelist_path),
                  'r', encoding='utf-8', newline='') as f:
            for filename, id_name in csv.reader(f):
                self.filenames[id_name].append(filename)

        self.pos_pair = list(chain.from_iterable(
            combinations(group, 2) for group in self.filenames.values())
        )

    def __getitem__(self, item):
        fname = self.pos_pair[item]
        image = tuple(Image.open(os.path.join(self.root, 'aligned images', f))
                      for f in fname)

        if self.transform is not None:
            image = tuple(self.transform(img) for img in image)
        return image

    def __len__(self):
        return len(self.pos_pair)

    @property
    def num_class(self):
        return len(self.filenames)


if __name__ == '__main__':
    dataset = CALFWDataset('../../data/calfw', 'ForTraining/CALFW_trainlist.csv',
                           transform=T.Compose([T.CenterCrop(112), T.ToTensor()]))
    print(f'There are {dataset.num_class} different people.')

    # if running with jupyter notebook, then the `num_workers` and `prefetch_factor` arguments should be removed
    # however it will become quite slow
    dataloader = DataLoader(dataset, 32, num_workers=2, prefetch_factor=8)

    for xs, y in tqdm.tqdm(dataloader):
        pass  # test the speed

    print(dataset[0][0].shape)
    plt.imshow(dataset[0][0].permute(1, 2, 0))
    plt.show()

    print()

    dataset = PairCALFWDataset('../../data/calfw', 'ForTraining/CALFW_trainlist.csv',
                               transform=T.Compose([T.CenterCrop(112), T.ToTensor()]))
    print(f'There are {len(dataset.pos_pair)} positive pairs.')
    pprint.pprint(dataset.pos_pair[:10])
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(dataset[0][0].permute(1, 2, 0))
    ax[1].imshow(dataset[0][1].permute(1, 2, 0))
    plt.show()


