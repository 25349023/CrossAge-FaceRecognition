import csv
import os.path

import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class CALFWDataset(Dataset):
    def __init__(self, root, filelist_path, transform=None, use_group=False, is_aug=False):
        self.root = root
        self.filenames = []
        self.identities = []
        self.name_to_id = {}
        self.transform = transform
        self.use_group = use_group
        self.is_aug = is_aug
        if self.use_group:
            self.groups = []

        with open(os.path.join(self.root, filelist_path),
                  'r', encoding='utf-8', newline='') as f:
            if not self.use_group:
                for filename, id_name in csv.reader(f):
                    if self.is_aug:
                        p = os.path.join(self.root, 'aligned_aug', filename)
                        if not os.path.isfile(p):
                            continue
                    id = self.name_to_id.setdefault(id_name, len(self.name_to_id))
                    self.filenames.append(filename)
                    self.identities.append(id)
            else:
                for filename, id_name, _, _, group in csv.reader(f):
                    id = self.name_to_id.setdefault(id_name, len(self.name_to_id))
                    self.filenames.append(filename)
                    self.identities.append(id)
                    self.groups.append(int(group))

    def __getitem__(self, item):
        middle_path = 'aligned images' if not self.is_aug else 'aligned_aug'
        image = Image.open(os.path.join(self.root, middle_path, self.filenames[item]))
        label = self.identities[item]

        if self.transform is not None:
            image = self.transform(image)
        
        if self.use_group:
            group = self.groups[item]
            return image, label, group
        
        return image, label

    def __len__(self):
        return len(self.filenames)

    @property
    def num_class(self):
        return len(self.name_to_id)


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
