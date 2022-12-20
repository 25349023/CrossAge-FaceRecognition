import csv
import os.path

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


if __name__ == '__main__':
    dataset = CALFWDataset('..\\..\\data\\calfw', 'ForTraining\\CALFW_trainlist.csv',
                           transform=T.Compose([T.CenterCrop(112), T.ToTensor()]))

    # if running with jupyter notebook, then the `num_workers` and `prefetch_factor` arguments should be removed
    # however it will become quite slow
    dataloader = DataLoader(dataset, 32, num_workers=2, prefetch_factor=8)

    for xs, y in tqdm.tqdm(dataloader):
        pass  # test the speed

    print(dataset[0][0].shape)
    plt.imshow(dataset[0][0].permute(1, 2, 0))
    plt.show()
