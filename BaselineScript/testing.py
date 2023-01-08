import argparse
import csv
import os
import pathlib

from PIL import Image
from torchvision import transforms as T

from Evaluation.Matrix import get_cosine_similarity
from ModelFactory import FaceFeatureExtractor


class TestCALFWDataset:
    def __init__(self, root, filelist_path, out_path, model, transform=None):
        self.root = root
        self.model = model
        self.model.eval()
        self.features = {}
        self.transform = transform
        self.gen_all_features_from(os.path.join(root, 'FGNet_500'))
        self.output_similarity(filelist_path, out_path)

    def gen_all_features_from(self, image_path):
        self.model.eval()

        for file in pathlib.Path(image_path).glob('*.jpg'):
            image = Image.open(str(file))
            if self.transform is not None:
                image = self.transform(image)

            self.features[file.name] = self.model(image)
        print(self.features)

    def output_similarity(self, filelist_path, out_path):
        with open(os.path.join(self.root, filelist_path), 'r', encoding='utf-8', newline='') as f, \
                open(os.path.join(self.root, out_path), 'r', encoding='utf-8', newline='') as f2:
            writer = csv.writer(f2)
            for filename1, filename2 in csv.reader(f):
                feat1, feat2 = self.features[filename1], self.features[filename2]
                sim = get_cosine_similarity(feat1, feat2)
                writer.writerow([sim])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt')
    args = parser.parse_args()

    model = FaceFeatureExtractor.insightFace("mobilefacenet", args.ckpt).model
    TestCALFWDataset(
        '../data/ForTesting', 'Test_PairList.csv', 'Test_team09_results.csv', model,
        transform=T.Compose([T.Resize(112), T.ToTensor(),
                             T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    )
