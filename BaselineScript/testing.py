import argparse
import collections
import csv
import heapq
import os
import pathlib

import numpy as np
import torch
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

        with torch.no_grad():
            for file in pathlib.Path(image_path).glob('*.jpg'):
                image = Image.open(str(file))
                if self.transform is not None:
                    image = self.transform(image)

                self.features[file.name] = self.model(image[None], 0)
                # print(self.features[file.name].shape)

    def output_similarity(self, filelist_path, out_path):
        with open(os.path.join(self.root, filelist_path), 'r', encoding='utf-8', newline='') as f, \
                open(os.path.join(self.root, out_path), 'w', encoding='utf-8', newline='') as f2:
            writer = csv.writer(f2)
            for filename1, filename2 in csv.reader(f):
                feat1, feat2 = self.features[filename1], self.features[filename2]
                sim = get_cosine_similarity(feat1, feat2)[0, 0]
                writer.writerow([sim])


class GroupingCALFWDataset:
    def __init__(self, root, filelist_path, out_path, model, transform=None):
        self.root = root
        self.model = model
        self.id_per_group = 6
        self.model.eval()
        self.features = []
        self.transform = transform
        self.gen_all_features_from(filelist_path)
        # self.output_grouping(out_path)
        self.k_means(out_path)

    def gen_all_features_from(self, filelist_path):
        self.model.eval()

        with torch.no_grad(), open(os.path.join(self.root, filelist_path), 'r', encoding='utf-8', newline='') as f:
            for file, in csv.reader(f):
                image = Image.open(os.path.join(self.root, 'Bonus_affine', file))
                if self.transform is not None:
                    image = self.transform(image)

                self.features.append(self.model(image[None], 0))

    def output_grouping(self, out_path):
        num_faces = len(self.features)
        group_num = [-1] * num_faces
        picked = np.zeros(num_faces, dtype=bool)
        current_group = -1

        for i, feat in enumerate(self.features):
            if picked[i]:
                continue
            current_group += 1
            group_num[i] = current_group

            # find 6 the most similar feat
            similarity = -np.ones(num_faces - i - 1)
            for _ in range(self.id_per_group - 1):
                new_simi = np.array([get_cosine_similarity(feat, feat2)[0, 0]
                                     for feat2 in self.features[i + 1:]])
                to_be_updated = (~picked[i + 1:]) & (new_simi > similarity)
                similarity[to_be_updated] = new_simi[to_be_updated]
                next_id = i + 1 + np.nanargmax(similarity)
                # print(f'{current_group}: {next_id} - {similarity[next_id - i - 1]}')

                feat = self.features[next_id]
                group_num[next_id] = current_group
                picked[next_id] = True
                similarity[next_id - i - 1] = np.nan

        with open(os.path.join(self.root, out_path), 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for g in group_num:
                writer.writerow([g])

    def k_means(self, out_path):
        from nltk.cluster.kmeans import KMeansClusterer
        import nltk
        NUM_CLUSTERS = 20
        data = np.concatenate(self.features, axis=0)  # .toarray()

        kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=50)
        assigned_clusters = kclusterer.cluster(data, assign_clusters=True)

        with open(os.path.join(self.root, out_path), 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for g in assigned_clusters:
                writer.writerow([g])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt')
    args = parser.parse_args()

    model = FaceFeatureExtractor.insightFace("mobilefacenet", args.ckpt, sigma=0).model
    TestCALFWDataset(
        '../data/ForTesting', 'Test_PairList.csv',
        f'Test_team09_results_{args.ckpt.split(".")[0].split("-")[-1]}.csv', model,
        transform=T.Compose([T.Resize(112), T.ToTensor(),
                             T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    )
    GroupingCALFWDataset(
        '../data/ForTesting', 'Test_BonusList.csv',
        f'Test_team09_bonus_results_{args.ckpt.split(".")[0].split("-")[-1]}.csv', model,
        transform=T.Compose([T.Resize(112), T.ToTensor(),
                             T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    )
