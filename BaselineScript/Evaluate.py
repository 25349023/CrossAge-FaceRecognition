import sys

from Evaluation import Matrix
import numpy as np
import tqdm
import torch


def evaluate(model, dataset, dataloader=None):
    model.eval()

    if dataloader is None:
        dataloader = DataLoader(dataset, 8, num_workers=2)

    with torch.no_grad():
        features = []
        for i, (pics, labels) in enumerate(tqdm.tqdm(dataloader, file=sys.stdout, desc='Evaluating: ', leave=False)):
            pics = pics.to('cuda')
            features.append(model(pics))
        features = torch.cat(features, dim=0).cpu()

    '''
    gt_pair: ground truth of if pair i are the same person. 0 is false and 1 is true.
    pd_pair: prediction of the similarity of pair i. value should between [-1, 1]
    '''
    gt_pair = []
    pd_pair = []
    ids = dataset.identities

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            gt_pair.append(1 * (ids[i] == ids[j]))
        pd_pair.append(Matrix.get_cosine_similarity(features[i:i + 1], features[i + 1:], batch=True)[0])

    pd_pair = np.concatenate(pd_pair, axis=0)

    auc = Matrix.get_auc(pd_pair, gt_pair)
    r1_acc = Matrix.get_rank_one(pd_pair, gt_pair, len(dataset))

    return auc, r1_acc


if __name__ == "__main__":
    from dataloader.dataset import CALFWDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from ModelFactory import FaceFeatureExtractor

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='Model/model_mobilefacenet.pth')
    args = parser.parse_args()

    facerecognition = FaceFeatureExtractor.insightFace("mobilefacenet", args.ckpt)
    model = facerecognition.model.to('cuda')

    dataset = CALFWDataset('../data/calfw', 'ForTraining/CALFW_validationlist.csv',
                           transform=T.Compose([T.CenterCrop(112), T.ToTensor(),
                                                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    auc, r1_acc = evaluate(model, dataset)
    print(f"AUC: {auc:.3f}")
    print(f"rank-1 ACC: {r1_acc:.3f}")
