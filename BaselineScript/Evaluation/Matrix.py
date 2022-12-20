import numpy as np


def get_cosine_similarity(feat1, feat2):
    feat1 = feat1 / np.linalg.norm(feat1)
    feat2 = feat2 / np.linalg.norm(feat2)
    return np.dot(feat1, feat2.T)[0, 0]  # get one number


'''
    follow the verification on resnet-face-pytorch
'''


def get_dis_score(feat1, feat2):
    feat1 = feat1 / np.linalg.norm(feat1)
    feat2 = feat2 / np.linalg.norm(feat2)
    feat_dist = np.linalg.norm(feat1 - feat2)
    return -feat_dist
