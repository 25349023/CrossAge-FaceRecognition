import numpy as np
import torch

def get_cosine_similarity(feat1, feat2):
    # feat1 = feat1 / np.linalg.norm(feat1)
    # feat2 = feat2 / np.linalg.norm(feat2)
    # result = np.dot(feat1, feat2.T)

    feat1 = feat1 / torch.linalg.norm(feat1)
    feat2 = feat2 / torch.linalg.norm(feat2)
    # print(feat1.shape)
    # print(feat2.T.shape)
    result = torch.matmul(feat1, feat2.T)
    # result = torch.nn.CosineSimilarity()(feat1, feat2)
    return result


'''
    follow the verification on resnet-face-pytorch
'''


def get_dis_score(feat1, feat2):
    # feat1 = feat1 / np.linalg.norm(feat1)
    # feat2 = feat2 / np.linalg.norm(feat2)
    # feat_dist = np.linalg.norm(feat1 - feat2)

    feat1 = feat1 / torch.linalg.norm(feat1)
    feat2 = feat2 / torch.linalg.norm(feat2)
    feat_dist = torch.linalg.norm(feat1 - feat2)
    return -feat_dist


def get_auc(pred_similarity, ground_truth):
    ''' this function return the AUC of the input
    
    Args:
        pred_similarity (numpy_array, size:(B, 1)): 
            the predicted similarity results

        ground_truth (numpy_array, size:(B, 1)): 
            the ground truth of whether each pair is same or not (same=1, not_same=0)
    '''

    from sklearn import metrics

    return metrics.roc_auc_score(ground_truth, pred_similarity)


def get_rank_one(pred_similarity, ground_truth, num_faces):
    ''' this function return the rank-1 accuracy of the input
    
    Args:
        pred_similarity (numpy_array, size:(B, 1)): 
            the predicted similarity results
            B should be C(num_faces, 2)

        ground_truth (numpy_array, size:(B, 1)): 
            the ground truth of whether each pair is same or not (same=1, not_same=0)
            B should be C(num_faces, 2)

        num_faces (int):
            the number of the faces
    '''

    # Rebuild nxn similarity and ground_truth matrix
    sim_map = torch.zeros((num_faces, num_faces))
    gt_map = torch.zeros((num_faces, num_faces))
    used_pairs = 0
    for i in range(num_faces - 1):
        cur_num_pairs = num_faces - i - 1
        target = pred_similarity[used_pairs:used_pairs + cur_num_pairs]
        target_truth = ground_truth[used_pairs:used_pairs + cur_num_pairs]

        sim_map[i, i + 1:] = target
        sim_map[i + 1:, i] = target
        gt_map[i, i + 1:] = target_truth
        gt_map[i + 1:, i] = target_truth

        used_pairs += cur_num_pairs

    total_success = 0
    for i in range(num_faces):
        j = torch.argmax(sim_map[i]) # most similar to face i

        success = gt_map[i][j]
        total_success += success

    return float(total_success) / num_faces

def get_tar_at_far(pred_similarity, ground_truth, levels=[1e-4, 1e-3, 1e-2, 1e-1]):
    ''' this function return the tar@far of the input
    
    Args:
        pred_similarity (numpy_array, size:(B, 1)): 
            the predicted similarity results

        ground_truth (numpy_array, size:(B, 1)): 
            the ground truth of whether each pair is same or not (same=1, not_same=0)
    '''

    from sklearn import metrics
    from scipy import interpolate

    fpr, tpr, thresholds = metrics.roc_curve(ground_truth, pred_similarity)
    f_interp = interpolate.interp1d(fpr, tpr)
    tpr_at_fpr = [ f_interp(x).item() for x in levels ]

    return tpr_at_fpr