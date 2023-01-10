import numpy as np


def get_cosine_similarity(feat1, feat2, batch=False):
    if batch:
        feat1 = feat1 / np.linalg.norm(feat1, axis=1)[..., None]
        feat2 = feat2 / np.linalg.norm(feat2, axis=1)[..., None]
    else:
        feat1 = feat1 / np.linalg.norm(feat1)
        feat2 = feat2 / np.linalg.norm(feat2)
    result = np.dot(feat1, feat2.T)

    return result


'''
    follow the verification on resnet-face-pytorch
'''


def get_dis_score(feat1, feat2):
    feat1 = feat1 / np.linalg.norm(feat1)
    feat2 = feat2 / np.linalg.norm(feat2)
    feat_dist = np.linalg.norm(feat1 - feat2)
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
    sim_map = np.zeros((num_faces, num_faces))
    gt_map = np.zeros((num_faces, num_faces))
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
        j = np.argmax(sim_map[i]) # most similar to face i

        success = gt_map[i][j]
        total_success += success

    return float(total_success) / num_faces
