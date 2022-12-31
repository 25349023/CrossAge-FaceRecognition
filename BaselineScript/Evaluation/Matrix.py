import numpy as np


def get_cosine_similarity(feat1, feat2):
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

def get_rank_one(pred_similarity, ground_truth, len_features):
    ''' this function return the rank-1 accuracy of the input
    
    Args:
        pred_similarity (numpy_array, size:(B, 1)): 
            the predicted similarity results
            B should be C(len_features, 2)

        ground_truth (numpy_array, size:(B, 1)): 
            the ground truth of whether each pair is same or not (same=1, not_same=0)
            B should be C(len_features, 2)

        len_features (int):
            the number of length of the features(faces)
    '''

    used_pairs = 0
    total_success = 0
    for cur_num_pairs in range(len_features-1, 1, -1):
        # find out current face pairs with others
        target = pred_similarity[used_pairs:used_pairs+cur_num_pairs]
        # check if the most similar pair is actually the same person 
        success = ground_truth[np.argmax(target)+used_pairs]

        total_success += success
    
    return float(total_success) / len_features