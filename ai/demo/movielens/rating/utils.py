import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return np.sqrt(mean_squared_error(pred, actual))

def cal_means(ratings):
    all_mean = np.mean(ratings[ratings!=0])
    user_mean = sum(ratings.T) / sum((ratings!=0).T)
    item_mean = sum(ratings) / sum((ratings!=0))
    user_mean = np.where(np.isnan(user_mean), all_mean, user_mean)
    item_mean = np.where(np.isnan(item_mean), all_mean, item_mean)
    return (all_mean, user_mean, item_mean)

def cal_user_similarity(ratings):
    epsilon = 1e-9
    sim = ratings.dot(ratings.T) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def cal_item_similarity(ratings): 
    epsilon = 1e-9
    sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def cal_user_similarity_norm(ratings, user_mean):
    epsilon = 1e-9
    rating_user_diff = ratings.copy()
    for i in range(ratings.shape[0]):
        nzero = ratings[i].nonzero()
        rating_user_diff[i][nzero] = ratings[i][nzero] - user_mean[i]
    sim = rating_user_diff.dot(rating_user_diff.T) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def cal_item_similarity_norm(ratings, item_mean):
    epsilon = 1e-9
    rating_item_diff = ratings.copy()
    for j in range(ratings.shape[1]):
        nzero = ratings[:,j].nonzero()
        rating_item_diff[:,j][nzero] = ratings[:,j][nzero] - item_mean[j]
    sim = rating_item_diff.T.dot(rating_item_diff) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)
