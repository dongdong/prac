import numpy as np
import utils

class Model:
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
    def _setup(self, train_df):
        self.ratings = np.zeros((self.num_users, self.num_items))
        for row in train_df.itertuples():
            self.ratings[row[1]-1, row[2]-1] = row[3]
    def _train(self):
        pass
    def _predict(self, user, item):
        pass
    def fit(self, train_df):
        print("%s.fit." % (self.__class__.__name__))
        self._setup(train_df)
        self._train()
    def evaluate(self, test_df):
        predictions = []
        targets = []
        for row in test_df.itertuples():
            user, item, actual = row[1]-1, row[2]-1, row[3]
            predictions.append(self._predict(user, item))
            targets.append(actual)
        rmse = utils.rmse(np.array(predictions), np.array(targets))
        print("%s.evaluate. rmse:%f" % (self.__class__.__name__, rmse))
        
class BaselineModel(Model):
    def _train(self):
        self.all_mean, self.user_mean, self.item_mean = utils.cal_means(self.ratings)
    def _predict(self, user, item):
        prediction = self.item_mean[item] + self.user_mean[user] - self.all_mean
        return prediction


class ItemCFModel(Model):
    def _train(self):
        self.item_sim = utils.cal_item_similarity(self.ratings)
    def _predict(self, user, item):
        nzero = self.ratings[user].nonzero()[0]
        prediction = (self.ratings[user, nzero].dot(self.item_sim[item, nzero]) / 
            sum(self.item_sim[item, nzero]))
        return prediction


class UserCFModel(Model):
    def _train(self):
        self.user_sim = utils.cal_user_similarity(self.ratings)
        self.all_mean, self.user_mean, self.item_mean = utils.cal_means(self.ratings)
    def _predict(self, user, item):
        nzero = self.ratings[:, item].nonzero()[0]
        prediction = (self.ratings[nzero, item].dot(self.user_sim[user, nzero]) / 
            sum(self.user_sim[user, nzero]))
        if np.isnan(prediction):
           prediction = self.item_mean[item] + self.user_mean[user] - self.all_mean 
        return prediction


class ItemCFMeanModel(Model):
    def _train(self):
        self.item_sim = utils.cal_item_similarity(self.ratings)
        self.all_mean, self.user_mean, self.item_mean = utils.cal_means(self.ratings)
    def _predict(self, user, item):
        item_mean_adj = self.item_mean + self.user_mean[user] - self.all_mean
        nzero = self.ratings[user].nonzero()[0]
        prediction = ((self.ratings[user, nzero] - item_mean_adj[nzero]).dot(
            self.item_sim[item, nzero]) / sum(self.item_sim[item, nzero]))
        prediction += item_mean_adj[item]
        return prediction


class UserCFMeanModel(Model):
    def _train(self):
        self.user_sim = utils.cal_user_similarity(self.ratings)
        self.all_mean, self.user_mean, self.item_mean = utils.cal_means(self.ratings)
    def _predict(self, user, item):
        user_mean_adj = self.user_mean + self.item_mean[item] - self.all_mean
        nzero = self.ratings[:, item].nonzero()[0]
        prediction = ((self.ratings[nzero, item] - user_mean_adj[nzero]).dot(
            self.user_sim[user, nzero]) / 
            sum(self.user_sim[user, nzero]))
        prediction += user_mean_adj[user]
        if np.isnan(prediction): prediction = user_mean_adj[user]
        return prediction


class ItemCFMeanScaleModel(Model):
    def _train(self):
        self.item_sim = utils.cal_item_similarity(self.ratings)
        self.all_mean, self.user_mean, self.item_mean = utils.cal_means(self.ratings)
    def _predict(self, user, item):
        item_mean_adj = self.item_mean + self.user_mean[user] - self.all_mean
        nzero = self.ratings[user].nonzero()[0]
        prediction = ((self.ratings[user, nzero] - item_mean_adj[nzero]).dot(
            self.item_sim[item, nzero]) / sum(self.item_sim[item, nzero]))
        prediction += item_mean_adj[item]
        if prediction > 5: prediction = 5
        if prediction < 1: prediction = 1
        return prediction


class UserCFMeanScaleModel(Model):
    def _train(self):
        self.user_sim = utils.cal_user_similarity(self.ratings)
        self.all_mean, self.user_mean, self.item_mean = utils.cal_means(self.ratings)
    def _predict(self, user, item):
        user_mean_adj = self.user_mean + self.item_mean[item] - self.all_mean
        nzero = self.ratings[:, item].nonzero()[0]
        prediction = ((self.ratings[nzero, item] - user_mean_adj[nzero]).dot(
            self.user_sim[user, nzero]) / 
            sum(self.user_sim[user, nzero]))
        prediction += user_mean_adj[user]
        if np.isnan(prediction): prediction = user_mean_adj[user]
        if prediction > 5: prediction = 5
        if prediction < 1: prediction = 1
        return prediction


class ItemCFTopKModel(Model):
    def __init__(self, num_users, num_items, topK):
        Model.__init__(self, num_users, num_items)
        self.topK = topK
    def _train(self):
        self.item_sim = utils.cal_item_similarity(self.ratings)
        self.all_mean, self.user_mean, self.item_mean = utils.cal_means(self.ratings)
    def _predict(self, user, item):
        item_mean_adj = self.item_mean + self.user_mean[user] - self.all_mean
        nzero = self.ratings[user].nonzero()[0]
        choice = nzero[self.item_sim[item, nzero].argsort()[::-1][:self.topK]]
        prediction = ((self.ratings[user, choice] - item_mean_adj[choice]).dot(
            self.item_sim[item, choice]) / sum(self.item_sim[item, choice]))
        prediction += item_mean_adj[item]
        if prediction > 5: prediction = 5
        if prediction < 1: prediction = 1
        return prediction


class UserCFTopKModel(Model):
    def __init__(self, num_users, num_items, topK):
        Model.__init__(self, num_users, num_items)
        self.topK = topK
    def _train(self):
        self.user_sim = utils.cal_user_similarity(self.ratings)
        self.all_mean, self.user_mean, self.item_mean = utils.cal_means(self.ratings)
    def _predict(self, user, item):
        user_mean_adj = self.user_mean + self.item_mean[item] - self.all_mean
        nzero = self.ratings[:, item].nonzero()[0]
        choice = nzero[self.user_sim[user, nzero].argsort()[::-1][:self.topK]]
        prediction = ((self.ratings[choice, item] - user_mean_adj[choice]).dot(
            self.user_sim[user, choice]) / 
            sum(self.user_sim[user, choice]))
        prediction += user_mean_adj[user]
        if np.isnan(prediction): prediction = user_mean_adj[user]
        if prediction > 5: prediction = 5
        if prediction < 1: prediction = 1
        return prediction


class ItemCFNormTopKModel(ItemCFTopKModel):
    def _train(self):
        self.all_mean, self.user_mean, self.item_mean = utils.cal_means(self.ratings)
        self.item_sim = utils.cal_item_similarity_norm(self.ratings, self.item_mean)
    

class UserCFNormTopKModel(UserCFTopKModel):
    def _train(self):
        self.all_mean, self.user_mean, self.item_mean = utils.cal_means(self.ratings)
        self.user_sim = utils.cal_user_similarity_norm(self.ratings, self.user_mean)


class BlendModel(Model):
    def __init__(self, num_users, num_items, topK1, topK2, alpha):
        self.itemModel = ItemCFNormTopKModel(num_users, num_items, topK1)
        self.userModel = UserCFNormTopKModel(num_users, num_items, topK2)
        self.alpha = alpha
    def fit(self, train_df):
        print("%s.fit." % (self.__class__.__name__))
        self.itemModel.fit(train_df)
        self.userModel.fit(train_df)
    def _predict(self, user, item):
        prediction1 = self.itemModel._predict(user, item)
        prediction2 = self.userModel._predict(user, item)
        prediction = self.alpha * prediction1 + (1 - self.alpha) * prediction2
        if prediction > 5: prediction = 5
        if prediction < 1: prediction = 1
        return prediction

    



