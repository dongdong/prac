import pandas as pd
from model import BaselineModel, ItemCFModel, UserCFModel, ItemCFMeanModel, UserCFMeanModel
from model import ItemCFMeanScaleModel, ItemCFTopKModel, ItemCFNormTopKModel
from model import UserCFMeanScaleModel, UserCFTopKModel, UserCFNormTopKModel
from model import BlendModel

if __name__ == "__main__":
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    trainingset_file = '../dataset/ml-100k/u3.base'
    testset_file= '../dataset/ml-100k/u3.test'
    n_users = 943
    n_items = 1682
    
    train_df = pd.read_csv(trainingset_file, sep='\t', names=names)
    test_df = pd.read_csv(testset_file, sep='\t', names=names)

    model = BaselineModel(n_users, n_items)
    model.fit(train_df)
    model.evaluate(test_df)    

    model = ItemCFModel(n_users, n_items)
    model.fit(train_df)
    model.evaluate(test_df)    

    model = UserCFModel(n_users, n_items)
    model.fit(train_df)
    model.evaluate(test_df)    

    model = ItemCFMeanModel(n_users, n_items)
    model.fit(train_df)
    model.evaluate(test_df)    

    model = UserCFMeanModel(n_users, n_items)
    model.fit(train_df)
    model.evaluate(test_df)    

    model = ItemCFMeanScaleModel(n_users, n_items)
    model.fit(train_df)
    model.evaluate(test_df)    

    model = UserCFMeanScaleModel(n_users, n_items)
    model.fit(train_df)
    model.evaluate(test_df)    

    ks = [8, 12, 16, 20, 24, 28, 32]
    for k in ks:
        print("K = %d " % k)
        model = ItemCFTopKModel(n_users, n_items, k)
        model.fit(train_df)
        model.evaluate(test_df)    

    ks = [24, 28, 32, 36, 40, 44, 48]
    for k in ks:
        print("K = %d " % k)
        model = UserCFTopKModel(n_users, n_items, k)
        model.fit(train_df)
        model.evaluate(test_df)    

    ks = [8, 12, 16, 20, 24, 28, 32]
    for k in ks:
        print("K = %d " % k)
        model = ItemCFNormTopKModel(n_users, n_items, k)
        model.fit(train_df)
        model.evaluate(test_df)    

    ks = [24, 28, 32, 36, 40, 44, 48]
    for k in ks:
        print("K = %d " % k)
        model = UserCFNormTopKModel(n_users, n_items, k)
        model.fit(train_df)
        model.evaluate(test_df)    

    k1 = 20
    k2 = 28
    alpha = 0.6
    model = BlendModel(n_users, n_items, k1, k2, alpha)
    model.fit(train_df)
    model.evaluate(test_df)    

