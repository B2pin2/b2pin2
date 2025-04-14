import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from sklearn.neighbors import NearestNeighbors


class kNN_AD():
    def __init__(self, X_df, alpha=0.95, metric='euclidean'):
        self.X_df = pd.DataFrame(X_df)
        self.alpha = alpha
        self.metric = metric
        # kの値はサンプル数の平方根
        self.k_number = round(len(X_df) ** 0.5)
        # 標準偏差が0の特徴量は削除
        self.std_selected_features_bool = X_df.std()!=0
        self.X_df_new = pd.DataFrame(X_df.loc[:, self.std_selected_features_bool])
        # Xの標準化
        self.X_std = self.X_df_new.std()
        self.X_mean = self.X_df_new.mean()
        scaled_X = (self.X_df_new - self.X_mean) / self.X_std
        # モデルの作成
        self.ad_model = NearestNeighbors(n_neighbors = self.k_number, metric = self.metric)
        self.ad_model.fit(scaled_X)        
        # k最近傍サンプルとの距離(自身も含む), k最近傍サンプルのインデックス(自身も含む)
        self.knn_dist_between_X, knn_index_X = self.ad_model.kneighbors(scaled_X, n_neighbors = self.k_number + 1)   
        # DataFrame型に変換して、自分以外のk個の距離の平均を求める
        self.knn_dist_df = pd.DataFrame(self.knn_dist_between_X)
        self.mean_of_knn_dist = self.knn_dist_df.iloc[:, 1:].mean(axis=1)
        # AD内外を分ける、平均距離の閾値を決める
        self.sorted_mean_of_knn_dist = self.mean_of_knn_dist.sort_values()
        self.ad_threshold = self.sorted_mean_of_knn_dist.iloc[round(len(self.X_df) * self.alpha) - 1]
        # X_dfのうち、AD内のものをTrueとしたboolシリーズ、AD外のものをTrueとしたboolシリーズ
        self.X_df_in_ad_bool = self.mean_of_knn_dist <= self.ad_threshold
        self.X_df_out_ad_bool = self.mean_of_knn_dist > self.ad_threshold
        
    def distinguish_ad(self, test_df):
        test_df = pd.DataFrame(test_df)
        # 標準偏差が0の特徴量削除したのちに標準化
        test_df_new = pd.DataFrame(test_df.loc[:, self.std_selected_features_bool])
        scaled_test_df_new = (self.X_df_new - self.X_mean) / self.X_std
        # k最近傍サンプルとの距離(自身も含む), k最近傍サンプルのインデックス(自身も含む)
        test_knn_dist, test_knn_index = self.ad_model.kneighbors(scaled_test_df_new, n_neighbors = self.k_number + 1)   
        # DataFrame型に変換して、自分以外のk個の距離の平均を求める
        test_knn_dist = pd.DataFrame(test_knn_dist)
        mean_of_test_knn_dist = test_knn_dist.iloc[:, 1:].mean(axis=1)
        # X_dfのうち、AD内のものをTrueとしたboolシリーズ、AD外のものをTrueとしたboolシリーズ
        test_df_in_ad_bool = mean_of_test_knn_dist <= self.ad_threshold
        test_df_out_ad_bool = mean_of_test_knn_dist > self.ad_threshold
        # AD内のデータセット、AD外のデータセットを返す
        return test_df.loc[test_df_in_ad_bool, :], test_df.loc[test_df_out_ad_bool, :]

