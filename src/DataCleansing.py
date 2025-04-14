#!/usr/bin/env python
# coding: utf-8
# def Cleansing(X_df)で、新しいデータフレームと、削除した特徴量数のリストを返す

import yaml
import sys
import pandas as pd
import numpy as np
sys.path.append('../')
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
import warnings
warnings.simplefilter('ignore', FutureWarning)

from rdkit import Chem

def DelSameValues(X_df, threshold):
    threshold_of_rate_of_same_values = threshold # 同じ値をもつサンプルの割合の閾値
    deleting_variable_numbers_in_same_values = [] # 削除する特徴量の番号を入れるリスト

    for x_number in range(X_df.shape[1]):
        same_value_numbers = X_df.iloc[:, x_number].value_counts()
        if same_value_numbers.iloc[0] / X_df.shape[0] >= threshold_of_rate_of_same_values:
            deleting_variable_numbers_in_same_values.append(x_number)
    deleting_variable_numbers_in_same_values_df = pd.DataFrame(deleting_variable_numbers_in_same_values) # DataFrame 型に変換
    deleting_variable_numbers_in_same_values_df.index = X_df.columns[deleting_variable_numbers_in_same_values] # 元のインデックス番号を行名にする
#    deleting_variable_numbers_in_same_values_df.columns = ['deleting variable numbers'] # 一つも削除しない場合にエラーになる
    X_df_new = X_df.drop(deleting_variable_numbers_in_same_values_df.index, axis=1) # 特徴量の削除
    # 新しいデータフレームと削除した特徴量数を返す
    return X_df_new, deleting_variable_numbers_in_same_values

def DelHighCorr(X_df, threshold):
    threshold_of_r = threshold # 相関係数の絶対値の閾値
    deleting_variable_numbers_in_r = []  # 相関係数の絶対値が閾値以上となる特徴量の一方の番号を入れるリスト
    r_in_x = X_df.corr() # 相関行列
    r_in_x = abs(r_in_x) # 絶対値を取り、0 から 1 の間にする
    for i in range(r_in_x.shape[0]):
        r_in_x.iloc[i, i] = 0 # 相関行列における対角線の要素を 0 にする
    for i in range(r_in_x.shape[0]):
        r_max = r_in_x.max() # 特徴量ごとの、他の特徴量との間における相関係数の絶対値の最大値
        r_max = list(r_max) # list に変換
        if max(r_max) >= threshold_of_r: # 相関係数の絶対値が閾値以上の特徴量の組があるとき
            variable_number_1 = r_max.index(max(r_max)) # 相関係数の絶対値が最大となる特徴量の組における、一方の番号
            r_in_variable_1 = list(r_in_x.iloc[:, variable_number_1]) # 上で選ばれた特徴量における相関係数の最大値
            variable_number_2 = r_in_variable_1.index(max(r_in_variable_1)) # 相関係数の絶対値が最大となる特徴量の組における、もう一方の番号
            # variable_number_1 を削除するか、variable_number_2 を削除するか、他の特徴量との相関係数の絶対値の和を計算し、それが大きい方とする
            r_sum_1 = r_in_x.iloc[:, variable_number_1].sum()
            r_sum_2 = r_in_x.iloc[:, variable_number_2].sum()
            if r_sum_1 >= r_sum_2:
                delete_x_number = variable_number_1
            else:
                delete_x_number = variable_number_2
                deleting_variable_numbers_in_r.append(delete_x_number) # 削除する特徴量の番号を追加
                # 削除する特徴量の影響をなくすため、対応する相関係数の絶対値を 0 に
                r_in_x.iloc[:, delete_x_number] = 0
                r_in_x.iloc[delete_x_number, :] = 0
        else: # 相関係数の絶対値が閾値以上の特徴量の組がなくなったら終了
            break
    deleting_variable_numbers_in_r_df = pd.DataFrame(deleting_variable_numbers_in_r) # DataFrame 型に変換
    deleting_variable_numbers_in_r_df.index = X_df.columns[deleting_variable_numbers_in_r]
#    deleting_variable_numbers_in_r_df.columns = ['deleting variable numbers'] # 一つも削除しない場合にエラーになる
    X_df_new = X_df.drop(deleting_variable_numbers_in_r_df.index, axis=1)
    # 新しいデータフレームと削除した特徴量を返す
    return X_df_new, deleting_variable_numbers_in_r

def chi_square(X_df, y_df, threshold_of_p_value):
    del_variable_li = [] # 削除する特徴名を入れるリスト
    X_df = pd.DataFrame(X_df)
    y_df = pd.DataFrame(y_df)
    y = [int(y_df.iloc[i,0]) for i in range(len(y_df))]
    selector = SelectKBest(chi2, k=1)
    X_df_tmp = selector.fit_transform(X_df, y)
    for i, p in enumerate(selector.pvalues_):
        if p > threshold_of_p_value:
            del_variable_li.append(X_df.columns.values[i])
        elif np.isnan(p):
            del_variable_li.append(X_df.columns.values[i])
    X_df_new = X_df.drop(del_variable_li, axis=1)
    # 新しいデータフレームと削除した特徴量を返す
    return X_df_new, del_variable_li
    
def anova(X_df, y, threshold_of_p_value):
    del_variable_li = [] # 削除する特徴名を入れるリスト
    X_df = pd.DataFrame(X_df)
    selector = SelectKBest(f_regression, k=1)
    X_df_tmp = selector.fit_transform(X_df, y)
    for i, p in enumerate(selector.pvalues_):
        if p < threshold_of_p_value and p > 0:
            continue
        else:
            del_variable_li.append(X_df.columns.values[i])
    X_df_new = X_df.drop(del_variable_li, axis=1)
    # 新しいデータフレームと削除した特徴量を返す
    return X_df_new, del_variable_li

# ステップワイズ回帰による特徴量選択
def stepwise_OLS_regression(X_df, y, threshold_in=0.01, threshold_out=0.05, initial_list=[], verbose=True):
    included = list(initial_list)
    scaler = StandardScaler()
    X_df = pd.DataFrame(X_df)
    X_df_new = X_df.loc[:, X_df.std() != 0]
    scaled_features = scaler.fit_transform(X_df_new)
    # 上書きしてます！！
    X_df_new = pd.DataFrame(scaled_features, columns=X_df_new.columns)
    while True:
        # 前進選択
        excluded = list(set(X_df_new.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            # 新しい変数を追加してモデルをフィット
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X_df_new[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        # p値がthreshold_in以下の変数を追加
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            if verbose:
                print(f'Adding {best_feature} with p-value {best_pval}')
        # 後退選択
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X_df_new[included]))).fit()
        pvalues = model.pvalues.iloc[1:]  # 定数項（intercept）は無視
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f'Dropping {worst_feature} with p-value {worst_pval}')
        # 終了条件
        if best_pval >= threshold_in and worst_pval <= threshold_out:
            excluded = list(set(X_df_new.columns) - set(included))
            break
    # 選択された特徴量のデータフレーム、選択された特徴量名、削除された特徴量名
    return pd.DataFrame(X_df[included]), included, excluded

def embedded_linear(X_df, y, model_name):
    del_variable_li = [] # 削除する特徴名を入れるリスト
    X_df = pd.DataFrame(X_df)
    X_df_new = X_df.loc[:, X_df.std() != 0]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X_df_new) # 標準化された特徴量を含むデータフレームの作成 
    scaled_df = pd.DataFrame(scaled_features, columns=X_df_new.columns)
    if model_name == 'Lasso':
        model = Lasso()
        param_grid = {'alpha': np.logspace(-9, 9, base=2)}
    elif model_name == 'ElasticNet':
        model = ElasticNet()
        param_grid = {
            'alpha': np.logspace(-9, 9, base=2),
            'l1_ratio': np.linspace(0.1, 0.9, 5)
            }
    grid_search = GridSearchCV(model, param_grid, cv=10)
    grid_search.fit(scaled_df, y)
    best_model = grid_search.best_estimator_
    if model_name == 'Lasso' or model_name == 'ElasticNet':
        # 係数がゼロではない特徴量を選択
        selected_features = scaled_df.columns[best_model.coef_ != 0]
    X_df_return = X_df[selected_features]
    del_variable_li = list(set(X_df.columns) - set(selected_features))
    # 新しいデータフレームと削除した特徴量を返す
    return X_df_return, del_variable_li

def Cleansing(X_df, y_df=None):
    with open('../config/config.yaml', 'r', encoding = 'utf-8') as file:
        config = yaml.safe_load(file)
    del_variable_numbers_li = []
    method = config['feature_engineering']['cleansing']['method']
    delete_variable_numbers_Val = []
    delete_variable_numbers_Corr = []
    delete_variable_others = []
    
    # ここから特徴量選択
    if method == 'Val' or method == 'Val_Corr':
        threshold_of_Val = config['feature_engineering']['cleansing']['same_value_threshold']
        X_df, delete_variable_numbers_Val = DelSameValues(X_df, threshold_of_Val)
    if method == 'Val_Corr':
        threshold_of_Corr = config['feature_engineering']['cleansing']['correlation_coefficient_threshold']
        X_df, delete_variable_numbers_Corr = DelHighCorr(X_df, threshold_of_Corr)
    if method == 'Chi_Square':
        threshold_of_p_value = config['feature_engineering']['cleansing']['p_value_of_chi_square']
        X_df, delete_variable_others = chi_square(X_df, y_df, threshold_of_p_value)
    if method == 'ANOVA':
        threshold_of_p_value = config['feature_engineering']['cleansing']['p_value_of_ANOVA']
        X_df, delete_variable_others = anova(X_df, y_df, threshold_of_p_value)
    if method == 'Stepwise_OLS':
        threshold_of_p_value_of_in = config['feature_engineering']['cleansing']['p_value_of_OLS_in']
        threshold_of_p_value_of_out = config['feature_engineering']['cleansing']['p_value_of_OLS_out']
        X_df, included, delete_variable_others = stepwise_OLS_regression(X_df, y_df, threshold_of_p_value_of_in, threshold_of_p_value_of_out)
    if method == 'Lasso':
        X_df, delete_variable_others = embedded_linear(X_df, y_df, 'Lasso')
    if method == 'ElasticNet':
        X_df, delete_variable_others = embedded_linear(X_df, y_df, 'ElasticNet')
    # ここまで
    
    del_variable_numbers_li += delete_variable_numbers_Val
    del_variable_numbers_li += delete_variable_numbers_Corr
    del_variable_numbers_li += delete_variable_others

    # 新しいデータフレームと、削除した特徴量のインデックス（[Valで削除 + Corrで削除]）を返す
    return X_df, del_variable_numbers_li

def Cleansing_CreateModel(X_df, y_df=None):
    # モデル作成の際は、config_create_model.yamlを読み込む
    with open('../config/config_create_model.yaml', 'r', encoding = 'utf-8') as file:
        config_create_model = yaml.safe_load(file)
    del_variable_numbers_li = []
    method = config_create_model['feature_engineering']['cleansing']['method']
    delete_variable_numbers_Val = []
    delete_variable_numbers_Corr = []
    delete_variable_others = []
    
    # ここから特徴量選択
    if method == 'Val' or method == 'Val_Corr':
        threshold_of_Val = config_create_model['feature_engineering']['cleansing']['same_value_threshold']
        X_df, delete_variable_numbers_Val = DelSameValues(X_df, threshold_of_Val)
    if method == 'Val_Corr':
        threshold_of_Corr = config_create_model['feature_engineering']['cleansing']['correlation_coefficient_threshold']
        X_df, delete_variable_numbers_Corr = DelHighCorr(X_df, threshold_of_Corr)
    if method == 'Chi_Square':
        threshold_of_p_value = config_create_model['feature_engineering']['cleansing']['p_value_of_chi_square']
        X_df, delete_variable_others = chi_square(X_df, y_df, threshold_of_p_value)
    if method == 'ANOVA':
        threshold_of_p_value = config_create_model['feature_engineering']['cleansing']['p_value_of_ANOVA']
        X_df, delete_variable_others = anova(X_df, y_df, threshold_of_p_value)
    if method == 'Stepwise_OLS':
        threshold_of_p_value_of_in = config_create_model['feature_engineering']['cleansing']['p_value_of_OLS_in']
        threshold_of_p_value_of_out = config_create_model['feature_engineering']['cleansing']['p_value_of_OLS_out']
        X_df, included, delete_variable_others = stepwise_OLS_regression(X_df, y_df, threshold_of_p_value_of_in, threshold_of_p_value_of_out)
    if method == 'Lasso':
        X_df, delete_variable_others = embedded_linear(X_df, y_df, 'Lasso')
    if method == 'ElasticNet':
        X_df, delete_variable_others = embedded_linear(X_df, y_df, 'ElasticNet')
    # ここまで
    
    del_variable_numbers_li += delete_variable_numbers_Val
    del_variable_numbers_li += delete_variable_numbers_Corr
    del_variable_numbers_li += delete_variable_others
    return X_df, del_variable_numbers_li

# テストデータ候補を未知のものと訓練データに含まれているものに分ける
def X_candidates_split_to_unknown_and_known(train_df, candidates_df):
    train_df = pd.DataFrame(train_df)
    candidates_df = pd.DataFrame(candidates_df)
    not_rem_li = [] # 未知のもの    
    rem_li = [] # 訓練データにあるもの
    set_li = list(set(train_df.iloc[:,0].values.tolist())) # 訓練データ
    candidates_li = candidates_df.iloc[:,0].values.tolist()
    for candidate in candidates_li:
        if candidate in set_li:
            rem_li.append(candidate)
        elif candidate in not_rem_li:
            print(f'重複しています: {candidate}')
        else:
            not_rem_li.append(candidate)
    not_removed_df = pd.DataFrame(not_rem_li, columns=pd.DataFrame(candidates_df.iloc[:,0]).columns.values) # 未知のもののデータフレーム
    removed_df = pd.DataFrame(rem_li, columns=list(pd.DataFrame(candidates_df.iloc[:,0]).columns.values)) # 訓練データに含まれる（candidateから除いた）もののデータフレーム
    # 新しい説明変数(index = 0)、訓練データに含まれる説明変数(index = 1)のデータフレームを格納したリスト
    return [not_removed_df, removed_df]

def substructure_search(smiles, target_smarts='c1ccccc1'):
    molecule = Chem.MolFromSmiles(smiles)
    target = Chem.MolFromSmiles(target_smarts)
    if molecule.HasSubstructMatch(target):
        return True # 部分構造(target_smarts)が含まれる場合
    else:
        return False # 含まれない場合
def substructure_search_df(smiles_df, target_smarts='c1ccccc1', column_idx=0):
    smiles_df = pd.DataFrame(smiles_df)
    bool_li = [substructure_search(smiles, target_smarts) for smiles in smiles_df.iloc[:,column_idx]]
    return smiles_df[bool_li] # 部分構造が含まれる行を抽出したデータフレームを返す
