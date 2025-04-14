import numpy as np
import pandas as pd
import yaml
import sys
sys.path.append('../')
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import ADASYN

# SMOTEを使ってデータを増強
def SMOTE_gen(df, target_name):
    # 設定の読み込み
    with open('../config/config.yaml', 'r', encoding = 'utf-8') as file:
        config = yaml.safe_load(file)
    smote_method = config['learning']['SMOTE']
    random_state = config['learning']['random_state']
    threshold = config['learning']['SMOTE_threshold']
    label=[]
    # SMOTEの手法を選択
    if smote_method == 'Normal':
        sm = SMOTE()
    elif smote_method == 'Borderline':
        sm = BorderlineSMOTE(kind='borderline-2', random_state = random_state)
    elif smote_method == 'SVM':
        sm = SVMSMOTE(random_state = random_state)
    elif smote_method == 'ADASYN':
        sm = ADASYN(random_state = random_state)
    else:
        raise ValueError('error in SMOTE_method')
    # 閾値に基づいてラベリング
    for i in df[target_name]:
        if i <= threshold:
            label.append(1)        
        else:
            label.append(0)
    df['label']=label
    # SMOTEに使う特徴量とクラス
    num_features = len(df.T) - 1
    features = df.iloc[:, : num_features]
    class_output = df.iloc[: ,num_features]
    # リサンプルした特徴量とクラス
    features_res, class_res = sm.fit_resample(features, class_output)
    X_resampled_df = features_res.drop([target_name], axis = 1)
    y_resampled_ser = features_res[target_name]
    return X_resampled_df, y_resampled_ser






 