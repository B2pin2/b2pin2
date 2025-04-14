feature_engineering
  - cleansing
    - method: ('Val', 'Val_Corr', 'Nothing')
    - same_value_threshold: (0 ~ 1 の値, 0.9とか0.95, 0.99が一般的)
    - correlation_coefficient_threshold: (0 ~ 1 の値, 0.9とか0.95, 0.99が一般的)
learning
  - scaler_X: ('S_Standard', 'U_Standard', 'Nothing', Sが標準分散でUが不偏分散)
  - scaler_Y: ('S_Standard', 'U_Standard', 'Nothing', Sが標準分散でUが不偏分散)
  - cross_validation: ('DCV', クロスバリデーションの方法をしていとりあえずはダブルクロスバリデーション)
  - DCV
    - : ('LOO', 'k-Fold', 'k-Fold_2', 外部CV方法の指定)
    - : ('LOO', 'k-Fold', 'k-Fold_2', 内部CV方法の指定)
  - k-Fold: (自然数, 上で'k-Fold'と指定した際のホールド数)
  - k-Fold_2: (自然数, 上で'k-Fold_2'と指定した際のホールド数)
  - loss_function: ('MAE', 'MSE', 'RMSE', 'RMSLE', 'r2')
  - param_search: ('Grid', 'Bayes', ハイパーパラメータの探索法, 決定木系の重いのはBayesを選択した方がよい)
  - Bayes_n_iter: (自然数, 20ぐらいあれば十分？, ベイズサーチの際の試行数)
  - option: ('None', 'PLS', 'SMOTE')
  - SMOTE: ('Normal', 'Borderline', 'SVM', 'ADASYN', SMOTEを用いる場合の手法の選択)
  - PLS_components: (自然数, PLSを用いる場合の変換後成分数)
  - random_state: (整数)
  
params
  各モデルにおいて探索するハイパーパラメータの設定ができる。範囲を変更したい場合のみ触る。



以下メモ
'''
encoder:  NotEn, MACCS, Aval, RDK, LinearRDK, AP, Topo, MHFP, Mor_1, Mor_2, Mor_3, Mor_4, FC_1, FC_2, FC_3, FC_4, MorFea_1, MorFea_2, MorFea_3, MorFea_4, 2DMordred

    NotEn: NotEncoding,
    MACCS: MACCS FP,
    Aval: Avalon FP,
    RDK(RDKitFP),
    LinearRDK(LinearRDKitFP),     
    AP(AtomPairsFP)
    Topo(TopolFP),
    MHFP(MHFP),
    Mor_1(MorganFP, radius = 1),
    Mor_2(MorganFP, radius = 2),
    Mor_3(MorganFP, radius = 3),
    Mor_4(MorganFP, radius = 4),
    MorFea_1: Encoders.MorFeaFPFromSmiles_1,
    MorFea_2: Encoders.MorFeaFPFromSmiles_2,
    MorFea_3: Encoders.MorFeaFPFromSmiles_3,
    MorFea_4: Encoders.MorFeaFPFromSmiles_4,
    2DMordred: (mordred, ignore_3D)
'''
