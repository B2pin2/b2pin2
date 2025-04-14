import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from src import Encoders

def FPGen(SMILES_df, selected_fp):
    encoder_dict = {
        'NotEn': Encoders.NotEncoding,
        'MACCS': Encoders.MACCSFPFromSmiles,
        'Aval': Encoders.AvalonFPFromSmiles,
        'RDK': Encoders.RDKitFPFromSmiles,
        'LinearRDK': Encoders.LinearRDKitFPFromSmiles,
        'AP': Encoders.AtomPairsFPFromSmiles,
        'Topo': Encoders.TopolFPFromSmiles,
        'MHFP': Encoders.MHFPFromSmiles,
        'Mor_1': Encoders.MorganFPFromSmiles_1,
        'Mor_2': Encoders.MorganFPFromSmiles_2,
        'Mor_3': Encoders.MorganFPFromSmiles_3,
        'Mor_4': Encoders.MorganFPFromSmiles_4,
        'MorFea_1': Encoders.MorFeaFPFromSmiles_1,
        'MorFea_2': Encoders.MorFeaFPFromSmiles_2,
        'MorFea_3': Encoders.MorFeaFPFromSmiles_3,
        'MorFea_4': Encoders.MorFeaFPFromSmiles_4,
        '2DMordred': Encoders.Mordred2DFPFromSmiles,
        'EState_1': Encoders.EStateFPFromSmiles_1,
        'EState_2': Encoders.EStateFPFromSmiles_2
    }
    encoder = encoder_dict[selected_fp]
    SMILES_df = pd.DataFrame(SMILES_df)
    X_li = [encoder(SMILES_df.iloc[n,0]) for n in range(len(SMILES_df))]
    column_name_li = [f'{selected_fp}_{n + 1}' for n in range(len(X_li[0]))] 
    X_df = pd.DataFrame(X_li, columns = column_name_li)
    # bool型をint型に変換
    X_df = X_df.applymap(lambda x: int(x) if isinstance(x, bool) else x)
    # 念のため、int型かfloat型以外を除く
    X_df = X_df.loc[:, (X_df.dtypes == int)|(X_df.dtypes == float)] 
    return X_df
