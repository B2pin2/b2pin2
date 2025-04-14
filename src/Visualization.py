#!/usr/bin/env python
# coding: utf-8
# いずれも、対応するnumpy配列を返す

# ライブラリーのインポート
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

def draw_structures_from_df_smiles(df, col_index=0): #引数はデータフレームと、列名
    df = pd.DataFrame(df)
    # SMILES列から分子molオブジェクトを取得
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in df.iloc[:,col_index].values]
    # 構造を描写
    Draw.rdDepictor.SetPreferCoordGen(True)
    img = Draw.MolsToGridImage(mol_list, molsPerRow=10, maxMols=1000, subImgSize=(300,300))
    return img

def draw_structure_from_smiles(smiles):
    # SMILES列から分子molオブジェクトを取得
    mol = Chem.MolFromSmiles(smiles)
    # 構造を描写
    Draw.rdDepictor.SetPreferCoordGen(True)
    img = Draw.MolToImage(mol, size=(100, 100))
    return img



