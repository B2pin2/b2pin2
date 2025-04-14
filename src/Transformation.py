#!/usr/bin/env python
# coding: utf-8

# ライブラリーのインポート
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# 部分構造の削除（例：BODIPYの骨格を削除して置換基のSMILESを生成する）
def remove_substructure(smiles, substructure_smarts="F[B-]1(F)[N+]=2C(=Cc3n1c(cc3C))\\C=CC=2"):
    molecule = Chem.MolFromSmiles(smiles)
    substructure = Chem.MolFromSmarts(substructure_smarts) # 削除したい部分構造
    # 部分構造をマッチングして削除
    if molecule.HasSubstructMatch(substructure):
        edmol = Chem.EditableMol(molecule)
        match_atoms = molecule.GetSubstructMatch(substructure)
        # マッチした原子を削除
        for atom_idx in sorted(match_atoms, reverse=True):
            edmol.RemoveAtom(atom_idx)
        # 新しい分子を生成してSMILESに変換
        modified_molecule = edmol.GetMol()
        return Chem.MolToSmiles(modified_molecule, canonical=True)
    else:
        # 部分構造が見つからない場合は元のSMILESを返す
        print(f'Can not generate fragment (SMILES: {smiles})')
        return gen_canonical_smiles(smiles)
def remove_substructure_df(df, substructure_smarts="F[B-]1(F)[N+]=2C(=Cc3n1c(cc3C))\\C=CC=2", col_index=0):
    sub_SMILES_li = []
    df = pd.DataFrame(df)
    for smiles in df.iloc[:,col_index].values:
        sub_SMILES_li.append(remove_substructure(smiles, substructure_smarts))
    new_df = pd.DataFrame(sub_SMILES_li, columns=['substruct_SMILES'])
    return new_df

# canonical SMILESへ変換  
def gen_canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    return canonical_smiles
def gen_canonical_smiles_df(df, col_index=0):
    con_SMILES_li = []
    df = pd.DataFrame(df)
    for smiles in df.iloc[:,col_index].values:
        con_SMILES_li.append(gen_canonical_smiles(smiles))
    new_df = pd.DataFrame(con_SMILES_li, columns=['con_SMILES'])
    return new_df

# 部分構造の変換（例：ホルミル基をエチレンに置換した化合物のconical smilesを生成）
def substructure_replacement(smiles, from_smarts='[CX3H1](=O)', to_smarts='C=C'):
    molecule = Chem.MolFromSmiles(smiles)
    from_group = Chem.MolFromSmarts(from_smarts) # 置換したい部分構造  
    # 部分構造の置換
    if molecule.HasSubstructMatch(from_group):
        # 部分構造を置換
        rxn = AllChem.ReactionFromSmarts(f"[*:1]{from_smarts}>>[*:1]{to_smarts}")
        products = rxn.RunReactants((molecule,))
        # 最初の生成物をSMILES形式で返す
        if products:
            return Chem.MolToSmiles(products[0][0], canonical=True)
    print(f'Can not be replaced (SMILES: {smiles})')
    return gen_canonical_smiles(smiles) # 部分構造が見つからない場合は元のcanonical SMILESを返す
def substructure_replacements_df(df, col_index=0, from_smarts='[CX3H1](=O)', to_smarts='C=C'):
    new_con_SMILES_li = []
    df = pd.DataFrame(df)
    for smiles in df.iloc[:,col_index].values:
        new_con_SMILES_li.append(substructure_replacement(smiles, from_smarts, to_smarts))
    new_df = pd.DataFrame(new_con_SMILES_li, columns=['replaced_con_SMILES'])
    return new_df

