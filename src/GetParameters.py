import numpy as np
import pandas as pd
import sys
sys.path.append('../')

from src.sterimol import sterimol_2
from rdkit.Chem import Descriptors, SDMolSupplier, rdFreeSASA, rdMolDescriptors

'''
files # 解析したいlogファイル名
radii # "cpk"または"bondi"
atom1 # 対象の1つ目の原子の番号
atom2 # 対象の2つ目の原子の番号
sterimolの場合、jobtype = 1
'''
def get_sterimol(log_file, atom1, atom2, radii = 'cpk'):
    if log_file[-4:] == '.log' or log_file[-4:] == '.out':
        log_file = log_file[:-4]
    L, B1, B5 = sterimol_2.run_sterimol(log_file, radii=radii, atom1=atom1, atom2=atom2, jobtype = 1)
    # リストとして返す
    return [L, B1, B5]

def sdf_to_mol(sdf_file):
    suppl = SDMolSupplier(sdf_file)
    mol = suppl[0]  
    return mol

# RDKit Descriptor
def getMolDescriptors(mol, missingVal=None):
    res = {}
    for nm,fn in Descriptors._descList:
        val = fn(mol)
        res[nm] = val
    return res
# free solvent accessible surface area, 溶媒接触可能表面積
def get_SASA(sdf_file):
    mol = sdf_to_mol(sdf_file)
    atomTypes = rdFreeSASA.classifyAtoms(mol)
    sasa = rdFreeSASA.CalcSASA(mol, atomTypes)
    print(f'SASA: {round(sasa, 2)}')
    return round(sasa, 2)

def get_MW(sdf_file):
    mol = sdf_to_mol(sdf_file)
    res = getMolDescriptors(mol)
    mw = res['MolWt']
    print(f'MW: {round(mw, 2)}')
    return round(mw, 2)

# 極性表面積（TPSA）
def get_TPSA(sdf_file):
    mol = sdf_to_mol(sdf_file)
    res = getMolDescriptors(mol)
    tpsa = res['TPSA']
    print(f'TPSA: {round(tpsa, 2)}')
    return round(tpsa, 2)
        




 