#!/usr/bin/env python
# coding: utf-8
# いずれも、対応するnumpy配列を返す

# ライブラリーのインポート
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint, rdMHFPFingerprint, rdMolDescriptors
from rdkit.Avalon.pyAvalonTools import GetAvalonFP  # type: ignore
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.EState import Fingerprinter
from mordred import Calculator, descriptors

# エンコーディングが必要ない場合
def NotEncoding(x):
    return np.array(x)

# MACCSフィンガープリント （167 bit）
def MACCSFPFromSmiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp)
    else:
        return np.zeros(167)

# RDKit Fingerprint (Topological フィンガープリント, 2048 bit)
def RDKitFPFromSmiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = RDKFingerprint(mol)
        return np.array(fp)
    else:
        return np.zeros(2048)

# Morganフィンガープリント (ECFP (Extended Connectivity Fingerprint), たとえばradius=2のものはECEP4と呼ばれる, 2048 bitだけど変更可能)
def MorganFPFromSmiles_1(smiles, n_bits=2048): # radius = 1
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 1, n_bits)
        return np.array(fp)
    else:
        return np.zeros(n_bits)
def MorganFPFromSmiles_2(smiles, n_bits=2048): # radius = 2
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, n_bits)
        return np.array(fp)
    else:
        return np.zeros(n_bits)
def MorganFPFromSmiles_3(smiles, n_bits=2048): # radius = 3
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, n_bits)
        return np.array(fp)
    else:
        return np.zeros(n_bits)
def MorganFPFromSmiles_4(smiles, n_bits=2048): # radius = 4
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, n_bits)
        return np.array(fp)
    else:
        return np.zeros(n_bits)

# MHFP (MinHash Fingerprint, 2048 bit)
def MHFPFromSmiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    encoder = rdMHFPFingerprint.MHFPEncoder()
    if mol is not None:
        fp = encoder.EncodeMol(mol)
        return np.array(fp)
    else:
        return np.zeros(2048)

# Avalonフィンガープリント (2048 bitだけど変更可能)
def AvalonFPFromSmiles(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = GetAvalonFP(mol, nBits=n_bits)
        return np.array(fp)
    else:
        return np.zeros(n_bits)

# Linear RDKit Fingerprint (2048 bit)
def LinearRDKitFPFromSmiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = RDKFingerprint(mol, branchedPaths=False)
        return np.array(fp)
    else:
        return np.zeros(2048)

# Atom-Pairs Fingerprint (2048 bit)
def AtomPairsFPFromSmiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol)
        return np.array(fp)
    else:
        return np.zeros(2048)

# Topological Torsion Fingerprint (2048 bit)
def TopolFPFromSmiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)
        return np.array(fp)
    else:
        return np.zeros(2048)

# Morgan Feature Fingerprint (Morgan FingerprintのuseFeatures=Trueバージョン, 2048 bitだけど変更可能)
def MorFeaFPFromSmiles_1(smiles, n_bits=2048): # radius = 1
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = 1, useFeatures=True, nBits=n_bits)
        return np.array(fp)
    else:
        return np.zeros(n_bits)
def MorFeaFPFromSmiles_2(smiles, n_bits=2048): # radius = 2
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = 2, useFeatures=True, nBits=n_bits)
        return np.array(fp)
    else:
        return np.zeros(n_bits)
def MorFeaFPFromSmiles_3(smiles, n_bits=2048): # radius = 3
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = 3, useFeatures=True, nBits=n_bits)
        return np.array(fp)
    else:
        return np.zeros(n_bits)
def MorFeaFPFromSmiles_4(smiles, n_bits=2048): # radius = 4
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = 4, useFeatures=True, nBits=n_bits)
        return np.array(fp)
    else:
        return np.zeros(n_bits)

# 2DMordredフィンガープリント （1613 bit）
def Mordred2DFPFromSmiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        calc = Calculator(descriptors, ignore_3D=True)
        fp = calc.pandas([mol])
        return fp.to_numpy().flatten()
    else:
        return np.zeros(calc.descriptors.__len__())
    
# EStateフィンガープリント （79 bit）
def EStateFPFromSmiles_1(smiles): # EState1
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        estate_fp = Fingerprinter.FingerprintMol(mol)
        return np.array(estate_fp[0])
    else:
        return np.zeros(79)
def EStateFPFromSmiles_2(smiles): # EState2
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        estate_fp = Fingerprinter.FingerprintMol(mol)
        return np.array(estate_fp[1])
    else:
        return np.zeros(79)
