#!/usr/bin/env python

# >>>>>>
# [DESCRIPTION]:
#
# [AUTHOR]: Chen Chen, Penn State Univ, 2021
# <<<<<<

import sys, os, copy, math, re, shutil, glob, json, argparse
import subprocess as sub, numpy as np, matplotlib.pyplot as plt

sys.path.append(os.getenv("HOME") + "/python")
import cpy, cpy_reaxff

from Bio import SeqIO
from sklearn.metrics import matthews_corrcoef, r2_score, accuracy_score

# All amino acids in the following functions are listed in this order.
aas = 'ARNDCQEGHILKMFPSTWYV'
naas = len(aas)

# --------------------------------------------------

def group_residue(aa_str):

  categs = ['g1', 'g2', 'g3', 'g4', 'g5']

  # 3-groups: (g1) ploar, (g2) neutral, (g3) hydrophobicity
  # AAIndex: PRAM900101
  gres_raw1 = {'g1':'RKEDQN', 'g2':'GASTPHY', 'g3':'CLVIMFW'}
  # AAIndex: ARGP820101
  gres_raw2 = {'g1':'QSTNGDE', 'g2':'RAHCKMV', 'g3':'LYPFIW'}
  # AAIndex: ZIMJ680101
  gres_raw3 = {'g1':'QNGSWTDERA', 'g2':'HMCKV', 'g3':'LPFYI'}
  # AAIndex: PONP930101
  gres_raw4 = {'g1':'KPDESNQT', 'g2':'GRHA', 'g3':'YMFWLCVI'}
  # AAIndex: CASG920101
  gres_raw5 = {'g1':'KDEQPSRNTG', 'g2':'AHYMLV', 'g3':'FIWC'}
  # AAIndex: ENGD860101
  gres_raw6 = {'g1':'RDKENQHYP', 'g2':'SGTAW', 'g3':'CVLIMF'}
  # AAIndex: FASG890101
  gres_raw7 = {'g1':'KERSQD', 'g2':'NTPG', 'g3':'AYHWVMFLIC'}

  # 3-groups: (g1) low, (g2) medium, (g3) high
  # Normalized vdw vol
  gres_raw8 = {'g1':'GASTPDC', 'g2':'NVEQIL', 'g3':'MHKFRYW'}
  # Polarity
  gres_raw9 = {'g1':'LIFWCMVY', 'g2':'PATGS', 'g3':'HQRKNED'}
  # Polarizability
  gres_raw10 = {'g1':'GASDT', 'g2':'GPNVEQIL', 'g3':'KMHFRYW'}
  # Charge
  gres_raw11 = {'g1':'KR', 'g2':'ANCQGHILMFPSTWYV', 'g3':'DE'}
  # Secondary structure
  gres_raw12 = {'g1':'EALMQKRH', 'g2':'VIYCWFT', 'g3':'GNPSD'}
  # Solvent accessibility
  gres_raw13 = {'g1':'ALFCGIVW', 'g2':'PKQEND', 'g3':'MPSTHY'}

  # 5-groups: (g1) aliphatic group, (g2) aromatic group, (g3) positive charged group, (g4) netative charged group, (g5) uncharged group.
  gres_raw14 = {'g1':'GAVLMI', 'g2':'FYW', 'g3':'KRH', 'g4':'DE', 'g5':'STCPNQ'} 

  gres_raw_db = [gres_raw1, gres_raw2, gres_raw3, gres_raw4, gres_raw5, gres_raw6, gres_raw7, gres_raw8, gres_raw9, gres_raw10, gres_raw11, gres_raw12, gres_raw13, gres_raw14]
  gres_db = [[0.0 for j in range(len(gres_raw_db[i]))] for i in range(len(gres_raw_db))]

  if aa_str not in ['all', 'empty']:
    for i in range(len(gres_raw_db)):
      gres_raw = gres_raw_db[i]
      for ic in range(len(gres_raw)):
        categ = categs[ic]
        categ_str = gres_raw[categ]
        count_categ = sum([aa_str.count(x) for x in categ_str])
        gres_db[i][ic] = count_categ
      #gres_db[i] = [x/sum(gres_db[i]) for x in gres_db[i]] # Use fraction rather than number

  gres_db = np.concatenate(gres_db, axis=None) # Flatten the list of list
  avg_gres = np.zeros(len(gres_db))
  std_gres = np.ones(len(gres_db))
  ngres = len(avg_gres)

  if aa_str in ["all"]:
    return gres_db, avg_gres, std_gres, ngres
  else:
    return gres_db
  
# --------------------------------------------------

def spairs(aa_list):
  
  # Macromolecules 9, 945-950 (1976)
  pot_raw1 = [
  "-2.6",
  "-3.4 -4.3",
  "-3.1 -4.1 -3.2",
  "-2.8 -3.9 -3.1 -2.7",
  "-4.2 -5.3 -4.9 -4.2 -7.1",
  "-3.5 -4.5 -3.8 -3.2 -5.0 -3.4",
  "-3.0 -4.2 -3.4 -3.3 -4.4 -3.6 -2.8",
  "-3.8 -4.5 -4.0 -3.7 -5.4 -4.4 -3.8 -3.9",
  "-4.0 -4.9 -4.4 -4.3 -5.6 -4.7 -4.5 -4.7 -4.9",
  "-5.9 -6.2 -5.8 -5.4 -7.3 -5.9 -5.7 -6.3 -6.6 -8.2",
  "-4.8 -5.1 -4.6 -4.3 -6.2 -5.0 -4.6 -5.2 -5.6 -7.5 -6.0",
  "-3.1 -3.6 -3.3 -3.2 -4.4 -3.7 -3.8 -3.8 -4.1 -5.6 -4.6 -2.7",
  "-4.6 -5.0 -4.2 -4.3 -6.2 -3.5 -4.6 -5.1 -5.4 -7.4 -6.3 -4.7 -5.8",
  "-5.1 -5.8 -5.0 -4.9 -6.8 -5.3 -5.0 -5.6 -6.4 -8.0 -7.0 -4.9 -6.6 -7.1",
  "-3.4 -4.2 -3.6 -3.3 -5.3 -4.0 -3.5 -4.2 -4.5 -6.0 -4.8 -3.6 -5.1 -5.2 -3.5",
  "-2.9 -3.8 -3.1 -2.7 -4.6 -3.6 -3.2 -3.8 -4.3 -5.5 -4.4 -3.0 -4.1 -4.7 -3.4 -2.5",
  "-3.3 -4.0 -3.5 -3.1 -4.8 -3.7 -3.3 -4.1 -4.5 -5.9 -4.8 -3.3 -4.6 -5.1 -3.6 -3.3 -3.1",
  "-5.2 -5.8 -5.3 -5.1 -6.9 -5.8 -5.2 -5.8 -6.5 -7.8 -6.8 -5.0 -6.9 -7.4 -5.6 -5.0 -5.1 -6.8",
  "-4.7 -5.6 -5.0 -4.7 -6.6 -5.2 -4.9 -5.4 -6.1 -7.4 -6.2 -4.9 -6.1 -6.6 -5.2 -4.7 -4.9 -6.8 -6.0",
  "-4.3 -4.9 -4.3 -4.0 -6.0 -4.7 -4.2 -5.1 -5.3 -7.3 -6.2 -4.2 -6.0 -6.5 -4.7 -4.2 -4.4 -6.5 -5.9 -5.5",
  ]

  # Proc. Natl. Acad. Sci. USA 93, 11628-11633 (1996)
  pot_raw2 = [
  "-0.08",
  " 0.07  0.23",
  "-0.14  0.04 -0.86",
  " 0.10 -0.15 -0.12  0.60",
  "-0.30 -0.40 -0.32  0.55 -1.79",
  "-0.11  0.62 -0.05  0.46 -0.49 -0.08",
  " 0.03 -0.26 -0.25  0.68  0.04  0.62  0.21",
  "-0.09 -0.15 -0.18 -0.06 -0.42  0.12  0.40  0.04",
  "-0.15 -0.01  0.06 -0.06 -0.82  0.05 -0.53  0.00  0.14",
  "-0.64 -0.08  0.39  0.04 -0.48 -0.39 -0.20  0.40 -0.52 -0.71",
  "-0.57 -0.10 -0.10  0.50 -0.69 -0.13 -0.05 -0.08 -0.36 -1.04 -1.14",
  " 0.00  0.30  0.18 -0.09  0.00  0.04 -0.09  0.10  0.14 -0.26  0.10  1.45",
  " 0.05 -0.43  0.31  1.07 -1.23 -0.54  0.02  0.00 -0.35 -0.41 -0.31  0.55  0.36",
  "-0.05 -0.22 -0.02  0.20 -0.98  0.10  0.19  0.21 -0.75 -0.66 -1.02 -0.17 -1.03 -0.61",
  " 0.41 -0.02  0.11  0.84  0.07 -0.21  0.33  0.40 -0.22  0.25  0.09  0.51 -0.25 -0.43  0.28",
  "-0.01  0.61  0.37 -0.09 -0.20  0.40  0.30 -0.04 -0.59 -0.13 -0.07  0.18 -0.47  0.14  0.44 -0.13",
  "-0.22 -0.17 -0.27 -0.03 -0.38 -0.17  0.15  0.13 -0.27 -0.29 -0.39  0.09  0.06 -0.19  0.36  0.05  0.26",
  "-0.08 -0.78 -0.68  0.24 -0.30  0.40  0.32 -0.14 -0.41 -0.89 -0.97 -0.30 -0.07 -0.89 -0.44 -0.20  0.07  0.02",
  "-0.37  0.21 -0.74  0.11 -0.96 -0.39  0.22 -0.32 -0.67 -0.87 -0.60 -0.20 -1.10 -0.82 -0.45  0.25 -0.23 -0.99  0.35",
  "-0.60 -0.48 -0.24  0.25 -0.94 -0.09 -0.02 -0.20 -0.35 -0.98 -1.03 -0.08 -0.94 -0.78 -0.08 -0.31  0.06 -0.60 -0.70 -1.15",
  ]

  # Proteins 16, 92-112 (1993)
  pot_raw3 = [
  " 0.230",
  " 0.237 -0.145",
  " 0.159 -0.303 -0.206",
  " 0.140 -0.527 -0.534  0.539",
  " 0.159  0.174  0.105 -0.137 -0.524",
  "-0.008  0.222 -0.187 -0.128  0.487 -0.430",
  " 0.182 -0.717 -0.262 -0.178  0.121 -0.728  1.060",
  " 0.282 -0.037 -0.097 -0.336 -0.187 -0.072 -0.041 -0.068",
  "-0.079  0.111 -0.141 -0.270 -0.187  0.247  0.072  0.127  0.392",
  "-0.535 -0.014  0.456  0.229 -0.415  0.020  0.385  0.127 -0.062 -0.051",
  "-0.245  0.374  0.430  0.316 -0.061  0.391  0.480 -0.004  0.029 -0.223 -0.070",
  " 0.063  0.734  0.222 -0.759  1.115 -0.181 -0.782  0.065 -0.235 -0.058 -0.015  0.567",
  "-0.364  0.509  0.405  0.097 -0.032  0.013  0.188  0.084  0.293 -0.103 -0.253  0.347 -0.006",
  " 0.132 -0.018  0.336  0.118 -0.095  0.283  0.270  0.457 -0.141 -0.348 -0.460  0.259 -0.501 -0.163",
  " 0.022 -0.176  0.109  0.175 -0.199 -0.545 -0.064  0.127  0.037  0.382 -0.170  0.068 -0.104 -0.083  0.033",
  " 0.115 -0.263 -0.334 -0.565 -0.164  0.298 -0.207 -0.100  0.080  0.254  0.390 -0.246  0.259  0.357  0.057 -0.150",
  " 0.002  0.250 -0.018 -0.433  0.021  0.060 -0.006 -0.106  0.061  0.167  0.075 -0.170  0.087  0.058  0.108 -0.430 -0.315",
  " 0.117 -0.244  0.538  1.287  0.080  0.459 -0.089  0.036 -0.098 -0.505 -0.635 -0.822 -0.783 -0.185 -0.312  0.298  0.040  0.707",
  "-0.030  0.044 -0.491  0.694  0.069 -0.036 -0.004  0.121 -0.305 -0.100 -0.289 -0.294 -0.218 -0.335  0.071  0.442  0.492 -0.105  0.211",
  "-0.359 -0.011  0.158  0.442 -0.026 -0.191  0.255  0.141  0.006 -0.253 -0.454  0.253 -0.265 -0.288  0.358  0.316  0.251 -0.034  0.075 -0.641",
   ]

  # Proteins 34, 82-95 (1999) (contacts within 0-5 Angstroms)
  pot_raw4 = [
  "-0.13571",
  " 0.37121  0.23245",
  " 0.25935 -0.27050 -0.61455",
  " 0.33397 -0.78243 -0.41830  0.06704",
  " 0.23079  0.49103  0.32481  0.53024 -1.79243",
  " 0.26575 -0.25307 -0.26143 -0.00061  0.25200 -0.24068",
  " 0.26471 -0.78607 -0.18010  0.24572  0.66360 -0.05835  0.51101",
  "-0.01467 -0.08319 -0.37069 -0.22435  0.20423 -0.01890  0.14922 -0.48115",
  " 0.32413 -0.10894 -0.00420 -0.47402  0.24383 -0.03046 -0.10674  0.08603 -0.23317",
  "-0.22176  0.33584  0.38282  0.44972  0.12534  0.20555  0.21945  0.36527  0.10553 -0.3170",
  "-0.15025  0.19784  0.35359  0.38200  0.10747  0.07523  0.19892  0.30617  0.11443 -0.1261 -0.19983",
  " 0.39894  0.55155 -0.07038 -0.90014  0.73178 -0.24804 -0.92364  0.00501  0.00361  0.2170  0.21292  0.56407",
  " 0.03521  0.12999  0.02882  0.32317  0.04462 -0.03542  0.15161  0.14609  0.01416 -0.0879 -0.12860  0.15363 -0.13998",
  " 0.08139  0.03136  0.26608  0.53784  0.09641  0.14340  0.25134  0.21293  0.03923 -0.1911 -0.22682  0.04828 -0.23360 -0.24651",
  " 0.03615 -0.06999 -0.20175 -0.21449 -0.04477 -0.16569 -0.14194 -0.18438 -0.27877  0.5603  0.35217  0.04081  0.11287 -0.10484 -0.04170",
  " 0.10475 -0.02548 -0.28825 -0.50285  0.11283 -0.06140 -0.35312 -0.27119 -0.11302  0.3130  0.27135  0.02715  0.19696  0.28005 -0.10791 -0.20955",
  "-0.04679 -0.05313 -0.28531 -0.36579  0.27539 -0.23014 -0.29144 -0.24551 -0.25624  0.2867  0.31011 -0.13219  0.30090  0.37472  0.02844 -0.32381 -0.19546",
  " 0.20001 -0.33116  0.28602  0.50378  0.09401 -0.04570  0.16071  0.24344 -0.17229 -0.1598 -0.16843 -0.24586 -0.09998 -0.13588 -0.55908  0.36554  0.33614  0.05462",
  " 0.12835 -0.14488  0.12638  0.46473  0.16464 -0.03777  0.18883  0.10640  0.02691 -0.1696 -0.20609 -0.16896 -0.22924 -0.01526 -0.41613  0.31614  0.36576 -0.03280 -0.01669",
  "-0.27134  0.33279  0.47451  0.44658  0.09778  0.08581  0.22764  0.23105  0.15787 -0.1963 -0.10641  0.23784  0.00637 -0.08226  0.39761  0.15369  0.14755  0.12174  0.02059 -0.29733",
  ] 

  # Proteins 34, 82-95 (1999), (contacts within 5-7.5 Angstroms)
  pot_raw5 = [
  "-0.02226",
  " 0.09517  0.07551",
  " 0.02956 -0.20477 -0.43720",
  " 0.13313 -0.75762 -0.36779 -0.21301",
  "-0.04411  0.21192  0.05061  0.16627 -0.35421",
  " 0.08891 -0.10846 -0.29003 -0.22107  0.06070 -0.25399",
  " 0.15220 -0.73107 -0.29050 -0.05421  0.46641 -0.22637 -0.06802",
  "-0.05662 -0.11493 -0.20722 -0.13454 -0.15618 -0.16506 -0.01456 -0.14051",
  " 0.02143  0.02841 -0.22095 -0.51082 -0.08947 -0.13022 -0.38576 -0.17065 -0.55055",
  "-0.03193  0.34774  0.49079  0.64069  0.08276  0.31459  0.51696  0.27635  0.51369 -0.38531",
  " 0.01865  0.31116  0.49954  0.67107  0.03309  0.32215  0.47591  0.33203  0.31403 -0.35708 -0.36593",
  " 0.14682  0.05722 -0.28908 -0.83773  0.30183 -0.21644 -0.84899 -0.13111  0.15045  0.37502  0.42522  0.06908",
  "-0.08634  0.23747  0.13447  0.34501 -0.03164  0.13752  0.21535  0.15193  0.04739 -0.15359 -0.14099  0.32782 -0.16514",
  "-0.14905  0.27997  0.27039  0.33380 -0.08872  0.11689  0.29542  0.05265  0.09431 -0.09249 -0.12690  0.31158 -0.11997 -0.19925",
  " 0.08000 -0.11752 -0.22235 -0.10799 -0.09590 -0.18800 -0.08512 -0.05692 -0.05316  0.19667  0.20002 -0.01420  0.08185  0.16121 -0.20538",
  " 0.00350 -0.18824 -0.23763 -0.17464 -0.06203 -0.19343 -0.21143 -0.20971 -0.19378  0.35197  0.34207 -0.13103  0.10816  0.09814 -0.11199 -0.18928",
  " 0.00526 -0.05800 -0.15047 -0.04369 -0.02885 -0.16004 -0.12650 -0.08635 -0.08904  0.11428  0.15723 -0.12997  0.10441  0.14806 -0.13377 -0.07783 -0.08219",
  "-0.11261  0.04019  0.03693 -0.00891 -0.17325 -0.03032 -0.09799 -0.09435 -0.02796  0.13403  0.13555  0.14321  0.00104 -0.05715 -0.13591 -0.03151  0.03798 -0.05100",
  "-0.05314  0.01551  0.05800 -0.12981  0.00201 -0.02419 -0.03815 -0.00928  0.00739  0.08237  0.03594 -0.03110 -0.01117 -0.03704 -0.10661 -0.02162  0.06792 -0.00889  0.00158",
  "-0.02354  0.33249  0.40194  0.49624 -0.01849  0.28120  0.41691  0.14201  0.34177 -0.27482 -0.27941  0.33110 -0.07385 -0.12269  0.15123  0.25468  0.08308  0.07499  0.04100 -0.25929",
  ] 


  pot_raw_db = [pot_raw1, pot_raw2, pot_raw3, pot_raw4, pot_raw5]
  pot_db = []
  # Create empty matrix
  for i in range(naas):
    pot_db.append([])
    for j in range(naas):
      pot_db[i].append([])
  # Fill lower triangular matrix
  for idx in range(len(pot_raw_db)):
    pot_raw = pot_raw_db[int(idx)]
    for i in range(naas):
      row_tmp = pot_raw[i]
      pot_tmp = row_tmp.strip().split()
      for j in range(i+1):
        pot_db[i][j].append(float(pot_tmp[j]))
  # Fill the full matrix
  for i in range(naas):
    for j in range(i+1,naas):
      pot_db[i][j] = pot_db[j][i]

  pot_db = np.array(pot_db)
  avg_sp = np.mean(pot_db[:][:][0], axis=0)
  std_sp = np.std(pot_db[:][:][0], axis=0)
  nsp = len(avg_sp)

  if aa_list in["all"]:
    return pot_db, avg_sp, std_sp, nsp
  else:
    aai, aaj = aa_list
    if isinstance(aai, int):
      iaa = aai
    else:
      iaa = aas.index(aai)
    if isinstance(aaj, int):
      jaa = aaj
    else:
      jaa = aas.index(aaj)
    return pot_db[iaa][jaa]

# --------------------------------------------------

def interface_spot(idx, interf_spot):

  # Feature: interface spot
  intspot_raw = { 'A':0.0,'R':0.0,'N':0.0,'D':0.0,'C':0.0,
                  'Q':0.0,'E':0.0,'G':0.0,'H':0.0,'I':0.0,
                  'L':0.0,'K':0.0,'M':0.0,'F':0.0,'P':0.0,
                  'S':0.0,'T':0.0,'W':0.0,'Y':0.0,'V':0.0, }

  intspot = [[x] for x in intspot_raw.values()]

  if idx < 0 or len(interf_spot) == 0:
    avg_intspot = np.zeros(1)
    std_intspot = np.ones(1)
    return intspot, avg_intspot, std_intspot
  else:
    return int(idx in interf_spot)

# --------------------------------------------------

def aa_hydropathy_index(aa):

  # Feature: hydropathy index.
  hi_raw = { 'A': 1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C': 2.5,
             'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I': 4.5,
             'L': 3.8,'K':-3.9,'M': 1.9,'F': 2.8,'P':-1.6,
             'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V': 4.2, }

  hi = []    
  for hi_tmp in hi_raw:
    hi.append([hi_raw[hi_tmp]])

  if aa in ["all"]: 
    avg_hi = np.mean(hi)
    std_hi = np.std(hi)
    return hi, avg_hi, std_hi
  else:
    return hi[aas.index(aa)]

# --------------------------------------------------

def aa_volume(aa):

  # Feature: volume
  vol_raw = { 'A': 88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,
              'Q':143.8,'E':138.4,'G': 60.1,'H':153.2,'I':166.7,
              'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,
              'S': 89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0, }

  vol = []    
  for vol_tmp in vol_raw:
    vol.append([vol_raw[vol_tmp]])

  if aa in ["all"]: 
    avg_vol = np.mean(vol)
    std_vol = np.std(vol)
    return vol, avg_vol, std_vol 
  else:
    return vol[aas.index(aa)]

# --------------------------------------------------

def aa_zscales(aa):

  zs_raw = [
  "0.24 -2.32 0.60 -0.14 1.30",
  "3.52 2.50 -3.50 1.99 -0.17",
  "3.05 1.62 1.04 -1.15 1.61",
  "3.98 0.93 1.93 -2.46 0.75",
  "0.84 -1.67 3.71 0.18 -2.65",
  "1.75 0.50 -1.44 -1.34 0.66",
  "3.11 0.26 -0.11 -3.04 -0.25",
  "2.05 -4.06 0.36 -0.82 -0.38",
  "2.47 1.95 0.26 3.90 0.09",
  "-3.89 -1.73 -1.71 -0.84 0.26",
  "-4.28 -1.30 -1.49 -0.72 0.84",
  "2.29 0.89 -2.49 1.49 0.31",
  "-2.85 -0.22 0.47 1.94 -0.98",
  "-4.22 1.94 1.06 0.54 -0.62",
  "-1.66 0.27 1.84 0.70 2.00",
  "2.39 -1.07 1.15 -1.39 0.67",
  "0.75 -2.18 -1.12 -1.46 -0.40",
  "-4.36 3.94 0.59 3.44 -1.59",
  "-2.54 2.44 0.43 0.04 -1.47",
  "-2.59 -2.64 -1.54 -0.85 -0.02",
  ]

  zs = []
  for zs_tmp in zs_raw:
    zs.append([float(x) for x in zs_tmp.strip().split()])

  if aa in ["all"]: 
    avg_zs = np.mean(zs, axis=0)
    std_zs = np.std(zs, axis=0)
    return zs, avg_zs, std_zs
  else:
    return zs[aas.index(aa)]

# --------------------------------------------------

def aa_vhse(aa):

  vhse_raw = [
  "0.15 -1.11 -1.35 -0.92 0.02 -0.91 0.36 -0.48",
  "-1.47 1.45 1.24 1.27 1.55 1.47 1.30 0.83",
  "-0.99 0.00 -0.37 0.69 -0.55 0.85 0.73 -0.80",
  "-1.15 0.67 -0.41 -0.01 -2.68 1.31 0.03 0.56",
  "0.18 -1.67 -0.46 -0.21 0.00 1.20 -1.61 -0.19",
  "-0.96 0.12 0.18 0.16 0.09 0.42 -0.20 -0.41",
  "-1.18 0.40 0.10 0.36 -2.16 -0.17 0.91 0.02",
  "-0.20 -1.53 -2.63 2.28 -0.53 -1.18 2.01 -1.34",
  "-0.43 -0.25 0.37 0.19 0.51 1.28 0.93 0.65",
  "1.27 -0.14 0.30 -1.80 0.30 -1.61 -0.16 -0.13",
  "1.36 0.07 0.26 -0.80 0.22 -1.37 0.08 -0.62",
  "-1.17 0.70 0.70 0.80 1.64 0.67 1.63 0.13",
  "1.01 -0.53 0.43 0.00 0.23 0.10 -0.86 -0.68",
  "1.52 0.61 0.96 -0.16 0.25 0.28 -1.33 -0.20",
  "0.22 -0.17 -0.50 0.05 -0.01 -1.34 -0.19 3.56",
  "-0.67 -0.86 -1.07 -0.41 -0.32 0.27 -0.64 0.11",
  "-0.34 -0.51 -0.55 -1.06 -0.06 -0.01 -0.79 0.39",
  "1.50 2.06 1.79 0.75 0.75 -0.13 -1.01 -0.85",
  "0.61 1.60 1.17 0.73 0.53 0.25 -0.96 -0.52",
  "0.76 -0.92 -0.17 -1.91 0.22 -1.40 -0.24 -0.03",
  ]

  vhse = []
  for vhse_tmp in vhse_raw:
    vhse.append([float(x) for x in vhse_tmp.strip().split()])

  if aa in ["all"]: 
    avg_vhse = np.mean(vhse, axis=0)
    std_vhse = np.std(vhse, axis=0)
    return vhse, avg_vhse, std_vhse
  else:
    return vhse[aas.index(aa)]


# --------------------------------------------------
def info_protein_seq(species):

  ips = {
  'humancov2':   {'tag_ace2':'6LZG_1', 'tag_rbd':'6LZG_2', 'head_ace2':1 , 'tail_ace2':596, 'head_rbd':15, 'tail_rbd':209, 'offset_ace2':18 , 'offset_rbd':318},
  'pig':         {'tag_ace2':'PIG'   , 'tag_rbd':'6LZG_2', 'head_ace2':19, 'tail_ace2':614, 'head_rbd':15, 'tail_rbd':209, 'offset_ace2':0  , 'offset_rbd':0  },
  'cattle':      {'tag_ace2':'BOBOX' , 'tag_rbd':'6LZG_2', 'head_ace2':18, 'tail_ace2':613, 'head_rbd':15, 'tail_rbd':209, 'offset_ace2':0  , 'offset_rbd':0  },
  'chicken':     {'tag_ace2':'CHICK' , 'tag_rbd':'6LZG_2', 'head_ace2':19, 'tail_ace2':614, 'head_rbd':15, 'tail_rbd':209, 'offset_ace2':0  , 'offset_rbd':0  },
  'deer':        {'tag_ace2':'ODOVR' , 'tag_rbd':'6LZG_2', 'head_ace2':18, 'tail_ace2':613, 'head_rbd':15, 'tail_rbd':209, 'offset_ace2':0  , 'offset_rbd':0  },
  }

  if species == 'all':
    return ips
  else: 
    return ips[species]

# --------------------------------------------------
def head_tail_rbd(species):

  # Here, chain_A is ACE2 (length 596), chain_B is RBD (length 195), and the length should add up to 791.
  # The 1st and 2nd number are the residue ID in PDB file, exactly the same as the number shown.
  
  ips = info_protein_seq(species)
  range_A = [ips['head_ace2']+ips['offset_ace2'], ips['tail_ace2']+ips['offset_ace2']]
  range_B = [ips['head_rbd'] +ips['offset_rbd'],  ips['tail_rbd'] +ips['offset_rbd']]
  ht = {'A':range_A, 'B':range_B}

  len_A = ht['A'][1] - ht['A'][0] + 1
  len_B = ht['B'][1] - ht['B'][0] + 1

  return [ht, len_A, len_B]

# --------------------------------------------------
def resi2resn_rbd(mutant, if_ace2=0):

  # General info of PDB 6lzg
  fn_inpt = '/gpfs/scratch/czc325/cov2/mater/6lzg.fasta'
  ft_inpt = fn_inpt.split('.')[-1]
  for record in SeqIO.parse(fn_inpt, ft_inpt):
    if '6LZG_2' in record.id:
      seq_rbd = record.seq[14:]
    elif '6LZG_1' in record.id:
      seq_ace2 = record.seq

  seq_wt = seq_ace2 + seq_rbd
  ht, len_A, len_B = head_tail_rbd('humancov2')

  # Get the original residue name.
  if 'wt' in mutant.lower():
    resn_orig = 'WT'
  else:
    resi = re.findall(r'\d+', mutant)[0]
    if if_ace2:
      idx = int(resi) - ht['A'][0]
    else:
      idx = int(resi) - ht['B'][0] + len_A
    resn_orig = seq_wt[idx]

  return resn_orig

# --------------------------------------------------
def ref_dict():

  fn_ref = "/gpfs/scratch/czc325/cov2/ml/gbsa/mater/labels/exp_data_all_RBD.csv" # For mutations in RBD
  ref = np.loadtxt(fn_ref, delimiter=",", skiprows=1, dtype=np.str)
  ref_dict = {x[0]:10**float(x[1]) for x in ref}

  return ref_dict

# --------------------------------------------------
def spot_interf():

  spot_interf = [417,439,446,449,453,455,456,475,486,487,489,493,494,496,498,500,501,502,503,505]

  return [str(x) for x in spot_interf]

# --------------------------------------------------
def var_preset(nvar, categ):

  var_108_pos = ['439K', '453F', '453K', '455M', '493A', '493F', '493G', '493K', '493L', '493M', '493V', '493Y', '498F', '498H', '498W', '498Y', '501F', '501T', '501V', '501W', '501Y', '503I', '503K', '503L', '503M', '503R', '505W', '346H', '358F', '359Q', '362T', '366N', '367A', '367F', '367W', '378R', '383E', '385R', '406Q', '414A', '440K', '452K', '452Q', '458D', '460K', '468M', '477D', '484R', '490K', '508H', '517M', '518S', '527M', '527Q']

  var_108_neg = ['439E', '439G', '439H', '439Q', '453D', '453P', '453Q', '453R', '455A', '455H', '455S', '455Y', '493D', '498I', '498L', '501E', '505G', '505L', '505P', '505R', '505T', '417E', '417R', '417V', '446E', '446N', '446Q', '449F', '449N', '449P', '456A', '456H', '456P', '475P', '475R', '475Y', '486A', '486D', '486M', '487D', '487L', '487R', '489D', '489K', '489R', '496A', '496D', '496T', '500A', '500N', '500V', '502P', '502Q', '502S']

  var_108_neu = ['WT']

  var_275_pos = ['439K', '453F', '453K', '455M', '493A', '493F', '493G', '493K', '493L', '493M', '493V', '493Y', '498F', '498H', '498W', '498Y', '501F', '501T', '501V', '501W', '501Y', '503I', '503K', '503L', '503M', '503R', '505W', '346H', '358F', '359Q', '362T', '366N', '367A', '367F', '367W', '378R', '383E', '385R', '406Q', '414A', '440K', '452K', '452Q', '458D', '460K', '468M', '477D', '484R', '490K', '508H', '517M', '518S', '527M', '527Q', '335A', '335M', '337D', '339H', '373Q', '427A', '430I', '445P', '492I', '494A', '522T', '333P', '333Q', '341I', '348S', '357K', '384V', '384W', '415S', '428N', '450K', '459D', '469T', '483N', '483R', '494H', '494K', '494R', '519P', '334E', '354K', '354T', '369I', '369M', '369Q', '369T', '394S', '402V', '441I', '441V', '470Q', '481K', '481R', '482Q', '340D', '370Q', '371D', '388D', '389E', '392W', '434F', '444Q', '478K', '478Q', '479E', '514T', '520H', '521S', '523F', '338W', '354R', '356T', '356V', '366D', '367K', '385H', '390M', '392Y', '452M', '452T', '452V', '490N', '520E', '527D', '334Q', '335E', '346N', '346T', '386Q', '390Y', '415H', '430M', '459G', '460R', '460V', '470E', '490R', '490Y', '527T']

  var_275_neg = ['439E', '439G', '439H', '439Q', '453D', '453P', '453Q', '453R', '455A', '455H', '455S', '455Y', '493D', '498I', '498L', '501E', '505G', '505L', '505P', '505R', '505T', '417E', '417R', '417V', '446E', '446N', '446Q', '449F', '449N', '449P', '456A', '456H', '456P', '475P', '475R', '475Y', '486A', '486D', '486M', '487D', '487L', '487R', '489D', '489K', '489R', '496A', '496D', '496T', '500A', '500N', '500V', '502P', '502Q', '502S', '356E', '360T', '363F', '386Y', '403K', '409N', '410E', '417N', '417T', '424H', '438K', '443I', '462N', '465C', '472C', '474R', '525K', '526D', '439I', '439P', '439R', '446D', '446M', '449W', '456C', '456R', '475Q', '494Y', '496I', '498C', '498K', '500Q', '501G', '417W', '449E', '449T', '453C', '453M', '475V', '486N', '486R', '486W', '489G', '494V', '496H', '498M', '501S', '502K', '417A', '446W', '453L', '453S', '455C', '456K', '456S', '475D', '486S', '487H', '496F', '500W', '502C', '502H', '503F', '449H', '453N', '456Y', '486Y', '493P', '494D', '498A', '502W', '502Y', '503C', '503N', '505I', '505K', '505M', '505N'] 

  var_275_neu = ['WT']

  var_dict = {"var_108_pos": var_108_pos, "var_108_neg": var_108_neg, "var_108_neu": var_108_neu, "var_275_pos": var_275_pos, "var_275_neg": var_275_neg, "var_275_neu": var_275_neu}
  key = f"var_{str(nvar)}_{categ}"

  return var_dict[key]

# --------------------------------------------------
def res_pair_prefactor(idx1, idx2, len_chain_A):

  prefactor = 1.0

  # Increase binding affinity if on different sides, decrease binding affinity if on the same side.
  if (idx1 < len_chain_A) != (idx2 < len_chain_A):
    prefactor *= 1.0
  else:
    #prefactor *= -1.0
    prefactor *= 1.0

  '''
  if abs(idx1 - idx2) == 1:
    prefactor *= 0.50
  elif abs(idx1 - idx2) == 2:
    prefactor *= 0.75
  '''

  return prefactor

# --------------------------------------------------
def res_pair_prefactor_scale(idx1, idx2, len_chain_A):

  prefactor = 1.0

  # Increase binding affinity if on different sides, decrease binding affinity if on the same side.
  if (idx1 < len_chain_A) != (idx2 < len_chain_A):
    prefactor *= 1.0
  else:
    prefactor *= -1.0

  if abs(idx1 - idx2) == 1:
    prefactor *= 0.50
  elif abs(idx1 - idx2) == 2:
    prefactor *= 0.75

  return prefactor

# --------------------------------------------------
def interpret_variable(var_str):

  spec = 'humancov2'
  chain = 'rbd'
  muts = ['wt']

  var_sect = var_str.split('_')
    
  var_muts = var_sect[-1]
  if '_wt' not in var_muts.lower() and var_muts.lower() != 'wt' :
    muts = sorted(re.findall(r'\d+[a-zA-Z]', var_sect[-1]))

  if len(var_sect) > 1:
    var_label = var_sect[0].split('-')
    var_label_lower = [x.lower() for x in var_label]

    if 'ace2' in var_label_lower:
      chain = 'ace2'

    if (len(var_label_lower) == 1 and var_label_lower[0] not in ['ace2', 'rbd']) or (len(var_label_lower) > 1):
      spec = var_label_lower[0]
    else:
      spec = 'humancov2'

  return spec, chain, muts

# --------------------------------------------------
def vc_percent(y_true, y_pred, thld):

  n = len(y_true)
  yt, yp = [], []
  for i in range(n):
    true = float(y_true[i])
    pred = float(y_pred[i])
    if true == thld:
      continue
    else:
      yt.append(true < thld) 
      yp.append(pred < thld)

  arr_yt = np.array(yt)
  arr_yp = np.array(yp)
  
  vcp = accuracy_score(arr_yt, arr_yp)

  return vcp*100.0
