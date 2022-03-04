#!/usr/bin/env python

# >>>>>>
# [DESCRIPTION]:
#
# [AUTHOR]: Chen Chen, Penn State Univ, 2022
# <<<<<<

import sys, os, copy, math, re, shutil, glob, json, argparse, random, collections, time, itertools, gc
import subprocess as sub, numpy as np, pandas as pd, matplotlib.pyplot as plt

sys.path.append('Mater/python')
from cpy_ml import *

from Bio import SeqIO
from Bio.Data.IUPACData import *
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef, r2_score, accuracy_score

# Torch distributed and multiprocessing settings.
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP

# ----------
# Input args
# ----------
# [Running Mode]
# 1: train with preset params;
# 2: predict unknown variants and those in the blind test set;
# 3: train with all positive examples plus certain negative variants;
# 4: train with parameters provided as input (hyperopt optimization);
# 5: predict only unknown variants;
mode = sys.argv[1]

debug = 0 # [Debug flag] 0: normal mode; 1: debug mode.
nseq_target = 791 # Total length of the sequence, ACE2[596] + RBD[195]
dir_multi_mutant = 'Mater/expt_multi_mutant/'
if mode in ['1','3'] and len(sys.argv) > 2 and sys.argv[2] == 'all': # Check if use all data for training.
  if_all = 1
else:
  if_all = 0

# Unknown variants to be predicted.
var_unk = []
var_chk = np.loadtxt('./var_chk.csv', delimiter=',', skiprows=0, dtype=np.str, comments='#')
var_unk.extend(var_chk)

# Blind test set to be excluded.
if mode not in ['5']:
  var_bts = np.loadtxt('./var_blind_test_set.csv', delimiter=',', skiprows=0, dtype=np.str, comments='#')
  var_unk.extend(var_bts)
  print(f'# of variants in blind test set: {len(var_bts)}')

# Specific for automatic scanning process.
if mode in ['2','5'] and len(sys.argv) > 2:
  dir_var_unk = './'
  fn_var_unk = dir_var_unk + 'var_scan_mutant_' + sys.argv[2] + '.csv'
  species = sys.argv[3]
  if sys.argv[2] == '1':
    sub.call(f'cp /gpfs/scratch/czc325/cov2/mater/scan/var_scan_mutant_{species}_1.csv ./{fn_var_unk}', shell=True)
  var_unk_raw = np.loadtxt(fn_var_unk, delimiter=',', skiprows=0, dtype=np.str)
  var_unk = var_unk_raw

nvar_unk = len(var_unk)

# ------------------
# General parameters
# ------------------
if mode in ['1','3']:
  kfold = 5
else:
  kfold = 25
dev = 'gpu' # 'gpu' or 'cpu'
max_epochs = 2000
bat_size = 1000
ep_log_interval = max_epochs//10 # Log interval for epochs, 10%, 20%, 30%, ..., 100%
loss_obj = torch.nn.MSELoss()
thld = 1.0 # Threshold value, correspond to variant with unchanged/neutral binding affinity w.r.t WT.
avg_tgt = 0.8
std_tgt = 0.5
rcut = 12.0
ratio_neg_pos = 3.0 # The ratio between negative and positive training examples.
std_ratio_ace2_rbd = 1.1148508062298277/0.4057055684244194 # The ratio between std value of single-mutant for ace2 and rbd, to match the distribution of two source of data.
ratio_global = 0.1 # Use certain fraction of data, when doing quickly debuging. Default: 1.

# --------------
# CNN parameters
# --------------
# Conv layer 1
ks_conv1 = 7 # kernel size
stride_conv1 = 1
dilation_conv1 = 1
# Conv layer 2
ks_conv2 = 3 # kernel size
stride_conv2 = 1
dilation_conv2 = 1
# Max-pooling layer
ks_pool = 2 # kernel size
stride_pool = 2
padding_pool = 0
dilation_pool = 1
# Fully-connected layer
loupt_conv1 = 10 # n_channels for output
loupt_fc1 = 256

# Quick check using instant hyperopt info
params = {'dropout_rate': 0.5044560593454535, 'ks_conv1': 6.0, 'ks_conv2': 12.0, 'loupt_conv1': 48.0, 'loupt_fc1': 192.0, 'lrn_rate': 0.00021899812515298127, 'wt_decay': 0.0005564168599064129}

dropout_rate = params['dropout_rate']
ks_conv1 = int(params['ks_conv1'])
ks_conv2 = int(params['ks_conv2'])
loupt_conv1 = int(params['loupt_conv1'])
loupt_fc1 = int(params['loupt_fc1'])
lrn_rate = params['lrn_rate']
wt_decay = params['wt_decay']
 
padding_conv1 = ks_conv1//2
padding_conv2 = ks_conv2//2

# ---------------------
# Hyperopt optimization
# ---------------------
if mode in ['4']:
  dropout_rate = float(sys.argv[2])
  lrn_rate = float(sys.argv[3])
  wt_decay = float(sys.argv[4])
  ks_conv1 = int(sys.argv[5])
  ks_conv2 = int(sys.argv[6])
  loupt_conv1 = int(sys.argv[7])
  loupt_fc1 = int(sys.argv[8])

# To make sure results can be reproduced, turn off dropout.
if mode in ['2','5']:
  dropout_rate = 0.0

loupt_conv2 = loupt_conv1//2 # n_channels for output
loupt_fc2 = loupt_fc1//4

# Specify device to use
if dev in ['gpu']:
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')

# ----------------------
# Feature initialization
# ----------------------
# AA-based features
# Feature: interface_spot
intspot, avg_intspot, std_intspot = interface_spot(-1, [])
# Feature: zscales
zs, avg_zs, std_zs = aa_zscales('all')
# Feature: vhse
vhse, avg_vhse, std_vhse = aa_vhse('all')
# Feature: hydropathy index
hi, avg_hi, std_hi = aa_hydropathy_index('all')
# Feature: volume
vol, avg_vol, std_vol = aa_volume('all')
# -----
# Composition features
# Feature: group_residue
gres, avg_gres, std_gres, ngres = group_residue('all')
# -----
# AA-pair-based features
# Feature: spairs
sp, avg_sp, std_sp, nsp = spairs('all')

nskip = ngres + nsp # Lines to skip when performing standardization

# -----------------
# Distance map info
# -----------------
seqs = info_protein_seq('all')
species  = seqs.keys()
seq_keys = seqs['humancov2'].keys()
seq_ace2 = dict()
seq_rbd = dict()
seq_patch = dict()
nseq_spec = dict()
ht = dict()
len_A = dict()
len_B = dict()
distmap = dict()
root_residue = dict()
root_hbond = dict() 
root_saltb = dict()
contact_dict = dict()
interf_pair = dict()
interf_spot = dict()

for s in species:
  # Info from structure: distance map and contact map.
  dir_species = './Mater/species/' + s + '/'
  fn_distmap = dir_species + 'DataTable.txt'
  fn_residue = dir_species + 'residue0.xml'
  fn_hbond = dir_species + 'hydrogenbond0.xml'
  fn_saltb = dir_species + 'saltbridge0.xml'
  tree_residue = ET.parse(fn_residue)
  tree_hbond = ET.parse(fn_hbond)
  tree_saltb = ET.parse(fn_saltb)

  ht[s], len_A[s], len_B[s] = head_tail_rbd(s)
  distmap[s] = np.loadtxt(fn_distmap, delimiter='\t', skiprows=0, dtype=np.float)
  root_residue[s] = tree_residue.getroot()
  root_hbond[s] = tree_hbond.getroot()
  root_saltb[s] = tree_saltb.getroot()

  interf_pair[s] = dict()
  interf_spot[s] = set()
  contact_dict[s] = {'Hydrogen bond': root_hbond[s], 'Salt bridge': root_saltb[s]}
  if debug:
    print()
    print(f'[Species]: {s}')
    print('%6s %6s %6s %6s %6s %6s %6s' % ('res1', 'res2', 'aa1', 'aa2', 'idx1', 'idx2', 'dist'))
    print('--------------------------------------')
  for contact_type, root in contact_dict[s].items():
    if debug:
      print(f'{contact_type}:')
    for struct in root:
      if s in ['humancov2']: # For some reason, 6lzg is different from homology models in naming of structures.
        struct1 = struct.find('STRUCTURE1').text #  RBD, chain B
        struct2 = struct.find('STRUCTURE2').text # ACE2, chain A
      else:
        struct1 = struct.find('STRUCTURE2').text # ACE2, chain A
        struct2 = struct.find('STRUCTURE1').text #  RBD, chain B
      s1s = struct1.split(':')[1].split('[')[0].split()
      s2s = struct2.split(':')[1].split('[')[0].split()
      s1resn = protein_letters_3to1[s1s[0].capitalize()]
      s2resn = protein_letters_3to1[s2s[0].capitalize()]
      s1 = s1resn + s1s[1]
      s2 = s2resn + s2s[1]
      dist_idx1 = int(s1s[1]) - ht[s]['B'][0] + len_A[s] # index of spot on RBD
      dist_idx2 = int(s2s[1]) - ht[s]['A'][0] # index of spot on ACE2
      interf_spot[s].add(dist_idx1)
      interf_spot[s].add(dist_idx2)
      spidx1, spidx2 = [aas.index(s1resn), aas.index(s2resn)]
      val_sp = sp[spidx1][spidx2]
      interf_pair[s]['-'.join([s1,s2])] = [s1resn, s2resn, dist_idx1, dist_idx2, distmap[s][dist_idx1][dist_idx2]]
      if debug:
        print('%6s %6s %6s %6s %6s %6s %6.2f' % (s1, s2, s1resn, s2resn, dist_idx1, dist_idx2, distmap[s][dist_idx1][dist_idx2]))
  if debug:
    print(f'\nIndexes of spots at interface: {interf_spot[s]}')

  # Info from sequence
  fn_inpt = dir_species + '/complex.fasta'
  ft_inpt = fn_inpt.split('.')[-1]
  for record in SeqIO.parse(fn_inpt, ft_inpt):
    if seqs[s]['tag_ace2'] in record.id:
      seq_ace2[s] = record.seq[seqs[s]['head_ace2']-1:seqs[s]['tail_ace2']]
    elif seqs[s]['tag_rbd'] in record.id:
      seq_rbd[s] = record.seq[seqs[s]['head_rbd']-1:seqs[s]['tail_rbd']]
  nseq_ace2  = len(seq_ace2[s])
  nseq_rbd   = len(seq_rbd[s])
  nseq_patch = nseq_target - nseq_ace2 - nseq_rbd
  if nseq_patch > 0:
    seq_patch[s] = 'Z'*nseq_patch # CCATTN
  else:
    seq_patch[s] = ''
  nseq_spec[s] = nseq_ace2 + nseq_rbd
  if debug:
    print(f'\nACE2-RBD complex\nACE2:{len(seq_ace2[s])}\n{seq_ace2[s]}\nRBD:{len(seq_rbd[s])}\n{seq_rbd[s]}')

# All features
sp_zeros = np.zeros((naas, nsp))
gres_zeros = np.zeros((naas, ngres)) # group of residues
feature_dict = collections.OrderedDict() # Using ordered dict is important.
feature_dict['intspot'] = [intspot, avg_intspot, std_intspot] # 1d
feature_dict['zs']      = [zs, avg_zs, std_zs] # 5d
feature_dict['vhse']    = [vhse, avg_vhse, std_vhse] # 8d
feature_dict['hi']      = [hi, avg_hi, std_hi] # 1d
feature_dict['vol']     = [vol, avg_vol, std_vol] # 1d
feature_dict['gres']    = [gres_zeros, avg_gres, std_gres] # 3d*13 + 5d*1
feature_dict['sp']      = [sp_zeros, avg_sp, std_sp] # 1d*5
feature_list = [v[0] for k,v in feature_dict.items()]
feature_avg  = [v[1] for k,v in feature_dict.items()]
feature_std  = [v[2] for k,v in feature_dict.items()]
feature = np.hstack(feature_list)
avg_all = np.hstack(feature_avg)
std_all = np.hstack(feature_std)
nfea = len(feature[0])

# Prepare reference dict to get experiment kd-ratio.
fn_dict = {
1:'exp_data_all_RBD.csv', 
2:'expt_multi_mutant_2.csv', 
3:'expt_multi_mutant_3.csv', 
4:'expt_multi_mutant_4.csv',
5:'expt_multi_mutant_5.csv',
6:'expt_multi_mutant_6.csv',
11:'expt_multi_mutant_ace2_1.csv',
}

fn_keys = fn_dict.keys()
fn_keys = [1,2,3,4,5,6,11] # Only include variants in these csv files.
ref = dict()
ref_dict = dict()
for k in sorted(fn_dict.keys()):
  fn_ref = dir_multi_mutant + fn_dict[k]
  ref_raw = np.loadtxt(fn_ref, delimiter=',', skiprows=1, dtype=np.str)
  if k in [11]:
    ref_dict_tmp = {x[0]:(10**float(x[1])-thld)/std_ratio_ace2_rbd+thld for x in ref_raw} # The scale of ACE2 data are larger than RBD data, scaling is needed.
  else:
    ref_dict_tmp = {x[0]:10**float(x[1]) for x in ref_raw}
  ref.update(ref_dict_tmp)
  ref_dict[k] = ref_dict_tmp

# Variants with AA change at spots beyond the ACE2 or RBD range.
var_beyond = []
for var_str_full in ref.keys():
  if_within_range = 1
  spec, chain, muts = interpret_variable(var_str_full)
  for m in muts:
    spot = int(re.findall(r'\d+', m)[0])
    if chain == 'ace2':
      if spot < ht['humancov2']['A'][0] or spot > ht['humancov2']['A'][1]:
        if_within_range = 0 
    else:
      if spot < ht['humancov2']['B'][0] or spot > ht['humancov2']['B'][1]:
        if_within_range = 0 
  if not if_within_range:
    var_beyond.append(var_str_full)

# Categorize the variants.
var_all, var_all_pos, var_all_neg, var_all_neu = [], [], [], []
var_curr, var_curr_pos, var_curr_neg, var_curr_neu = {}, {}, {}, {}
var_full, var_full_pos, var_full_neg, var_full_neu = {}, {}, {}, {}
var_rest, var_rest_pos, var_rest_neg, var_rest_neu = {}, {}, {}, {}
var_extra, var_extra_pos, var_extra_neg, var_extra_neu = {}, {}, {}, {}
var_avail, var_avail_pos, var_avail_neg, var_avail_neu = {}, {}, {}, {}
nvar_curr_pos, nvar_curr_neg, nvar_curr_neu = {}, {}, {}
nvar_full_pos, nvar_full_neg, nvar_full_neu = {}, {}, {}
nvar_rest_pos, nvar_rest_neg, nvar_rest_neu = {}, {}, {}
nvar_extra_pos, nvar_extra_neg, nvar_extra_neu = {}, {}, {}
nvar_avail_pos, nvar_avail_neg, nvar_avail_neu = {}, {}, {}
nvar_single, nvar_multi, nvar_avail = 0, 0, 0

for k in fn_keys:
  var_curr_pos[k] = []
  var_curr_neg[k] = []
  var_curr_neu[k] = []
  var_curr_pos[k] = list(set(var_curr_pos[k]).difference(set(var_unk)))
  var_curr_neg[k] = list(set(var_curr_neg[k]).difference(set(var_unk)))
  var_curr_neu[k] = list(set(var_curr_neu[k]).difference(set(var_unk)))
  nvar_curr_pos[k] = len(var_curr_pos[k])
  nvar_curr_neg[k] = len(var_curr_neg[k])
  nvar_curr_neu[k] = len(var_curr_neu[k])
  var_curr[k] = var_curr_pos[k] + var_curr_neg[k] + var_curr_neu[k]
  ref_dict_tmp      = ref_dict[k]
  var_full[k]       = ref_dict_tmp
  var_full_pos[k]   = [x for x in var_full[k] if ref_dict_tmp[x] >  thld]
  var_full_neg[k]   = [x for x in var_full[k] if ref_dict_tmp[x] <  thld]
  var_full_neu[k]   = [x for x in var_full[k] if ref_dict_tmp[x] == thld]
  nvar_full_pos[k]  = len(var_full_pos[k])
  nvar_full_neg[k]  = len(var_full_neg[k])
  nvar_full_neu[k]  = len(var_full_neu[k])
  var_rest[k]       = set(var_full[k]).difference(set(var_curr[k])).difference(set(var_beyond)).difference(set(var_unk))
  var_rest_pos[k]   = [x for x in var_rest[k] if ref_dict_tmp[x] >  thld]
  var_rest_neg[k]   = [x for x in var_rest[k] if ref_dict_tmp[x] <  thld]
  var_rest_neu[k]   = [x for x in var_rest[k] if ref_dict_tmp[x] == thld]
  nvar_rest_pos[k]  = len(var_rest_pos[k])
  nvar_rest_neg[k]  = len(var_rest_neg[k])
  nvar_rest_neu[k]  = len(var_rest_neu[k])
  nvar_extra_pos[k] = len(var_rest_pos[k]) # Use all the positive variants.
  nvar_extra_neg[k] = min(int((nvar_curr_pos[k] + nvar_extra_pos[k])*ratio_neg_pos) - nvar_curr_neg[k], nvar_rest_neg[k])
  nvar_extra_neu[k] = min(int((nvar_curr_pos[k] + nvar_extra_pos[k])*ratio_neg_pos) - nvar_curr_neu[k], nvar_rest_neu[k])
  var_extra_pos[k]  = np.random.choice(var_rest_pos[k], size=nvar_extra_pos[k], replace=False)
  var_extra_neg[k]  = np.random.choice(var_rest_neg[k], size=nvar_extra_neg[k], replace=False)
  var_avail_pos[k]  = set(var_full_pos[k]).difference(set(var_beyond))
  var_avail_neg[k]  = set(var_full_neg[k]).difference(set(var_beyond))
  var_avail_neu[k]  = set(var_full_neu[k]).difference(set(var_beyond))
  nvar_avail_pos[k] = len(var_avail_pos[k])
  nvar_avail_neg[k] = len(var_avail_neg[k])
  nvar_avail_neu[k] = len(var_avail_neu[k])
  if debug:
    print(nvar_curr_neu, nvar_full_neu, nvar_rest_neu)
  try:
    var_extra_neu[k] = np.random.choice(var_rest_neu[k], size=nvar_extra_neu[k], replace=False)
  except: # The neutral database could be empty.
    var_extra_neu[k] = np.array([])
  var_extra[k]      = var_extra_pos[k].tolist() + var_extra_neg[k].tolist() + var_extra_neu[k].tolist()
  if k not in [1,11]:
    nvar_multi += nvar_curr_pos[k] + nvar_curr_neg[k] + nvar_curr_neu[k] + nvar_extra_pos[k] + nvar_extra_neg[k] + nvar_extra_neu[k]
  var_all_pos.extend(var_curr_pos[k])
  var_all_neg.extend(var_curr_neg[k])
  var_all_neu.extend(var_curr_neu[k])
  var_all.extend(var_curr[k])
  if mode in ['2', '3', '4']:
    var_all_pos.extend(var_extra_pos[k].tolist())
    var_all_neg.extend(var_extra_neg[k].tolist())
    var_all_neu.extend(var_extra_neu[k].tolist())
    var_all.extend(var_extra[k])

for k in [1,11]:
  try:
    nvar_single += nvar_curr_pos[k] + nvar_curr_neg[k] + nvar_curr_neu[k] + nvar_extra_pos[k] + nvar_extra_neg[k] + nvar_extra_neu[k]
  except:
    pass
navail_pos = sum(nvar_avail_pos.values())
navail_neg = sum(nvar_avail_neg.values())
navail_neu = sum(nvar_avail_neu.values())
navail = navail_pos + navail_neg + navail_neu

var_all_sorted = sorted(var_all)
var_unk_sorted = sorted(var_unk)
varMapDct = {}
varMapDctRev = {}
for i in range(len(var_all)):
  varMapDct[i] = var_all_sorted[i]
  varMapDctRev[var_all_sorted[i]] = i
for i in range(len(var_all),len(var_all)+len(var_unk)):
  varMapDct[i] = var_unk_sorted[i-len(var_all)]
  varMapDctRev[var_unk_sorted[i-len(var_all)]] = i

var_trn = [[] for _ in range(kfold)]
var_tst = [[] for _ in range(kfold)]
posKey = np.copy(var_all_pos)
negKey = np.copy(var_all_neg)
neuKey = np.copy(var_all_neu)
np.random.shuffle(posKey)
np.random.shuffle(negKey)
np.random.shuffle(neuKey)
# Use only a faction of data to accelerate the training process.
posKey = posKey[:int(ratio_global*len(posKey))]
negKey = negKey[:int(ratio_global*len(negKey))]
neuKey = neuKey[:int(ratio_global*len(neuKey))]
var_all = list(np.concatenate((posKey, negKey, neuKey), axis=0))
nvar_all = len(var_all)
nvar_all_pos = len(posKey)
nvar_all_neg = len(negKey)
nvar_all_neu = len(neuKey)
nvar_multi = int(nvar_multi*ratio_global)
nvar_single = nvar_all - nvar_multi
posKeySplit = np.array_split(posKey,kfold)
negKeySplit = np.array_split(negKey,kfold)
neuKeySplit = np.array_split(neuKey,kfold)

# Construct trn/tst sets.
for ifold in range(kfold):
  var_tst[ifold] = np.concatenate((posKeySplit[ifold], negKeySplit[ifold], neuKeySplit[ifold]), axis=0)
  var_trn[ifold] = [x for x in var_all if x not in var_tst[ifold]]

# Build blind test set.
#print(f'{len(var_all)}, {len(posKeySplit[ifold])}, {len(negKeySplit[ifold])}, {len(neuKeySplit[ifold])}')
#fn_bts = 'var_blind_test_set.csv'
#np.savetxt(fn_bts, var_tst[0], delimiter=',', fmt='%s')

# --------------------------------------------------

def setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '99999'

  # initialize the process group
  dist.init_process_group("gloo", rank=rank, world_size=world_size)

# --------------------------------------------------

def cleanup():
  dist.destroy_process_group()

# --------------------------------------------------

def run_mp(mp_fn, world_size, var_mp):
  mp.spawn(mp_fn,
           args=(world_size, var_mp),
           nprocs=world_size,
           join=True)

# --------------------------------------------------

def feature_init_wt(seq, s):

  nseq_curr = len(seq)
  nseq = min(nseq_spec[s], nseq_curr, nseq_target)
  fea = np.zeros([nseq_target, nfea], dtype=float) 
  #print(s, nseq_curr, nseq, len(distmap[s]), '\n')
  for idx in range(nseq):
    aa0 = seq[idx]
    fea[idx] = feature[aas.index(aa0)]
    fea[idx][0] = interface_spot(idx, interf_spot[s]) # Indicate if this is an interface spot.
    neighbors = [[j,v,seq[j]] for j,v in enumerate(distmap[s][idx][:nseq]) if v < rcut and v > 0.0]
    sp_items = []
    sp_weights = []
    for neigh in neighbors:
      idx_aa2 = neigh[0] # Neighbor of current exchanged AA
      dist_02 = neigh[1]
      aa2 = neigh[2]
      neigh_aa_str = ''
      if dist_02 < rcut:
        spidx0, spidx2 = [aas.index(aa0), aas.index(aa2)]
        val_sp02 = sp[spidx0][spidx2]
        prefactor = res_pair_prefactor_scale(idx, idx_aa2, len_A[s]) # Comment this line when no scaling is needed.
        sp_items.append(val_sp02)
        sp_weights.append(prefactor/dist_02)
        neigh_aa_str += aa2
    sp_items = np.array(sp_items)
    sp_weights = np.tile(np.transpose([sp_weights]),(1,len(sp_items[0])))
    fea3 = np.average(sp_items, weights=sp_weights, axis=0)
    fea[idx][-nsp:] = fea3[:]
    fea[idx][-nskip:-nsp] = group_residue(neigh_aa_str)
  fea[:,1:-nskip] = (fea[:,1:-nskip] - avg_all[1:-nskip])/std_all[1:-nskip] 

  return fea
  
# Apply features to sequence
seq_wt = dict()
fea_wt = dict()
for s in species: 
  #print(f'{seq_ace2[s]}\n{seq_rbd[s]}')
  seq_wt[s] = seq_ace2[s] + seq_rbd[s] + seq_patch[s]
  fea_wt[s] = feature_init_wt(seq_wt[s], s)
# Other CNN parameters
linpt_conv1 = nfea # n_channels for input, determined by the number of features
l_tmp = len(seq_wt['humancov2'])
l_tmp = (l_tmp + 2*padding_conv1 - dilation_conv1*(ks_conv1-1) - 1)//stride_conv1 + 1
l_tmp = (l_tmp + 2*padding_pool - dilation_pool*(ks_pool-1) - 1)//stride_pool + 1
l_tmp = (l_tmp + 2*padding_conv2 - dilation_conv2*(ks_conv2-1) - 1)//stride_conv2 + 1
l_tmp = (l_tmp + 2*padding_pool - dilation_pool*(ks_pool-1) - 1)//stride_pool + 1
linpt_fc1 = l_tmp*loupt_conv2

# ---------------------------------------------------------

class Dataset(torch.utils.data.Dataset):

  def __init__(self, vars, num_rows=None):
    
    feas = []
    tgts = []
    lbls = []
    for i in range(len(vars)):
      var_str_full = vars[i]
      s, chain, muts = interpret_variable(var_str_full)
      fea_var = np.copy(fea_wt[s])
      for m in muts:
        if m == 'wt':
          continue
        resid = int(re.findall(r'\d+', m)[0])
        if chain == 'ace2':
          idx = resid - ht[s]['A'][0]
        else:
          idx = resid - ht[s]['B'][0] + len_A[s]
        dist_idx_tmp = idx
        aa0 = seq_wt[s][idx] # AA before substitution
        aa1 = m[-1] # AA after substitution

        # [cc]: Needs to be cleaned up here.
        neighbors = [[j,v, seq_wt[s][j]] for j,v in enumerate(distmap[s][dist_idx_tmp]) if v < rcut and v > 0.0]
        neigh_aa0_str = '' #aa0
        neigh_aa1_str = '' #aa1
        sp_items = []
        sp_weights = []
        for neigh in neighbors:
          idx_aa2 = neigh[0] # Neighbor of current exchanged AA
          dist_02 = neigh[1] # Distance between aa0 and aa2
          aa2 = neigh[2]

          # fea2(for aa2): feature about distribution of polar/neutral/hydrophobicity in environment
          if dist_02 < rcut:
            neighbors_aa2 = [[j,v, seq_wt[s][j]] for j,v in enumerate(distmap[s][idx_aa2]) if v < rcut and v > 0.0]
            sp_items_aa2 = []
            sp_weights_aa2 = []
            for neigh_aa2 in neighbors_aa2:
              idx_aa3 = neigh_aa2[0]
              dist_23 = neigh_aa2[1]
              aa3 = neigh_aa2[2]
              spidx2, spidx3 = [aas.index(aa2), aas.index(aa3)]
              val_sp23 = sp[spidx2][spidx3]
              prefactor = res_pair_prefactor_scale(idx_aa2, idx_aa3, len_A[s]) # Comment this line when no scaling is needed.
              sp_items_aa2.append(val_sp23)
              sp_weights_aa2.append(prefactor/dist_23)
            # fea3: AA-pair-based features
            sp_items_aa2 = np.array(sp_items_aa2)
            sp_weights_aa2 = np.tile(np.transpose([sp_weights_aa2]),(1,len(sp_items_aa2[0])))
            fea3_aa2 = np.average(sp_items_aa2, weights=sp_weights_aa2, axis=0)
            fea_var[idx_aa2][-nsp:] = fea3_aa2[:]
            spidx1, spidx2 = [aas.index(aa1), aas.index(aa2)]
            val_sp12 = sp[spidx1][spidx2]
            # Increase binding affinity if on different sides, decrease binding affinity if on the same side.
            prefactor = res_pair_prefactor_scale(idx, idx_aa2, len_A[s]) # Comment this line when no scaling is needed.
            sp_items.append(val_sp12)
            sp_weights.append(prefactor/dist_02)
            neigh_aa0_str += aa2
            neigh_aa1_str += aa2
            fea2 = group_residue(aa1) - group_residue(aa0)
            fea_var[idx_aa2][-nskip:-nsp] += fea2[:]

        sp_items = np.array(sp_items)
        sp_weights = np.tile(np.transpose([sp_weights]),(1,len(sp_items[0])))
        fea3_aa1 = np.average(sp_items, weights=sp_weights, axis=0)
        fea_var[idx][-nsp:] = fea3_aa1[:]
        # fea2(for aa01): feature about distribution of polar/neutral/hydrophobicity in environment
        fea2 = group_residue(neigh_aa1_str) - group_residue(neigh_aa0_str)
        #print(var_str, fea_var[idx][-nskip:-nsp], fea2)
        fea_var[idx][-nskip:-nsp] += fea2[:]

        # fea1: the AA-based features
        fea1 = (feature[aas.index(aa1)][1:-nskip] - avg_all[1:-nskip])/std_all[1:-nskip]
        fea_var[idx][1:-nskip] = fea1[:]
      fea_var = fea_var.transpose()
      try:
        tgt_var = ref[var_str_full]
      except:
        tgt_var = 1.0
        #if 'wt' in muts:
        #  tgt_var = 1.0
        #else:
        #  tgt_var = 5.0
      # Standardization of target value
      tgt_var = (tgt_var - avg_tgt)/std_tgt

      # CCATTN
      #if s in ['pig']:
      #  fea_var = np.delete(fea_var, 136, 1)

      feas.append(fea_var[:])
      tgts.append(tgt_var)
      lbls.append(varMapDctRev[var_str_full])

    if mode in ['5']:
      self.x_data = torch.tensor(feas, dtype=torch.float)
      self.y_data = torch.tensor(tgts, dtype=torch.float)
      self.label = torch.tensor(lbls, dtype=torch.float)
    else:
      self.x_data = torch.tensor(feas, dtype=torch.float).to(device)
      self.y_data = torch.tensor(tgts, dtype=torch.float).to(device)
      self.label = torch.tensor(lbls, dtype=torch.float).to(device)
    self.y_data = self.y_data.reshape(-1,1)
    self.label = self.label.reshape(-1,1)
    del feas, tgts, lbls
    gc.collect()

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    fea = self.x_data[idx,:]
    tgt = self.y_data[idx,:]
    lbl = self.label[idx,:]
    sample = {'feature':fea, 'target':tgt, 'label':lbl}

    return sample

# --------------------------------------------------

class Net(torch.nn.Module):

  def __init__(self):
    super(Net, self).__init__()

    self.conv1 = torch.nn.Conv1d(in_channels=linpt_conv1, out_channels=loupt_conv1, kernel_size=ks_conv1, stride=stride_conv1, padding=padding_conv1, dilation=dilation_conv1)
    self.conv2 = torch.nn.Conv1d(in_channels=loupt_conv1, out_channels=loupt_conv2, kernel_size=ks_conv2, stride=stride_conv2, padding=padding_conv2, dilation=dilation_conv2)

    #self.pool = torch.nn.AvgPool1d(kernel_size=ks_pool, stride=stride_pool, padding=padding_pool, dilation=dilation_pool, count_include_pad=False) # MaxPool1d, AvgPool1d
    self.pool = torch.nn.AvgPool1d(kernel_size=ks_pool, stride=stride_pool, padding=padding_pool, count_include_pad=False) # MaxPool1d, AvgPool1d

    self.fc1 = torch.nn.Linear(linpt_fc1, loupt_fc1)
    self.fc2 = torch.nn.Linear(loupt_fc1, loupt_fc2)
    self.oupt = torch.nn.Linear(loupt_fc2, 1)

    self.layers = [self.fc1, self.fc2] + [self.oupt]

    for layer in self.layers:
      torch.nn.init.xavier_uniform_(layer.weight)
      torch.nn.init.zeros_(layer.bias)

    # Define proportion or neurons to dropout
    self.dropout = torch.nn.Dropout(dropout_rate)
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    z = x
    # Conv1 + Relu + Pool1
    z = self.conv1(z)
    z = self.relu(z)
    z = self.pool(z)
    # Conv2 + Relu + Pool2
    z = self.conv2(z)
    z = self.relu(z)
    z = self.pool(z)
    # Flatten
    z = torch.flatten(z, 1)
    # Fully-Connected1 + Relu + Dropout
    z = self.fc1(z)
    z = self.relu(z)
    z = self.dropout(z)
    # Fully-Connected2 + Relu + Dropout
    z = self.fc2(z)
    z = self.relu(z)
    z = self.dropout(z)
    # Output
    z = self.oupt(z)

    return z

# --------------------------------------------------

def accuracy(model, ds):

  ndat = len(ds)
  tgts, prds = [], []
  tgts_mcc, prds_mcc = [], []
  tgts_multi, prds_multi = [], []
  tgts_mcc_multi, prds_mcc_multi = [], []
  rst_multi = []
  dct_rst = {}
  varMap = {}
  loss_val = 0.0

  for idat in range(ndat):
    lbl = ds[idat]['label']
    fea = ds[idat]['feature']
    tgt = ds[idat]['target']
    lbl = int(torch.round(lbl))

    fea.unsqueeze_(0) # Transform 2d tensor to 3d

    with torch.no_grad():
      prd = model(fea).reshape(-1) # Transform 2d tensor to 1d

    # Recover the original value
    tgt *= std_tgt
    prd *= std_tgt
    tgt += avg_tgt
    prd += avg_tgt
    varMap[lbl] = tgt.item()

    tgts.append(tgt.item())
    prds.append(prd.item())

    loss_val += loss_obj(prd, tgt).item()

    if tgt < thld:
      tgts_mcc.append(0)
    elif tgt > thld:
      tgts_mcc.append(1)

    if tgt == thld: # Neutral examples
      pass
    else:
      if prd < thld:
        prds_mcc.append(0)
      elif prd >= thld:
        prds_mcc.append(1)

    if lbl in dct_rst:
      dct_rst[lbl][0] += 1
      dct_rst[lbl][1].append(prd.item())
    else:
      dct_rst[lbl] = [1,[prd.item()]]

  tgts = np.array(tgts)
  prds = np.array(prds)
  rst = np.array(list(zip(tgts, prds)))
  tgts_mcc = np.array(tgts_mcc)
  prds_mcc = np.array(prds_mcc)

  acc = np.count_nonzero(tgts_mcc == prds_mcc) * 1.0/ndat
  mcc_cov = np.cov(tgts_mcc, prds_mcc)
  if mcc_cov[0][0] * mcc_cov[1][1] == 0:
    mcc = 0.0
  else:
    mcc = matthews_corrcoef(tgts_mcc, prds_mcc) # Matthews correlation coefficient
  mse = ((tgts - prds)**2).mean()
  r2  = r2_score(tgts, prds)
  r   = np.corrcoef(tgts, prds)[0,1]

  for lbl in dct_rst:
    rst_multi.append([varMap[lbl], np.median(dct_rst[lbl][1]), lbl]) # Or use np.mean()
  ndat_multi = len([x for x in rst_multi if x[0] != thld])

  for x in rst_multi:
    if x[0] == thld:
      continue
    tgts_multi.append(x[0])
    if x[0] < thld:
      tgts_mcc_multi.append(0)
    elif x[0] >= thld:
      tgts_mcc_multi.append(1)
    prds_multi.append(x[1])
    if x[1] < thld:
      prds_mcc_multi.append(0)
    elif x[1] >= thld:
      prds_mcc_multi.append(1)

  tgts_multi = np.array(tgts_multi)
  prds_multi = np.array(prds_multi)
  tgts_mcc_multi = np.array(tgts_mcc_multi)
  prds_mcc_multi = np.array(prds_mcc_multi)
  mse_multi = ((tgts_multi - prds_multi)**2).mean()
  acc_multi = np.count_nonzero(tgts_mcc_multi == prds_mcc_multi) * 1.0/ndat_multi
  mcc_multi_cov = np.cov(tgts_mcc_multi, prds_mcc_multi)
  if mcc_multi_cov[0][0] * mcc_multi_cov[1][1] == 0:
    mcc_multi = 0.0
  else:
    mcc_multi = matthews_corrcoef(tgts_mcc_multi, prds_mcc_multi)

  rstarr   = np.array(rst_multi)
  r2_multi = r2_score(rstarr[:,0], rstarr[:,1])
  r_multi  = np.corrcoef(rstarr[:,0], rstarr[:,1])[0,1]

  return rst, acc, mcc, mse, r2, r, rst_multi, acc_multi, mcc_multi, mse_multi, r2_multi, r_multi

# --------------------------------------------------

def pred_mp(rank, world_size, var_mp):

  setup(rank, world_size)
  rst_mp_kfold = [[] for ifold in range(kfold)]

  for ifold in range(kfold):
    model = Net()
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    del model
    map_location = {'cuda:0': f'cuda:{rank}'}
    path = f'./Models/model_{str(ifold)}.pth'
    state_dict = torch.load(path, map_location=map_location)
    new_state_dict = collections.OrderedDict()

    for k, v in state_dict.items():
      if 'module.' not in k:
        name = 'module.' + k
      else:
        name = k
      new_state_dict[name] = v

    ddp_model.load_state_dict(new_state_dict)
    nchunk = 20000
    nsplit = int(np.ceil(len(var_mp[rank])/nchunk))
    var_mp_split = [var_mp[rank][i*nchunk:(i+1)*nchunk] for i in range(nsplit)]

    for isplit in range(nsplit):
      ds = Dataset(np.copy(var_mp_split[isplit]))
      for i in range(len(ds)):
        dat = ds[i]
        lbl = dat['label']
        fea = dat['feature'].unsqueeze_(0)
        tgt = dat['target'].item()
        lbl = int(torch.round(lbl))
        prd = ddp_model(fea).item()
        # Recover original value
        tgt *= std_tgt
        prd *= std_tgt
        tgt += avg_tgt
        prd += avg_tgt

        rst_mp_kfold[ifold].append([str(lbl), str(tgt), str(prd)])
    
    rst_mp_ifold = np.array(rst_mp_kfold[ifold])
    fn_mp_ifold = f'./Results/pred_ifold{ifold}_rank{rank}.csv'
    f_mp_ifold = open(fn_mp_ifold, 'w')
    np.savetxt(f_mp_ifold, rst_mp_ifold, delimiter=',', fmt='%s')
    f_mp_ifold.close()

    del state_dict, new_state_dict, ds, rst_mp_ifold, var_mp_split
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------

def main():
 
  net = Net()
  net.to(device)
  net = net.train()

  optimizer = torch.optim.Adam(net.parameters(), lr=lrn_rate, weight_decay=wt_decay, amsgrad=False)
  print(f'Device: {dev}')
  print(f'kfold: {kfold}')
  print(f'Predicting binding affinity using PyTorch')
  print(f'NN configuration\n{str(net)}')
  print(f'Loss function: {str(loss_obj)}')
  print(f'Optimizer: {str(optimizer)}')
  print(f'Batch size: {bat_size}')
  print(f'Max epochs: {str(max_epochs)}')

  if mode in ['1','3','4']:
    print(f'\nDataset composition:')
    for k in fn_keys:
      print(f'# of positive, negative, neutral {k:2d}-mutated variants in [curr/extra/avail/full] dataset: [{nvar_curr_pos[k]}/{nvar_extra_pos[k]}/{nvar_avail_pos[k]}/{nvar_full_pos[k]}], [{nvar_curr_neg[k]}/{nvar_extra_neg[k]}/{nvar_avail_neg[k]}/{nvar_full_neg[k]}], [{nvar_curr_neu[k]}/{nvar_extra_neu[k]}/{nvar_avail_neu[k]}/{nvar_full_neu[k]}]')
    print(f'# of all, positive, negative, neutral variants in the available dataset: {navail}, {navail_pos}({100.0*navail_pos/navail:.2f}%), {navail_neg}({100.0*navail_neg/navail:.2f}%), {navail_neu}({100.0*navail_neu/navail:.2f}%)')
    print(f'# of all, positive, negative, neutral variants in the final dataset: {nvar_all}, {nvar_all_pos}({100.0*nvar_all_pos/nvar_all:.2f}%), {nvar_all_neg}({100.0*nvar_all_neg/nvar_all:.2f}%), {nvar_all_neu}({100.0*nvar_all_neu/nvar_all:.2f}%)')
    print(f'# of all, single-, multi-mutated variants: {nvar_all}, {nvar_single}({100.0*nvar_single/nvar_all:.2f}%), {nvar_multi}({100.0*nvar_multi/nvar_all:.2f}%)')

  if mode in ['1', '3', '4']:

    acc_bat = [[] for ifold in range(kfold)]
    acc_multi_bat = [[] for ifold in range(kfold)]
    mcc_bat = [[] for ifold in range(kfold)]
    mcc_multi_bat = [[] for ifold in range(kfold)]
    mse_bat = [[] for ifold in range(kfold)]
    mse_multi_bat = [[] for ifold in range(kfold)]
    rst_bat = [[] for ifold in range(kfold)]
    rst_multi_bat = [[] for ifold in range(kfold)]
    rst_multi_unk_bat = [[] for ifold in range(kfold)]
    r2_bat = [[] for ifold in range(kfold)]
    r2_multi_bat = [[] for ifold in range(kfold)]
    r_bat = [[] for ifold in range(kfold)]
    r_multi_bat = [[] for ifold in range(kfold)]

    for ifold in range(kfold):
      #if if_all and ifold > 0:
      #  acc_bat[ifold]       = acc_bat[0]
      #  acc_multi_bat[ifold] = acc_multi_bat[0]
      #  mcc_bat[ifold]       = mcc_bat[0]      
      #  mcc_multi_bat[ifold] = mcc_multi_bat[0]
      #  mse_bat[ifold]       = mse_bat[0]      
      #  mse_multi_bat[ifold] = mse_multi_bat[0]
      #  rst_bat[ifold]       = rst_bat[0]      
      #  rst_multi_bat[ifold] = rst_multi_bat[0]
      #  r2_bat[ifold]        = r2_bat[0]       
      #  r2_multi_bat[ifold]  = r2_multi_bat[0] 
      #  r_bat[ifold]         = r_bat[0]        
      #  r_multi_bat[ifold]   = r_multi_bat[0]  
      #  rst_multi_unk_bat[ifold] = rst_multi_unk_bat[0]
      #  path = f'./Models/model_{str(ifold)}.pth'
      #  print(f'\nSaving trained model state_dict for ifold = {ifold+1}...')
      #  torch.save(net.state_dict(), path)
      #  continue

      time_0 = time.time()
      net = Net()
      net = DP(net)
      net.to(device)
      net = net.train()
      optimizer = torch.optim.Adam(net.parameters(), lr=lrn_rate, weight_decay=wt_decay, amsgrad=False)

      print('\n[ifold = %s/%s]' % (ifold+1, kfold))
      dstps = ['trn', 'tst', 'unk', 'all']
      ndstp = len(dstps)
      if if_all:
        varsMap = {'trn':var_all, 'tst':var_tst[ifold], 'all':var_all, 'unk':var_unk} # File names of different datasets: 'trn', 'tst'
      else:
        varsMap = {'trn':var_trn[ifold], 'tst':var_tst[ifold], 'all':var_all, 'unk':var_unk} # File names of different datasets: 'trn', 'tst'
      dsMap = {} # Datasets for different types: 'trn', 'tst'
      for dstp in dstps:
        vars = varsMap[dstp]
        ds = Dataset(np.copy(vars))
        dsMap[dstp] = ds
      ntrn = len(varsMap['trn'])
      ntst = len(varsMap['tst'])
      frac_trn = ntrn/nvar_all*100.0
      frac_tst = ntst/nvar_all*100.0
      print(f"# of variants in trn, tst, all, unk datasets: {ntrn}({frac_trn:.2f}%), {ntst}({frac_tst:.2f}%), {nvar_all}, {nvar_unk}")

      trn_ldr = torch.utils.data.DataLoader(dsMap['trn'], batch_size=bat_size, shuffle=True)

      for epoch in range(0, max_epochs):
        epoch_loss = 0.0
        for (batch_idx, batch) in enumerate(trn_ldr):
          fea = batch['feature']
          tgt = batch['target']
          prd = net(fea)
          loss_val = loss_obj(prd, tgt)
          epoch_loss += loss_val.item()
          optimizer.zero_grad()
          loss_val.backward()
          optimizer.step()
        if (epoch+1) % ep_log_interval == 0:
          epoch_pct = 100*(epoch+1)/max_epochs
          time_1 = time.time()
          print('epoch%% = %3d%%   loss = %0.4f   time_elapsed = %.2fs' % (epoch_pct, epoch_loss, time_1-time_0))
          time_0 = time_1


      # 3. Save model
      path = f'./Models/model_{str(ifold)}.pth'
      print(f'\nSaving trained model state_dict for ifold = {ifold+1}...')
      torch.save(net.state_dict(), path)

      # 4. Evaluate model
      net = net.eval()

      for itp in range(len(dstps)):
        dstp = dstps[itp]
        ds = dsMap[dstp]

        if dstp in ['trn', 'tst', 'unk']:
          rst, acc, mcc, mse, r2, r, rst_multi, acc_multi, mcc_multi, mse_multi, r2_multi, r_multi = accuracy(net, ds)
          print('[%s]: %%VC_multi = %.2f%%, MCC_multi = %.2f, MSE_multi = %.2f, r2_multi = %.2f, r_multi = %.2f' % \
        (dstp, acc_multi*100.0, mcc_multi, mse_multi, r2_multi, r_multi))
        fn_rst = 'cout.rslt_' + dstp + '_' + str(ifold) + '.csv'
        fn_rst_multi = 'cout.rslt_multi_' + dstp + '_' + str(ifold) + '.csv'
        np.savetxt(fn_rst, rst, delimiter=',')
        np.savetxt(fn_rst_multi, rst_multi, delimiter=',')
        if dstp in ['tst']:
          rst_bat[ifold].append(rst[:])
          rst_multi_bat[ifold].append(rst_multi[:])
        if dstp in ['unk']:
          rst_multi_unk_bat[ifold].append(rst_multi[:])
        acc_bat[ifold].append(acc*100.0)
        acc_multi_bat[ifold].append(acc_multi*100.0)
        mcc_bat[ifold].append(mcc)
        mcc_multi_bat[ifold].append(mcc_multi)
        mse_bat[ifold].append(mse)
        mse_multi_bat[ifold].append(mse_multi)
        r2_bat[ifold].append(r2)
        r_bat[ifold].append(r)
        r2_multi_bat[ifold].append(r2_multi)
        r_multi_bat[ifold].append(r_multi)

    # Average over kfold results.
    acc_avg = np.mean(acc_bat, axis=0)
    acc_std = np.std(acc_bat, axis=0)
    acc_multi_avg = np.mean(acc_multi_bat, axis=0)
    acc_multi_std = np.std(acc_multi_bat, axis=0)
    mcc_avg = np.mean(mcc_bat, axis=0)
    mcc_std = np.std(mcc_bat, axis=0)
    mcc_multi_avg = np.mean(mcc_multi_bat, axis=0)
    mcc_multi_std = np.std(mcc_multi_bat, axis=0)
    mse_avg = np.mean(mse_bat, axis=0)
    mse_std = np.std(mse_bat, axis=0)
    mse_multi_avg = np.mean(mse_multi_bat, axis=0)
    mse_multi_std = np.std(mse_multi_bat, axis=0)
    r2_avg  = np.mean(r2_bat, axis=0)
    r2_std  = np.std(r2_bat, axis=0)
    r2_multi_avg = np.mean(r2_multi_bat, axis=0)
    r2_multi_std = np.std(r2_multi_bat, axis=0)
    r_avg  = np.mean(r_bat, axis=0)
    r_std  = np.std(r_bat, axis=0)
    r_multi_avg = np.mean(r_multi_bat, axis=0)
    r_multi_std = np.std(r_multi_bat, axis=0)

    print('\nResults average over k-fold:\n')
    fn_kfold_bat = 'cout.kfold_bat'
    for itp in range(ndstp):
      dstp = dstps[itp]
      if dstp in ['all']: continue
      #print('[%s]: %%VC = %.2f(%.2f)%%, MCC = %.2f(%.2f), MSE = %.2f(%.2f), r2 = %.2f(%.2f), r = %.2f(%.2f)' % \
      #  (dstp, acc_avg[itp], acc_std[itp], mcc_avg[itp], mcc_std[itp], mse_avg[itp], mse_std[itp], r2_avg[itp], r2_std[itp], r_avg[itp], r_std[itp]))
      print('[%s]: %%VC_multi = %.2f(%.2f)%%, MCC_multi = %.2f(%.2f), MSE_multi = %.2f(%.2f), r2_multi = %.2f(%.2f), r_multi = %.2f(%.2f)' % \
        (dstp, acc_multi_avg[itp], acc_multi_std[itp], mcc_multi_avg[itp], mcc_multi_std[itp], mse_multi_avg[itp], mse_multi_std[itp], r2_multi_avg[itp], r2_multi_std[itp], r_multi_avg[itp], r_multi_std[itp]))
      if dstp in ['tst']:
        with open(fn_kfold_bat, 'a') as f:
          f.write('%.2f %.2f %.2f %.2f %.2f\n' % (acc_multi_avg[itp], mcc_multi_avg[itp], mse_multi_avg[itp], r2_multi_avg[itp], r_multi_avg[itp]))

    # Average using all kfold data for test set
    rst_multi_kfold = []
    for x in rst_multi_bat:
      rst_multi_kfold.extend(np.vstack(x))
    rst = np.array(rst_multi_kfold)

    ndat = len(rst)
    tgts = rst[:,0]
    prds = rst[:,1]
    tgts_mcc = []
    prds_mcc = []
    for i in range(ndat):
      tgt = rst[i][0]
      prd = rst[i][1]
      if tgt < thld:
        tgts_mcc.append(0)
      else:
        tgts_mcc.append(1)

      if prd < thld:
        prds_mcc.append(0)
      else:
        prds_mcc.append(1)

    tgts_mcc = np.array(tgts_mcc)
    prds_mcc = np.array(prds_mcc)

    acc = np.count_nonzero(tgts_mcc == prds_mcc) * 1.0/ndat
    mcc = matthews_corrcoef(tgts_mcc, prds_mcc) # Matthews correlation coefficient
    mse = ((tgts - prds)**2).mean()
    r2  = r2_score(tgts, prds)
    r   = np.corrcoef(tgts, prds)[0,1]

    print(f'\nFinal results (recalculate using multi results) with {ndat} data points:\n')
    print('[%s]: %%VC = %.2f%%, MCC = %.2f, MSE = %.2f, r2 = %.2f r = %.2f' % \
      ('tst', acc*100.0, mcc, mse, r2, r))

    rst_final = [[str(x[0]),str(x[1]),str(int(x[2])),varMapDct[int(x[2])]] for x in rst]
    fn_rst = 'cout.rslt_multi_tst_recalc.csv'
    np.savetxt(fn_rst, rst_final, fmt='%s', delimiter=',')

    # Average using all kfold data for unk set
    rst_multi_unk_bat = np.array(rst_multi_unk_bat)
    rst_unk = np.mean(rst_multi_unk_bat, axis=0)[0]

    rst_unk_final = [[str(x[0]),str(x[1]),str(int(x[2])),varMapDct[int(x[2])],str(var_all.count(varMapDct[int(x[2])]))] for x in rst_unk]
    fn_rst_unk = 'cout.rslt_multi_unk_recalc.csv'
    np.savetxt(fn_rst_unk, rst_unk_final, fmt='%s', delimiter=',')

    # To be used in hyperparameter optimization
    opt_score = [ \
      np.mean([acc_multi_avg[0] - acc_multi_avg[3]]), \
      np.mean([mcc_multi_avg[0] - mcc_multi_avg[3]]), \
      np.mean([mse_multi_avg[3] , mse_multi_avg[0]]), \
      np.mean([ r2_multi_avg[0] -  r2_multi_avg[3]]), \
      np.mean([  r_multi_avg[0] -   r_multi_avg[3]]), \
    ]
    fn_opt_score = 'cout.opt_score'
    np.savetxt(fn_opt_score, opt_score, fmt='%s', delimiter=',')

  elif mode in ['2','5']:

    world_size = torch.cuda.device_count()
    print(f'world_size = {world_size}')
    print(f'# of variants: {nvar_unk}')
    var_unk_mp = np.array_split(var_unk, world_size)
    # Parallel prediction
    run_mp(pred_mp, world_size, var_unk_mp)
    rst_kfold = {}
    for ifold in range(kfold):
      for irank in range(world_size):
        fn_tmp = f'Results/pred_ifold{ifold}_rank{irank}.csv'
        rst_tmp = np.loadtxt(fn_tmp, delimiter=",", dtype=np.str, ndmin=2)
        for r in rst_tmp:
          var_tmp = varMapDct[int(r[0])]
          kdr_tmp = r[2]
          if var_tmp in rst_kfold:
            rst_kfold[var_tmp].append(kdr_tmp)
          else:
            rst_kfold[var_tmp] = [kdr_tmp]
      
    # Calculate avg and std
    if nvar_unk <= 500:
      print()
      print(f'Results for unknown variants from {kfold} models [first 5 are shown]:')
      print('%30s: %6s %6s %6s' % ('var', 'tgt', 'avg', 'std'))
    rst_final = []
    ncorrect, ntotal = 0, 0
    for k, v in rst_kfold.items():
      v = [float(x) for x in rst_kfold[k]]
      s, chain, muts = interpret_variable(k)
      k_full = s + '-' + chain + '_'
      for m in muts:
        if 'wt' in m.lower():
          aa0 = ''
        else:
          resid = int(re.findall(r'\d+', m)[0])
          if chain == 'ace2':
            idx = resid - ht[s]['A'][0]
          else:
            idx = resid - ht[s]['B'][0] + len_A[s]
          aa0 = seq_wt[s][idx] # AA before substitution
        k_full += aa0 + m
      mean = np.mean(v)
      std = np.std(v)
      try:
        k = k.replace('humancov2_', '', 1)
        kdratio = ref[k]
        if np.round(kdratio,2) == 1.0:
          symbol = '-'
        elif (mean > thld) == (kdratio > thld):
          symbol = 'O'
          ncorrect += 1
          ntotal += 1
        else:
          symbol = 'X'
          ntotal += 1
        if nvar_unk <= 500:
          print('%30s: %6.2f %6.2f %6.2f %3s %s' % (k, kdratio, mean, std, symbol, list(map(lambda n: '%.2f' % n, v[:5]))))
      except:
        symbol = '-'
        if k.lower() in ['wt']:
          kdratio = '1.00'
        else:
          kdratio = '-'
        if nvar_unk <= 500:
          print('%30s: %6s %6.2f %6.2f %3s %s' % (k, kdratio, mean, std, symbol, list(map(lambda n: '%.2f' % n, v[:5]))))
      rst_final.append([str(kdratio), str(mean), str(std), k, k, k_full])
    try:
      print(f'\n%VC = {ncorrect/ntotal*100:.2f} [{ncorrect}/{ntotal}]')
    except:
      pass
    fn_rst = 'cout.rslt_multi_unk_recalc.csv'
    np.savetxt(fn_rst, rst_final, fmt='%s', delimiter=',')


if __name__ == '__main__':

  time_start = time.time()
  main()
  time_end = time.time()
  print(f'Time elapsed: {time_end - time_start:.2f}s')
