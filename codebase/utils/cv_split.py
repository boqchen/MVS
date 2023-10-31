import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from sklearn.model_selection import KFold, GroupKFold

from codebase.utils.constants import TRAIN_SAMPLES, VALID_SAMPLES, TEST_SAMPLES, MVS_SAMPLES
from codebase.utils.dataset_utils import get_patient_samples

# load all the MVS samples
# originally used samples (all well-aligned 80 samples)
# all_samples = ['MACOLUD', 'MADAJEJ', 'MADEGOD', 'MADUBIP', 'MADUFEM', 'MAFIBAF', 'MAHEFOG', 'MAHOBAM', 'MAJEFEV', 'MAJOFIJ', 'MAKYGIW', 'MANOFYB',
# 'MEBIGAL', 'MECADAC', 'MECUGUH', 'MECYGYR', 'MEDEFUC', 'MEDYCAR', 'MEFYFAK', 'MEGEGUJ', 'MEHEHUM', 'MEKAKYD', 'MEKOBAB', 'MELYPEB', 'MEMEMUH', 'MEMIGOG',
# 'MEVIXYV', 'MEZYWEG', 'MIBAFUK', 'MICEGOK', 'MIDEKOG', 'MIFOGIL', 'MIGEKUT', 'MIHIFIB', 'MIJYDYP', 'MIKOBID', 'MIPYNAP', 'MISYPUP', 'MOBICUN', 'MOCELOJ',
# 'MODALEG', 'MODIGOS', 'MODOHOX', 'MOGYHOJ', 'MOJYMOC', 'MOPYPUS', 'MOQAVIJ', 'MOVAZYQ', 'MUBIBIT', 'MUBOMEF', 'MUBYJOF', 'MUCADOP', 'MUDEFAW', 'MUDIFOB',
# 'MUDUKEF', 'MUFYDUM', 'MUGAHEK', 'MUGAKOL', 'MUHYBAF', 'MUKAGOX', 'MULELEZ', 'MULYMUP', 'MYBYHER', 'MYDACIM', 'MYGIFUD', 'MYHAJIS', 'MYJILAS', 'MYJUFAJ',
# 'MYKOKIG', 'MYKYPAZ', 'MYLURAZ', 'MYNELIC', 'MEFOCUR', 'MOBUBOT', 'MUMIFAD', 'MYBYFUW','MALYLEJ', 'MEHUFEF', 'MOBYLUD', 'MOMUSIG']


def get_json(samples):
    ''' Get a list of alignment results for a given sample list
    samples: list of sample IDs
    '''
    manual_aligns_path = '/cluster/work/grlab/projects/projects2021-imc_morphology/template_matching/manual_aligns/'
    align_results_paths = [os.path.join(manual_aligns_path, f.name) for f in os.scandir(manual_aligns_path) if f.is_file() and f.name.endswith(".json") and f.name.split('_')[0] in samples]
    align_results = []
    for sample in samples:
        for arp in [x for x in align_results_paths if x.split('/')[-1].split('_')[0]==sample]:
            with open(arp) as json_file:
                align_results.append(json.load(json_file))
                json_file.close()
    return align_results


# TODO make it more efficient - find json files once and then subset
# TODO: account for the fact that samples have different number of ROIs aligned OR align all the ROIs
def train_valid_test_split(all_samples=MVS_SAMPLES, random_seed=456, test_frac=0.05, valid_frac=0.05, n_folds=5, grouping_dict=None):
    ''' Get train, valid, split (outputs a dictionary)
    all_samples: list of all sample IDs to sample from
    random_seed: seed for sampling
    test_frac: fraction of test samples (test_frac + valid_frac cannot exceed 0.8)
    valid_frac: fraction of valid_samples (test_frac + valid_frac cannot exceed 0.8)
    n_folds: number of folds (splits) to output, independent of test_frac and valid_frac
    '''
    
    assert test_frac+valid_frac<0.8, print('test and valid fraction together exceed 80% of the data!')
    np.random.seed(random_seed)
    K = int(1/test_frac)
    
    #print('performing kfold split with')
    #print('test size', int(len(all_samples)*test_frac))
    #print('valid size', int(len(all_samples)*valid_frac))
    
    if grouping_dict is not None:
        kf = GroupKFold(n_splits=K)
        groups = [grouping_dict[x] for x in all_samples]
        splits = [x for x in kf.split(all_samples,[1 for i in range(len(all_samples))],groups)]
    else:
        kf = KFold(n_splits=K, random_state=random_seed, shuffle=True)
        splits = [x for x in kf.split(all_samples)]
    
    kfold_splits = dict()
    valid_all = []
    # select n_folds splits and add validation (from train samples)
    sel_splits = np.random.choice(range(len(splits)), n_folds, replace=False)
    for i,run in enumerate(sel_splits):
        kfold_splits['split'+str(i+1)] = dict()
        kfold_splits['split'+str(i+1)]['test'] = get_json([all_samples[x] for x in splits[run][1]])
        train = [all_samples[x] for x in splits[run][0]]
        train_to_sel = [x for x in train if x not in valid_all]
        valid_size = int(np.ceil(len(all_samples)*valid_frac))
        if len(train_to_sel)<valid_size:
            print('Assignment not possible, please change the seed and retry')
        valid = list(np.random.choice(train_to_sel, valid_size, replace=False))
        valid_all.extend(valid)
        kfold_splits['split'+str(i+1)]['valid'] = get_json(valid)
        kfold_splits['split'+str(i+1)]['train'] = get_json([x for x in train if x not in valid])
    
    return kfold_splits

    
# load sample - patient dictionary
sample_patient_df = get_patient_samples(indication='melanoma')
sample_patient_df = sample_patient_df.loc[sample_patient_df['sample'].isin(MVS_SAMPLES),:]
sample_patient_dict = dict(zip(sample_patient_df['sample'], sample_patient_df['patient']))


kfold_splits = train_valid_test_split(MVS_SAMPLES, random_seed=456, test_frac=0.05, valid_frac=0.05, n_folds=5, grouping_dict=sample_patient_dict)

# report split
kfold_splits['split0'] = dict()
kfold_splits['split0']['train'] = get_json(TRAIN_SAMPLES)
kfold_splits['split0']['valid'] = get_json(VALID_SAMPLES)
kfold_splits['split0']['test'] = get_json(TEST_SAMPLES)

# Uncomment to save the split
# with open("/cluster/work/grlab/projects/projects2021-multivstain/data/tupro/mvs-cv_split-dict-v1.json", 'w') as fout:
#     json_dumps_str = json.dumps(kfold_splits, indent=2)
#     print(json_dumps_str, file=fout)
    
kfold_splits_samples = dict()
for split in kfold_splits.keys():
    kfold_splits_samples[split] = dict()
    for split_set in kfold_splits[split].keys():
        split_set_samples = []
        for i in range(len(kfold_splits[split][split_set])):
            split_set_samples.append(kfold_splits[split][split_set][i]['sample'])
        kfold_splits_samples[split][split_set] = list(np.unique(split_set_samples))
        
# Uncomment to save the split (sample assignment)
# with open("/cluster/work/grlab/projects/projects2021-multivstain/data/tupro/mvs-cv_split-samples-v1.json", 'w') as fout:
#     json_dumps_str = json.dumps(kfold_splits_samples, indent=2)
#     print(json_dumps_str, file=fout)
        
# test_all = []
# for run in kfold_splits.keys():
#     print(run, len(kfold_splits[run]['test']))
#     for ar in kfold_splits[run]['test']:
#         sample = ar['sample']
#         test_all.append(sample)