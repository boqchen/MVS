
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier


def get_manual_aggregation(rf, x, types):
    '''
    This function is needed because the sklearn RF for some reason returned all-zero prediction results in some cases.
    This function makes sure this doesn't happen.
    types: last type must be 'other'
    '''
    assert 'other' in types, 'One of the cell types must be "other" to capture unassigned cells'
    idx_other = [i for i,x in enumerate(types) if x == "other"]
    
    raw_result = np.zeros((x.shape[0], len(types)))
    for i in range(len(rf.estimators_)):
        raw_result += rf.estimators_[i].predict(x)
        
    # do max-voting:
    aggr_result = np.zeros(raw_result.shape)    
    
    for i in range(raw_result.shape[0]):
        is_maximum = (raw_result[i, :] == raw_result[i, :].max())
        num_of_max_entries = is_maximum.sum()
        
        if num_of_max_entries == 1:
            aggr_result[i, np.argmax(raw_result[i, :])] = 1
        else:
            # if no clear candidate found, then set to "other"
            aggr_result[i, idx_other] = 1
#             candidates = np.nonzero(is_maximum)[0].tolist()
#             aggr_result[i, random.choice(candidates)] = 1
            
    return aggr_result
