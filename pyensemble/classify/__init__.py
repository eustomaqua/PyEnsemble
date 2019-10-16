# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# X_trn, X_tst:     list, [[nb_feat] nb_trn/tst]
# y_trn, y_tst:     list, [nb_trn/tst]
#
# Y \in {0, 1}
# y_insp:           list, [[nb_trn] nb_cls],    inspect
# y_pred:           list, [[nb_tst] nb_cls],    predict
# coefficient:              list, [nb_cls]
# weights (in resample):    list, [nb_y/X]
#

AVAILABLE_ABBR_CLS = ['DT', 'NB', 'SVM', 'LSVM', 'KNNu', 'KNNd', 'LM1', 'LM2']
