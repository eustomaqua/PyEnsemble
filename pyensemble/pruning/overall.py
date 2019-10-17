# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import numpy as np


from pyensemble.pruning import RANKING_BASED
from pyensemble.pruning import OPTIMIZATION_BASED
from pyensemble.pruning import COMPOSABLE_CORE_SETS
from pyensemble.pruning import AVAILABLE_NAME_PRUNE

from pyensemble.pruning.ranking_based import Early_Stopping as ES
from pyensemble.pruning.ranking_based import KL_divergence_Pruning as KL
from pyensemble.pruning.ranking_based import KL_divergence_Pruning_modify as KLplus
from pyensemble.pruning.ranking_based import Kappa_Pruning as KP
from pyensemble.pruning.ranking_based import Orientation_Ordering_Pruning as OO
from pyensemble.pruning.ranking_based import Reduce_Error_Pruning as RE

from pyensemble.pruning.composable import GMM_Algorithm as GMM
from pyensemble.pruning.composable import Local_Search_Alg as LCS

from pyensemble.pruning.optimization_based import DREP as DREP
from pyensemble.pruning.ranking_based import OEP_inPEP as OEP
from pyensemble.pruning.optimization_based import SEP_inPEP as SEP
from pyensemble.pruning.optimization_based import PEP_inPEP as PEP
from pyensemble.pruning.optimization_based import PEP_modify as PEPplus



#==================================
# Overall interface
#==================================


#----------------------------------
# Overall interface
#----------------------------------



def existing_contrastive_pruning_method(name_pru, yt, y, nb_cls, nb_pru, rho=None, epsilon=1e-3):
    #@ params:  name_prune
    # self._name_pru_set = ['ES','KL','KL+','KP','OO','RE', 'GMM','LCS', 'DREP','SEP','OEP','PEP','PEP+']
    rho = nb_pru / nb_cls if not rho else rho
    # epsilon = 1e-6
    # print("rho     = {}\nepsilon = {}".format(rho, epsilon))
    #
    if name_pru == 'ES':
        yo, P       = ES.Early_Stopping(              yt,    nb_cls, nb_pru)
    elif name_pru == 'KL':
        yo, P       = KL.KL_divergence_Pruning(       yt,    nb_cls, nb_pru)
    elif name_pru == 'KL+':
        yo, P       = KLplus.KL_divergence_Pruning_modify(yt,    nb_cls, nb_pru)
    elif name_pru == 'KP':
        yo, P       = KP.Kappa_Pruning(               yt, y, nb_cls, nb_pru)
    elif name_pru == 'OO':
        yo, P, flag = OO.Orientation_Ordering_Pruning(yt, y)
    elif name_pru == 'RE':
        yo, P       = RE.Reduce_Error_Pruning(        yt, y, nb_cls, nb_pru)
    elif name_pru == 'GMM':  # 'GMM_Algorithm'
        yo, P       = GMM.GMM_Algorithm(               yt, y, nb_cls, nb_pru)
    elif name_pru == 'LCS':  # 'Local_Search':
        yo, P       = LCS.Local_Search(                yt, y, nb_cls, nb_pru, epsilon)
    elif name_pru == 'DREP':
        yo, P       = DREP.DREP_Pruning(                yt, y, nb_cls,         rho)
    elif name_pru == 'SEP':
        yo, P       = SEP.PEP_SEP(                     yt, y, nb_cls,         rho)
    elif name_pru == 'OEP':
        yo, P       = OEP.PEP_OEP(                     yt, y, nb_cls)
    elif name_pru == 'PEP':
        yo, P       = PEP.PEP_PEP(                     yt, y, nb_cls,         rho)
    elif name_pru == 'PEP+':
        yo, P       = PEPplus.PEP_PEP_modify(              yt, y, nb_cls,         rho)
    else:
        raise UserWarning("LookupError! Check the `name_prune`.")
    #
    if name_pru != 'OO':
        flag = None
    P = np.where(np.array(P) == True)[0].tolist()
    return deepcopy(yo), deepcopy(P), flag


