import os
import sys
from itertools import product

import joblib
from tqdm import tqdm

sys.path.append(os.path.abspath('..'))

from ReproducibilityContext import *
from ReproducibilityStrategy import MeanDistanceStrategy, VectorDistanceStrategy, CorrelationStrategy, SpecificityStrategy
from FilterStrategy import *

threshs = list(range(2, 21))  # 19 Options
filters = {'Absolute Threshold': (AbsoluteFilterStrategy, np.arange(0.05, 0.81, 0.05)),
           'Top K hits': (TopKFromDupFilterStrategy, range(50, 1001, 50))}  # 2 Options
reps = {
    'Mean Distance': MeanDistanceStrategy,
    'Euc-Distance of Channels` Vector': VectorDistanceStrategy,
    # 'Pearson Correlation': lambda: CorrelationStrategy('pearson', corr_sign=-1),
    'Specificity': SpecificityStrategy,
} # 3 Options


if __name__ == '__main__':
    inp = int(sys.argv[1])
    params = list(product(threshs, filters.keys(), reps.keys()))
    t, filter_key, rep_key = params[inp]
    filter_cls, filter_rng = filters[filter_key]
    rep_cls = reps[rep_key]

    res_pth = '/storage/users/g-and-n/tabular_models_results/30000/results/z_scores/frac_score'
    out_pth = f'/storage/users/g-and-n/tabular_models_results/30000/results/z_scores/frac_score/reproduce_intersect'
    out_pth = os.path.join(out_pth, str(t), filter_key, rep_key)
    os.makedirs(out_pth, exist_ok=True)
    plates = os.listdir(os.path.join(res_pth, str(t)))
    plates.sort()

    zscores = {p: pd.read_csv(os.path.join(res_pth, str(t), p), index_col=[0,1,2]) for p in plates}
    joined = pd.concat(zscores.values())

    rep_cont = ReproducibilityContext(None, None)

    methods = ['map', 'raw', 'raw1to1']
    for method in methods:
        cur_out_pth = os.path.join(out_pth, method)
        os.makedirs(cur_out_pth, exist_ok=True)
        cur = joined.filter(regex=f'_{method}$')

        metrics = []
        for filter_arg in tqdm(filter_rng):
            # filter_strg = DerivedFilterDecorator(filter_cls(filter_arg), joined.filter(regex=f'_map$'), 'ALL_map')
            # filter_strg = UnionFilterDecorator(filter_cls(filter_arg),
            #                                    [joined.filter(regex=f'_{m}$') for m in methods],
            #                                    [f'ALL_{m}' for m in methods])

            filter_strg = IntersectFilterDecorator(filter_cls(filter_arg),
                                               [joined.filter(regex=f'_{m}$') for m in methods],
                                               [f'ALL_{m}' for m in methods])
            rep_strg = rep_cls()

            rep_cont.filter_strategy = filter_strg
            rep_cont.reproducibility_strategy = rep_strg

            cur_res = rep_cont.compare_triplets(cur, f'ALL_{method}')
            metric = cur_res['Metric'].sum().sum() / cur_res['Triplet Count'].sum().sum()
            joblib.dump(cur_res, os.path.join(cur_out_pth, str(filter_arg)+'.sav'))
            del cur_res
            metrics.append(metric)

        joblib.dump(metrics, os.path.join(cur_out_pth, 'summary.sav'))
