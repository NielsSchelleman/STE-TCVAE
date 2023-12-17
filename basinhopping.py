import pandas as pd
import numpy as np
import basinhopper_utils
from scipy.optimize import basinhopping


class BasinHopper:

    def __init__(self, target_var, phi, n_iter = 250, temperature = 0.05, repeats = 4, min_contribution = 0.025):
        self.target = target_var
        self.phi = phi

        self.n_iter = n_iter
        self.temperature = temperature
        self.repeats = repeats
        self.min_contribution = min_contribution

    def perform_step(self, x, cols):
        subgroups = pd.DataFrame(np.hstack([np.packbits(cols > x, axis=1), np.expand_dims(self.target, 1)])).groupby(0)
        size = subgroups.count(1)
        tp = subgroups.sum(1)
        acc = tp / size

        return - float(np.max(self.phi(size, acc), axis=0))

    def hop(self, cols):
        top_result = ([0], 1)
        for _ in range(self.repeats):
            result = basinhopping(self.perform_step, list(cols.mean()), niter=self.n_iter, T=self.temperature,
                                  minimizer_kwargs={'args': cols})
            if result.fun < top_result[1]:
                top_result = (np.array(result.x), result.fun)

        # tries to greedily remove columns below the contribution threshold
        dcols = len(cols.keys()) + 1
        while (len(cols.keys()) > 1) and (dcols != len(cols.keys())):
            dcols -= 1
            keys = cols.keys()
            for i in range(len(keys)):
                temp = list(keys).copy()
                del temp[i]
                rescopy = list(top_result[0]).copy()
                rescopy = rescopy[:i]+rescopy[i+1:]

                score = self.perform_step(rescopy, cols[temp])
                if score - self.min_contribution < top_result[1]:
                    top_result = (rescopy, score)
                    cols = cols[temp]
                    break
        return tuple(sorted(cols.keys())), top_result[0], -top_result[1]

    def predict(self, latent_space, colnames, nlargest=20):
        to_evaluate = basinhopper_utils.powerset(colnames)
        best_sets = {}
        for combination in to_evaluate:
            print('\r', f'checking: {combination}', end='')
            c, p, w = self.hop(pd.DataFrame(latent_space)[list(combination)])

            if c not in best_sets.keys():
                best_sets[c] = (p, w)
            elif w > best_sets[c][1]:
                best_sets[c] = (p, w)

        return dict(sorted(best_sets.items(), key=lambda x: x[1][1], reverse=True)[:nlargest])

