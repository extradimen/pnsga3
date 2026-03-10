import warnings

import numpy as np
from numpy.linalg import LinAlgError

import os
import time

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.survival import Survival
from pymoo.core.population import Population
try:
    from pymoo.docs import parse_doc_string
except ImportError:
    def parse_doc_string(*args, **kwargs):
        pass
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.functions import load_function
from pymoo.util.misc import intersect, has_feasible
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util import default_random_state


# =========================================================================================================
# Implementation
# =========================================================================================================

@default_random_state
def comp_by_cv_then_random(pop, P, random_state=None, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible just set random
        else:
            S[i] = random_state.choice([a, b])

    return S[:, None].astype(int)


class NSGA3(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 pop_size=None,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                 crossover=SBX(eta=30, prob=1.0),
                 mutation=PM(eta=20),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 output=MultiObjectiveOutput(),
                 **kwargs):
        """

        Parameters
        ----------

        ref_dirs : {ref_dirs}
        pop_size : int (default = None)
            By default the population size is set to None which means that it will be equal to the number of reference
            line. However, if desired this can be overwritten by providing a positive number.
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """

        self.ref_dirs = ref_dirs

        # in case of R-NSGA-3 they will be None - otherwise this will be executed
        if self.ref_dirs is not None:

            if pop_size is None:
                pop_size = len(self.ref_dirs)

            if pop_size < len(self.ref_dirs):
                print(
                    f"WARNING: pop_size={pop_size} is less than the number of reference directions ref_dirs={len(self.ref_dirs)}.\n"
                    "This might cause unwanted behavior of the algorithm. \n"
                    "Please make sure pop_size is equal or larger than the number of reference directions. ")

        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            survival = ReferenceDirectionSurvival(ref_dirs)

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         output=output,
                         advance_after_initial_infill=True,
                         **kwargs)

    def _setup(self, problem, **kwargs):

        if self.ref_dirs is not None:
            if self.ref_dirs.shape[1] != problem.n_obj:
                raise Exception(
                    "Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                    (self.ref_dirs.shape[1], problem.n_obj))

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            if len(self.survival.opt):
                self.opt = self.survival.opt


class ParallelNSGA3(NSGA3):

    def __init__(self,
                 ref_dirs,
                 pop_size=None,
                 n_islands=4,
                 migration_interval=10,
                 migration_rate=0.1,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                 crossover=SBX(eta=30, prob=1.0),
                 mutation=PM(eta=20),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 output=MultiObjectiveOutput(),
                 **kwargs):
        """

        Parameters
        ----------

        ref_dirs : {ref_dirs}
        pop_size : int (default = None)
            Total population size across all islands. If None, it is set to
            the number of reference directions.
        n_islands : int
            Number of parallel sub-populations (islands).
        migration_interval : int
            Perform migration every `migration_interval` generations. If set to
            None or a non-positive value, migration is disabled.
        migration_rate : float
            Fraction of each island population to migrate during a migration
            step. For example, 0.1 migrates 10%% of each island.

        Other parameters follow the standard NSGA3 interface.

        """

        self.n_islands = max(1, int(n_islands))
        self.migration_interval = migration_interval if migration_interval is not None else -1
        self.migration_rate = float(migration_rate)
        self.hall_of_fame = Population()

        # determine total population size (same semantics as NSGA3)
        if pop_size is None and ref_dirs is not None:
            pop_size = len(ref_dirs)

        if pop_size is None or pop_size <= 0:
            raise ValueError("pop_size must be a positive integer.")

        self.total_pop_size = int(pop_size)

        # population size per island (last island may have +1 if not divisible)
        base = self.total_pop_size // self.n_islands
        rem = self.total_pop_size % self.n_islands
        self._island_sizes = [base + (1 if i < rem else 0) for i in range(self.n_islands)]
        self.max_island_size = max(self._island_sizes)

        if n_offsprings is None:
            n_offsprings = self.total_pop_size

        super().__init__(ref_dirs=ref_dirs,
                         pop_size=self.total_pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         output=output,
                         **kwargs)

    def _initialize_infill(self):
        pop = super()._initialize_infill()

        # assign island ids to each individual
        island_ids = []
        for i, size in enumerate(self._island_sizes):
            island_ids.extend([i] * size)
        island_ids = np.array(island_ids, dtype=int)

        if len(island_ids) != len(pop):
            # in case initialization produced a slightly different number (due to duplicate elimination),
            # fall back to truncation or padding with last island id.
            if len(island_ids) > len(pop):
                island_ids = island_ids[:len(pop)]
            else:
                pad = np.full(len(pop) - len(island_ids), self.n_islands - 1, dtype=int)
                island_ids = np.concatenate([island_ids, pad])

        pop.set("island", island_ids)
        return pop

    def _infill(self):
        # generate offspring independently for each island
        off_all = []

        for island in range(self.n_islands):
            mask = self.pop.get("island") == island
            sub = self.pop[mask]
            if len(sub) == 0:
                continue

            n_off = len(sub)
            off = self.mating.do(self.problem, sub, n_off, algorithm=self, random_state=self.random_state)

            if len(off) == 0:
                continue

            off.set("island", np.full(len(off), island, dtype=int))
            off_all.append(off)

        if not off_all:
            self.termination.force_termination = True
            return None

        # if only a single island has offspring just return it directly
        if len(off_all) == 1:
            return off_all[0]

        return Population.merge(*off_all)

    def _advance(self, infills=None, **kwargs):
        _timing = os.environ.get("PYMOO_TIMING", "").strip() == "1"

        t_merge_init = 0.0
        t_survival_loop = 0.0
        t_merge_final = 0.0
        t_migrate = 0.0
        t_set_opt = 0.0

        t0 = time.perf_counter() if _timing else 0
        pop = self.pop
        if infills is not None:
            pop = Population.merge(self.pop, infills)
        if _timing:
            t_merge_init = time.perf_counter() - t0

        t1 = time.perf_counter() if _timing else 0
        new_islands = []
        for island in range(self.n_islands):
            mask = pop.get("island") == island
            sub = pop[mask]
            if len(sub) == 0:
                continue

            target_size = self._island_sizes[island]
            surv = self.survival.do(self.problem, sub, n_survive=target_size,
                                    algorithm=self, random_state=self.random_state, **kwargs)
            surv.set("island", np.full(len(surv), island, dtype=int))
            new_islands.append(surv)
        if _timing:
            t_survival_loop = time.perf_counter() - t1

        if not new_islands:
            self.termination.force_termination = True
            return False

        t2 = time.perf_counter() if _timing else 0
        if len(new_islands) == 1:
            self.pop = new_islands[0]
        else:
            self.pop = Population.merge(*new_islands)
        if _timing:
            t_merge_final = time.perf_counter() - t2

        t3 = time.perf_counter() if _timing else 0
        if self.migration_interval is not None and self.migration_interval > 0:
            if self.n_iter is not None and self.n_iter % self.migration_interval == 0:
                self._migrate()
        if _timing:
            t_migrate = time.perf_counter() - t3

        t4 = time.perf_counter() if _timing else 0
        self._set_optimum()
        if _timing:
            t_set_opt = time.perf_counter() - t4
            self.data["advance_timing"] = dict(
                merge_init=t_merge_init,
                survival_loop=t_survival_loop,
                merge_final=t_merge_final,
                migrate=t_migrate,
                set_optimum=t_set_opt,
            )

        return True

    def _migrate(self):
        if self.migration_rate <= 0.0 or self.n_islands <= 1:
            return

        hof = []
        replace_indices = []

        # collect top and bottom individuals per island
        for island in range(self.n_islands):
            mask = self.pop.get("island") == island
            sub = self.pop[mask]
            if len(sub) == 0:
                continue

            island_size = len(sub)
            n_mig = max(1, int(np.floor(self.migration_rate * island_size)))

            # use rank (and then distance) as in NSGA-III to sort
            F = sub.get("F")
            fronts, rank = NonDominatedSorting().do(F, return_rank=True)
            # flatten fronts to get global ordering within island
            order = np.concatenate(fronts)

            top_idx_local = order[:n_mig]
            bottom_idx_local = order[-n_mig:]

            hof.append(sub[top_idx_local])

            # map local indices back to global indices in self.pop
            island_global_indices = np.where(mask)[0]
            replace_indices.extend(island_global_indices[bottom_idx_local].tolist())

        if not hof or not replace_indices:
            return

        hof_pop = Population.merge(*hof)

        # sample from hall of fame to replace worst individuals globally
        k = min(len(replace_indices), len(hof_pop))
        selected = self.random_state.choice(len(hof_pop), size=k, replace=False)

        for dst, src in zip(replace_indices[:k], selected):
            # copy individual (deep copy of object) to keep attributes consistent
            self.pop[dst] = hof_pop[src]


# =========================================================================================================
# Survival
# =========================================================================================================


class ReferenceDirectionSurvival(Survival):

    def __init__(self, ref_dirs):
        super().__init__(filter_infeasible=True)
        self.ref_dirs = ref_dirs
        self.opt = None
        self.norm = HyperplaneNormalization(ref_dirs.shape[1])

    def _do(self, problem, pop, n_survive, D=None, random_state=None, **kwargs):

        # optional: survival step timing when PYMOO_TIMING=1
        _timing = os.environ.get("PYMOO_TIMING", "").strip() == "1"
        _algo = kwargs.get("algorithm")

        # attributes to be set after the survival
        F = pop.get("F")

        # calculate the fronts of the population
        t0 = time.perf_counter() if _timing else 0
        fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
        t_nds = (time.perf_counter() - t0) if _timing else 0
        non_dominated, last_front = fronts[0], fronts[-1]

        # update the hyperplane based boundary estimation
        hyp_norm = self.norm
        t1 = time.perf_counter() if _timing else 0
        hyp_norm.update(F, nds=non_dominated)
        t_norm = (time.perf_counter() - t1) if _timing else 0
        ideal, nadir = hyp_norm.ideal_point, hyp_norm.nadir_point

        #  consider only the population until we come to the splitting front
        I = np.concatenate(fronts)
        pop, rank, F = pop[I], rank[I], F[I]

        # update the front indices for the current population
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1
        last_front = fronts[-1]

        # associate individuals to niches
        t2 = time.perf_counter() if _timing else 0
        niche_of_individuals, dist_to_niche, dist_matrix = \
            associate_to_niches(F, self.ref_dirs, ideal, nadir)
        t_assoc = (time.perf_counter() - t2) if _timing else 0

        # attributes of a population
        pop.set('rank', rank,
                'niche', niche_of_individuals,
                'dist_to_niche', dist_to_niche)

        # set the optimum, first front and closest to all reference directions
        closest = np.unique(dist_matrix[:, np.unique(niche_of_individuals)].argmin(axis=0))
        self.opt = pop[intersect(fronts[0], closest)]
        if len(self.opt) == 0:
            self.opt = pop[fronts[0]]

        # if we need to select individuals to survive
        if len(pop) > n_survive:

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = n_survive
                until_last_front = np.array([], dtype=int)
                niche_count = np.zeros(len(self.ref_dirs), dtype=int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[until_last_front])
                n_remaining = n_survive - len(until_last_front)

            t3 = time.perf_counter() if _timing else 0
            S = niching(pop[last_front], n_remaining, niche_count, niche_of_individuals[last_front],
                        dist_to_niche[last_front], random_state=random_state)
            t_niching = (time.perf_counter() - t3) if _timing else 0

            survivors = np.concatenate((until_last_front, last_front[S].tolist()))
            pop = pop[survivors]
        else:
            t_niching = 0.0 if _timing else 0

        if _timing and _algo is not None:
            st = _algo.data.get("survival_timing_last")
            if st is not None:
                st["nds"] += t_nds
                st["norm"] += t_norm
                st["associate"] += t_assoc
                st["niching"] += t_niching
            else:
                _algo.data["survival_timing_last"] = dict(nds=t_nds, norm=t_norm, associate=t_assoc, niching=t_niching)

        return pop


@default_random_state
def niching(pop, n_remaining, niche_count, niche_of_individuals, dist_to_niche, random_state=None):
    survivors = []

    # boolean array of elements that are considered for each iteration
    mask = np.full(len(pop), True)

    while len(survivors) < n_remaining:

        # number of individuals to select in this iteration
        n_select = n_remaining - len(survivors)

        # all niches where new individuals can be assigned to and the corresponding niche count
        next_niches_list = np.unique(niche_of_individuals[mask])
        next_niche_count = niche_count[next_niches_list]

        # the minimum niche count
        min_niche_count = next_niche_count.min()

        # all niches with the minimum niche count (truncate randomly if there are more niches than remaining individuals)
        next_niches = next_niches_list[np.where(next_niche_count == min_niche_count)[0]]
        next_niches = next_niches[random_state.permutation(len(next_niches))[:n_select]]

        for next_niche in next_niches:

            # indices of individuals that are considered and assign to next_niche
            next_ind = np.where(np.logical_and(niche_of_individuals == next_niche, mask))[0]

            # shuffle to break random tie (equal perp. dist) or select randomly
            random_state.shuffle(next_ind)

            if niche_count[next_niche] == 0:
                next_ind = next_ind[np.argmin(dist_to_niche[next_ind])]
            else:
                # already randomized through shuffling
                next_ind = next_ind[0]

            # add the selected individual to the survivors
            mask[next_ind] = False
            survivors.append(int(next_ind))

            # increase the corresponding niche count
            niche_count[next_niche] += 1

    return survivors


def associate_to_niches(F, niches, ideal_point, nadir_point, utopian_epsilon=0.0):
    utopian_point = ideal_point - utopian_epsilon

    denom = nadir_point - utopian_point
    denom[denom == 0] = 1e-12

    # normalize by ideal point and intercepts
    N = (F - utopian_point) / denom
    dist_matrix = load_function("calc_perpendicular_distance")(N, niches)

    niche_of_individuals = np.argmin(dist_matrix, axis=1)
    dist_to_niche = dist_matrix[np.arange(F.shape[0]), niche_of_individuals]

    return niche_of_individuals, dist_to_niche, dist_matrix


def calc_niche_count(n_niches, niche_of_individuals):
    niche_count = np.zeros(n_niches, dtype=int)
    index, count = np.unique(niche_of_individuals, return_counts=True)
    niche_count[index] = count
    return niche_count


# =========================================================================================================
# Normalization
# =========================================================================================================


class HyperplaneNormalization:

    def __init__(self, n_dim) -> None:
        super().__init__()
        self.ideal_point = np.full(n_dim, np.inf)
        self.worst_point = np.full(n_dim, -np.inf)
        self.nadir_point = None
        self.extreme_points = None

    def update(self, F, nds=None):
        # find or usually update the new ideal point - from feasible solutions
        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

        # this decides whether only non-dominated points or all points are used to determine the extreme points
        if nds is None:
            nds = np.arange(len(F))

        # find the extreme points for normalization
        self.extreme_points = get_extreme_points_c(F[nds, :], self.ideal_point,
                                                   extreme_points=self.extreme_points)

        # find the intercepts for normalization and do backup if gaussian elimination fails
        worst_of_population = np.max(F, axis=0)
        worst_of_front = np.max(F[nds, :], axis=0)

        self.nadir_point = get_nadir_point(self.extreme_points, self.ideal_point, self.worst_point,
                                           worst_of_front, worst_of_population)


def get_extreme_points_c(F, ideal_point, extreme_points=None):
    # calculate the asf which is used for the extreme point decomposition
    weights = np.eye(F.shape[1])
    weights[weights == 0] = 1e6

    # add the old extreme points to never lose them for normalization
    _F = F
    if extreme_points is not None:
        _F = np.concatenate([extreme_points, _F], axis=0)

    # use __F because we substitute small values to be 0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F * weights[:, None, :], axis=2)

    I = np.argmin(F_asf, axis=1)
    extreme_points = _F[I, :]

    return extreme_points


def get_nadir_point(extreme_points, ideal_point, worst_point, worst_of_front, worst_of_population):
    try:

        # find the intercepts using gaussian elimination
        M = extreme_points - ideal_point
        b = np.ones(extreme_points.shape[1])
        plane = np.linalg.solve(M, b)

        warnings.simplefilter("ignore")
        intercepts = 1 / plane

        nadir_point = ideal_point + intercepts

        # check if the hyperplane makes sense
        if not np.allclose(np.dot(M, plane), b) or np.any(intercepts <= 1e-6):
            raise LinAlgError()

        # if the nadir point should be larger than any value discovered so far set it to that value
        # NOTE: different to the proposed version in the paper
        b = nadir_point > worst_point
        nadir_point[b] = worst_point[b]

    except LinAlgError:

        # fall back to worst of front otherwise
        nadir_point = worst_of_front

    # if the range is too small set it to worst of population
    b = nadir_point - ideal_point <= 1e-6
    nadir_point[b] = worst_of_population[b]

    return nadir_point


parse_doc_string(NSGA3.__init__)
