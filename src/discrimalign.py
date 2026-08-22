"""
Main functions of DiscrimAlign for learning alignment parameters from
labeled pairs of biological sequences.
"""
from Bio.Align import PairwiseAligner, substitution_matrices
import numpy as np
from numpy import random as rd
from scipy.optimize import minimize
from copy import deepcopy
from math import ceil
from .optimization import get_initial_estimate, get_first_alignment
from .logit_link import logit_partial_scores, logit_logL, logit_subgradient


def _align_pair_chunk(pair_chunk, aligner):
    """Align a chunk of sequence pairs in one joblib task.

    Dispatching one pair per task creates substantial scheduling overhead when
    this function is called at every optimization iteration. Chunking keeps the
    returned alignment order stable while reducing per-iteration joblib overhead.
    """
    return [get_first_alignment(seqA, seqB, aligner) for seqA, seqB in pair_chunk]


def _pair_chunks(seqlistA, seqlistB, chunk_size):
    pair_count = len(seqlistA)
    for start in range(0, pair_count, chunk_size):
        stop = min(start + chunk_size, pair_count)
        yield list(zip(seqlistA[start:stop], seqlistB[start:stop]))


def _align_pairs(seqlistA, seqlistB, aligner, num_threads):
    pair_count = len(seqlistA)
    if num_threads == 1 or pair_count == 0:
        return [get_first_alignment(seqA, seqB, aligner) for seqA, seqB in zip(seqlistA, seqlistB)]

    from joblib import Parallel, delayed

    n_jobs = min(int(num_threads), pair_count)
    # Aim for a few chunks per worker so the work is balanced, without creating
    # one joblib task per sequence pair at every fitting iteration.
    chunk_size = max(1, ceil(pair_count / (n_jobs * 4)))
    parallel = Parallel(n_jobs=n_jobs, prefer="threads", return_as="list")
    chunked_alignments = parallel(
        delayed(_align_pair_chunk)(pair_chunk, aligner)
        for pair_chunk in _pair_chunks(seqlistA, seqlistB, chunk_size)
    )
    return [alignment for chunk in chunked_alignments for alignment in chunk]


def _matrix_alphabet(matrix):
    alphabet = getattr(matrix, 'alphabet', None)
    if alphabet is None:
        return None
    return ''.join(alphabet)


def _warm_start_alphabet(baseline_aligner=None, initial_parameters=None):
    if initial_parameters is not None:
        matrix = initial_parameters.get('substitution_matrix')
        alphabet = _matrix_alphabet(matrix)
        if alphabet:
            return alphabet
    if baseline_aligner is not None:
        try:
            matrix = getattr(baseline_aligner, 'substitution_matrix')
        except Exception:
            matrix = None
        alphabet = _matrix_alphabet(matrix)
        if alphabet:
            return alphabet
    return None


def discrimalign(seqlistA, seqlistB,
                 labels,
                 baseline_aligner=None,
                 aligner_mode='local',
                 gap_mode='affine',
                 substitution_mode='symmetric',
                 alphabet=None,
                 stochastic_factor=None,
                 stepfunction=None,
                 max_iter=1000, tol=1e-3,
                 num_threads=1,
                 subgradient_scale=1.0,
                 initial_parameters=None,
                 return_alignments=True,
                 verbose=False):
    # TODO: Implement tol and additional stepfunctions.
    assert aligner_mode in {'local', 'global'}
    assert gap_mode in {'affine', 'linear'}
    assert substitution_mode in {'general', 'symmetric', 'simple'}
    if alphabet is None:
        alphabet = _warm_start_alphabet(baseline_aligner, initial_parameters)
    if alphabet is None:
        charsetA = set(char for seq in seqlistA for char in seq)
        charsetB = set(char for seq in seqlistB for char in seq)
        alphabet = charsetA | charsetB
        alphabet = ''.join(sorted(alphabet))

    if verbose:
        print('Alphabet:')
        print(alphabet)

    # Stochastic trick function
    if stochastic_factor is None:
        def add_noise(shape, niter):
            if shape == 1:
                return 0.
            else:
                return np.zeros(shape)
            # Note: For consistency with numpy, returning a number should
            # happen when shape == None, shape 1 should return an array.
            # This would require modifying the code elsewhere
    else:
        def add_noise(shape, niter):
            if shape == 1:
                return rd.normal(scale=stochastic_factor/(niter+1))
            else:
                return rd.normal(size=shape, scale=stochastic_factor/(niter+1))

    # Initial alignments
    if baseline_aligner is not None:
        aligner = deepcopy(baseline_aligner)
    else:
        aligner = PairwiseAligner()
        aligner.mode = aligner_mode
        if gap_mode == 'affine':
            aligner.open_gap_score = -8
            aligner.extend_gap_score = -0.5
        elif gap_mode == 'linear':
            aligner.gap_score = -6
        if substitution_mode == 'simple':
            aligner.match_score = 5
            aligner.mismatch_score = -4
        else:
            aligner.substitution_matrix = substitution_matrices.Array(data=9*np.eye(len(alphabet))-4,
                                                                      alphabet=alphabet)

    alnlist = _align_pairs(seqlistA, seqlistB, aligner, num_threads)
    alignment_scores = [aln.score for aln in alnlist]

    # Initial logistic estimation, unless caller provides fitted parameters for a warm start.
    if initial_parameters is not None:
        updated_parameters = deepcopy(initial_parameters)
    else:
        updated_parameters = get_initial_estimate(alnlist, labels,
                                                  substitution_mode=substitution_mode,
                                                  gap_mode=gap_mode,
                                                  alphabet=alphabet)
    if verbose:
        print('Initial parameters:')
        print(updated_parameters)
    if gap_mode == 'affine':
        aligner.open_gap_score = updated_parameters['open_gap_score']
        aligner.extend_gap_score = updated_parameters['extend_gap_score']
    else:
        aligner.gap_score = updated_parameters['gap_score']

    if substitution_mode == 'simple':
        aligner.match_score = updated_parameters['match_score']
        aligner.mismatch_score = updated_parameters['mismatch_score']
    else:
        aligner.substitution_matrix = updated_parameters['substitution_matrix']

    # Subgradient refinement
    loglik_trajectory = []
    subgradient_l2_trajectory = []
    loglik_expectation = []
    loglik_sd = []
    theta_trajectory = []
    
    for iternb in range(max_iter):
        if verbose:
            print('Start of iteration', iternb)
        # Set new aligner parameters
        if gap_mode == 'affine':
            aligner.open_gap_score = updated_parameters['open_gap_score']
            aligner.extend_gap_score = updated_parameters['extend_gap_score']
        elif gap_mode == 'linear':
            aligner.gap_score = updated_parameters['gap_score']

        if substitution_mode == 'simple':
            aligner.match_score = updated_parameters['match_score']
            aligner.mismatch_score = updated_parameters['mismatch_score']
        else:
            aligner.substitution_matrix = updated_parameters['substitution_matrix']

        # Realign with the new parameters
        alnlist = _align_pairs(seqlistA, seqlistB, aligner, num_threads)
        alignment_scores = [aln.score for aln in alnlist]
        logit_scores = logit_partial_scores(alignment_scores,
                                            updated_parameters['alpha'])
        new_logL = logit_logL(logit_scores, labels)
        loglik_trajectory.append(new_logL)

        if substitution_mode == 'simple':
            theta_n = np.array([updated_parameters['match_score'], updated_parameters['mismatch_score'], updated_parameters['open_gap_score'], updated_parameters['extend_gap_score']])

        elif substitution_mode == 'general':
            theta_n = list(np.array(updated_parameters['substitution_matrix']).flatten())
            theta_n.append(updated_parameters['open_gap_score'])
            theta_n.append(updated_parameters['extend_gap_score'])

            theta_n = np.array(theta_n)

        ## might need a case for subst mode = symmetric

        theta_trajectory.append(theta_n)

        if verbose:
            print("Current alpha:", updated_parameters['alpha'])
            print('Current logL:', new_logL)
##        EL = 0
##        VL = 0
##        for ls in logit_scores:
##            if 1e-30 < ls < 1-1e-30:
##                EL += ls*np.log(ls) + (1-ls)*np.log(1-ls)
##                VL += ls*(1-ls)*(np.log(ls)**2 + np.log(1-ls)**2)
##        SDL = np.sqrt(VL)
##        loglik_expectation.append(EL)
##        loglik_sd.append(SDL)

        # Optimize the logistic intercept (alpha)
        def alpha_target(alpha):
            logit_scores = logit_partial_scores(alignment_scores, alpha)
            return -logit_logL(logit_scores, labels)

        def alpha_fprime(alpha):
            logit_scores = logit_partial_scores(alignment_scores, alpha)
            return -np.sum(labels - logit_scores)

        def alpha_fsec(alpha):
            logit_scores = logit_partial_scores(alignment_scores, alpha)
            return np.sum(logit_scores*(1 - logit_scores))

        new_alpha = minimize(alpha_target,
                             updated_parameters['alpha'],
                             jac=alpha_fprime,
                             # hess=alpha_fsec,
                             # method='Newton-CG'
                             )['x'][0]

        logit_scores = logit_partial_scores(alignment_scores, new_alpha)
        if verbose:
            new_logL = logit_logL(logit_scores, labels)
            print("Updated alpha:", new_alpha)
            print('Updated logL:', new_logL)

        updated_parameters['alpha'] = new_alpha

        # Make a subgradient step
        subgradient = logit_subgradient(alnlist, logit_scores,
                                        labels, new_alpha,
                                        alphabet)
        if subgradient_scale != 1.0:
            subgradient['Gap opens'] *= subgradient_scale
            subgradient['Gap extends'] *= subgradient_scale
            subgradient['Substitutions'] *= subgradient_scale

        stepsize = stepfunction(iternb)
        if verbose:
            print('New subgradient:')
            print(subgradient)
            print('Stepsize:', stepsize)

        subgradient_square_norm = 0
        if gap_mode == 'affine':
            updated_parameters['open_gap_score'] += stepsize*subgradient['Gap opens'] + add_noise(1, iternb)
            updated_parameters['extend_gap_score'] += stepsize*subgradient['Gap extends'] + add_noise(1, iternb)
            subgradient_square_norm += subgradient['Gap opens']**2
            subgradient_square_norm += subgradient['Gap extends']**2
            if verbose:
                print('Gap open step:', stepsize*subgradient['Gap opens'])
                print('Gap extend step:', stepsize*subgradient['Gap extends'])
        elif gap_mode == 'linear':
            gapnb = subgradient['Gap opens'] + subgradient['Gap extends']
            updated_parameters['gap_score'] += stepsize*gapnb + add_noise(1, iternb)
            subgradient_square_norm += gapnb**2
            if verbose:
                print('Linear gap step:', stepsize*gapnb)

        if substitution_mode == 'simple':
            match_subgradient = np.sum(np.diag(subgradient['Substitutions']))
            mismatch_subgradient = np.sum(subgradient['Substitutions']) - match_subgradient
            updated_parameters['match_score'] += stepsize*match_subgradient + add_noise(1, iternb)
            updated_parameters['mismatch_score'] += stepsize*mismatch_subgradient + add_noise(1, iternb)
            subgradient_square_norm += match_subgradient**2
            subgradient_square_norm += mismatch_subgradient**2
            if verbose:
                print('Match step:', stepsize*match_subgradient)
                print('Mismatch step:', stepsize*mismatch_subgradient)
        elif substitution_mode == 'symmetric':
            subsM = subgradient['Substitutions']
            subsM = subsM + add_noise(subsM.shape, iternb)
            # Summing mismatch counts to keep the matrix symmetric
            # with off-diagonal elements corresponding to mismatches,
            # subtracting the diagonal to avoid counting matches twice
            subsM = subsM + subsM.T - np.diag(np.diag(subsM))
            # subsM = subsM + subsM.T
            updated_parameters['substitution_matrix'] += stepsize*subsM
            subgradient_square_norm += np.sum(subsM**2)
        else:
            subsM = subgradient['Substitutions']
            subsM = subsM + add_noise(subsM.shape, iternb)
            updated_parameters['substitution_matrix'] += stepsize*subsM
            subgradient_square_norm += np.sum(subsM**2)
        subgradient_l2_trajectory.append(np.sqrt(subgradient_square_norm))
        if verbose:
            print('New parameters:')
            print(updated_parameters)
            print('Subgradient norm:', subgradient_l2_trajectory[-1])
        if verbose:
            print('End of iteration', iternb)
            print()

    # Set final parameters
    results = {}
    if gap_mode == 'affine':
        aligner.open_gap_score = updated_parameters['open_gap_score']
        aligner.extend_gap_score = updated_parameters['extend_gap_score']
        results['open_gap_score'] = updated_parameters['open_gap_score']
        results['extend_gap_score'] = updated_parameters['extend_gap_score']
    elif gap_mode == 'linear':
        aligner.gap_score = updated_parameters['gap_score']
        results['gap_score'] = updated_parameters['gap_score']

    if substitution_mode == 'simple':
        aligner.match_score = updated_parameters['match_score']
        aligner.mismatch_score = updated_parameters['mismatch_score']
        results['match_score'] = updated_parameters['match_score']
        results['mismatch_score'] = updated_parameters['mismatch_score']
    else:
        aligner.substitution_matrix = updated_parameters['substitution_matrix']
        results['substitution_matrix'] = updated_parameters['substitution_matrix']

    # Realign with the new parameters
    alnlist = _align_pairs(seqlistA, seqlistB, aligner, num_threads)
    alignment_scores = [aln.score for aln in alnlist]
    logit_scores = logit_partial_scores(alignment_scores,
                                        updated_parameters['alpha'])
    new_logL = logit_logL(logit_scores, labels)
##    EL = 0
##    VL = 0
##    for ls in logit_scores:
##        if 1e-30 < ls < 1-1e-30:
##            EL += ls*np.log(ls) + (1-ls)*np.log(1-ls)
##            VL += ls*(1-ls)*(np.log(ls)**2 + np.log(1-ls)**2)
##    SDL = np.sqrt(VL)
##    loglik_expectation.append(EL)
##    loglik_sd.append(SDL)
    loglik_trajectory.append(new_logL)

    if substitution_mode == 'simple':
        theta_n = np.array([updated_parameters['match_score'], updated_parameters['mismatch_score'], updated_parameters['open_gap_score'], updated_parameters['extend_gap_score']])

    elif substitution_mode == 'general':
        theta_n = list(np.array(updated_parameters['substitution_matrix']).flatten())
        theta_n.append(updated_parameters['open_gap_score'])
        theta_n.append(updated_parameters['extend_gap_score'])

        theta_n = np.array(theta_n)

    theta_trajectory.append(theta_n)

    results['theta_trajectory'] = np.array(theta_trajectory) ## currently a 2d array (rows are theta_n vector)
    results['loglik_trajectory'] = loglik_trajectory
    results['subgradient_l2_trajectory'] = subgradient_l2_trajectory
    results['final_loglik'] = new_logL
    results['aligner'] = aligner
    if return_alignments:
        results['alignments'] = alnlist
    results['alignment_logit_scores'] = logit_scores
    results['alpha'] = updated_parameters['alpha']
    # results['loglik_expectation_trajectory'] = loglik_expectation
    # results['loglik_sd_trajectory'] = loglik_sd
    return results
