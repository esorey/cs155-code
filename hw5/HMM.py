########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang, Avishek Dutta
# Description:  Set 5 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding.s If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        # Initialize: the most likely length one sequence ending in state i is
        # just i, and the corresponding probability is just the probability of
        # transitioning from the start state to state i, times the probability
        # of state i emitting the first observation.
        for i in range(self.L):
            probs[1][i] = self.A_start[i] * self.O[i][x[0]]
            seqs[1][i] += str(i)

        # Fill in the tables using dynamic programming
        for t in range(2, M+1):
            # Iterate over the states for the current time t
            for state in range(self.L):
                max_prob = -1
                max_prob_prev_state = None

                # Find the maximum score from the previous time step. This
                # means looking at the previous time step, and multiplying by
                # the transition probability and observation probability.
                for prev_state in range(self.L):
                    prob = probs[t-1][prev_state] * self.A[prev_state][state] \
                                                  * self.O[state][x[t-1]]
                    if prob > max_prob:
                        max_prob = prob
                        max_prob_prev_state = prev_state

                # Quick sanity check
                assert max_prob > -1
                assert max_prob_prev_state != None

                # Update this entry in the two tables
                probs[t][state] = max_prob
                seqs[t][state] = seqs[t-1][max_prob_prev_state] + str(state)

        # Get the most likely sequence
        max_prob = -1
        max_prob_state = None
        for state in range(self.L):
            if probs[M][state] > max_prob:
                max_prob = probs[M][state]
                max_prob_state = state

        # Quick sanity check again
        assert max_prob > -1
        assert max_prob_state != None

        print(max_prob)
        max_seq = seqs[M][max_prob_state]
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Initialize the alphas matrix. The probability of being in state i
        # after the first observation is just the probability of transitioning
        # from the start state to state i, times the probability of seeing the
        # first observation while in state i.
        for i in range(self.L):
            alphas[1][i] = self.A_start[i] * self.O[i][x[0]]
        if normalize:
            first_row_sum = sum(alphas[1])
            for i in range(self.L):
                alphas[1][i] /= first_row_sum
        # Fill in the alphas matrix using dynamic programming.
        for t in range(2, M + 1):
            # Iterate over the states for the current time t
            for state in range(self.L):
                # Add each previous state's contribution
                for prev_state in range(self.L):
                    alphas[t][state] += alphas[t-1][prev_state] * \
                                        self.A[prev_state][state] * \
                                        self.O[state][x[t-1]]
            if normalize:
                normalizer = sum(alphas[t])
                for i in range(self.L):
                    alphas[t][i] /= normalizer

        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Initialize the beta matrix.
        if normalize:
            betas[M] = [1/self.L for _ in range(self.L)]
        else:
            for i in range(self.L):
                betas[M][i] = 1



        # Fill in the rest of the beta matrix using dynamic programming.
        for t in range(M - 1, 0, -1):
            for state in range(self.L):
                for state_after in range(self.L):
                    betas[t][state] += betas[t+1][state_after] * \
                                       self.A[state][state_after] * \
                                       self.O[state_after][x[t]]

            if normalize:
                normalizer = sum(betas[t])
                for i in range(self.L):
                    betas[t][i] /= normalizer

        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.

        # Don't need random initialization of A and O for counting
        self.A = [[0. for _ in range(self.L)] for _ in range(self.L)]
        self.O = [[0. for _ in range(self.D)] for _ in range(self.L)]

        # Count the state transitions
        for state_seq in Y:
            for i in range(1, len(state_seq)):
                prev_state = state_seq[i-1]
                curr_state = state_seq[i]
                self.A[prev_state][curr_state] += 1

        # Go back and normalize by the sum of each row (ie, the total number
        # of transitions from each state).
        for i in range(self.L):
            normalizer = sum(self.A[i])
            for j in range(self.L):
                self.A[i][j] /= normalizer


        # Calculate each element of O using the M-step formulas.

        # Sanity check
        assert len(X) == len(Y)
        # Count the emissions
        for i in range(len(X)):
            inp_seq = X[i]
            state_seq = Y[i]

            # Sanity check
            assert len(inp_seq) == len(state_seq)

            for j in range(len(inp_seq)):
                self.O[state_seq[j]][inp_seq[j]] += 1

        # Go back and normalize by the sum of each row (ie, the total number of
        # times we were in state i)
        for i in range(self.L):
            normalizer = sum(self.O[i])
            for j in range(self.D):
                self.O[i][j] /= normalizer


    def unsupervised_learning(self, X):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.
        '''

        NUM_ITERS = 1000
        for itr in range(NUM_ITERS):
            A_update_numerator = [[0. for _ in range(self.L)] for _ in range(self.L)]
            A_update_denominator = [[0. for _ in range(self.L)] for _ in range(self.L)]
            O_update_numerator = [[0. for _ in range(self.D)] for _ in range(self.L)]
            O_update_denominator = [[0. for _ in range(self.D)] for _ in range(self.L)]

            # Expectation step: run forward-backward to compute marginal
            # probabilities with the current model parameters.
            for x in X:
                M = len(x)
                # Get the alphas and betas
                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)

                # Compute P(y^idx = state, x) and
                # P(y^t = state1, y^{t+1} = state2, x). These
                # probabilities are one_state_probs[t][state] and
                # two_state_probs[t][state1][state2], respectively.
                # Think of t as a 'time', indicating how far we've stepped
                # through our input x. Think of state as one of the hidden
                # states in our HMM.
                assert M + 1 == len(alphas)
                one_state_probs = [[alphas[t][state] * betas[t][state]
                                    for state in range(self.L)]
                                    for t in range(M + 1)]

                # Normalize by sum of rows
                for t in range(1, M + 1):
                    row_sum = sum(one_state_probs[t])
                    for state in range(self.L):
                        one_state_probs[t][state] /= row_sum


                two_state_probs = [[[alphas[t][state1] * \
                                     self.O[state2][x[t]] * \
                                     self.A[state1][state2] * \
                                     betas[t+1][state2]
                                    for state2 in range(self.L)]
                                    for state1 in range(self.L)]
                                    for t in range(M)]

                # Normalize.
                for t in range(1, M):
                    normalizer = 0.
                    for state1 in range(self.L):
                        for state2 in range(self.L):
                            normalizer += alphas[t][state1] * \
                            self.O[state2][x[t]] * self.A[state1][state2] * \
                            betas[t+1][state2]
                    for state1 in range(self.L):
                        for state2 in range(self.L):
                            two_state_probs[t][state1][state2] /= normalizer


                # Maximization step: update A and O accordingly.

                # First compute the self.A update
                for state1 in range(self.L):
                    for state2 in range(self.L):
                        for t in range(1, M + 1):
                            A_update_numerator[state1][state2] += \
                                two_state_probs[t - 1][state1][state2]
                            A_update_denominator[state1][state2] += one_state_probs[t - 1][state1]

                # Now compute the self.O update
                for state in range(self.L):
                    for symbol in range(self.D):
                        for t in range(1, M + 1):
                            if x[t - 1] == symbol:
                                O_update_numerator[state][symbol] += one_state_probs[t][state]
                            O_update_denominator[state][symbol] += one_state_probs[t][state]

            # After we've gone through all the observation sequences, we update
            # self.A and self.O
            for state1 in range(self.L):
                for state2 in range(self.L):
                    self.A[state1][state2] = A_update_numerator[state1][state2] \
                                                / A_update_denominator[state1][state2]

            for state in range(self.L):
                for symbol in range(self.D):
                    self.O[state][symbol] = O_update_numerator[state][symbol] \
                                                / O_update_denominator[state][symbol]


    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a string.
        '''

        emission = ''

        # Choose initial state uniformly at random.
        state = random.choice(range(self.L))

        for _ in range(M):
            # Emit an observation from this state.
            emission_probs = self.O[state]
            rand = random.random()
            mark = 0.0
            emission_idx = 0
            while True:
                mark += emission_probs[emission_idx]
                if mark > rand:
                    emission += str(emission_idx)
                    break
                emission_idx += 1

            # Transition to a new state.
            transition_probs = self.A[state]
            rand = random.random()
            mark = 0.0
            next_state = 0
            while True:
                mark += transition_probs[next_state]
                if mark > rand:
                    state = next_state
                    break
                next_state += 1

        return emission


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X)

    return HMM
