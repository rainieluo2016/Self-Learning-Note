# %% COMMAND
import tensorflow as tf


# %%
class HiddenMarkovModel:

    def __init__(self, T, E, T0, epsilon=0.001, maxStep=10):

        with tf.name_scope("Inital_Parameters"):
            with tf.name_scope("Scalar_constants"):
                # Max number of iteration
                self.maxStep = maxStep

                # convergence criteria
                self.epsilon = epsilon

                # Number of possible states
                self.S = T.shape[0]

                # Number of possible observations
                self.O = E.shape[0]

                self.prob_state_1 = []

            with tf.name_scope("Model_Parameters"):
                # Emission probability
                self.E = tf.Variable(E, dtype=tf.float64, name="emission_matrix")

                # Transition matrix
                self.T = tf.Variable(T, dtype=tf.float64, name="transition_matrix")

                # Initial state vector
                self.T0 = tf.Variable(
                    tf.constant(T0, dtype=tf.float64, name="inital_state_vector")
                )

    def initialize_viterbi_variables(self, shape):

        # Path states - N steps x S states
        pathStates = tf.Variable(tf.zeros(shape), name="pathStates")

        # Path scores - N steps x S states
        pathScores = tf.Variable(tf.zeros(shape), name="pathScores")

        # States sequence - N steps
        states_seq = tf.Variable(tf.zeros([self.N], dtype=tf.int32), name="states_seq")

        return pathStates, pathScores, states_seq

    def belif_propagation(self, scores):
        score_reshape = tf.reshape(scores, (-1, 1))
        return tf.add(score_reshape, tf.log(self.T))

    def viterbi_inference(self, obs_seq):

        # Number of observations - n steps
        self.N = len(obs_seq)
        # shape of path variable n steps x s states
        shape = [self.N, self.S]

        # observed sequence
        x = tf.constant(obs_seq, dtype=tf.int32, name="obs_seq")

        with tf.name_scope("Init_viterbi_variables"):
            # Initialize variables - define the shape of the variables
            pathStates, pathScores, states_seq = self.initialize_viterbi_variables(
                shape
            )

        with tf.name_scope("Emission_seq_"):
            # Emission sequence
            # find the emission probability of the observed sequence
            obs_prob_seq = tf.gather(self.E, x)
            # split into list fo tensors
            obs_prob_list = tf.split(obs_prob_seq, self.N, 0)

        with tf.name_scope("Starting_log-priors"):
            # initialize with state starting log-priors
            pathScores = tf.scatter_update(
                pathScores, 0, tf.log(self.T0) + tf.squeeze(obs_prob_list[0])
            )

        with tf.name_scope("Belief_propagation"):
            for step, obs_prob in enumerate(obs_prob_list[1:]):
                # compute the belief propagation
                with tf.name_scope(f"Step_{step}"):
                    # compute the transition probability
                    belief = self.belief_propagation(pathScores[step, :])
                # compute the transition probability
