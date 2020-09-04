import numpy as np
import matplotlib.pyplot as plt


I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
SIG_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SIG_Y = np.array([[0, -1j], [1j, 0]])
SIG_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

def op_tensor(op, i):
    """
    Returns an operator acting on one spin as a (2^N, 2^N) tensor acting on N spins.
    :param op: a (2, 2) shaped array representing an operator.
    :param i: the index of the spin on which it acts.
    :return: a (2^N, 2^N) shaped array representing the tensor operator acting on N spins.
    """
    ten = 1
    for j in range(N):
        curr = I
        if j == i:
            curr = op
        ten = np.kron(ten, curr)
    return ten

def get_hamiltonian(J_x, J_y, h):
    """
    Calculates the XY hamiltonian with periodic boundary conditions.
    :param J_x: a scalar. Coupling in X axis.
    :param J_y: a scalar. Coupling in Y axis.
    :return: the XY hamiltonian as a 2**N_SPINS matrix.
    """
    H = np.zeros((2 ** N, 2 ** N), dtype=np.complex128)
    for spin in range(N):
        i = spin
        j = spin + 1
        if i == N - 1:
            j = 0
        H += J_x * np.dot(op_tensor(SIG_X, i), op_tensor(SIG_X, j))
        H += J_y * np.dot(op_tensor(SIG_Y, i), op_tensor(SIG_Y, j))
        H += h * op_tensor(SIG_Z, i)
    return H

def get_ground_state(H):
    """
    Gets the ground state of a Hamiltonian
    :param H: a 2D array of shape (N, N) representing the Hamiltonian.
    :return: an array of shape (N,) representing the ground state.
    """
    eig_vals, eig_states = np.linalg.eig(H)
    eig_vals = np.real(eig_vals)
    gs = eig_states[:, np.argmin(eig_vals)]
    return gs

def ex_value(ten, state):
    """
    Gets the expected value of a tensor operator and a state.
    :param ten: a (N, N) shaped array representing the operator as a tensor acting on N spins.
    :param state: a (N,) shaped array representing a state of N spins.
    :return: the expectation value as a scalar.
    """
    return np.real(np.dot(state.T, np.dot(ten, state)))

def get_correlation(op, state):
    """
    Gets the correlation matrix between two operators for a given state.
    :param op: a (2, 2) shaped array representing an operator acting on one spin.
    :param state: an (N,) shaped array representing the state.
    :return: an (N, N) shaped array C where C[i, j] represents the correlation between A_i and A_j.
    """
    N = state.shape[0]
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            op_i, op_j = op_tensor(op, i), op_tensor(op, j)
            C[i, j] = ex_value(np.dot(op_i, op_j), state) - ex_value(op_i, state) * ex_value(op_j, state)
    return C


# def get_corr_data(J_x, J_y, h):
#     H = get_hamiltonian(J_x, J_y, h)
#     gs = get_ground_state(H)
#     C = get_correlation(SIG_Z, gs)
#     print(C)
#     num_samples = N * (N - 1)
#     x = np.zeros((num_samples, N, 4))
#     y = np.zeros((num_samples,))
#     x[:, :, 1] = J_x
#     x[:, :, 2] = J_y
#     x[:, :, 3] = h
#     count = 0
#     for i in range(N):
#         for j in range(N):
#             if i == j:
#                 continue
#             x[count, i, 0] = 1
#             x[count, j, 0] = 1
#             y[count] = C[i, j]
#             count += 1
#     return [x, y]


def simple_jw():
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j + 1 or (i == N - 1 and j == 0):
                A[i, j] = 0.5
            if j == i + 1 or (j == N - 1 and i == 0):
                A[i, j] = 0.5
    print(A)


def get_fermionic_hamiltonian(N, J, h):
    """
    Returns matrix A such that the fermionic Hamiltonian H satisfies H = c_dagger_i * A_i_j * c_i
    :param N: The number of spins in the chain.
    :param J: A vector of shape (N,) representing the coupling coefficient between consecutive spins.
    :param h: A vector of shape (N,) representing the magnetic field at each site.
    :return: A matrix of shape (N, N).
    """
    A = 0.5 * h * np.eye(N)
    for i in range(N - 1):
        A[i, i + 1] += 0.5 * J[i]
        A[i + 1, i] += 0.5 * J[i]
    # Introducing boundary conditions
    A[N - 1, 0] += 0.5 * J[N - 1]
    A[0, N - 1] += 0.5 * J[N - 1]
    return A


def calc_quadruple_term(i, j, eigvals, U):
    sum = 0
    for ind1, eigval1 in enumerate(eigvals):
        for ind2, eigval2 in enumerate(eigvals):
            if eigval1 < 0 and eigval2 < 0:
                sum += (np.abs(U[:, ind1][i]) ** 2) * (np.abs(U[:, ind2][j]) ** 2)
    return sum


def get_z_correlation(i, j, eigvals, U, P):
    # # These two lines seem equivalent.
    # c = 4 * (P[i, i] * P[j, j] - P[i, j] * P[j, i])
    # # c = 4 * calc_quadruple_term(i, j, eigvals, U)
    # c += 1 - 2 * (P[i, i] + P[j, j])
    c = -4 * P[i, j] * P[j, i]
    return c


def get_random_distance(num_spins):
    distances = np.arange(1, int(num_spins / 2))
    weights = [(1 / d) for d in distances]
    total = sum(weights)
    probs = [weight / total for weight in weights]
    distance = np.random.choice(distances, p=probs)
    sign = np.random.choice([1, -1])
    diff = distance * sign
    return diff


def plot_correlation(num_spins, phase_rates, dists, J_val=0.1):
    dist_corrs = np.zeros(shape=(len(dists), len(phase_rates)))
    for config_ind, phase_rate in enumerate(phase_rates):
        J = np.full(num_spins, J_val)
        h = np.full(num_spins, J_val / phase_rate)
        A = get_fermionic_hamiltonian(num_spins, J, h)
        eigvals, U = np.linalg.eigh(A)
        proj = (eigvals < 0) * np.eye(num_spins)
        P = np.dot(U, np.dot(proj, np.matrix(U).H))
        for dist_ind, dist in enumerate(dists):
            corr = get_z_correlation(0, dist, eigvals, U, P)
            dist_corrs[dist_ind, config_ind] = corr
    # for i, phase_rate in enumerate(phase_rates):
    #     plt.plot(dists, np.log(dist_corrs[:, i] + 1e-30), label="J/h = " + str(phase_rate))
    # plt.xlabel("dist")
    # plt.ylabel("log(Spin Z Correlation)")
    # plt.legend()
    # plt.show()
    for i, dist in enumerate(dists):
        plt.plot(phase_rates, dist_corrs[i, :], label="d = " + str(dist))
    plt.xlabel("J / h")
    plt.ylabel("Spin Z Correlation")
    plt.legend()
    plt.show()




def get_correlation_data(num_spins, samples, samples_per_config, repetitions=1, disorder=True,
                         J_mean=0, J_std=0.3, h_mean=0, h_std=0.3, custom_dist=False):
    """
    Generates correlation data for the spin_z correlation neural network. The data contains the coupling coefficients
    vector J, the field strength vector h and a one-hot vector for two spin sites i and j. The ground truth for each
    sample is the spin-z correlation between these sites.
    :param num_spins: The number of spins in the chain.
    :param samples: The number of samples to generate.
    :param samples_per_config: The number of samples from each Hamiltonian configuration.
    :param repetitions: The number of times the J, h and one-hot vectors will be repeated. Used to simulate cyclic
    boundary condition for the convolutional network.
    :param disorder: Whether to include disorder in the J and h vectors. If there is disorder, the coupling
    coefficient J[i] and field strength h[i] at each site will be drawn individually for each configuration and each,
    according to the means and std values specified. If there is no disorder, the J and h vectors for each configuration
    will be uniform, but their values will be drawn from the means and stds specified.
    :param J_mean: Mean value for drawing coupling coefficients.
    :param J_std: Standard deviation for drawing coupling coefficients.
    :param h_mean: Mean value for drawing field strengths.
    :param h_std: Standard deviation for drawing field strengths.
    :return:
        x - An array of shape (samples, num_spins, 3): the first channel is a one-hot vector for the sites where the
        correlation is measured, and the second and third channels are coupling and field coefficients vectors J and h.
        y - An array of shape (samples,) that specifies the correlation between the sites marked by the one-hot vector.
    """
    x = np.zeros((samples, num_spins, 3))
    y = np.zeros(samples)
    # a = []
    for i in range(int(samples / samples_per_config)):
        if disorder:
            J = np.random.normal(J_mean, J_std, num_spins)
            h = np.random.normal(h_mean, h_std, num_spins)
        else:
            J = np.full(num_spins, np.random.normal(J_mean, J_std))
            h = np.full(num_spins, np.random.normal(h_mean, h_std))
        A = get_fermionic_hamiltonian(num_spins, J, h)
        eigvals, U = np.linalg.eigh(A)
        proj = (eigvals < 0) * np.eye(num_spins)
        P = np.dot(U, np.dot(proj, np.matrix(U).H))
        for j in range(samples_per_config):
            n = i * samples_per_config + j
            x[n, :, 1] = J
            x[n, :, 2] = h
            # Do we need to make sure the indices are different?
            k = l = 0
            while(k == l):
                k = np.random.randint(num_spins)
                if custom_dist:
                    l = (k + get_random_distance(num_spins)) % num_spins
                else:
                    l = np.random.randint(num_spins)
            x[n, k, 0] = 1
            x[n, l, 0] = 1
            y[n] = get_z_correlation(k, l, eigvals, U, P)
            if (n % 100 == 0 and n > 0):
                print("Created " + str(n) + " samples")
    x = np.tile(x, reps=(1, repetitions, 1))
    return x, y


class Scaler(object):

    def __init__(self, y=None):
        self.mean = None
        self.std = None
        if y is not None:
            self.fit(y)

    def fit(self, y):
        y = np.log(-y + 1e-30)
        self.mean = y.mean()
        self.std = y.std()

    def transform(self, y):
        y = np.log(-y + 1e-30)
        return (y - self.mean) / self.std

    def inverse_transform(self, y):
        y = y * self.std + self.mean
        return np.exp(-(y - 1e-30))


if __name__ == '__main__':
    plot_correlation(num_spins=256, phase_rates=np.arange(0.5, 5.0, 0.01), dists=[4, 10, 20])
