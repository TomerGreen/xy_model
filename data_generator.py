import numpy as np
import matplotlib.pyplot as plt
import random


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
            C[i, j] = ex_value(np.dot(op_i, op_j), state) - ex_value(
                op_i, state
            ) * ex_value(op_j, state)
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


def get_z_correlation(i, j, P):
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
    spectra = np.zeros(shape=(num_spins, len(phase_rates)))
    for config_ind, phase_rate in enumerate(phase_rates):
        J = np.full(num_spins, J_val)
        h = np.full(num_spins, J_val / phase_rate)
        A = get_fermionic_hamiltonian(num_spins, J, h)
        eigvals, U = np.linalg.eigh(A)
        spectra[:, config_ind] = eigvals
        proj = (eigvals < 0) * np.eye(num_spins)
        P = np.dot(U, np.dot(proj, np.array(U).conj().T))
        for dist_ind, dist in enumerate(dists):
            corr = get_z_correlation(0, dist, P)
            dist_corrs[dist_ind, config_ind] = corr
    print(dist_corrs)
    for i, phase_rate in enumerate(phase_rates):
        # plt.plot(dists, np.log(-dist_corrs[:, i] + 1e-30), label="J/h = " + str(phase_rate))
        plt.plot(dists, dist_corrs[:, i], label="J/h = " + str(phase_rate))
    plt.xlabel("dist")
    plt.ylabel("Spin Z Correlation")
    plt.legend()
    plt.show()

    # for i, phase_rate in enumerate(phase_rates):
    #     plt.scatter([phase_rate] * len(spectra[:, i]), spectra[:, i], s=0.1)
    # plt.xlabel("J / h")
    # plt.ylabel("Energy Levels")
    # plt.show()

    # for i, dist in enumerate(dists):
    #     plt.plot(phase_rates, dist_corrs[i, :], label="d = " + str(dist))
    # plt.xlabel("J / h")
    # plt.ylabel("Spin Z Correlation")
    # plt.legend()
    # plt.show()


def get_gaussian_kernel(length=11, sigma=1):
    r = range(-int(length / 2), int(length / 2) + 1)
    return [
        1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-float(x) ** 2 / (2 * sigma ** 2))
        for x in r
    ]


def get_smooth_correlations(P, kernel):
    num_spins = P.shape[0]
    one_hot_mat = np.zeros(shape=(num_spins - 1, num_spins))
    corrs = np.zeros(shape=num_spins)
    for d in range(0, num_spins - 1):
        one_hot_mat[d, 0] = 1
        one_hot_mat[d, d + 1] = 1
        corrs[d] = get_z_correlation(0, d, P)
    # plt.plot(range(1, 128), -corrs[1:128] + 1e-30)
    # plt.suptitle("Spin Correlation by Distance")
    # plt.ylabel("-corr(i, j)")
    # plt.xlabel("r(i, j)")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()
    corrs = np.tile(corrs, reps=3)
    smoothed_corrs = np.convolve(corrs, kernel, mode="same")
    smoothed_corrs = smoothed_corrs[num_spins - 1 : 2 * (num_spins - 1)]
    # plt.plot(range(1, 128), -smoothed_corrs[1:128])
    # plt.suptitle("Smoothed Spin Correlation by Distance")
    # plt.ylabel("-corr(i, j) * g(r)")
    # plt.xlabel("r(i, j)")
    # plt.yscale("log")
    # plt.show()
    return one_hot_mat, smoothed_corrs


class RandomValueGenerator:
    """Generates a random number when called, based on a given distribution type and parameters."""

    def __init__(self, dist_type: str, a: float, b: float):
        """
        Defines the distribution
        :param dist_type: One of the distribution types used in __call__.
        :param a: The first parameter for the distribution type. For example, if dist_type="uniform", a is the
        lowest value and b in the highest, and if dist_type="normal", a is the mean and b is the std.
        :param b: The second parameter for the distribution type.
        """
        self.dist_type = dist_type
        self.a = a
        self.b = b

    def __call__(self, shape=None, *args, **kwargs):
        """
        :param shape: The shape of the requested array filled with random values. If not provided, will return a scalar.
        :return: An array or value from the requested distribution.
        """
        if self.dist_type == "uniform":
            val = (
                np.random.uniform(low=self.a, high=self.b)
                if shape is None
                else np.random.uniform(low=self.a, high=self.b, size=shape)
            )
        elif self.dist_type == "normal":
            val = (
                np.random.normal(loc=self.a, scale=self.b)
                if shape is None
                else np.random.normal(loc=self.a, scale=self.b, size=shape)
            )
        else:
            raise ValueError(
                "RandomValueGenerator dist_type must be either 'uniform' or 'normal'."
            )
        return val


class CorrelationDataGenerator:
    """
    Generates image-like data for the XY model solver.
    """

    def __init__(
        self,
        repetitions=3,
        J_val_gen=RandomValueGenerator("normal", 0.0, 1.0),
        h_val_gen=RandomValueGenerator("normal", 0.0, 1.0),
        disorder=True,
        custom_dist=False,
        gaussian_smoothing_sigma=0,
        toy_model=False,
    ):
        """
        Defines the data being generated.
        :param num_spins: The number of spins in the simulated spin chain.
        :param repetitions: The number of times the J, h and one-hot vectors will be repeated.
        Used to simulate cyclic boundary condition for the convolutional network.
        :param J_val_gen: The random value generator used to generate values for coupling.
        :param h_val_gen: The random value generator used to generate values for field.
        :param disorder: Whether to include disorder in the J and h vectors. If there is
        disorder, the coupling coefficient J[i] and field strength h[i] at each site will be
        drawn individually for each site in the configuration. Otherwise, the J and h vectors for
        each configuration will be a full vector with a value from rand_val_gen.
        :param custom_dist: Whether we are using a custom distribution for distances between
        examined sites.
        :param gaussian_smoothing_sigma: An odd int. The sigma value for gaussian smoothing. A
        kernel with this sigma
        will be convolved with the by-distance data if gaussian_smoothing_sigma > 0.
        :param toy_model: Whether to generate data from the toy model (|J/h|)/r.
        """
        self.repetitions = repetitions
        self.J_val_gen = J_val_gen
        self.h_val_gen = h_val_gen
        self.disorder = disorder
        self.custom_dist = custom_dist
        self.gaussian_smoothing_sigma = gaussian_smoothing_sigma
        self.toy_model = toy_model

    def get_data(
        self,
        num_spins,
        samples,
        samples_per_config,
    ):
        """
        Generates correlation data for the spin_z correlation neural network. The data contains
        the coupling coefficients vector J, the field strength vector h and a one-hot vector for
        two spin sites i and j. The ground truth for each sample is the spin-z correlation
        between these sites.
        :param samples: The number of samples to generate.
        :param samples_per_config: The number of samples from each Hamiltonian configuration.
        :return:
            x - An array of shape (samples, num_spins, 3): the first channel is a one-hot vector
            for the sites where the correlation is measured, and the second and third channels
            are coupling and field coefficients vectors J and h.
            y - An array of shape (samples,) that specifies the correlation between the sites
            marked by the one-hot vector.
        """
        x = np.zeros((samples, num_spins, 3))
        y = np.zeros(samples)
        for i in range(int(samples / samples_per_config)):
            if self.disorder:
                J = self.J_val_gen(shape=num_spins)
                h = self.h_val_gen(shape=num_spins)
            else:
                J = np.full(num_spins, self.J_val_gen())
                h = np.full(num_spins, self.h_val_gen())
            A = get_fermionic_hamiltonian(num_spins, J, h)
            eigvals, U = np.linalg.eigh(A)
            proj = (eigvals < 0) * np.eye(num_spins)
            P = np.dot(U, np.dot(proj, np.matrix(U).H))
            if self.gaussian_smoothing_sigma > 0:
                gaussian_kernel = get_gaussian_kernel(
                    length=101, sigma=self.gaussian_smoothing_sigma
                )
                one_hot_mat, corrs = get_smooth_correlations(P, kernel=gaussian_kernel)
            for j in range(samples_per_config):
                n = i * samples_per_config + j
                x[n, :, 1] = J
                x[n, :, 2] = h
                if self.gaussian_smoothing_sigma > 0:
                    rand_int = np.random.randint(low=0, high=num_spins - 1)
                    x[n, :, 0] = one_hot_mat[rand_int, :]
                    y[n] = corrs[rand_int]
                else:
                    # Do we need to make sure the indices are different?
                    k, l = 0, 0
                    while k == l:
                        k = np.random.randint(num_spins)
                        if self.custom_dist:
                            l = (k + get_random_distance(num_spins)) % num_spins
                        else:
                            l = np.random.randint(num_spins)
                    x[n, k, 0] = 1
                    x[n, l, 0] = 1
                    if self.toy_model:
                        distance = min((l - k) % num_spins, (k - l) % num_spins)
                        y[n] = -np.abs(J[0] / h[0]) / distance
                    else:
                        y[n] = get_z_correlation(k, l, P)
                if n % 1000 == 0 and n > 0:
                    print("Created " + str(n) + "/" + str(samples) + " samples")
        x = np.tile(x, reps=(1, self.repetitions, 1))
        temp = list(zip(x, y))
        random.shuffle(temp)
        x, y = zip(*temp)
        return np.array(x), np.array(y)


# def get_correlation_data(
#     num_spins,
#     samples,
#     samples_per_config,
#     repetitions=1,
#     disorder=True,
#     J_mean=None,
#     J_sigma=None,
#     h_mean=None,
#     h_sigma=None,
#     J_low=None,
#     J_high=None,
#     h_low=None,
#     h_high=None,
#     custom_dist=False,
#     gaussian_smoothing_sigma=0,
#     toy_model=False,
# ):
#     """
#     Generates correlation data for the spin_z correlation neural network. The data contains the coupling coefficients
#     vector J, the field strength vector h and a one-hot vector for two spin sites i and j. The ground truth for each
#     sample is the spin-z correlation between these sites.
#     :param num_spins: The number of spins in the chain.
#     :param samples: The number of samples to generate.
#     :param samples_per_config: The number of samples from each Hamiltonian configuration.
#     :param repetitions: The number of times the J, h and one-hot vectors will be repeated. Used to simulate cyclic
#     boundary condition for the convolutional network.
#     :param disorder: Whether to include disorder in the J and h vectors. If there is disorder, the coupling
#     coefficient J[i] and field strength h[i] at each site will be drawn individually for each configuration and each,
#     according to the means and std values specified. If there is no disorder, the J and h vectors for each configuration
#     will be uniform, but their values will be drawn from the means and stds specified.
#     :param J_mean: Mean value for drawing coupling coefficients.
#     :param J_sigma: Standard deviation for drawing coupling coefficients.
#     :param h_mean: Mean value for drawing field strengths.
#     :param h_sigma: Standard deviation for drawing field strengths.
#     :return:
#         x - An array of shape (samples, num_spins, 3): the first channel is a one-hot vector for the sites where the
#         correlation is measured, and the second and third channels are coupling and field coefficients vectors J and h.
#         y - An array of shape (samples,) that specifies the correlation between the sites marked by the one-hot vector.
#     """
#     x = np.zeros((samples, num_spins, 3))
#     y = np.zeros(samples)
#     for i in range(int(samples / samples_per_config)):
#         if all([x is not None for x in [J_mean, J_sigma, h_mean, h_sigma]]):
#             if disorder:
#                 J = np.random.normal(J_mean, J_sigma, num_spins)
#                 h = np.random.normal(h_mean, h_sigma, num_spins)
#             else:
#                 J = np.full(num_spins, np.random.normal(J_mean, J_sigma))
#                 h = np.full(num_spins, np.random.normal(h_mean, h_sigma))
#         elif all([x is not None for x in [J_low, J_high, h_low, h_high]]):
#             if disorder:
#                 J = np.random.uniform(J_low, J_high, num_spins)
#                 h = np.random.uniform(h_low, h_high, num_spins)
#             else:
#                 J = np.full(num_spins, np.random.uniform(J_low, J_high))
#                 h = np.full(num_spins, np.random.uniform(h_low, h_high))
#         else:
#             raise ValueError(
#                 "Either (h_low, h_high, J_low, J_high) must be provided for uniform distribution,"
#                 + " or (h_mean, h_std, J_mean, J_std) for normal distribution"
#             )
#         A = get_fermionic_hamiltonian(num_spins, J, h)
#         eigvals, U = np.linalg.eigh(A)
#         proj = (eigvals < 0) * np.eye(num_spins)
#         P = np.dot(U, np.dot(proj, np.matrix(U).H))
#         if gaussian_smoothing_sigma > 0:
#             gaussian_kernel = get_gaussian_kernel(
#                 length=101, sigma=gaussian_smoothing_sigma
#             )
#             one_hot_mat, corrs = get_smooth_correlations(P, kernel=gaussian_kernel)
#         for j in range(samples_per_config):
#             n = i * samples_per_config + j
#             x[n, :, 1] = J
#             x[n, :, 2] = h
#             if gaussian_smoothing_sigma > 0:
#                 rand_int = np.random.randint(low=0, high=num_spins - 1)
#                 x[n, :, 0] = one_hot_mat[rand_int, :]
#                 y[n] = corrs[rand_int]
#             else:
#                 # Do we need to make sure the indices are different?
#                 k = l = 0
#                 while k == l:
#                     k = np.random.randint(num_spins)
#                     if custom_dist:
#                         l = (k + get_random_distance(num_spins)) % num_spins
#                     else:
#                         l = np.random.randint(num_spins)
#                 x[n, k, 0] = 1
#                 x[n, l, 0] = 1
#                 if toy_model:
#                     distance = min((l - k) % num_spins, (k - l) % num_spins)
#                     y[n] = -np.abs(J[0] / h[0]) / distance
#                 else:
#                     y[n] = get_z_correlation(k, l, P)
#             if n % 1000 == 0 and n > 0:
#                 print("Created " + str(n) + "/" + str(samples) + " samples")
#     x = np.tile(x, reps=(1, repetitions, 1))
#     temp = list(zip(x, y))
#     random.shuffle(temp)
#     x, y = zip(*temp)
#     return np.array(x), np.array(y)


class Scaler(object):
    def __init__(self, y=None, log=True, normalize=True):
        self.log = log
        self.normalize = normalize
        self.mean = None
        self.std = None
        if y is not None:
            self.fit(y)

    def fit(self, y):
        if self.log:
            y = np.log(-y + 1e-30)
        self.mean = y.mean()
        self.std = y.std()

    def transform(self, y):
        if self.log:
            y = np.log(-y + 1e-30)
        if self.normalize:
            y = (y - self.mean) / self.std
        return y

    def inverse_transform(self, y):
        if self.normalize:
            y = y * self.std + self.mean
        if self.log:
            y = np.exp(y)
            y = -(y - 1e-30)
        return y


def get_distances_from_x(x, num_reps=3):
    def get_distance(x_vec):
        num_spins = len(x_vec)
        nonzero_inds = np.argwhere(x_vec == 1.0)
        k, l = nonzero_inds[0, 0], nonzero_inds[1, 0]
        distance = min((l - k) % num_spins, (k - l) % num_spins)
        return distance

    num_spins = int(x.shape[1] / num_reps)
    ind_start = num_spins * int(num_reps / 2)
    spin_locs = x[:, ind_start : ind_start + num_spins, 0]
    distances = np.apply_along_axis(get_distance, axis=1, arr=spin_locs)
    return distances


def analyze_corr_by_distance(distances, y):
    plt.scatter(distances, np.log(-y))
    # plt.scatter(distances, np.log(-y))
    plt.show()


if __name__ == "__main__":
    J_rand_gen = RandomValueGenerator("normal", 1.0, 0.0)
    h_rand_gen = RandomValueGenerator("normal", 1.0, 0.0)

    data_gen = CorrelationDataGenerator(
        J_val_gen=J_rand_gen,
        h_val_gen=h_rand_gen,
        disorder=False,
        gaussian_smoothing_sigma=3,
    )
    x, y = data_gen.get_data(256, 256, 255)
    # distances = get_distances_from_x(x)
    # analyze_corr_by_distance(distances, y)
