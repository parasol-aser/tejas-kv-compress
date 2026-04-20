import numpy as np
from numpy.linalg import qr
from scipy.integrate import quad
from scipy.special import gamma as gamma_fn
import warnings

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# Beta distribution PDF
# Calculates the centroids (rounding points) for a given dimension and number of bits
def _beta_pdf(x, d):
    if d > 300:
        # For high dimensions, use Gaussian approximation (Beta distribution converges to Gaussian)
        sigma = 1.0 / np.sqrt(d)
        # Mean is zero for a sphere
        val = np.exp(-0.5 * (x / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
        return val
    # For low dimensions, use the exact Beta distribution formula
    coeff = gamma_fn(d / 2) / (np.sqrt(np.pi) * gamma_fn((d - 1) / 2))
    base = np.maximum(1 - x**2, 0)
    return coeff * base ** ((d - 3) / 2)

def _solve_lloyd_max(d: int, bits: int, n_iter: int = 200, n_init_pts: int = 2000):
    # Number of centroids
    k = 2 ** bits

    # Initialize centroids using uniform sampling
    xs = np.linspace(-1 + 1e-9, 1 - 1e-9, n_init_pts)
    # For each point, calculate the probability density function and normalize
    fxs = np.array([_beta_pdf(xi, d) for xi in xs])
    fxs = fxs / fxs.sum()
    # Calculate the cumulative distribution function
    cdf = np.cumsum(fxs)

    # Get the quantiles and setup initial centroids
    quantiles = np.linspace(1 / (2 * k), 1 - 1 / (2 * k), k)
    centroids = np.interp(quantiles, cdf, xs)

    # Iteratively update centroids until convergence
    for _ in range(n_iter):

        boundaries = np.concatenate([[-1.0],
                                      (centroids[:-1] + centroids[1:]) / 2,
                                      [1.0]])

        new_centroids = np.empty(k)
        for i in range(k):
            lo, hi = boundaries[i], boundaries[i + 1]

            num, _ = quad(lambda x: x * _beta_pdf(x, d), lo, hi, limit=100)
            den, _ = quad(lambda x: _beta_pdf(x, d),     lo, hi, limit=100)
            new_centroids[i] = num / den if den > 1e-15 else (lo + hi) / 2
        # Early stopping if centroids converge
        if np.max(np.abs(new_centroids - centroids)) < 1e-10:
            break
        centroids = new_centroids

    return centroids

# Bookkeeping for codebooks
_CODEBOOK_CACHE: dict = {}

# Get codebook for a given dimension and number of bits
def get_codebook(d: int, bits: int) -> np.ndarray:
    key = (d, bits)
    if key not in _CODEBOOK_CACHE:
        _CODEBOOK_CACHE[key] = _solve_lloyd_max(d, bits)
    return _CODEBOOK_CACHE[key]

# Random rotation matrix
def _random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:

    G = rng.standard_normal((d, d))
    Q, R = qr(G)

    Q = Q * np.sign(np.diag(R))
    return Q

# ---------------------------------------------------------------------------
# Algorithm 1: TurboQuant with MSE distortion
# ---------------------------------------------------------------------------
class TurboQuantMSE:

    # Setup the dimension, number of bits, random rotation matrix, and codebook
    def __init__(self, dim: int, bits: int, seed: int = 42):
        if bits < 1:
            raise ValueError("bits must be >= 1")
        self.dim  = dim
        self.bits = bits

        rng       = np.random.default_rng(seed)
        self.Pi   = _random_rotation(dim, rng)
        self.Pi_T = self.Pi.T
        self.codebook = get_codebook(dim, bits)
        self._k = 2 ** bits

    # Quantize the input vector
    def quantize(self, x: np.ndarray) -> np.ndarray:
        # Check if the input vector is one-dimensional
        single = x.ndim == 1
        if single:
            x = x[None, :]

        # Width of the data must match the dimension
        n, d = x.shape
        assert d == self.dim, f"Expected dim {self.dim}, got {d}"

        # Rotate the input vector
        y = (self.Pi @ x.T).T

        # Find the nearest centroid
        diff = y[:, :, None] - self.codebook[None, None, :]

        # Get the index of the nearest centroid
        idx  = np.argmin(np.abs(diff), axis=-1).astype(np.int32)
        return idx[0] if single else idx

    # Dequantize the input vector
    def dequantize(self, idx: np.ndarray) -> np.ndarray:
        # Check if the input vector is one-dimensional
        single = idx.ndim == 1
        if single:
            idx = idx[None, :]

        # Get the actual centroids
        y_tilde = self.codebook[idx]

        # Rotate the input vector back
        x_tilde = (self.Pi_T @ y_tilde.T).T

        # Return the reconstructed vector
        return x_tilde[0] if single else x_tilde


    # Simulate the round trip of the input vector
    def round_trip(self, x: np.ndarray) -> np.ndarray:
        return self.dequantize(self.quantize(x))


    # Calculate the distortion
    def mse(self, x: np.ndarray) -> float:

        x_hat = self.round_trip(x)
        return float(np.mean((x - x_hat) ** 2))

    # Pretty print the object
    def __repr__(self):
        return (f"TurboQuantMSE(dim={self.dim}, bits={self.bits}, "
                f"codebook_size={self._k})")

# ---------------------------------------------------------------------------
# Algorithm 2: TurboQuant with product quantization
# ---------------------------------------------------------------------------
class TurboQuantProd:

    # Setup the dimension, number of bits, random rotation matrix, and codebook
    def __init__(self, dim: int, bits: int, seed: int = 42):
        if bits < 1:
            raise ValueError("bits must be >= 1")
        self.dim  = dim
        self.bits = bits

        rng = np.random.default_rng(seed)

        self._use_mse_stage = (bits > 1)
        mse_bits = max(bits - 1, 1)
        self.mse_q = TurboQuantMSE(dim, mse_bits, seed=seed)

        self.S = rng.standard_normal((dim, dim))

    # Quantize the input vector
    def quantize(self, x: np.ndarray):

        # Check if the input vector is one-dimensional
        single = x.ndim == 1
        if single:
            x = x[None, :]

        # Width of the data must match the dimension
        n, d = x.shape
        assert d == self.dim

        # Use MSE quantization for the first stage
        if self._use_mse_stage:
            idx    = self.mse_q.quantize(x)
            x_mse  = self.mse_q.dequantize(idx)
            r      = x - x_mse
        else:
            idx    = np.zeros((n, d), dtype=np.int32)
            x_mse  = np.zeros((n, d), dtype=np.float64)
            r      = x.copy()

        # Quantize the residual using the random matrix S
        Sr      = (self.S @ r.T).T
        qjl     = np.sign(Sr).astype(np.int8)

        qjl[qjl == 0] = 1

        # Calculate the norm of the residual
        gamma = np.linalg.norm(r, axis=1)

        if single:
            return idx[0], qjl[0], float(gamma[0])
        return idx, qjl, gamma

    # Dequantize the input vector
    def dequantize(self, idx, qjl, gamma) -> np.ndarray:

        # Check if the input vector is one-dimensional
        single = (idx.ndim == 1)
        if single:
            idx   = idx[None, :]
            qjl   = qjl[None, :]
            gamma = np.atleast_1d(gamma)

        n, d = idx.shape

        # Dequantize the MSE part
        if self._use_mse_stage:
            x_mse = self.mse_q.dequantize(idx)
        else:
            x_mse = np.zeros((n, d), dtype=np.float64)

        # Calculate the coefficient for the residual part
        coeff   = np.sqrt(np.pi / 2) / d

        # Dequantize the residual part
        St_qjl  = (self.S.T @ qjl.astype(np.float64).T).T
        x_qjl   = coeff * gamma[:, None] * St_qjl

        # Combine the MSE and residual parts
        x_hat = x_mse + x_qjl

        # Return the reconstructed vector
        return x_hat[0] if single else x_hat

    # Simulate the round trip of the input vector
    def round_trip(self, x: np.ndarray) -> np.ndarray:

        idx, qjl, gamma = self.quantize(x)
        return self.dequantize(idx, qjl, gamma)

    # Pretty print the object
    def __repr__(self):
        return (f"TurboQuantProd(dim={self.dim}, bits={self.bits}, "
                f"mse_bits={self.mse_q.bits})")


def _fmt(x):
    return f"{x:.6f}"

def run_tests():

    rng = np.random.default_rng(0)
    sep = "=" * 68

    print(sep)
    print("  TurboQuant Tests")
    print(sep)

    # TEST 1: Basic round-trip test for MSE-optimal quantization
    print("\n[TEST 1] TurboQuantMSE: single vector round-trip (d=128)")
    d = 128
    x = rng.standard_normal(d)
    x /= np.linalg.norm(x)

    expected_mse = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
    for bits in [1, 2, 3, 4]:
        q = TurboQuantMSE(d, bits, seed=7)
        x_hat = q.round_trip(x)
        mse_val = np.mean((x - x_hat) ** 2)
        exp = expected_mse[bits]
        status = "PASS" if mse_val < exp * 2.5 else "WARN"
        print(f"  bits={bits}  MSE={_fmt(mse_val)}  expected={exp}  [{status}]")

    # TEST 2: Unbiasedness check for the inner-product quantizer
    print("\n[TEST 2] TurboQuantProd: unbiasedness of inner-product estimator (d=256, trials=300)")
    d      = 256
    trials = 300
    x  = rng.standard_normal(d); x  /= np.linalg.norm(x)
    y  = rng.standard_normal(d)
    true_ip = np.dot(y, x)

    for bits in [1, 2, 3, 4]:
        estimates = []
        for seed_i in range(trials):
            q = TurboQuantProd(d, bits, seed=seed_i)
            x_hat = q.round_trip(x)
            estimates.append(np.dot(y, x_hat))
        mean_est = np.mean(estimates)
        bias = abs(mean_est - true_ip)
        rel_bias = bias / (abs(true_ip) + 1e-9)
        status = "PASS" if rel_bias < 0.15 else "WARN"
        print(f"  bits={bits}  true_IP={_fmt(true_ip)}  "
              f"mean_est={_fmt(mean_est)}  bias={_fmt(bias)}  [{status}]")

    # TEST 3: Visual inspection of original vs reconstructed values
    print("\n[TEST 3] Dequantization visual inspection (d=8, unit vector)")
    d = 8
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    x /= np.linalg.norm(x)

    print(f"  Original x:  {np.array2string(x, precision=4, suppress_small=True)}")
    for bits in [1, 2, 4]:
        q_mse  = TurboQuantMSE(d, bits, seed=0)
        q_prod = TurboQuantProd(d, bits, seed=0)
        x_mse  = q_mse.round_trip(x)
        x_prod = q_prod.round_trip(x)
        mse_m  = np.mean((x - x_mse)**2)
        mse_p  = np.mean((x - x_prod)**2)
        print(f"  bits={bits}  TurboQuantMSE  : "
              f"{np.array2string(x_mse,  precision=4, suppress_small=True)}"
              f"  MSE={mse_m:.4f}")
        print(f"  bits={bits}  TurboQuantProd : "
              f"{np.array2string(x_prod, precision=4, suppress_small=True)}"
              f"  MSE={mse_p:.4f}")

    print("\n" + sep)
    print("  All tests complete.")
    print(sep)

if __name__ == "__main__":
    run_tests()
