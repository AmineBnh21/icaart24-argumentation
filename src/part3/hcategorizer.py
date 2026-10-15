"""Weighted h-categorizer semantics (Amgoud, Ben-Naim, Doder & Vesic, IJCAI 2017).

For a weighted argumentation framework (A, R, w) with w : A -> [0, 1], the
acceptability degree of an argument a is the unique solution of

    Deg(a) = w(a) / (1 + sum_{b in Att(a)} Deg(b))

It is computed as the limit of the sequence

    f^0(a)     = w(a)
    f^{k+1}(a) = w(a) / (1 + sum_{b in Att(a)} f^k(b))

and the iteration stops once max_a |f^{k+1}(a) - f^k(a)| < epsilon.
"""

import numpy as np


class ArgumentationFramework:
    """An abstract argumentation framework (A, R).

    `arguments` is a list of names, `attacks` a list of pairs (b, a) meaning
    "b attacks a". The attack matrix M has M[i, j] = 1 iff arguments[i]
    attacks arguments[j], so the sum of attacker degrees is x @ M.
    """

    def __init__(self, arguments, attacks, name=None):
        self.arguments = list(arguments)
        if len(set(self.arguments)) != len(self.arguments):
            raise ValueError("argument names must be unique")
        self.index = {a: i for i, a in enumerate(self.arguments)}
        self.attacks = []
        for b, a in attacks:
            if b not in self.index or a not in self.index:
                raise ValueError(f"attack ({b}, {a}) uses an unknown argument")
            if (b, a) not in self.attacks:
                self.attacks.append((b, a))
        self.name = name or "AF"
        n = len(self.arguments)
        self.matrix = np.zeros((n, n))
        for b, a in self.attacks:
            self.matrix[self.index[b], self.index[a]] = 1.0

    def __len__(self):
        return len(self.arguments)

    def attackers(self, a):
        return [b for b, target in self.attacks if target == a]

    def __repr__(self):
        rel = ", ".join(f"{b}->{a}" for b, a in self.attacks) or "no attack"
        return f"{self.name}(A={{{', '.join(self.arguments)}}}, R={{{rel}}})"


def h_categorizer(af, weights, epsilon=1e-8, max_iter=100_000, return_iterations=False):
    """Acceptability degrees of a single weight vector.

    `weights` is a dict {argument: weight} or a sequence ordered like
    `af.arguments`. Returns a dict {argument: degree}.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if isinstance(weights, dict):
        w = np.array([weights[a] for a in af.arguments], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    if w.shape != (len(af),):
        raise ValueError(f"expected {len(af)} weights, got shape {w.shape}")
    if np.any(w < 0) or np.any(w > 1):
        raise ValueError("weights must lie in [0, 1]")

    x, iterations = h_categorizer_batch(af, w[None, :], epsilon, max_iter, return_iterations=True)
    degrees = dict(zip(af.arguments, x[0]))
    return (degrees, iterations) if return_iterations else degrees


def h_categorizer_batch(af, W, epsilon=1e-8, max_iter=100_000, return_iterations=False):
    """Vectorised h-categorizer for many weight vectors at once.

    `W` has shape (N, |A|), one weight vector per row. All rows are iterated
    together; the loop stops when every row has moved by less than epsilon
    on every argument. Returns X of shape (N, |A|).
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    W = np.asarray(W, dtype=float)
    M = af.matrix
    x = W.copy()
    for k in range(1, max_iter + 1):
        x_new = W / (1.0 + x @ M)
        delta = np.max(np.abs(x_new - x)) if x.size else 0.0
        x = x_new
        if delta < epsilon:
            return (x, k) if return_iterations else x
    raise RuntimeError(f"no convergence after {max_iter} iterations (last delta={delta:.3g})")
