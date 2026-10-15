"""Catalogue of argumentation frameworks used in the Part 3 experiments,
plus a parser so any framework (A, R) can be given on the command line."""

import itertools
import string

import numpy as np

from .hcategorizer import ArgumentationFramework


def _names(n):
    return list(string.ascii_lowercase[:n])


def isolated(n):
    return ArgumentationFramework(_names(n), [], name=f"isolated{n}")


def chain(n):
    a = _names(n)
    return ArgumentationFramework(a, [(a[i], a[i + 1]) for i in range(n - 1)], name=f"chain{n}")


def cycle(n):
    a = _names(n)
    return ArgumentationFramework(a, [(a[i], a[(i + 1) % n]) for i in range(n)], name=f"cycle{n}")


def complete(n):
    a = _names(n)
    return ArgumentationFramework(a, [(x, y) for x, y in itertools.permutations(a, 2)], name=f"complete{n}")


def star(n):
    """n-1 unattacked arguments all attacking the last one."""
    a = _names(n)
    return ArgumentationFramework(a, [(x, a[-1]) for x in a[:-1]], name=f"star{n}")


def random_af(n, p, seed):
    """Erdős–Rényi digraph without self-attacks."""
    rng = np.random.default_rng(seed)
    a = _names(n)
    att = [(x, y) for x, y in itertools.permutations(a, 2) if rng.random() < p]
    return ArgumentationFramework(a, att, name=f"random{n}_p{p}_s{seed}")


def parse(spec, name=None):
    """Parse 'a,b,c : a>b, b>c, c>a' into an ArgumentationFramework."""
    if ":" in spec:
        args_part, att_part = spec.split(":", 1)
    else:
        args_part, att_part = spec, ""
    arguments = [s.strip() for s in args_part.split(",") if s.strip()]
    attacks = []
    for item in att_part.split(","):
        item = item.strip()
        if not item:
            continue
        b, a = (s.strip() for s in item.split(">"))
        attacks.append((b, a))
    return ArgumentationFramework(arguments, attacks, name=name or "custom")


def small_frameworks():
    """Hand-made frameworks whose degrees have closed forms (used in tests)."""
    return {
        "isolated2": isolated(2),
        "attack2": ArgumentationFramework(["a", "b"], [("a", "b")], name="attack2"),
        "mutual2": ArgumentationFramework(["a", "b"], [("a", "b"), ("b", "a")], name="mutual2"),
        "self1": ArgumentationFramework(["a"], [("a", "a")], name="self1"),
        "chain3": chain(3),
        "cycle3": cycle(3),
        "star3": star(3),
    }


def experiment_frameworks():
    """Frameworks studied in Part B, grouped by number of arguments."""
    afs = [
        isolated(2),
        ArgumentationFramework(["a", "b"], [("a", "b")], name="attack2"),
        ArgumentationFramework(["a", "b"], [("a", "b"), ("b", "a")], name="mutual2"),
        ArgumentationFramework(["a", "b"], [("a", "b"), ("b", "b")], name="attack2_self"),
        chain(3),
        cycle(3),
        star(3),
        complete(3),
        chain(4),
        cycle(4),
        complete(4),
        random_af(4, 0.4, seed=1),
        chain(5),
        cycle(5),
        random_af(5, 0.3, seed=2),
        cycle(6),
        random_af(6, 0.3, seed=3),
    ]
    return afs
