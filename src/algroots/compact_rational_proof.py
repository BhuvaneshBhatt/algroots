"""Cross-prime CRT for compact polynomial-combination DAGs.

This module never assumes a prime image is a rational proof.  It only aligns
strictly identical, coefficient-free DAG layouts.  Every reconstructed rational
DAG must be independently evaluated against the original inputs over QQ.
Layouts with different supports are kept in separate CRT buckets.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RationalProofDAG:
    """Exact, topologically ordered polynomial-combination circuit.

    Node edges are (earlier node or negative original-generator index,
    sparse rational coefficient). `scales` convert original generators to
    the primitive integral generators used by the modular calculation.
    """

    nodes: tuple
    roots: tuple
    scales: tuple
    generator_count: int


def _edge_dict(factor, nvars, prime):
    if isinstance(factor, int):
        value = factor % prime
        return {(0,) * nvars: value} if value else {}
    return {
        tuple(power): int(value) % prime for power, value in factor.terms() if int(value) % prime
    }


def normalize_modular_topology(compact, prime):
    """Deterministically renumber reachable DAG nodes, without altering identities.

    Child traversal is ordered by coefficient-free descendant structure. Equal
    structural keys retain their stable original order; this is *not* a claim
    that all algebraically equal proof circuits have identical topology.
    Prime-specific coefficients are retained and checked at reconstruction.
    """
    nodes = compact.nodes
    roots = compact.roots
    structural = {}

    def shape(index):
        if index < 0:
            return (0, -index)
        if index not in structural:
            structural[index] = (1, tuple(sorted(shape(child) for child, _ in nodes[index])))
        return structural[index]

    # This postorder ensures every positive child has a smaller new index.
    order, seen = [], set()

    def visit(index):
        if index in seen:
            return
        seen.add(index)
        for child, _ in sorted(nodes[index], key=lambda edge: (shape(edge[0]), edge[0])):
            if child >= 0:
                visit(child)
        order.append(index)

    for root in sorted(roots, key=lambda index: (shape(index), index)):
        visit(root)
    numbering = {old: new for new, old in enumerate(order)}
    arranged = tuple(
        tuple(
            sorted(
                (
                    (numbering[child] if child >= 0 else child, factor)
                    for child, factor in nodes[old]
                ),
                key=lambda edge: edge[0],
            )
        )
        for old in order
    )
    return tuple(numbering[root] for root in roots), arranged


def encode_compact(proof, prime):
    """Take an immutable per-prime snapshot, without expanding witness rows."""
    nvars = len(proof.ring.symbols)
    roots, normalized = normalize_modular_topology(proof, prime)
    rows = tuple(
        tuple((child, _edge_dict(factor, nvars, prime)) for child, factor in edges)
        for edges in normalized
    )
    topology = (
        roots,
        tuple(
            tuple((child, tuple(sorted(coefficients))) for child, coefficients in edges)
            for edges in rows
        ),
    )
    return topology, rows


def crt_edges(previous, incoming, modulus, prime, crt_polynomial):
    """Combine matching proof-DAG edge coefficients by the Chinese remainder theorem."""
    return tuple(
        tuple(
            (left_child, crt_polynomial(left, right, modulus, prime))
            for (left_child, left), (right_child, right) in zip(left_row, right_row, strict=True)
        )
        for left_row, right_row in zip(previous, incoming, strict=True)
    )


def reconstruct_dag(edges, roots, modulus, scales, generator_count, reconstruct):
    """Reconstruct rational proof-DAG coefficients, returning None if reconstruction fails."""
    rows = []
    for row in edges:
        out = []
        for child, terms in row:
            rational = {}
            for monomial, residue in terms.items():
                coefficient = reconstruct(residue, modulus)
                if coefficient is None:
                    return None
                if coefficient:
                    rational[monomial] = coefficient
            out.append((child, rational))
        rows.append(tuple(out))
    return RationalProofDAG(tuple(rows), tuple(roots), tuple(scales), generator_count)


def verify_rational_dag(dag, original, candidate, ring, *, max_terms=100000):
    """Independently check full rational DAG identities by exact evaluation.

    The verifier never calls the modular kernel and never reconstructs a
    coefficient-polynomial *witness row*.  Each DAG node is evaluated to one
    QQ polynomial; every edge is checked for acyclicity and valid exponents.
    Reverse ideal inclusion and Buchberger's criterion are checked separately.
    """
    try:
        if not isinstance(dag, RationalProofDAG):
            return False
        nonzero = tuple(f for f in original if f)
        if (
            dag.generator_count != len(nonzero)
            or len(dag.scales) != len(nonzero)
            or len(dag.roots) != len(candidate)
        ):
            return False
        if len(dag.nodes) > max_terms or max_terms < 1:
            return False
        primitives = tuple(
            f * ring.domain.convert(scale) for f, scale in zip(nonzero, dag.scales, strict=True)
        )
        values = []
        variables = len(ring.symbols)
        for index, edges in enumerate(dag.nodes):
            value = ring.zero
            for child, terms in edges:
                if not isinstance(child, int) or child >= index or child < -len(primitives):
                    return False
                if not hasattr(terms, "items"):
                    return False
                coefficients = {}
                for powers, coefficient in terms.items():
                    if (
                        not isinstance(powers, tuple)
                        or len(powers) != variables
                        or any(not isinstance(p, int) or p < 0 for p in powers)
                    ):
                        return False
                    c = ring.domain.convert(coefficient)
                    if c:
                        coefficients[powers] = c
                if coefficients:
                    factor = ring.from_dict(coefficients)
                    value += factor * (primitives[-child - 1] if child < 0 else values[child])
                    if len(value) > max_terms:
                        return False
            values.append(value)
        if any(not isinstance(root, int) or root < 0 or root >= len(values) for root in dag.roots):
            return False
        return all(
            values[root] == target for root, target in zip(dag.roots, candidate, strict=True)
        )
    except (
        TypeError,
        ValueError,
        KeyError,
        ArithmeticError,
        AttributeError,
        OverflowError,
        ZeroDivisionError,
    ):
        return False


__all__ = [
    "RationalProofDAG",
    "encode_compact",
    "crt_edges",
    "reconstruct_dag",
    "verify_rational_dag",
    "normalize_modular_topology",
]


def canonicalize_rational_dag(dag, originals, ring, *, max_terms=100000):
    """Deduplicate **algebraically equal** rational proof nodes.

    For fixed original generators, two circuit nodes may have completely
    different topologies and yet represent the same QQ polynomial. Evaluate
    each node exactly, and replace later nodes by the earliest polynomial-
    equal predecessor. Merge parallel edges and delete zero coefficients.
    This is a semantic normalization, not a potentially unsafe cross-prime
    topology guess: no prime images or modular hashes authorize equality.

    Its cost is proportional to evaluating the entire circuit over QQ, and it
    is optional.  The independent certificate verifier still
    checks the final (possibly normalized) DAG over QQ.
    """
    if not isinstance(dag, RationalProofDAG):
        raise TypeError("expected RationalProofDAG")
    nonzero = tuple(f for f in originals if f)
    if len(nonzero) != dag.generator_count or len(nonzero) != len(dag.scales):
        raise ValueError("generator count or scales mismatch")
    primitive = tuple(
        f * ring.domain.convert(scale) for f, scale in zip(nonzero, dag.scales, strict=True)
    )
    nvars = len(ring.symbols)
    old_to_new, values, new_nodes, seen = {}, [], [], {}
    for i, edges in enumerate(dag.nodes):
        combined = {}
        for child, terms in edges:
            if not isinstance(child, int) or child >= i or child < -len(primitive):
                raise ValueError("invalid forward or out-of-range proof edge")
            destination = old_to_new[child] if child >= 0 else child
            merged = combined.setdefault(destination, {})
            for monomial, coefficient in terms.items():
                if (
                    not isinstance(monomial, tuple)
                    or len(monomial) != nvars
                    or any(not isinstance(v, int) or v < 0 for v in monomial)
                ):
                    raise ValueError("invalid proof exponent")
                q = ring.domain.convert(coefficient)
                merged[monomial] = merged.get(monomial, ring.domain.zero) + q
                if not merged[monomial]:
                    merged.pop(monomial)
        combined = {child: coeffs for child, coeffs in combined.items() if coeffs}
        if sum(len(poly) for poly in combined.values()) > max_terms:
            raise ValueError("proof normalization term budget exceeded")
        result = ring.zero
        for child, coeffs in sorted(combined.items()):
            factor = ring.from_dict(coeffs)
            result += factor * (primitive[-child - 1] if child < 0 else values[child])
            if len(result) > max_terms:
                raise ValueError("semantic proof normalization exceeded term budget")
        signature = tuple(result.terms())
        previous = seen.get(signature)
        if previous is not None:
            old_to_new[i] = previous
        else:
            old_to_new[i] = len(new_nodes)
            seen[signature] = len(new_nodes)
            values.append(result)
            new_nodes.append(tuple(sorted(combined.items())))
    if any(root not in old_to_new for root in dag.roots):
        raise ValueError("proof root out of bounds")
    normalized_roots = tuple(old_to_new[root] for root in dag.roots)
    reachable = set()
    pending = list(normalized_roots)
    while pending:
        node = pending.pop()
        if node in reachable:
            continue
        reachable.add(node)
        pending.extend(child for child in dict(new_nodes[node]) if child >= 0)
    ordered = sorted(reachable)
    remap = {old: i for i, old in enumerate(ordered)}
    packed = tuple(
        tuple((remap[child] if child >= 0 else child, terms) for child, terms in new_nodes[old])
        for old in ordered
    )
    return RationalProofDAG(
        packed, tuple(remap[root] for root in normalized_roots), dag.scales, dag.generator_count
    )


__all__.append("canonicalize_rational_dag")
