"""Deterministic structural cost policy; calibrated by the bundled benchmark corpus."""

from .domains import algebraic_polynomial, polynomial


def _candidate_order(equations, variables, *, policy):
    graph = {v: set() for v in variables}
    for row in equations:
        present = row.free_symbols.intersection(variables)
        for v in present:
            graph[v].update(present - {v})
    counts = {v: sum(v in row.free_symbols for row in equations) for v in variables}
    polynomials = tuple(algebraic_polynomial(row, variables, extension=False) for row in equations)
    degrees = {v: max((p.degree(v) for p in polynomials), default=0) for v in variables}
    result = []
    graph = {v: set(neighbours) for v, neighbours in graph.items()}
    while graph:

        def key(v):
            neighbours = list(graph[v])
            fill = sum(
                b not in graph[a] for i, a in enumerate(neighbours) for b in neighbours[i + 1 :]
            )
            return (fill if policy == "fill" else 0, counts[v], degrees[v], variables.index(v))

        selected = min(graph, key=key)
        neighbours = graph.pop(selected)
        for v in neighbours:
            graph[v].discard(selected)
            graph[v].update(neighbours - {v})
        result.append(selected)
    return tuple(result)


def graph_cost(equations, order):
    """Return elimination width and fill-edge count for a coordinate order."""
    graph = {v: set() for v in order}
    for row in equations:
        present = row.free_symbols.intersection(order)
        for v in present:
            graph[v].update(present - {v})
    total = 0
    width = 0
    for selected in order:
        neighbours = list(graph.pop(selected))
        width = max(width, len(neighbours))
        total += sum(
            b not in graph[a] for i, a in enumerate(neighbours) for b in neighbours[i + 1 :]
        )
        for v in neighbours:
            graph[v].discard(selected)
            graph[v].update(set(neighbours) - {v})
    return width, total


def ordered_variables(equations, variables, *, policy="auto"):
    """Choose a deterministic coordinate order from polynomial incidence and degree."""
    if policy not in {"auto", "fill", "frequency"}:
        raise ValueError("unknown ordering policy")
    if policy != "auto":
        return _candidate_order(equations, variables, policy=policy)
    baseline = _candidate_order(equations, variables, policy="frequency")
    candidate = _candidate_order(equations, variables, policy="fill")
    # Calibration found no advantage when orders coincide. Preserve the
    # established order on ties; change only on lower structural graph cost.
    return (
        candidate
        if graph_cost(equations, candidate) < graph_cost(equations, baseline)
        else baseline
    )


def prefer_action(quotient, max_dimension, equations=()):
    # A repeated univariate relation is a cost hint, never a radicality proof.
    """Select action-matrix extraction from quotient size and repeated-factor hints."""
    repeated = False
    for poly in quotient.groebner_basis.polys:
        expression = poly.as_expr()
        symbols = expression.free_symbols
        if len(symbols) == 1:
            univariate = algebraic_polynomial(expression, (next(iter(symbols)),))
            repeated |= univariate.gcd(univariate.diff()).degree() > 0
    repeated_input = False
    # The expanded corpus exposes repeated multivariate input factors hidden by
    # reduced Groebner relations. Restrict square-free analysis to small inputs.
    checked = 0
    for expression in equations:
        poly = polynomial(expression, quotient.variables, quotient.domain)
        if poly.total_degree() <= 8 and len(poly.terms()) <= 32:
            checked += 1
            repeated_input |= any(power > 1 for _, power in poly.sqf_list()[1])
        if repeated_input or checked >= 16:
            break
    dimension = quotient.dimension
    allowed = dimension <= min(64, max_dimension) and not (repeated or repeated_input)
    return allowed, {
        "quotient_dimension": dimension,
        "dense_action_entries": len(quotient.variables) * dimension**2,
        "action_cubic_work": dimension**3,
        "repeated_univariate_relation": bool(repeated),
        "repeated_input_factor": bool(repeated_input),
        "dispatch_reason": "repeated_factor"
        if repeated or repeated_input
        else "small_action"
        if allowed
        else "dimension_budget",
        "preferred_backend": "action" if allowed else "rur",
    }
