"""One deterministic recursive chain, with proof-bearing splits and open leaves.

Chain completeness concerns its leaves, not enumeration of every possible chain.
Algebraic search retains the coefficient-field scope of the two-factor engine.
Rational graphs are additionally decomposed as rational maps in either direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp
from sympy.polys.polyerrors import BasePolynomialError

from ._multivariate_decomposition import _degree, _normal
from .curves import CurveError
from .errors import QuotientAlgebraError
from .function_fields import _constant_domain

_RESOLVED = frozenset({"indecomposable", "degree_below_two"})
_OPEN = frozenset(
    {
        "max_depth",
        "max_nodes",
        "degree_limit",
        "search_incomplete",
        "unsupported",
        "repeated_component",
    }
)


def _fd():
    # Importing either this module or the public facade first must work.
    from . import functional_decomposition

    return functional_decomposition


def _signature(function, variables, constants, algebraic):
    if algebraic:
        return tuple(function.monic().terms())
    p, q = sp.fraction(_normal(function, variables, constants))
    return (
        len(variables),
        tuple(sp.Poly(p, *variables, domain=constants).terms()),
        tuple(sp.Poly(q, *variables, domain=constants).terms()),
    )


def _graph_map(relation):
    """Return (rational map, source coordinate, forward orientation), if linear."""
    x, z = relation.gens
    if relation.degree(x) == 1:
        p = sp.Poly(relation.as_expr(), x)
        return sp.cancel(-p.TC() / p.LC()), z, True
    if relation.degree(z) == 1:
        p = sp.Poly(relation.as_expr(), z)
        return sp.cancel(-p.TC() / p.LC()), x, False
    return None


def _graph_split(relation, decomposition, forward):
    fd = _fd()
    x, z = relation.gens
    t = decomposition.parameter
    p, q = sp.fraction(decomposition.outer)
    a, b = sp.fraction(decomposition.inner)
    if forward:
        outer, inner = x * q - p, t * b - a
    else:
        outer, inner = t * b - a, z * q - p
    G = fd._relation(outer, (x, t), relation.domain)
    H = fd._relation(inner, (t, z), relation.domain)
    resultant = sp.Poly(sp.resultant(G.as_expr(), H.as_expr(), t), x, z, domain=relation.domain)
    result = fd.AlgebraicDecomposition(relation, G, H, 1, resultant.LC() / relation.LC())
    if not result.verify():
        raise ArithmeticError("rational graph split failed its resultant identity")
    return result


@dataclass(frozen=True)
class RationalChainComponent:
    """A leaf map; only the last component can have several input coordinates."""

    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]


@dataclass(frozen=True)
class DecompositionChainNode:
    """A certified split, or a leaf with a proof or an explicit stopping reason.

    ``search`` retains the original two-factor result. For implicit rational
    graphs, ``rational_search`` records the additional graph-map search.
    Unsupported diagnostics certify no negative mathematical claim.
    """

    function: object
    variables: tuple[sp.Symbol, ...]
    constants: object
    search: object | None = None
    rational_search: object | None = None
    supplied: object | None = None
    split: object | None = None
    outer: DecompositionChainNode | None = None
    inner: DecompositionChainNode | None = None
    stop_reason: str | None = None
    detail: str | None = None

    @property
    def leaves(self):
        return (self,) if self.split is None else (*self.outer.leaves, *self.inner.leaves)

    @property
    def resolved(self):
        return self.stop_reason in _RESOLVED


def _rational_context(result, function, variables, constants):
    fd = _fd()
    if isinstance(result, fd.RationalDecompositionResult):
        inputs = (result.variable,)
    elif isinstance(result, fd.MultivariateRationalDecompositionResult):
        inputs = result.variables
    else:
        return False
    return (
        inputs == variables
        and result.original_expression == function
        and result.constants == constants
        and result.verify()
    )


def _initial_context(certificate, function, variables, constants, algebraic):
    fd = _fd()
    if algebraic:
        return (
            isinstance(certificate, fd.AlgebraicDecomposition)
            and certificate.relation == function
            and certificate.constants == constants
            and certificate.verify()
        )
    if isinstance(certificate, fd.RationalDecomposition):
        inputs = (certificate.variable,)
    elif isinstance(certificate, fd.MultivariateRationalDecomposition):
        inputs = certificate.variables
    else:
        return False
    return (
        inputs == variables
        and certificate.original_expression == function
        and certificate.constants == constants
        and certificate.verify()
    )


def _node_answers(node, algebraic):
    fd = _fd()
    if node.supplied is not None:
        if (
            node.search is not None
            or node.rational_search is not None
            or not _initial_context(
                node.supplied, node.function, node.variables, node.constants, algebraic
            )
        ):
            return None
        return (node.supplied,), False
    answers, covered = (), True
    if algebraic:
        graph = _graph_map(node.function)
        if node.rational_search is not None:
            if graph is None:
                return None
            expression, variable, forward = graph
            if not _rational_context(node.rational_search, expression, (variable,), node.constants):
                return None
            answers = tuple(
                _graph_split(node.function, d, forward) for d in node.rational_search.decompositions
            )
            covered = node.rational_search.complete
        elif graph is not None and node.search is not None:
            return None
        if node.search is not None:
            if (
                not isinstance(node.search, fd.AlgebraicDecompositionSearch)
                or node.search.relation != node.function
                or not node.search.verify()
            ):
                return None
            answers += node.search.decompositions
            covered = covered and node.search.complete
        elif not answers:
            covered = False
    else:
        if node.rational_search is not None:
            return None
        if node.search is not None:
            if not _rational_context(node.search, node.function, node.variables, node.constants):
                return None
            answers, covered = node.search.decompositions, node.search.complete
        else:
            covered = False
    return answers, covered


def _verify_tree(chain, algebraic):
    fd = _fd()
    seen, node_count, node_limit_leaves = set(), 0, []

    def visit(node, depth, ancestors):
        nonlocal node_count
        if not isinstance(node, DecompositionChainNode) or id(node) in seen:
            return False
        seen.add(id(node))
        node_count += 1
        if node.constants != chain.constants or not isinstance(node.variables, tuple):
            return False
        if algebraic:
            if (
                not isinstance(node.function, sp.Poly)
                or node.function.gens != node.variables
                or fd._input_relation(node.function.as_expr(), node.variables, node.constants)
                != node.function
            ):
                return False
        else:
            if not node.variables:
                return False
            fd._symbols(node.variables)
            _normal(node.function, node.variables, node.constants)
        signature = _signature(node.function, node.variables, node.constants, algebraic)
        evidence = _node_answers(node, algebraic)
        if evidence is None:
            return False
        answers, covered = evidence
        if node.split is not None:
            if (
                signature in ancestors
                or depth >= chain.max_depth
                or node.stop_reason is not None
                or node.detail is not None
                or not answers
                or node.split != answers[0]
                or not node.split.verify()
                or node.outer is None
                or node.inner is None
            ):
                return False
            if algebraic:
                expected = (
                    (node.split.outer, node.split.outer.gens),
                    (node.split.inner, node.split.inner.gens),
                )
            else:
                expected = (
                    (node.split.outer, (node.split.parameter,)),
                    (node.split.inner, node.variables),
                )
            return all(
                child.function == expression
                and child.variables == variables
                and visit(child, depth + 1, (*ancestors, signature))
                for child, (expression, variables) in zip(
                    (node.outer, node.inner), expected, strict=True
                )
            )
        if (
            node.outer is not None
            or node.inner is not None
            or node.stop_reason not in _RESOLVED | _OPEN
        ):
            return False
        reason = node.stop_reason
        if reason in _RESOLVED:
            degree = (
                max(node.function.degree(v) for v in node.variables)
                if algebraic
                else _degree(node.function, node.variables, node.constants)
            )
            return (
                covered
                and not answers
                and node.detail is None
                and reason == ("degree_below_two" if degree < 2 else "indecomposable")
            )
        if not isinstance(node.detail, str) or not node.detail:
            return False
        if reason == "max_depth":
            return depth >= chain.max_depth and node.search is None and node.rational_search is None
        if reason == "repeated_component":
            return signature in ancestors and node.search is None and node.rational_search is None
        if reason == "max_nodes":
            node_limit_leaves.append(node)
            return bool(answers)
        if reason == "search_incomplete":
            return (
                not covered
                and not answers
                and (node.search is not None or node.rational_search is not None)
            )
        if reason == "degree_limit":
            if node.supplied is not None:
                return False
            graph = _graph_map(node.function) if algebraic else None
            return (
                (algebraic and node.function.degree(node.variables[0]) > chain.max_degree)
                or (
                    graph is not None
                    and _degree(graph[0], (graph[1],), node.constants) > chain.max_rational_degree
                )
                or (
                    not algebraic
                    and _degree(node.function, node.variables, node.constants) > chain.max_degree
                )
            )
        return reason == "unsupported" and node.search is None and node.supplied is None

    return (
        visit(chain.root, 0, ())
        and node_count <= chain.max_nodes
        and (not node_limit_leaves or node_count + 2 > chain.max_nodes)
    )


@dataclass(frozen=True)
class RationalDecompositionChain:
    """One ordered outer-to-inner chain over the declared constant field."""

    root: DecompositionChainNode
    max_depth: int
    max_nodes: int
    max_degree: int

    @property
    def original_expression(self):
        return self.root.function

    @property
    def variables(self):
        return self.root.variables

    @property
    def constants(self):
        return self.root.constants

    @property
    def leaves(self):
        return self.root.leaves

    @property
    def components(self):
        return tuple(RationalChainComponent(n.function, n.variables) for n in self.leaves)

    @property
    def complete(self):
        return all(n.resolved for n in self.leaves)

    @property
    def status(self):
        return "complete" if self.complete else "incomplete"

    @property
    def input_guard(self):
        fd = _fd()
        return sp.And(*(sp.Ne(p, 0) for p in fd._visible_denominators(self.original_expression)))

    def compose(self):
        """Return the function-field composition, without pointwise pole claims."""
        components = self.components
        result = components[-1].expression
        for component in reversed(components[:-1]):
            result = component.expression.xreplace({component.variables[0]: result})
        return _normal(result, self.variables, self.constants)

    def verify(self):
        """Replay every split/search and the final rational composition."""
        try:
            _validate_limits(self.max_depth, self.max_nodes, self.max_degree)
            return (
                _constant_domain(self.constants) == self.constants
                and _verify_tree(self, False)
                and _normal(
                    self.compose() - self.original_expression, self.variables, self.constants
                )
                == 0
            )
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            BasePolynomialError,
            ArithmeticError,
            RecursionError,
        ):
            return False


@dataclass(frozen=True)
class AlgebraicDecompositionChain:
    """One implicit chain with exact resultant powers and unresolved leaves.

    Completeness inherits the minimal coefficient-field scope of automatic
    algebraic search, augmented by rational graph-map decomposition. It does
    not certify arbitrary algebraic-function indecomposability or all chains.
    """

    root: DecompositionChainNode
    max_depth: int
    max_nodes: int
    max_degree: int
    max_rational_degree: int

    @property
    def relation(self):
        return self.root.function

    @property
    def constants(self):
        return self.root.constants

    @property
    def variables(self):
        return self.root.variables

    @property
    def leaves(self):
        return self.root.leaves

    @property
    def components(self):
        return tuple(n.function for n in self.leaves)

    @property
    def complete(self):
        return all(n.resolved for n in self.leaves)

    @property
    def status(self):
        return "complete" if self.complete else "incomplete"

    def resultant(self):
        """Eliminate the ordered chain's internal coordinates, inner first."""
        components = self.components
        result = components[-1]
        for component in reversed(components[:-1]):
            result = sp.Poly(
                sp.resultant(component.as_expr(), result.as_expr(), component.gens[1]),
                component.gens[0],
                result.gens[1],
                domain=self.constants,
            )
        return result

    @property
    def exponent(self):
        resultant = self.resultant()
        return resultant.degree(self.variables[0]) // self.relation.degree(self.variables[0])

    @property
    def unit(self):
        return self.resultant().LC() / (self.relation**self.exponent).LC()

    def verify(self):
        """Replay the tree and its global resultant = unit * relation**exponent."""
        try:
            _validate_limits(self.max_depth, self.max_nodes, self.max_degree)
            _fd()._integer(self.max_rational_degree, "max_rational_degree", 1)
            if _constant_domain(self.constants) != self.constants or not _verify_tree(self, True):
                return False
            components = self.components
            coordinates = (components[0].gens[0], *(p.gens[1] for p in components))
            _fd()._symbols(coordinates)
            if any(
                a.gens[1] != b.gens[0] for a, b in zip(components, components[1:], strict=False)
            ):
                return False
            resultant = self.resultant()
            exponent, remainder = divmod(
                resultant.degree(self.variables[0]), self.relation.degree(self.variables[0])
            )
            unit = resultant.LC() / (self.relation**exponent).LC()
            return (
                not remainder
                and exponent >= 1
                and (resultant - self.relation**exponent * unit).is_zero
            )
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            BasePolynomialError,
            ArithmeticError,
            RecursionError,
        ):
            return False


def _validate_limits(max_depth, max_nodes, max_degree):
    fd = _fd()
    fd._integer(max_depth, "max_depth", 0)
    fd._integer(max_nodes, "max_nodes", 1)
    fd._integer(max_degree, "max_degree", 1)


def _build(function, variables, constants, *, algebraic, max_depth, max_nodes, options, initial):
    fd = _fd()
    allocated = 1
    used_names = {v.name for v in variables}
    if initial is not None:
        used_names.add(initial.intermediate.name if algebraic else initial.parameter.name)

    def visit(expression, inputs, depth, ancestors, supplied=None):
        nonlocal allocated
        common = dict(function=expression, variables=inputs, constants=constants)
        signature = _signature(expression, inputs, constants, algebraic)
        if signature in ancestors:
            return DecompositionChainNode(
                **common,
                stop_reason="repeated_component",
                detail="component repeats an ancestor after coordinate renaming",
            )
        if depth >= max_depth:
            return DecompositionChainNode(
                **common, stop_reason="max_depth", detail="recursive depth limit reached"
            )
        search, graph_search, answers = None, None, ()
        label = f"chain_intermediate_{len(used_names)}"
        while label in used_names:
            label = "_" + label
        used_names.add(label)
        parameter = sp.Dummy(label)
        try:
            if supplied is not None:
                answers = (supplied,)
            elif algebraic:
                graph = _graph_map(expression)
                if graph is not None:
                    value, source, forward = graph
                    graph_search = fd.rational_decompose(
                        value,
                        source,
                        parameter=parameter,
                        constants=constants,
                        max_candidates=options["max_candidates"],
                        max_degree=options["max_rational_degree"],
                    )
                    answers = tuple(
                        _graph_split(expression, d, forward) for d in graph_search.decompositions
                    )
                if not answers:
                    search = fd.algebraic_decompose(
                        expression.as_expr(),
                        inputs,
                        parameter=parameter,
                        constants=constants,
                        **{k: v for k, v in options.items() if k != "max_rational_degree"},
                    )
                    answers = search.decompositions
            else:
                search = fd.rational_decompose(
                    expression,
                    inputs[0] if len(inputs) == 1 else inputs,
                    parameter=parameter,
                    constants=constants,
                    **options,
                )
                answers = search.decompositions
        except fd.DecompositionLimitError as exc:
            return DecompositionChainNode(
                **common, rational_search=graph_search, stop_reason="degree_limit", detail=str(exc)
            )
        except (CurveError, QuotientAlgebraError, NotImplementedError) as exc:
            return DecompositionChainNode(
                **common,
                rational_search=graph_search,
                stop_reason="unsupported",
                detail=f"{type(exc).__name__}: {exc}",
            )
        evidence = dict(search=search, rational_search=graph_search, supplied=supplied)
        if not answers:
            covered = (search is not None and search.complete) and (
                graph_search is None or graph_search.complete
            )
            degree = (
                max(expression.degree(v) for v in inputs)
                if algebraic
                else _degree(expression, inputs, constants)
            )
            reason = (
                ("degree_below_two" if degree < 2 else "indecomposable")
                if covered
                else "search_incomplete"
            )
            return DecompositionChainNode(
                **common,
                **evidence,
                stop_reason=reason,
                detail=None if covered else "two-factor search did not resolve every candidate",
            )
        if allocated + 2 > max_nodes:
            return DecompositionChainNode(
                **common,
                **evidence,
                stop_reason="max_nodes",
                detail="two child nodes would exceed max_nodes",
            )
        split = answers[0]
        allocated += 2
        if algebraic:
            outer_inputs, inner_inputs = split.outer.gens, split.inner.gens
        else:
            outer_inputs, inner_inputs = (split.parameter,), inputs
        outer = visit(split.outer, outer_inputs, depth + 1, (*ancestors, signature))
        inner = visit(split.inner, inner_inputs, depth + 1, (*ancestors, signature))
        return DecompositionChainNode(**common, **evidence, split=split, outer=outer, inner=inner)

    return visit(function, variables, 0, (), initial)


def rational_decomposition_chain(
    expression,
    variable,
    *,
    constants=sp.QQ,
    initial=None,
    max_depth=8,
    max_nodes=63,
    max_candidates=256,
    max_degree=32,
):
    """Recursively choose the first certified split, for one or several inputs.

    ``max_nodes`` bounds the entire stored binary tree, ``max_depth`` bounds
    recursive edges. Two-factor budgets apply separately at each visited node;
    neither factorization nor certificate replay has a wall-clock bound.
    A valid incomplete chain remains usable as an exact composition certificate.
    ``initial`` optionally supplies a verified two-component certificate at the
    root; recursive searches then refine both of its components.
    """
    fd = _fd()
    _validate_limits(max_depth, max_nodes, max_degree)
    fd._integer(max_candidates, "max_candidates", 0)
    variables = tuple(variable) if isinstance(variable, (tuple, list)) else (variable,)
    fd._symbols(variables)
    if not variables or any(not v.is_commutative for v in variables):
        raise fd.FunctionalDecompositionError(
            "a nonempty sequence of commutative Symbols is required"
        )
    constants = _constant_domain(constants)
    original = expression.as_expr() if isinstance(expression, sp.Poly) else sp.sympify(expression)
    _normal(original, variables, constants)
    if initial is not None and not _initial_context(initial, original, variables, constants, False):
        raise fd.FunctionalDecompositionError("initial certificate does not verify for this input")
    root = _build(
        original,
        variables,
        constants,
        algebraic=False,
        max_depth=max_depth,
        max_nodes=max_nodes,
        options=dict(max_candidates=max_candidates, max_degree=max_degree),
        initial=initial,
    )
    return RationalDecompositionChain(root, max_depth, max_nodes, max_degree)


def algebraic_decomposition_chain(
    relation,
    variables,
    *,
    constants=sp.QQ,
    initial=None,
    max_depth=8,
    max_nodes=63,
    max_candidates=64,
    max_degree=6,
    max_rational_degree=32,
    max_factor_dimension=32,
    max_separating_attempts=64,
    max_steps=256,
    max_dimension=256,
):
    """Build one recursive implicit chain, preserving resultant powers and units.

    Linear-coordinate relations first use rational decomposition in their graph
    orientation. Other splits use automatic algebraic discovery. Repeated
    ancestor relations are open leaves, never indecomposability proofs.
    Completeness has the two-factor engine's declared coefficient-field scope.
    ``initial`` accepts an already verified algebraic decomposition, including
    results reconstructed from a supplied factor and extension field.
    """
    fd = _fd()
    _validate_limits(max_depth, max_nodes, max_degree)
    for name, value, minimum in (
        ("max_candidates", max_candidates, 0),
        ("max_rational_degree", max_rational_degree, 1),
        ("max_factor_dimension", max_factor_dimension, 1),
        ("max_separating_attempts", max_separating_attempts, 1),
        ("max_steps", max_steps, 1),
        ("max_dimension", max_dimension, 1),
    ):
        fd._integer(value, name, minimum)
    variables = tuple(variables)
    if len(variables) != 2:
        raise fd.FunctionalDecompositionError("variables must be (output, input)")
    fd._symbols(variables)
    constants = _constant_domain(constants)
    expression = relation.as_expr() if isinstance(relation, sp.Poly) else relation
    F = fd._input_relation(expression, variables, constants)
    if initial is not None and not _initial_context(initial, F, variables, constants, True):
        raise fd.FunctionalDecompositionError("initial certificate does not verify for this input")
    root = _build(
        F,
        variables,
        constants,
        algebraic=True,
        max_depth=max_depth,
        max_nodes=max_nodes,
        options=dict(
            max_candidates=max_candidates,
            max_degree=max_degree,
            max_rational_degree=max_rational_degree,
            max_factor_dimension=max_factor_dimension,
            max_separating_attempts=max_separating_attempts,
            max_steps=max_steps,
            max_dimension=max_dimension,
        ),
        initial=initial,
    )
    return AlgebraicDecompositionChain(root, max_depth, max_nodes, max_degree, max_rational_degree)
