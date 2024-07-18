from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # 中心差分: f'(x) = (f(x + eps) - f(x - eps)) / (2 * eps)
    vals = list(vals)
    vals[arg] += epsilon
    f_plus = f(*vals)
    vals[arg] -= 2 * epsilon
    f_minus = f(*vals)
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # 拓扑排序，注意不要把 leaf 放进去
    # dfs 算法
    from collections import defaultdict
    status = defaultdict(int)  # 0:to_be_visited, 1:visiting, 2: visited
    result = []

    def dfs(variable: Variable) -> bool:
        status[variable.unique_id] = 1
        for p in variable.parents:
            if status[p.unique_id] == 1 or (status[p.unique_id] == 0 and not dfs(p)):
                return False
        status[variable.unique_id] = 2
        if not variable.is_leaf():
            result.append(variable)
        return True

    dfs(variable)
    del status[variable.unique_id]
    result.reverse()  # dfs 得到的结果是反的，所以要 reverse 一下
    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # deriv_parents = variable.chain_rule(deriv)
    # for parent, parent_deriv in deriv_parents:
    #     if parent.is_leaf():
    #         parent.accumulate_derivative(parent_deriv)
    #     else:
    #         backpropagate(parent, parent_deriv)
    topo = topological_sort(variable)
    from collections import defaultdict
    var_deriv_map = defaultdict(float)
    var_deriv_map[variable.unique_id] = deriv
    for node in topo:
        derivs = node.chain_rule(var_deriv_map[node.unique_id])
        for vars, deriv in derivs:
            if vars.is_leaf():
                vars.accumulate_derivative(deriv)
            else:
                var_deriv_map[vars.unique_id] += deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
