def square(x: float) -> float:
    return x ** 2


res = map(square, [1, 2, 3, 4, 5])
print(list(res))


from typing import Callable

# 使用Callable类型提示
def operate_on_two_integers(op: Callable[[int, int], int], a: int, b: int) -> int:
    return op(a, b)

# 符合上面的 op 的一个实例
def add(x: int, y: int) -> int:
    return x + y
result = operate_on_two_integers(add, 2, 3)
print(result)