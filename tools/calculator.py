import ast, operator

ops = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def calculate(expr: str) -> str:
    node = ast.parse(expr, mode='eval')
    def _eval(n):
        if isinstance(n, ast.BinOp):
            return ops[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.Constant):
            return n.value
        raise ValueError("Unsupported")
    return str(_eval(node.body))
