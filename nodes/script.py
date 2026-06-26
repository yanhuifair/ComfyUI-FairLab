import ast
import operator

from comfy.comfy_types.node_typing import IO

DEFAULT_SCRIPT = "RESULT = (A, B, C, D)"


SAFE_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

SAFE_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

SAFE_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}

SAFE_FUNCTIONS = {
    "abs": abs,
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "round": round,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "list": list,
}


class SafeScriptEvaluator:
    def __init__(self, variables):
        self.variables = variables

    def evaluate(self, script):
        tree = ast.parse(script, mode="exec")
        if not tree.body:
            return None

        if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
            return self._eval_node(tree.body[0].value)

        if len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign):
            assign = tree.body[0]
            if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name) or assign.targets[0].id != "RESULT":
                raise ValueError("Only 'RESULT = ...' assignment is supported")
            return self._eval_node(assign.value)

        raise ValueError("Only a single expression or 'RESULT = ...' assignment is supported")

    def _eval_node(self, node):
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id in self.variables:
                return self.variables[node.id]
            if node.id in SAFE_FUNCTIONS:
                return SAFE_FUNCTIONS[node.id]
            raise ValueError(f"Unsupported name: {node.id}")

        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(element) for element in node.elts)

        if isinstance(node, ast.List):
            return [self._eval_node(element) for element in node.elts]

        if isinstance(node, ast.Set):
            return {self._eval_node(element) for element in node.elts}

        if isinstance(node, ast.Dict):
            return {self._eval_node(key): self._eval_node(value) for key, value in zip(node.keys, node.values)}

        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_UNARY_OPS:
            return SAFE_UNARY_OPS[type(node.op)](self._eval_node(node.operand))

        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_BIN_OPS:
            return SAFE_BIN_OPS[type(node.op)](self._eval_node(node.left), self._eval_node(node.right))

        if isinstance(node, ast.BoolOp):
            values = [self._eval_node(value) for value in node.values]
            if isinstance(node.op, ast.And):
                result = values[0]
                for value in values[1:]:
                    result = result and value
                return result
            if isinstance(node.op, ast.Or):
                result = values[0]
                for value in values[1:]:
                    result = result or value
                return result

        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                if type(op) not in SAFE_COMPARE_OPS:
                    raise ValueError(f"Unsupported compare operator: {type(op).__name__}")
                right = self._eval_node(comparator)
                if not SAFE_COMPARE_OPS[type(op)](left, right):
                    return False
                left = right
            return True

        if isinstance(node, ast.IfExp):
            return self._eval_node(node.body) if self._eval_node(node.test) else self._eval_node(node.orelse)

        if isinstance(node, ast.Subscript):
            value = self._eval_node(node.value)
            index = self._eval_slice(node.slice)
            return value[index]

        if isinstance(node, ast.Call):
            func = self._eval_node(node.func)
            if func not in SAFE_FUNCTIONS.values():
                raise ValueError("Only whitelisted functions are supported")
            args = [self._eval_node(arg) for arg in node.args]
            kwargs = {keyword.arg: self._eval_node(keyword.value) for keyword in node.keywords}
            return func(*args, **kwargs)

        raise ValueError(f"Unsupported syntax: {type(node).__name__}")

    def _eval_slice(self, node):
        if isinstance(node, ast.Slice):
            return slice(
                self._eval_node(node.lower) if node.lower else None,
                self._eval_node(node.upper) if node.upper else None,
                self._eval_node(node.step) if node.step else None,
            )
        return self._eval_node(node)


class PythonScriptNode:

    def __init__(self):
        pass

    RETURN_TYPES = (IO.ANY, IO.ANY, IO.ANY, IO.ANY)
    RETURN_NAMES = ("A", "B", "C", "D")
    FUNCTION = "run_script"
    OUTPUT_NODE = True
    CATEGORY = "Fair/utility"
    DESCRIPTION = "Safely evaluate a limited Python expression and return up to four outputs."
    SEARCH_ALIASES = ["safe python", "expression evaluator"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "text": (IO.STRING, {"default": DEFAULT_SCRIPT, "multiline": True}),
                "A": (IO.ANY, {}),
                "B": (IO.ANY, {}),
                "C": (IO.ANY, {}),
                "D": (IO.ANY, {}),
            },
        }

    def run_script(self, text=DEFAULT_SCRIPT, A=None, B=None, C=None, D=None):
        script = text if text is not None and len(text) > 0 else DEFAULT_SCRIPT
        evaluator = SafeScriptEvaluator({"A": A, "B": B, "C": C, "D": D, "RESULT": None})
        result = evaluator.evaluate(script)

        if isinstance(result, tuple):
            return result
        return (result, None, None, None)
