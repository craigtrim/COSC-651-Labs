# Issue #32: Diagnostic and arithmetic limits are enforced inside the worker.
"""Known diagnostic code in a subprocess. This is not an arbitrary Python sandbox."""

import ast
import csv
import io
import json
import operator
import sys


def diagnose(text):
    """Return CSV diagnostic codes within 100,000 characters and 1,000 rows."""
    if len(text) > 100000:
        return {"codes": ["SIZE_LIMIT"], "rows": 0}
    if not text.strip():
        return {"codes": ["EMPTY"], "rows": 0}
    codes = []
    if text.startswith("\ufeff"):
        codes.append("BOM_HEADER")
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    fields = reader.fieldnames or []
    if any(f != f.strip() for f in fields):
        codes.append("SPACED_HEADER")
    normalized = [f.strip() for f in fields]
    if not {"id", "invoice_date", "amount"} <= set(normalized):
        codes.append("MISSING_COLUMN")
    rows = []
    for row in reader:
        if len(rows) >= 1000:
            return {"codes": ["SIZE_LIMIT"], "rows": len(rows)}
        rows.append({str(k).strip(): v for k, v in row.items() if k is not None})
    seen = set()
    for row in rows:
        date = row.get("invoice_date", "") or ""
        if not date.strip():
            codes.append("MISSING_DATE")
        if "/" in date:
            codes.append("AMBIGUOUS_DATE")
        key = row.get("id", "")
        if key in seen:
            codes.append("DUPLICATE_ID")
        seen.add(key)
        amount = row.get("amount", "") or ""
        if amount.startswith("-"):
            codes.append("NEGATIVE_AMOUNT")
        if "," in amount:
            codes.append("COMMA_AMOUNT")
        for value in row.values():
            if isinstance(value, str) and value.startswith(("=", "+", "@")):
                codes.append("FORMULA_CELL")
    return {
        "codes": sorted(set(codes)) or ["VALID"],
        "rows": len(rows),
        "columns": normalized,
    }


def calculate(expression):
    """Evaluate bounded arithmetic with length, AST-size, and magnitude limits."""
    if len(expression) > 200:
        raise ValueError("Expression is too long")
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 64:
        raise ValueError("Expression is too complex")
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def visit(node):
        """Evaluate permitted numeric AST nodes and reject out-of-range results."""
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = -visit(node.operand)
        elif isinstance(node, ast.BinOp) and type(node.op) in operators:
            value = operators[type(node.op)](visit(node.left), visit(node.right))
        else:
            raise ValueError("Only bounded arithmetic is allowed")
        if not -1000000 <= value <= 1000000:
            raise ValueError("Magnitude limit exceeded")
        return value

    return {"value": visit(tree)}


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        result = (
            diagnose(request["text"])
            if request["task"] == "diagnose"
            else calculate(request["expression"])
        )
        print(json.dumps({"ok": True, "result": result}))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
