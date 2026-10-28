"""Classical table-driven LL(1) predictive parser (baseline)."""
from grammar import END


class Node:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []

    def size(self):
        return 1 + sum(c.size() for c in self.children)

    def pretty(self, indent=0):
        s = "  " * indent + str(self.label) + "\n"
        for c in self.children:
            s += c.pretty(indent + 1)
        return s


class ParseError(Exception):
    def __init__(self, msg, pos):
        super().__init__(msg)
        self.pos = pos


class LL1Parser:
    def __init__(self, grammar, allow_conflicts=False):
        self.g = grammar
        self.table, self.conflicts = grammar.ll1_table()
        if self.conflicts and not allow_conflicts:
            raise ValueError("Grammar is not LL(1): %d conflicts" % len(self.conflicts))

    def parse(self, tokens, trace=False):
        """tokens: list of (terminal, lexeme). Returns (tree, info)."""
        g = self.g
        toks = [t for t, _ in tokens] + [END]
        root = Node(g.start)
        stack = [(END, None), (g.start, root)]
        pos, steps, log = 0, 0, []
        while stack:
            sym, node = stack.pop()
            la = toks[pos]
            steps += 1
            if g.is_nt(sym):
                pi = self.table.get((sym, la))
                if pi is None:
                    raise ParseError(f"No rule for ({sym}, {la}) at token {pos}", pos)
                log.append({"nt": sym, "pos": pos, "la": la, "prod": pi})
                rhs = g.productions[pi][1]
                kids = [Node(s) for s in rhs]
                node.children = kids if rhs else [Node("ε")]
                for s, k in reversed(list(zip(rhs, kids))):
                    stack.append((s, k))
                if trace:
                    print(f"expand {g.prod_str(pi)}")
            else:
                if sym != la:
                    raise ParseError(f"Expected {sym!r}, found {la!r} at token {pos}", pos)
                if sym != END:
                    node.label = f"{sym}"
                pos += 1
        return root, {"steps": steps, "decisions": log}

    def accepts(self, tokens):
        try:
            self.parse(tokens)
            return True
        except ParseError:
            return False
