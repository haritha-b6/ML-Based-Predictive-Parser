"""SLR(1) shift-reduce parser (LR baseline)."""
from collections import defaultdict
from grammar import END
from ll1_parser import Node, ParseError

AUG = "S'"


class SLRParser:
    def __init__(self, grammar):
        self.g = grammar
        # augmented production stored as index -1
        self.prods = list(grammar.productions) + [(AUG, (grammar.start,))]
        self.aug = len(self.prods) - 1
        self._build()

    def _closure(self, items):
        items = set(items)
        work = list(items)
        while work:
            p, d = work.pop()
            rhs = self.prods[p][1]
            if d < len(rhs) and self.g.is_nt(rhs[d]):
                for q in self.g.by_lhs[rhs[d]]:
                    it = (q, 0)
                    if it not in items:
                        items.add(it)
                        work.append(it)
        return frozenset(items)

    def _goto(self, items, sym):
        moved = {(p, d + 1) for p, d in items
                 if d < len(self.prods[p][1]) and self.prods[p][1][d] == sym}
        return self._closure(moved) if moved else None

    def _build(self):
        g = self.g
        start = self._closure({(self.aug, 0)})
        self.states = [start]
        index = {start: 0}
        trans = {}
        work = [start]
        while work:
            s = work.pop()
            syms = {self.prods[p][1][d] for p, d in s if d < len(self.prods[p][1])}
            for x in sorted(syms):
                t = self._goto(s, x)
                if t not in index:
                    index[t] = len(self.states)
                    self.states.append(t)
                    work.append(t)
                trans[(index[s], x)] = index[t]
        self.action = {}
        self.goto = {}
        self.conflicts = []

        def put(st, a, act):
            old = self.action.get((st, a))
            if old is not None and old != act:
                self.conflicts.append((st, a, old, act))
            else:
                self.action[(st, a)] = act

        for i, s in enumerate(self.states):
            for p, d in s:
                lhs, rhs = self.prods[p]
                if d < len(rhs):
                    x = rhs[d]
                    j = trans[(i, x)]
                    if g.is_nt(x):
                        self.goto[(i, x)] = j
                    else:
                        put(i, x, ("s", j))
                elif p == self.aug:
                    put(i, END, ("acc",))
                else:
                    for a in g.follow[lhs]:
                        put(i, a, ("r", p))

    def parse(self, tokens):
        toks = [t for t, _ in tokens] + [END]
        stack = [(0, None)]
        pos, steps, log = 0, 0, []
        while True:
            st = stack[-1][0]
            la = toks[pos]
            act = self.action.get((st, la))
            steps += 1
            if act is None:
                raise ParseError(f"Syntax error at token {pos} ({la!r})", pos)
            if act[0] == "s":
                stack.append((act[1], Node(la)))
                pos += 1
            elif act[0] == "r":
                lhs, rhs = self.prods[act[1]]
                n = len(rhs)
                kids = [x[1] for x in stack[len(stack) - n:]] if n else [Node("ε")]
                if n:
                    del stack[-n:]
                node = Node(lhs, kids)
                stack.append((self.goto[(stack[-1][0], lhs)], node))
                log.append({"state": st, "pos": pos, "la": la, "prod": act[1]})
            else:
                return stack[-1][1], {"steps": steps, "decisions": log}

    def accepts(self, tokens):
        try:
            self.parse(tokens)
            return True
        except ParseError:
            return False
