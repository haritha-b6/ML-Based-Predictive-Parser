"""Random sentence generator: samples leftmost derivations from a grammar and
records every parsing decision (the training labels for the ML parser)."""
import random
from grammar import END

# relative weight of the epsilon alternative (controls sentence length)
EPS_WEIGHT = {"StmtList": 0.8, "ExprT": 1.2, "TermT": 1.5, "ElseOpt": 1.0,
              "ArgList": 1.0, "ArgTail": 1.2}


class Generator:
    def __init__(self, g, seed=0, max_depth=30, max_tokens=140):
        self.g, self.rng = g, random.Random(seed)
        self.max_depth, self.max_tokens = max_depth, max_tokens
        self.cost = self._min_costs()

    def _min_costs(self):
        cost = {nt: 10**9 for nt in self.g.nonterminals}
        changed = True
        while changed:
            changed = False
            for lhs, rhs in self.g.productions:
                c = sum(cost[s] if self.g.is_nt(s) else 1 for s in rhs)
                if c < cost[lhs]:
                    cost[lhs], changed = c, True
        return cost

    def _prod_cost(self, pi):
        return sum(self.cost[s] if self.g.is_nt(s) else 1 for s in self.g.productions[pi][1])

    def _expand(self, nt, depth, parent, toks, ev):
        cands = self.g.by_lhs[nt]
        if depth > self.max_depth or len(toks) > self.max_tokens:
            pi = min(cands, key=self._prod_cost)
        else:
            w = [EPS_WEIGHT.get(nt, 1.0) if not self.g.productions[i][1] else 1.0
                 for i in cands]
            pi = self.rng.choices(cands, weights=w)[0]
        ev.append({"nt": nt, "prod": pi, "pos": len(toks), "parent": parent})
        for s in self.g.productions[pi][1]:
            if self.g.is_nt(s):
                self._expand(s, depth + 1, nt, toks, ev)
            else:
                toks.append(s)

    def sentence(self):
        toks, ev = [], []
        self._expand(self.g.start, 0, "<root>", toks, ev)
        return toks, ev


def features(toks, ev):
    """Context features for each decision (computed from the token stream)."""
    n = len(toks)
    brace, paren, b, p = [0], [0], 0, 0
    for t in toks:
        b += (t == "{") - (t == "}")
        p += (t == "(") - (t == ")")
        brace.append(b); paren.append(p)
    tok = lambda i: toks[i] if 0 <= i < n else (END if i >= n else "<s>")
    rows = []
    for e in ev:
        i = e["pos"]
        rows.append({"nt": e["nt"], "la1": tok(i), "la2": tok(i + 1), "la3": tok(i + 2),
                     "prev": tok(i - 1), "parent": e["parent"],
                     "brace": str(min(brace[i], 4)), "paren": str(min(paren[i], 4)),
                     "label": e["prod"]})
    return rows


def build_dataset(g, n_sent, seed):
    gen = Generator(g, seed)
    rows, sents = [], []
    for sid in range(n_sent):
        toks, ev = gen.sentence()
        sents.append(toks)
        for r in features(toks, ev):
            r["sid"] = sid
            rows.append(r)
    return rows, sents
