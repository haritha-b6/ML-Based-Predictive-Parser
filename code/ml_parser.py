"""ML-based predictive parser: a trained classifier replaces / assists the parse table.

mode="pure"    : the classifier picks among ALL productions of the nonterminal.
mode="guarded" : the classifier may only pick among productions whose predict set
                 contains the lookahead (FIRST/FOLLOW). No candidate -> syntax error;
                 one candidate -> taken directly; several (an LL(1) conflict) -> ML decides.
"""
import pandas as pd
import numpy as np
from grammar import END
from ll1_parser import Node, ParseError

F3 = ["nt", "la1", "la2", "la3", "prev", "parent", "brace", "paren"]


class MLPredictiveParser:
    def __init__(self, grammar, model, cols=F3, cache=True, mode="guarded", str_labels=False):
        self.g, self.model, self.cols, self.mode = grammar, model, cols, mode
        self.use_cache, self.cache = cache, {}
        self.hits = self.misses = self.ml_calls = 0
        self.classes = list(model.named_steps["clf"].classes_)
        if str_labels:
            lookup = {grammar.prod_str(i): i for i in range(len(grammar.productions))}
            self.class_idx = np.array([lookup.get(c, -1) for c in self.classes])
        else:
            self.class_idx = np.array(self.classes)
        self.class_nt = np.array([grammar.productions[i][0] if i >= 0 else "?" for i in self.class_idx])
        # predict sets for the guard
        self.pset = []
        for lhs, rhs in grammar.productions:
            fs = grammar.first_of_seq(rhs)
            ps = fs - {"ε"}
            if "ε" in fs:
                ps |= grammar.follow[lhs]
            self.pset.append(ps)

    def candidates(self, nt, la):
        return [i for i in self.g.by_lhs[nt] if la in self.pset[i]]

    def _ml_choose(self, feat, allowed):
        key = tuple(feat[c] for c in self.cols) + (tuple(allowed),)
        if self.use_cache and key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        self.ml_calls += 1
        X = pd.DataFrame([{c: feat[c] for c in self.cols}])
        proba = self.model.predict_proba(X)[0]
        mask = np.isin(self.class_idx, allowed)
        pi = int(self.class_idx[int((proba * mask).argmax())]) if mask.any() else allowed[0]
        if self.use_cache:
            self.cache[key] = pi
        return pi

    def choose(self, feat):
        nt, la = feat["nt"], feat["la1"]
        if self.mode == "guarded":
            cands = self.candidates(nt, la)
            if not cands:
                return None
            if len(cands) == 1:
                return cands[0]
            return self._ml_choose(feat, cands)
        return self._ml_choose(feat, self.g.by_lhs[nt])

    def parse(self, tokens, recover=False, max_steps=None):
        g = self.g
        toks = [t for t, _ in tokens] + [END]
        n = len(toks)
        tok = lambda i: toks[i] if 0 <= i < n else (END if i >= n else "<s>")
        root = Node(g.start)
        stack = [(END, None, "<root>"), (g.start, root, "<root>")]
        pos, brace, paren, steps, decisions = 0, 0, 0, 0, 0
        errors = []
        cap = max_steps or (30 * n + 200)
        while stack:
            steps += 1
            if steps > cap:
                errors.append(pos)
                if not recover:
                    raise ParseError("Parse did not terminate (step cap)", pos)
                break
            sym, node, parent = stack.pop()
            la = toks[pos]
            err = False
            if g.is_nt(sym):
                feat = {"nt": sym, "la1": la, "la2": tok(pos + 1), "la3": tok(pos + 2),
                        "prev": tok(pos - 1), "parent": parent,
                        "brace": str(min(max(brace, 0), 4)), "paren": str(min(max(paren, 0), 4))}
                pi = self.choose(feat)
                if pi is None:
                    err = True
                else:
                    decisions += 1
                    rhs = g.productions[pi][1]
                    kids = [Node(s) for s in rhs]
                    node.children = kids if rhs else [Node("ε")]
                    for s, k in reversed(list(zip(rhs, kids))):
                        stack.append((s, k, sym))
            elif sym == la:
                if sym == END:
                    break
                brace += (sym == "{") - (sym == "}")
                paren += (sym == "(") - (sym == ")")
                node.label = sym
                pos += 1
            else:
                err = True
            if err:
                errors.append(pos)
                if not recover:
                    raise ParseError(f"Syntax error at token {pos} ({la!r})", pos)
                while pos < n - 1 and toks[pos] not in (";", "}"):     # panic mode
                    pos += 1
                if pos < n - 1 and toks[pos] == ";":
                    pos += 1
                while stack and stack[-1][0] not in ("StmtList", "}", END):
                    stack.pop()
                done = toks[:pos]
                brace = done.count("{") - done.count("}")
                paren = done.count("(") - done.count(")")
                if stack and stack[-1][0] == END and pos < n - 1:
                    errors.append(pos)
                    break
        return root, {"steps": steps, "decisions": decisions, "errors": errors}

    def accepts(self, tokens):
        try:
            self.parse(tokens)
            return True
        except ParseError:
            return False
