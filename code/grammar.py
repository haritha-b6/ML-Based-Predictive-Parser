"""Context-free grammar utilities: parsing, FIRST/FOLLOW, LL(1) table."""
from collections import defaultdict

EPS = "ε"
END = "$"


class Grammar:
    def __init__(self, text):
        self.productions = []          # list of (lhs, tuple(rhs)); rhs == () means epsilon
        self.start = None
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lhs, rhs = line.split("->")
            lhs = lhs.strip()
            if self.start is None:
                self.start = lhs
            for alt in rhs.split("|"):
                syms = tuple(s for s in alt.split() if s != EPS)
                self.productions.append((lhs, syms))
        self.nonterminals = []
        for lhs, _ in self.productions:
            if lhs not in self.nonterminals:
                self.nonterminals.append(lhs)
        nts = set(self.nonterminals)
        terms = []
        for _, rhs in self.productions:
            for s in rhs:
                if s not in nts and s not in terms:
                    terms.append(s)
        self.terminals = terms
        self.by_lhs = defaultdict(list)
        for i, (lhs, _) in enumerate(self.productions):
            self.by_lhs[lhs].append(i)
        self.first = {}
        self.follow = {}
        self._compute_first()
        self._compute_follow()

    @classmethod
    def from_file(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(f.read())

    def prod_str(self, i):
        lhs, rhs = self.productions[i]
        return f"{lhs} -> {' '.join(rhs) if rhs else EPS}"

    def is_nt(self, s):
        return s in self.by_lhs

    # ---------- FIRST ----------
    def _compute_first(self):
        first = {nt: set() for nt in self.nonterminals}
        changed = True
        while changed:
            changed = False
            for lhs, rhs in self.productions:
                before = len(first[lhs])
                first[lhs] |= self.first_of_seq(rhs, first)
                if len(first[lhs]) != before:
                    changed = True
        self.first = first

    def first_of_seq(self, seq, first=None):
        first = first if first is not None else self.first
        out = set()
        for s in seq:
            if not self.is_nt(s):
                out.add(s)
                return out
            out |= first[s] - {EPS}
            if EPS not in first[s]:
                return out
        out.add(EPS)
        return out

    # ---------- FOLLOW ----------
    def _compute_follow(self):
        follow = {nt: set() for nt in self.nonterminals}
        follow[self.start].add(END)
        changed = True
        while changed:
            changed = False
            for lhs, rhs in self.productions:
                for i, s in enumerate(rhs):
                    if not self.is_nt(s):
                        continue
                    fs = self.first_of_seq(rhs[i + 1:])
                    add = fs - {EPS}
                    if EPS in fs:
                        add |= follow[lhs]
                    if not add <= follow[s]:
                        follow[s] |= add
                        changed = True
        self.follow = follow

    # ---------- LL(1) table ----------
    def ll1_table(self):
        """Return (table, conflicts). table[(A, a)] = production index."""
        cells = defaultdict(list)
        for i, (lhs, rhs) in enumerate(self.productions):
            fs = self.first_of_seq(rhs)
            for a in fs - {EPS}:
                cells[(lhs, a)].append(i)
            if EPS in fs:
                for a in self.follow[lhs]:
                    cells[(lhs, a)].append(i)
        table = {k: v[0] for k, v in cells.items()}
        conflicts = {k: v for k, v in cells.items() if len(v) > 1}
        return table, conflicts

    def is_ll1(self):
        return not self.ll1_table()[1]
