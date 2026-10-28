"""Sanity checks on the generated data."""
from collections import defaultdict
from grammar import Grammar
from generator import build_dataset
from ll1_parser import LL1Parser

g1 = Grammar.from_file("grammars/minilang_ll.txt")
g2 = Grammar.from_file("grammars/minilang_g2.txt")
tab2, conf2 = g2.ll1_table()
print("G2 LL(1)?", g2.is_ll1(), "| conflicting cells:", sorted(conf2))

# 1) G1: generator derivations == classical parser decisions
rows, sents = build_dataset(g1, 300, seed=1)
p = LL1Parser(g1)
by_sid = defaultdict(list)
for r in rows: by_sid[r["sid"]].append(r["label"])
ok = all([d["prod"] for d in p.parse([(t, t) for t in s])[1]["decisions"]] == by_sid[i]
         for i, s in enumerate(sents))
print("G1 generator matches LL(1) parser on all 300 sentences:", ok)

# 2) G2: is (nt, la1, la2) enough to determine the production? (LL(2) check)
rows, sents = build_dataset(g2, 2000, seed=2)
seen = defaultdict(set)
for r in rows: seen[(r["nt"], r["la1"], r["la2"])].add(r["label"])
amb = {k: v for k, v in seen.items() if len(v) > 1}
print("G2 (nt,la1,la2) ambiguous contexts:", len(amb))
