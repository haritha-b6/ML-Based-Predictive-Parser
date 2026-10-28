from grammar import Grammar
from lexer import tokenize
from ll1_parser import LL1Parser
from slr_parser import SLRParser

gl = Grammar.from_file("grammars/minilang_ll.txt")
gr = Grammar.from_file("grammars/minilang_lr.txt")

print("== LL grammar ==")
print("productions:", len(gl.productions), "| nonterminals:", len(gl.nonterminals),
      "| terminals:", len(gl.terminals), "| LL(1):", gl.is_ll1())
for nt in ("Stmt", "Expr", "ExprT", "StmtList"):
    print(f"FIRST({nt}) =", sorted(gl.first[nt]), "| FOLLOW =", sorted(gl.follow[nt]))
ll = LL1Parser(gl)
print("LL(1) table cells:", len(ll.table))

print("\n== LR grammar ==")
slr = SLRParser(gr)
print("SLR states:", len(slr.states), "| ACTION entries:", len(slr.action),
      "| conflicts:", len(slr.conflicts))

good = [
    "x = 1 + 2 * 3;",
    "a = (b + c) * d - 4 / e;",
    "if (x < 10) { y = x + 1; } else { y = 0; }",
    "while (i != 5) { print(i); i = i + 1; }",
    "",
]
bad = [
    "x = ;", "x = 1 + ;", "if (x < ) { }", "while x < 3 { }",
    "x = (1 + 2;", "print(x)", "else { }", "x = 1 2;",
]
print("\n== Acceptance test (LL(1) vs SLR) ==")
ok = True
for src in good + bad:
    t = tokenize(src)
    a, b = ll.accepts(t), slr.accepts(t)
    expect = src in good
    flag = "OK " if (a == b == expect) else "BAD"
    ok &= (a == b == expect)
    print(f"[{flag}] LL={a!s:5} SLR={b!s:5}  {src!r}")
print("\nALL CONSISTENT:", ok)

tree, info = ll.parse(tokenize("x = 1 + 2 * 3;"))
print("\nLL(1): steps =", info["steps"], "| decisions =", len(info["decisions"]), "| tree nodes =", tree.size())
tree, info = slr.parse(tokenize("x = 1 + 2 * 3;"))
print("SLR  : steps =", info["steps"], "| reductions =", len(info["decisions"]), "| tree nodes =", tree.size())
print("\nFirst expansions (trace):")
ll.parse(tokenize("x = 1 + 2;"), trace=True)
