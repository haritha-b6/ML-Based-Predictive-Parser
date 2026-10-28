"""Part 3, batch 1: end-to-end correctness, error detection/localisation, recovery, speed."""
import json, random, time, warnings
warnings.filterwarnings("ignore")
import joblib
from grammar import Grammar
from generator import Generator
from ll1_parser import LL1Parser, ParseError
from slr_parser import SLRParser
from ml_parser import MLPredictiveParser
from oracle import Oracle

R = {}
g1 = Grammar.from_file("grammars/minilang_ll.txt")
g1r = Grammar.from_file("grammars/minilang_lr.txt")
g2 = Grammar.from_file("grammars/minilang_g2.txt")
M1, M2 = joblib.load("models/G1_DecisionTree.joblib"), joblib.load("models/G2_DecisionTree.joblib")
ml1 = MLPredictiveParser(g1, M1, mode="pure")
ml2 = MLPredictiveParser(g2, M2, mode="pure")
hy1 = MLPredictiveParser(g1, M1, mode="guarded")
hy2 = MLPredictiveParser(g2, M2, mode="guarded")
cl1 = LL1Parser(g1)
cl2 = LL1Parser(g2, allow_conflicts=True)
slr = SLRParser(g1r)
tup = lambda toks: [(t, t) for t in toks]

def ser(n):   # serialise tree for equality tests
    return (n.label, tuple(ser(c) for c in n.children))

# fresh test programs (seeds different from training: 42/43)
test1 = [Generator(g1, 9001 + i).sentence()[0] for i in range(400)]
test2 = [Generator(g2, 9501 + i).sentence()[0] for i in range(400)]

# ---------- A. end-to-end correctness on valid programs ----------
def acc_rate(parser, sents):
    ok = 0
    for s in sents:
        try: parser.parse(tup(s)); ok += 1
        except ParseError: pass
    return ok / len(sents)

A = {"G1": {"classical_LL1": acc_rate(cl1, test1), "ML_pure": acc_rate(ml1, test1), "ML_guarded": acc_rate(hy1, test1)},
     "G2": {"classical_LL1(first-wins)": acc_rate(cl2, test2), "ML_pure": acc_rate(ml2, test2), "ML_guarded": acc_rate(hy2, test2)}}
same_tree = sum(ser(cl1.parse(tup(s))[0]) == ser(ml1.parse(tup(s))[0]) for s in test1) / len(test1)
A["G1"]["identical_parse_tree_vs_classical"] = same_tree
R["A_valid_program_acceptance"] = A
print("A:", json.dumps(A, indent=1))

# ---------- B. error detection / localisation on mutated programs ----------
rng = random.Random(7)
def mutate(toks, terminals):
    t = list(toks); i = rng.randrange(len(t)); op = rng.choice(["del", "ins", "sub"])
    if op == "del": del t[i]
    elif op == "ins": t.insert(i, rng.choice(terminals))
    else: t[i] = rng.choice(terminals)
    return t

def eval_errors(g, parsers, sents, n=600):
    orc = Oracle(g); pool = [s for s in sents if 5 <= len(s) <= 45]
    cases = []
    while len(cases) < n:
        m = mutate(rng.choice(pool), g.terminals)
        ok, k = orc.analyse(m)
        cases.append((m, ok, k))
    out = {}
    n_invalid = sum(1 for _, ok, _ in cases if not ok)
    for name, p in parsers.items():
        agree = false_acc = false_rej = loc = 0
        for m, ok, k in cases:
            try: p.parse(tup(m)); res, pos = True, None
            except ParseError as e: res, pos = False, e.pos
            agree += (res == ok)
            false_acc += (res and not ok)
            false_rej += ((not res) and ok)
            if not ok and not res and pos == k: loc += 1
        out[name] = {"agreement_with_oracle": agree / n, "false_accepts": false_acc,
                     "false_rejects": false_rej,
                     "exact_error_position": loc / max(n_invalid, 1)}
    return {"mutants": n, "invalid_mutants": n_invalid, "results": out}

B = {"G1": eval_errors(g1, {"classical_LL1": cl1, "ML_pure": ml1, "ML_guarded": hy1}, test1),
     "G2": eval_errors(g2, {"classical_LL1(first-wins)": cl2, "ML_pure": ml2, "ML_guarded": hy2}, test2)}
R["B_error_detection"] = B
print("B:", json.dumps(B, indent=1))

# ---------- C. panic-mode recovery with several injected errors ----------
def recovery_eval(g, ml, sents, n=200):
    pool = [s for s in sents if 25 <= len(s) <= 80]
    rng2 = random.Random(11); orc = Oracle(g)
    stats = {"programs": 0, "no_crash": 0, "detected_any": 0, "reported_errors": 0,
             "injected_errors": 0, "recovered_to_end": 0}
    while stats["programs"] < n:
        s = rng2.choice(pool)
        m = list(s); k_inj = rng2.choice([1, 2, 3])
        for _ in range(k_inj):
            i = rng2.randrange(len(m)); m[i] = rng2.choice(g.terminals)
        if orc.analyse(m)[0]:
            continue                                # mutation accidentally valid
        stats["programs"] += 1; stats["injected_errors"] += k_inj
        try:
            _, info = ml.parse(tup(m), recover=True)
            stats["no_crash"] += 1
            stats["detected_any"] += bool(info["errors"])
            stats["reported_errors"] += len(info["errors"])
            stats["recovered_to_end"] += (max(info["errors"]) < len(m)) if info["errors"] else 0
        except Exception:
            pass
    return stats
R["C_recovery_G2"] = {"ML_pure": recovery_eval(g2, ml2, test2), "ML_guarded": recovery_eval(g2, hy2, test2)}
print("C:", R["C_recovery_G2"])

# ---------- G. speed ----------
def time_parser(f, sents, reps=3):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        for s in sents: f(tup(s))
        best = min(best, time.perf_counter() - t0)
    return best
def row(v, S):
    return {"ms_per_program": v / len(S) * 1e3, "tokens_per_sec": sum(len(x) for x in S) / v}
S1, S2 = test1[:200], test2[:200]
nc1 = MLPredictiveParser(g1, M1, cache=False, mode="pure")
nc2 = MLPredictiveParser(g2, M2, cache=False, mode="guarded")
for p in (ml1, ml2, hy1, hy2): p.cache.clear(); p.hits = p.misses = p.ml_calls = 0
G1s = {"classical_LL1": row(time_parser(cl1.parse, S1), S1), "SLR(1)": row(time_parser(slr.parse, S1), S1),
       "ML pure, no cache": row(time_parser(nc1.parse, S1[:30], reps=1) * len(S1) / 30, S1),
       "ML pure, cached": row(time_parser(ml1.parse, S1), S1),
       "ML guarded": row(time_parser(hy1.parse, S1), S1)}
G1s["pure cache hit rate"] = ml1.hits / max(ml1.hits + ml1.misses, 1)
G1s["guarded model calls"] = hy1.ml_calls
G2s = {"ML guarded, no cache": row(time_parser(nc2.parse, S2[:30], reps=1) * len(S2) / 30, S2),
       "ML guarded, cached": row(time_parser(hy2.parse, S2), S2),
       "ML pure, cached": row(time_parser(ml2.parse, S2), S2)}
G2s["guarded cache hit rate"] = hy2.hits / max(hy2.hits + hy2.misses, 1)
G2s["guarded model calls (cold)"] = hy2.ml_calls
R["G_speed"] = {"G1": G1s, "G2": G2s}
print("G:", json.dumps(R["G_speed"], indent=1))
json.dump(R, open("results/part3_exp1.json", "w"), indent=2)
