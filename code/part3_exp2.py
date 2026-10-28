"""Part 3, batch 2: learning curve, length generalisation, grammar adaptation, importance."""
import json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from common import *
from grammar import Grammar
from generator import build_dataset

R = {}
g1 = Grammar.from_file("grammars/minilang_ll.txt")
g2 = Grammar.from_file("grammars/minilang_g2.txt")
d2 = pd.read_csv("data/G2_decisions.csv", keep_default_na=False)
d2["brace"] = d2["brace"].astype(str); d2["paren"] = d2["paren"].astype(str)
tr_all, te = d2[d2.sid < 2000], d2[d2.sid >= 2000]
conf = lambda df: ((df.nt == "Stmt") | (df.nt == "Factor")) & (df.la1 == "id")

def score(model, cols, test, g=g2, str_labels=False, label_col="label"):
    pred = masked_predict(model, g, test[cols], str_labels)
    ok = (pred == test[label_col].values)
    c = conf(test).values
    return {"decision_acc": float(ok.mean()), "sentence_acc": sent_acc(test.sid.values, ok),
            "conflict_acc": float(ok[c].mean())}

# ---------- D. learning curve (G2) ----------
D = []
for n in [5, 10, 25, 50, 100, 250, 500, 2000]:
    tr = tr_all[tr_all.sid < n]
    row = {"train_programs": n, "train_decisions": len(tr)}
    for name, kind, cols in [("DT_F1", "DecisionTree", F1), ("DT_F2", "DecisionTree", F2),
                             ("DT_F3", "DecisionTree", F3), ("MLP_F3", "MLP", F3)]:
        m = make_model(kind, cols).fit(tr[cols], tr.label)
        row[name] = score(m, cols, te)
    D.append(row)
    print(n, {k: round(v["sentence_acc"], 3) for k, v in row.items() if isinstance(v, dict)})
R["D_learning_curve"] = D

# ---------- E. generalisation to longer programs ----------
from generator import Generator, features
def make(g, n, seed, max_tokens, min_len=0):
    gen = Generator(g, seed, max_tokens=max_tokens); rows = []; sid = 0
    while sid < n:
        toks, ev = gen.sentence()
        if len(toks) < min_len: continue
        for r in features(toks, ev): r["sid"] = sid; rows.append(r)
        sid += 1
    return pd.DataFrame(rows)
short_tr = make(g2, 1500, 100, 30)
long_te = make(g2, 150, 200, 400, min_len=100)
short_te = make(g2, 300, 300, 30)
E = {"train_avg_decisions_per_program": float(short_tr.groupby("sid").size().mean()),
     "long_test_avg_decisions_per_program": float(long_te.groupby("sid").size().mean())}
for name, kind, cols in [("DT_F2", "DecisionTree", F2), ("DT_F3", "DecisionTree", F3), ("MLP_F3", "MLP", F3)]:
    m = make_model(kind, cols).fit(short_tr[cols], short_tr.label)
    E[name] = {"short_test": score(m, cols, short_te), "long_test": score(m, cols, long_te)}
R["E_length_generalisation"] = E
print("E:", json.dumps(E, indent=1))

# ---------- F. adaptation: G1 -> G2 ----------
d1 = pd.read_csv("data/G1_decisions.csv", keep_default_na=False)
d1["brace"] = d1["brace"].astype(str); d1["paren"] = d1["paren"].astype(str)
d1["slabel"] = d1["label"].map(g1.prod_str)
d2["slabel"] = d2["label"].map(g2.prod_str); te = d2[d2.sid >= 2000]; tr_all = d2[d2.sid < 2000]
base = d1[d1.sid < 2000]
Fres = []
for k in [0, 5, 10, 25, 50, 100, 500]:
    mix = pd.concat([base, tr_all[tr_all.sid < k]]) if k else base
    t0 = time.time(); m = make_model("DecisionTree", F3).fit(mix[F3], mix.slabel); dt = time.time() - t0
    s = score(m, F3, te, g2, True, "slabel")
    s.update({"new_programs": k, "retrain_s": round(dt, 2)}); Fres.append(s)
    print("F", k, {a: round(b, 3) if isinstance(b, float) else b for a, b in s.items()})
t0 = time.time(); Grammar.from_file("grammars/minilang_g2.txt").ll1_table(); rebuild = time.time() - t0
R["F_adaptation"] = {"curve": Fres, "classical_table_rebuild_s": rebuild,
                     "classical_note": "table rebuild is instant but G2 has 2 LL(1) conflicts needing manual left-factoring"}
print("table rebuild s:", rebuild)

# ---------- H. feature importance at conflict points ----------
c_tr = tr_all[conf(tr_all)]
m = make_model("DecisionTree", F3).fit(c_tr[F3], c_tr.label)
names = m.named_steps["pre"].named_transformers_["oh"].get_feature_names_out(F3)
imp = m.named_steps["clf"].feature_importances_
grp = {}
for nme, v in zip(names, imp):
    key = nme.split("_")[0]; grp[key] = grp.get(key, 0) + float(v)
R["H_importance_conflict_points"] = dict(sorted(grp.items(), key=lambda kv: -kv[1]))
print("H:", R["H_importance_conflict_points"])
json.dump(R, open("results/part3_exp2.json", "w"), indent=2)
