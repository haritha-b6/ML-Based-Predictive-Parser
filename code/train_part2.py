"""Part 2: build datasets, train ML models, compare with the classical LL(1) table."""
import json, time
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from grammar import Grammar
from generator import build_dataset

SEED, N_SENT = 42, 2500
FEATSETS = {"F1 (nt, la1)": ["nt", "la1"],
            "F2 (+la2)": ["nt", "la1", "la2"],
            "F3 (full)": ["nt", "la1", "la2", "la3", "prev", "parent", "brace", "paren"]}

def make_model(kind, cols):
    clf = {"DecisionTree": DecisionTreeClassifier(random_state=SEED),
           "RandomForest": RandomForestClassifier(n_estimators=40, n_jobs=-1, random_state=SEED),
           "MLP": MLPClassifier(hidden_layer_sizes=(64,), max_iter=40, random_state=SEED)}[kind]
    pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), cols)])
    return Pipeline([("pre", pre), ("clf", clf)])

def masked_predict(model, g, X):
    """Choose the most probable production among those valid for the nonterminal."""
    proba = model.predict_proba(X)
    classes = model.named_steps["clf"].classes_
    nt_of_class = np.array([g.productions[c][0] for c in classes])
    out = np.empty(len(X), dtype=int)
    for nt in X["nt"].unique():
        idx = np.where(X["nt"].values == nt)[0]
        sub = proba[idx] * (nt_of_class == nt)
        out[idx] = classes[sub.argmax(axis=1)]
    return out

def sentence_acc(df, pred):
    ok = (df["label"].values == pred)
    return pd.Series(ok).groupby(df["sid"].values).all().mean()

results = {}
for gname, path in [("G1", "grammars/minilang_ll.txt"), ("G2", "grammars/minilang_g2.txt")]:
    g = Grammar.from_file(path)
    rows, sents = build_dataset(g, N_SENT, seed=SEED if gname == "G1" else SEED + 1)
    df = pd.DataFrame(rows)
    df.to_csv(f"data/{gname}_decisions.csv", index=False)
    train_ids = set(range(int(N_SENT * 0.8)))
    tr, te = df[df.sid.isin(train_ids)], df[~df.sid.isin(train_ids)]
    stats = {"sentences": N_SENT, "decisions": len(df), "train_decisions": len(tr),
             "test_decisions": len(te), "avg_tokens": float(np.mean([len(s) for s in sents])),
             "productions": len(g.productions), "ll1": g.is_ll1()}
    # classical baseline: LL(1) table (first production wins on conflicts)
    table, _ = g.ll1_table()
    base = te.apply(lambda r: table.get((r.nt, r.la1), -1), axis=1).values
    stats["classical_LL1"] = {"decision_acc": float((base == te.label.values).mean()),
                              "sentence_acc": float(sentence_acc(te, base))}
    stats["models"] = {}
    print(f"\n=== {gname}: {len(df)} decisions, LL(1)={g.is_ll1()} | classical table acc "
          f"{stats['classical_LL1']['decision_acc']:.4f} (sentence {stats['classical_LL1']['sentence_acc']:.4f})")
    for fname, cols in FEATSETS.items():
        for kind in ["DecisionTree", "RandomForest", "MLP"]:
            if fname != "F3 (full)" and kind != "DecisionTree":
                continue                      # ablation uses the decision tree only
            m = make_model(kind, cols)
            t0 = time.time(); m.fit(tr[cols], tr.label); t_train = time.time() - t0
            t0 = time.time(); pred = masked_predict(m, g, te[cols]); t_inf = time.time() - t0
            rec = {"decision_acc": float((pred == te.label.values).mean()),
                   "sentence_acc": float(sentence_acc(te, pred)),
                   "train_s": round(t_train, 2), "infer_us_per_decision": round(t_inf / len(te) * 1e6, 1)}
            stats["models"][f"{kind} | {fname}"] = rec
            print(f"{kind:13s} {fname:14s} dec={rec['decision_acc']:.4f} sent={rec['sentence_acc']:.4f} "
                  f"train={rec['train_s']}s infer={rec['infer_us_per_decision']}us")
            if fname == "F3 (full)":
                joblib.dump(m, f"models/{gname}_{kind}.joblib")
    # where do the classical errors come from?
    err = te[base != te.label.values].groupby(["nt", "la1"]).size().sort_values(ascending=False)
    stats["classical_errors_by_context"] = {f"{a},{b}": int(c) for (a, b), c in err.head(5).items()}
    results[gname] = stats

json.dump(results, open("results/part2_metrics.json", "w"), indent=2)
print("\nSaved: data/*.csv, models/*.joblib, results/part2_metrics.json")
