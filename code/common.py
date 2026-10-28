import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

SEED = 42
F1, F2 = ["nt", "la1"], ["nt", "la1", "la2"]
F3 = ["nt", "la1", "la2", "la3", "prev", "parent", "brace", "paren"]

def make_model(kind, cols):
    clf = {"DecisionTree": DecisionTreeClassifier(random_state=SEED),
           "RandomForest": RandomForestClassifier(n_estimators=40, n_jobs=-1, random_state=SEED),
           "MLP": MLPClassifier(hidden_layer_sizes=(64,), max_iter=40, random_state=SEED)}[kind]
    pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), cols)])
    return Pipeline([("pre", pre), ("clf", clf)])

def masked_predict(model, g, X, labels_are_str=False):
    proba = model.predict_proba(X)
    classes = model.named_steps["clf"].classes_
    if labels_are_str:
        cnt = np.array([c.split(" -> ")[0] for c in classes], dtype=object)
    else:
        cnt = np.array([g.productions[c][0] for c in classes], dtype=object)
    out = np.empty(len(X), dtype=object if labels_are_str else int)
    for nt in X["nt"].unique():
        idx = np.where(X["nt"].values == nt)[0]
        m = (cnt == nt)
        if not m.any():
            out[idx] = classes[0]; continue
        out[idx] = classes[(proba[idx] * m).argmax(axis=1)]
    return out

def sent_acc(sids, ok):
    return float(pd.Series(ok).groupby(np.asarray(sids)).all().mean())
