import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True, "grid.alpha": .3,
                     "axes.spines.top": False, "axes.spines.right": False})
p2 = json.load(open("results/part2_metrics.json"))
e1 = json.load(open("results/part3_exp1.json")); e2 = json.load(open("results/part3_exp2.json"))
C = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860"]
H = ["", "//", "..", "xx", "\\\\", "--"]

# 1. whole-program accuracy on G2
m = p2["G2"]["models"]
labels = ["Classical\nLL(1) table", "DT\n(la1)", "DT\n(la1,la2)", "DT\n(full)", "Random\nForest", "MLP"]
vals = [p2["G2"]["classical_LL1"]["sentence_acc"], m["DecisionTree | F1 (nt, la1)"]["sentence_acc"],
        m["DecisionTree | F2 (+la2)"]["sentence_acc"], m["DecisionTree | F3 (full)"]["sentence_acc"],
        m["RandomForest | F3 (full)"]["sentence_acc"], m["MLP | F3 (full)"]["sentence_acc"]]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
b = ax.bar(labels, [v * 100 for v in vals], color=C, edgecolor="k")
for r, h in zip(b, H): r.set_hatch(h)
for r, v in zip(b, vals): ax.text(r.get_x() + r.get_width() / 2, v * 100 + 1.5, f"{v*100:.1f}", ha="center", fontsize=9)
ax.set_ylabel("Programs parsed correctly (%)"); ax.set_ylim(0, 112)
ax.set_title("Grammar G2 (non-LL(1)): whole-program accuracy")
fig.tight_layout(); fig.savefig("figures/fig_accuracy_g2.png", dpi=200); plt.close()

# 2. learning curve
D = e2["D_learning_curve"]; xs = [d["train_programs"] for d in D]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for key, lab, mk, c in [("DT_F1", "Decision tree, lookahead 1", "s", C[3]), ("DT_F2", "Decision tree, lookahead 2", "o", C[0]),
                        ("DT_F3", "Decision tree, full context", "^", C[2]), ("MLP_F3", "MLP, full context", "D", C[1])]:
    ax.plot(xs, [d[key]["sentence_acc"] * 100 for d in D], marker=mk, label=lab, color=c)
ax.set_xscale("log"); ax.set_xlabel("Training programs (log scale)"); ax.set_ylabel("Programs parsed correctly (%)")
ax.set_title("Learning curve on G2"); ax.legend(fontsize=8, loc="center right")
fig.tight_layout(); fig.savefig("figures/fig_learning_curve.png", dpi=200); plt.close()

# 3. error localisation
B = e1["B_error_detection"]
names = [("classical_LL1", "Classical LL(1)"), ("classical_LL1(first-wins)", "Classical LL(1)"), ("ML_pure", "ML (pure)"), ("ML_guarded", "ML (guarded)")]
fig, axs = plt.subplots(1, 2, figsize=(7, 3.3), sharey=True)
for ax, gk in zip(axs, ["G1", "G2"]):
    res = B[gk]["results"]; ks = [k for k in res]
    ls = ["Classical\nLL(1)", "ML\n(pure)", "ML\n(guarded)"]
    bars = ax.bar(ls, [res[k]["exact_error_position"] * 100 for k in ks], color=[C[5], C[1], C[2]], edgecolor="k")
    for r, h in zip(bars, ["", "//", ".."]): r.set_hatch(h)
    for r, k in zip(bars, ks): ax.text(r.get_x() + r.get_width() / 2, res[k]["exact_error_position"] * 100 + 1.5, f"{res[k]['exact_error_position']*100:.0f}", ha="center", fontsize=9)
    ax.set_title(f"{gk}" + (" (LL(1))" if gk == "G1" else " (conflicts)")); ax.set_ylim(0, 112)
axs[0].set_ylabel("Errors reported at exact position (%)")
fig.tight_layout(); fig.savefig("figures/fig_error_localisation.png", dpi=200); plt.close()

# 4. speed (G1)
S = e1["G_speed"]["G1"]
ks = ["classical_LL1", "SLR(1)", "ML guarded", "ML pure, cached", "ML pure, no cache"]
lab = ["LL(1)\ntable", "SLR(1)", "ML\nguarded", "ML pure\ncached", "ML pure\nno cache"]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
b = ax.bar(lab, [S[k]["ms_per_program"] for k in ks], color=C[:5], edgecolor="k"); ax.set_yscale("log")
for r, h in zip(b, H): r.set_hatch(h)
for r, k in zip(b, ks): ax.text(r.get_x() + r.get_width() / 2, S[k]["ms_per_program"] * 1.15, f"{S[k]['ms_per_program']:.2f}", ha="center", fontsize=8)
ax.set_ylabel("Time per program (ms, log scale)"); ax.set_title("Parsing speed on G1 (avg. ~%d tokens)" % 64 if False else "Parsing speed on G1")
fig.tight_layout(); fig.savefig("figures/fig_speed.png", dpi=200); plt.close()

# 5. adaptation
F = e2["F_adaptation"]["curve"]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
xs = [max(f["new_programs"], 1) for f in F]
ax.plot(xs, [f["sentence_acc"] * 100 for f in F], "o-", color=C[0], label="Whole-program accuracy")
ax.plot(xs, [f["conflict_acc"] * 100 for f in F], "s--", color=C[3], label="Accuracy at conflict points")
ax.set_xscale("log"); ax.set_xlabel("New-grammar programs added to G1 training data (0 plotted at 1)")
ax.set_ylabel("%"); ax.set_title("Adapting a G1-trained model to the extended grammar G2"); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("figures/fig_adaptation.png", dpi=200); plt.close()

# 6. feature importance
I = e2["H_importance_conflict_points"]
order = ["la2", "parent", "nt", "la1", "la3", "prev", "brace", "paren"]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.barh(order[::-1], [I[k] * 100 for k in order[::-1]], color=C[0], edgecolor="k")
ax.set_xlabel("Importance (%)"); ax.set_title("Feature importance at LL(1) conflict points (G2)")
fig.tight_layout(); fig.savefig("figures/fig_importance.png", dpi=200); plt.close()
print("figures written")
