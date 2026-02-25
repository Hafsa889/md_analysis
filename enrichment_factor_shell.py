import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda

# Generic component names (edit locally)
SPECIES = ["A", "B", "C", "D"]
CUTOFF_A = 6.0
STRIDE = 10

REPLICATES = [
    {"name": "Replicate 1", "topology": "topology.gro", "trajectory": "rep1_aligned.xtc"},
    {"name": "Replicate 2", "topology": "topology.gro", "trajectory": "rep2_aligned.xtc"},
]

all_rep_means = {s: [] for s in SPECIES}

for rep in REPLICATES:
    print(f"--- Processing {rep['name']} ---")
    if not (os.path.exists(rep["topology"]) and os.path.exists(rep["trajectory"])):
        raise FileNotFoundError("Missing topology/trajectory for this replicate (edit filenames locally).")

    u = mda.Universe(rep["topology"], rep["trajectory"])

    all_sel = " or ".join([f"resname {s}" for s in SPECIES])
    all_components = u.select_atoms(all_sel)
    per_species = {s: u.select_atoms(f"resname {s}") for s in SPECIES}

    total_count = all_components.n_residues
    bulk_counts = {s: per_species[s].n_residues for s in SPECIES}
    bulk_frac = {s: (bulk_counts[s] / total_count) if total_count else 0.0 for s in SPECIES}

    frames_data = []
    for ts in u.trajectory[::STRIDE]:
        shell = u.select_atoms(f"({all_sel}) and around {CUTOFF_A} protein").residues
        n_shell = shell.n_residues
        if n_shell == 0:
            continue

        resnames = shell.resnames
        local_counts = {s: np.count_nonzero(resnames == s) for s in SPECIES}
        local_frac = {s: local_counts[s] / n_shell for s in SPECIES}

        ef = {}
        for s in SPECIES:
            ef[s] = (local_frac[s] / bulk_frac[s]) if bulk_frac[s] > 0 else 0.0
        frames_data.append(ef)

    df = pd.DataFrame(frames_data)
    df.to_csv(f"ef_frames_{rep['name'].replace(' ', '_').lower()}.csv", index=False)

    for s in SPECIES:
        all_rep_means[s].append(float(df[s].mean()) if len(df) else 0.0)

final = {}
for s in SPECIES:
    vals = np.array(all_rep_means[s], dtype=float)
    final[s] = {"mean": float(vals.mean()) if len(vals) else 0.0,
                "std": float(vals.std(ddof=0)) if len(vals) > 1 else 0.0,
                "n": int(len(vals))}

df_final = pd.DataFrame(final).T[["mean", "std", "n"]]
df_final.to_csv("enrichment_summary.csv")

labels = list(final.keys())
means = [final[k]["mean"] for k in labels]
stds = [final[k]["std"] for k in labels]

plt.figure(figsize=(10, 7))
plt.bar(labels, means, yerr=stds, capsize=5)
plt.axhline(y=1, linestyle="--", linewidth=2, label="No enrichment")
plt.title("Enrichment factor in shell around receptor (mean across replicates)")
plt.xlabel("Species")
plt.ylabel("Enrichment factor")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("enrichment_factor.png", dpi=300)
print("Saved: enrichment_factor.png and enrichment_summary.csv")