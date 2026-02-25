import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

REPLICATES = [
    {"name": "Replicate 1", "topology": "topology.gro", "trajectory": "rep1_aligned.xtc"},
    {"name": "Replicate 2", "topology": "topology.gro", "trajectory": "rep2_aligned.xtc"},
]

LIGAND_RESNAME = "LIG"          # anonymized
PROTEIN_SELECTION = "protein"
CONTACT_CUTOFF_A = 6.0          # Å
STRIDE = 10
TOP_N = 20

total_counts = Counter()
total_frames = 0

print("--- Starting residue contact frequency analysis ---")

for rep in REPLICATES:
    print(f"\n--- {rep['name']} ---")
    u = mda.Universe(rep["topology"], rep["trajectory"])

    for ts in u.trajectory[::STRIDE]:
        contacting = u.select_atoms(
            f"({PROTEIN_SELECTION}) and around {CONTACT_CUTOFF_A} resname {LIGAND_RESNAME}"
        ).residues

        # Unique per frame (avoid counting same residue multiple times in same frame)
        unique = {f"{r.resname}-{r.resid}" for r in contacting}
        total_counts.update(unique)
        total_frames += 1

print(f"\nProcessed frames: {total_frames}")

freq = {k: v / total_frames for k, v in total_counts.items()} if total_frames else {}
top = Counter(freq).most_common(TOP_N)

labels = [x[0] for x in top]
values = [x[1] for x in top]

plt.figure(figsize=(10, 12))
y = np.arange(len(labels))
plt.barh(y, values, align="center")
plt.yticks(y, labels)
plt.gca().invert_yaxis()
plt.xlabel("Contact Frequency (occupancy)")
plt.ylabel("Residue")
plt.title(f"Top {TOP_N} contacting residues (cutoff={CONTACT_CUTOFF_A} Å)")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("top_residue_contacts.png", dpi=300)
print("Saved: top_residue_contacts.png")