import argparse
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--lig_sel", default="resname LIG")
    ap.add_argument("--prot_sel", default="protein")
    ap.add_argument("--cutoff_A", type=float, default=4.5)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out_npy", default="contact_map.npy")
    ap.add_argument("--out_png", default="contact_map.png")
    args = ap.parse_args()

    u = mda.Universe(args.top, args.traj)
    lig = u.select_atoms(args.lig_sel).select_atoms("not name H*")
    prot = u.select_atoms(args.prot_sel).select_atoms("not name H*")

    prot_res = list(prot.residues)
    lig_res = list(lig.residues)

    M = np.zeros((len(prot_res), len(lig_res)), dtype=float)
    nframes = 0

    for ts in u.trajectory[::args.stride]:
        nframes += 1
        for i, pr in enumerate(prot_res):
            pra = pr.atoms.select_atoms("not name H*")
            if pra.n_atoms == 0:
                continue
            for j, lr in enumerate(lig_res):
                lra = lr.atoms.select_atoms("not name H*")
                if lra.n_atoms == 0:
                    continue
                dmin = np.min(distance_array(pra.positions, lra.positions))
                if dmin < args.cutoff_A:
                    M[i, j] += 1

    if nframes > 0:
        M /= nframes  # occupancy 0..1

    np.save(args.out_npy, M)

    plt.figure(figsize=(8, 6))
    plt.imshow(M, aspect="auto", origin="lower")
    plt.colorbar(label="Contact occupancy")
    plt.xlabel("Ligand residue index")
    plt.ylabel("Protein residue index")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    print(f"Saved: {args.out_npy}, {args.out_png}")

if __name__ == "__main__":
    main()