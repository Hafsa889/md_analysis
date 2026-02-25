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
    ap.add_argument("--rec_sel", default="protein")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out_csv", default="min_distance.csv")
    ap.add_argument("--out_png", default="min_distance.png")
    args = ap.parse_args()

    u = mda.Universe(args.top, args.traj)
    lig = u.select_atoms(args.lig_sel).select_atoms("not name H*")
    rec = u.select_atoms(args.rec_sel).select_atoms("not name H*")

    times_ns = []
    dmins = []

    for ts in u.trajectory[::args.stride]:
        d = distance_array(lig.positions, rec.positions)
        dmin = float(np.min(d))
        times_ns.append(ts.time / 1000.0)  # ps -> ns (MDAnalysis convention)
        dmins.append(dmin)

    arr = np.column_stack([times_ns, dmins])
    np.savetxt(args.out_csv, arr, delimiter=",", header="time_ns,min_distance_A", comments="")

    plt.figure(figsize=(10, 5))
    plt.plot(times_ns, dmins, linewidth=1.5)
    plt.xlabel("Time (ns)")
    plt.ylabel("Min distance (Å)")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    print(f"Saved: {args.out_csv}, {args.out_png}")

if __name__ == "__main__":
    main()