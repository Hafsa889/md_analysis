#!/usr/bin/env python3
"""
flip_contact_classifier.py

Classify translocation (flip-flop) events as "receptor-driven" based on contact occupancy
in a ±time window around each event.

Contact definition: any heavy-atom pair within CONTACT_CUTOFF Å between ligand and receptor.
A flip is called receptor-driven if occupancy >= MIN_OCCUPANCY within the window.

This script is intentionally generic and does not assume a specific biological system.
"""

import re
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict
import sys
import numpy as np

import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

# Generic log line example:
# EVENT: ligand_resid 123 (LIG) replicate 1 frame 4567 direction UPPER
EVENT_RE = re.compile(
    r"EVENT:\s+ligand_resid\s+(?P<lig>\d+)\s+\((?P<resname>\w+)\)\s+replicate\s+(?P<rep>\d+)\s+frame\s+(?P<frame>\d+)\s+direction\s+(?P<dir>UPPER|LOWER)",
    re.IGNORECASE
)

@dataclass
class Event:
    ligand_resid: int
    ligand_resname: str
    replicate: int
    frame: int
    direction: str

def parse_event_log(path: str) -> List[Event]:
    events: List[Event] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = EVENT_RE.search(line)
            if m:
                events.append(Event(
                    ligand_resid=int(m.group("lig")),
                    ligand_resname=m.group("resname"),
                    replicate=int(m.group("rep")),
                    frame=int(m.group("frame")),
                    direction=m.group("dir").upper(),
                ))
    return events

def frames_in_window(center: int, window_ps: float, dt_ps: float, n_frames: int) -> np.ndarray:
    half = int(round(window_ps / dt_ps))
    start = max(0, center - half)
    end = min(n_frames - 1, center + half)
    return np.arange(start, end + 1, dtype=int)

def contact_occupancy(u: mda.Universe,
                      ligand_resid: int,
                      receptor_sel: str,
                      cutoff_A: float,
                      frames: np.ndarray) -> Tuple[float, Dict[Tuple[str, int], int]]:
    ligand = u.select_atoms(f"resid {ligand_resid}")
    if ligand.n_atoms == 0:
        raise ValueError(f"Ligand resid {ligand_resid} not found.")

    ligand = ligand.select_atoms("not name H*")
    receptor = u.select_atoms(receptor_sel)
    if receptor.n_atoms == 0:
        raise ValueError(f"Receptor selection '{receptor_sel}' yielded 0 atoms.")
    receptor = receptor.select_atoms("not name H*") if receptor.select_atoms("not name H*").n_atoms else receptor

    receptor_residues = list(receptor.residues)
    residue_frame_counts: Dict[Tuple[str, int], int] = {}
    n_contact = 0

    for fr in frames:
        u.trajectory[fr]
        touched = []

        for r in receptor_residues:
            ra = r.atoms
            ra = ra.select_atoms("not name H*") if ra.select_atoms("not name H*").n_atoms else ra
            if ra.n_atoms == 0:
                continue
            dmin = np.min(distance_array(ligand.positions, ra.positions))
            if dmin < cutoff_A:
                touched.append((r.resname, int(r.resid)))

        if touched:
            n_contact += 1
            for key in set(touched):
                residue_frame_counts[key] = residue_frame_counts.get(key, 0) + 1

    occ = n_contact / float(len(frames)) if len(frames) else 0.0
    return occ, residue_frame_counts

def estimate_dt_ps(u: mda.Universe) -> float:
    times = []
    for ts in u.trajectory[:10]:
        times.append(ts.time)
    if len(times) >= 2:
        return (times[-1] - times[0]) / (len(times) - 1)
    return 1000.0  # fallback

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--event_log", required=True)
    ap.add_argument("--receptor_sel", default="protein")
    ap.add_argument("--ligand_resname", default="LIG")
    ap.add_argument("--time_window_ps", type=float, default=20000.0)
    ap.add_argument("--contact_cutoff_A", type=float, default=4.5)
    ap.add_argument("--min_occupancy", type=float, default=0.30)
    ap.add_argument("--out_csv", default="flip_contact_classification.csv")
    ap.add_argument("--report_txt", default=None)
    ap.add_argument("--dt_ps_override", type=float, default=None)
    args = ap.parse_args()

    events = parse_event_log(args.event_log)
    if not events:
        print("No events parsed. Check your log format vs regex.", file=sys.stderr)
        sys.exit(2)

    u = mda.Universe(args.top, args.traj)
    dt_ps = args.dt_ps_override if args.dt_ps_override else estimate_dt_ps(u)
    n_frames = len(u.trajectory)

    import csv
    rows = []

    rpt = None
    if args.report_txt:
        rpt = open(args.report_txt, "w", encoding="utf-8")
        rpt.write("# Flip contact classification report\n")
        rpt.write(f"# window_ps={args.time_window_ps} cutoff_A={args.contact_cutoff_A} min_occ={args.min_occupancy}\n\n")

    for ev in events:
        if ev.ligand_resname.upper() != args.ligand_resname.upper():
            continue

        frs = frames_in_window(ev.frame, args.time_window_ps, dt_ps, n_frames)
        occ, res_counts = contact_occupancy(u, ev.ligand_resid, args.receptor_sel, args.contact_cutoff_A, frs)
        receptor_driven = occ >= args.min_occupancy

        top = sorted(res_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        top_str = "; ".join([f"{r}:{i}({c}/{len(frs)})" for (r, i), c in top])

        rows.append({
            "replicate": ev.replicate,
            "ligand_resid": ev.ligand_resid,
            "direction": ev.direction,
            "event_frame": ev.frame,
            "contact_occupancy": f"{occ:.3f}",
            "receptor_driven": receptor_driven,
            "top_contact_residues": top_str
        })

        if rpt:
            rpt.write(f"Event resid={ev.ligand_resid} dir={ev.direction} frame={ev.frame}\n")
            rpt.write(f"  occupancy={occ:.3f} -> {'RECEPTOR-DRIVEN' if receptor_driven else 'OTHER'}\n")
            rpt.write(f"  top: {top_str}\n\n")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["replicate","ligand_resid","direction","event_frame","contact_occupancy","receptor_driven","top_contact_residues"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if rpt:
        rpt.close()

    print(f"Saved: {args.out_csv}")
    if args.report_txt:
        print(f"Saved: {args.report_txt}")

if __name__ == "__main__":
    main()