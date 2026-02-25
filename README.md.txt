# MDAnalysis Drug Discovery Toolkit (Public)

A small collection of Python scripts for common molecular dynamics (MD) analyses used in structure-based drug discovery:
- RMSD across replicates
- Per-residue contact frequency (protein–ligand)
- Enrichment factor of components in a shell around a receptor
- Classification of translocation/flip events based on contact occupancy
- Min distance time series
- RMSF per residue
- Protein–ligand residue contact maps

## Notes on data
This repo contains scripts only (no project trajectories/structures). Use your own local topology/trajectory files.

## Installation
pip install -r requirements.txt

## Example usage
python scripts/rmsd_replicates.py
python scripts/min_distance_timeseries.py --top topology.gro --traj rep1_aligned.xtc --lig_sel "resname LIG" --rec_sel "protein"
python scripts/flip_contact_classifier.py --top topology.gro --traj rep1_aligned.xtc --event_log events.log --ligand_resname LIG