This folder contains the code to utilize PANNA networks as potentials in LAMMPS, and a minimal example.

To enable PANNA potentials in LAMMPS, just copy the files pair_panna.* in your LAMMPS src directory, and compile as usual.

The minimal example provided contains a sample LAMMPS input file with the definition of a PANNA potential, i.e.:

pair_style      panna
pair_coeff network panna.in

The parameters of pair_coeff are the name of the folder containing the potential and the name of the parameters file.

The potential files can be generated automatically from a trained network by using the code extract_weights.py with the flag

output_type = LAMMPS

See the code description in the Tutorial for more information (in this case the gvector input file is a mandatory field).