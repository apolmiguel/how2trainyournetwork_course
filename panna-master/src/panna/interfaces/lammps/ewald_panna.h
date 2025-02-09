/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS

KSpaceStyle(ewald/panna,EwaldPANNA)

#else

#ifndef LMP_EWALD_PANNA_H
#define LMP_EWALD_PANNA_H

#include "kspace.h"
#include <string>
#include <fstream>

namespace LAMMPS_NS {

class EwaldPANNA : public KSpace {
 public:
  EwaldPANNA(class LAMMPS *);
  virtual ~EwaldPANNA();
  void init();
  void setup();
  virtual void settings(int, char **);
  virtual void compute(int, int);
  double memory_usage();

  void compute_group_group(int, int, int);
  virtual void compute_A_dot_v(double *, double *, double *);

  int get_parameters(char*, char *);

  int get_input_line(std::ifstream*, std::string*, std::string*);

protected:
  // kspace parameters
  int kxmax,kymax,kzmax;
  int kcount,kmax,kmax3d,kmax_created;
  double gsqmx,volume;
  int nmax;

  double unitk[3];
  int *kxvecs,*kyvecs,*kzvecs;
  int kxmax_orig,kymax_orig,kzmax_orig;
  double *ug;
  double **eg,**vg, **vg_v3;
  double **ek;
  double *sfacrl,*sfacim,*sfacrl_all,*sfacim_all;
  double *sfacrl_g,*sfacim_g,*sfacrl_all_g,*sfacim_all_g;
  double ***cs,***sn;
  int allocated;
  //long_range 
  double *gaussian_width;
  double *hardness;
  int Nspecies;

  // group-group interactions

  int group_allocate_flag;
  double *sfacrl_A,*sfacim_A,*sfacrl_A_all,*sfacim_A_all;
  double *sfacrl_B,*sfacim_B,*sfacrl_B_all,*sfacim_B_all;

  double rms(int, double, bigint, double);
  virtual void eik_dot_r(double *);
  void coeffs();
  virtual void allocate();
  void deallocate();
  void slabcorr();

  // triclinic

  int triclinic;
  void eik_dot_r_triclinic(double *);
  void coeffs_triclinic();

  // group-group interactions

  void slabcorr_groups(int,int,int);
  void allocate_groups();
  void deallocate_groups();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Cannot use Ewald with 2d simulation

The kspace style ewald cannot be used in 2d simulations.  You can use
2d Ewald in a 3d simulation; see the kspace_modify command.

E: Kspace style requires atom attribute q

The atom style defined does not have these attributes.

E: Cannot use non-periodic boundaries with Ewald

For kspace style ewald, all 3 dimensions must have periodic boundaries
unless you use the kspace_modify command to define a 2d slab with a
non-periodic z dimension.

E: Incorrect boundaries with slab Ewald

Must have periodic x,y dimensions and non-periodic z dimension to use
2d slab option with Ewald.

E: Cannot (yet) use Ewald with triclinic box and slab correction

This feature is not yet supported.

E: KSpace style is incompatible with Pair style

Setting a kspace style requires that a pair style with matching
long-range Coulombic or dispersion components be used.

E: KSpace accuracy must be > 0

The kspace accuracy designated in the input must be greater than zero.

E: Must use 'kspace_modify gewald' for uncharged system

UNDOCUMENTED

E: Cannot (yet) use K-space slab correction with compute group/group for triclinic systems

This option is not yet supported.

*/
