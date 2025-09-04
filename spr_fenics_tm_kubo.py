#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"SPR in Kretschmann configuration (TM polarization) with graphene monolayer
FEniCS implementation — angle sweep and reflectivity curve.
This variant uses a Kubo-model for graphene's optical conductivity (intraband + approx interband).
Generated: 2025-09-03
\"\"\"

from dolfin import *
import numpy as np
import math, os, cmath
import csv

# -----------------------
# User-tunable graphene Kubo params
# -----------------------
Ef_eV = 0.0      # Fermi level in eV (Ef = 0 -> undoped)
tau = 1e-13      # relaxation time (s)
T = 300.0        # Temperature (K)

# -----------------------
# Physical constants
# -----------------------
e = 1.602176634e-19
hbar = 1.054571817e-34
eps0 = 8.854187817e-12
mu0 = 4e-7*np.pi
kB = 1.380649e-23
c0 = 299792458.0

# -----------------------
# Optical parameters (wavelength)
# -----------------------
lambda0 = 633e-9                 # wavelength [m]
omega   = 2*np.pi*c0/lambda0    # angular frequency [rad/s]
k0      = 2*np.pi/lambda0        # free-space wavenumber

# -----------------------
# Layers (thickness along z)
# -----------------------
t_prism    = 1.0e-6
t_metal    = 50e-9               # choose 40-60 nm
t_graphene = 0.34e-9
t_sense    = 1.2e-6

# Width in x (core + PMLs)
core_width = 2.4e-6
t_pml      = 0.3e-6
x_min, x_max = -t_pml, core_width + t_pml
z_min, z_max = 0.0, t_prism + t_metal + t_graphene + t_sense

# Refractive indices (materials)
n_prism   = 1.5 + 0j
n_sense   = 1.33 + 0j
n_metal   = 0.19 + 1j*3.59       # Au at 633 nm (given)

# -----------------------
# Kubo conductivity for graphene (in S)
def kubo_sigma(omega_rad, Ef_eV=0.0, tau_s=1e-13, T_K=300.0):
    # convert Ef to Joules
    Ef = Ef_eV * e
    # Intraband (Drude-like) term (finite-T):
    pre_intra = 2.0 * (e**2) * kB * T_K / (np.pi * hbar**2)
    denom = omega_rad + 1j*(1.0/tau_s)
    arg = Ef/(2.0*kB*T_K) if T_K>0 else np.sign(Ef)*np.inf
    # safe log-cosh
    lncosh = np.log(2.0*np.cosh(arg)) if abs(arg) < 50 else abs(arg) + np.log(2.0)
    sigma_intra = 1j * pre_intra * lncosh / denom

    # Interband term (approximate finite-T smoothing)
    hbar_omega = hbar * omega_rad
    Gamma_energy = 2.0 * kB * T_K    # energy broadening [J]
    Gamma = Gamma_energy
    term1 = 0.5
    x_atan = (hbar_omega - 2.0*Ef) / (Gamma + 1e-40)
    atan_part = (1.0/np.pi) * np.arctan(x_atan)
    num = (hbar_omega + 2.0*Ef)**2 + Gamma**2
    den = (hbar_omega - 2.0*Ef)**2 + Gamma**2
    log_part = -1j/(2.0*np.pi) * np.log(num/den + 0j)
    sigma_inter = (e**2/(4.0*hbar)) * (term1 + atan_part + log_part)

    sigma_tot = sigma_intra + sigma_inter
    return sigma_tot

sigma_check = kubo_sigma(omega, Ef_eV=Ef_eV, tau_s=tau, T_K=T)
print("Graphene Kubo sigma (S) at λ={:.0f} nm: {:.3e} + {:.3e}j".format(lambda0*1e9, sigma_check.real, sigma_check.imag))

# Effective permittivity for thin-film model
eps_bg = 1.0
eps_g_eff = eps_bg + 1j * sigma_check / (eps0 * omega * t_graphene)
print("Effective eps_g (complex):", eps_g_eff)

# -----------------------
# Mesh and subdomains (same approach as before)
nx_core = 320
nz      = 400
nx_pml  = 80
nx_total = nx_core + 2*nx_pml

mesh = RectangleMesh(Point(x_min, z_min), Point(x_max, z_max), nx_total, nz)

subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
subdomains.set_all(0)

class Region(SubDomain):
    def __init__(self, x0, x1, z0, z1):
        super().__init__()
        self.x0, self.x1, self.z0, self.z1 = x0, x1, z0, z1
    def inside(self, x, on_boundary):
        return (self.x0 - DOLFIN_EPS <= x[0] <= self.x1 + DOLFIN_EPS and
                self.z0 - DOLFIN_EPS <= x[1] <= self.z1 + DOLFIN_EPS)

z_prism_top    = t_prism
z_metal_top    = t_prism + t_metal
z_graph_top    = t_prism + t_metal + t_graphene
z_sense_top    = z_max

x0_core, x1_core = 0.0, core_width
MARK_PRISM, MARK_METAL, MARK_GRAPH, MARK_SENSE, MARK_PML_L, MARK_PML_R = 1,2,3,4,5,6

Region(x0_core, x1_core, z_min,         z_prism_top).mark(subdomains, MARK_PRISM)
Region(x0_core, x1_core, z_prism_top,   z_metal_top).mark(subdomains, MARK_METAL)
Region(x0_core, x1_core, z_metal_top,   z_graph_top).mark(subdomains, MARK_GRAPH)
Region(x0_core, x1_core, z_graph_top,   z_sense_top).mark(subdomains, MARK_SENSE)
Region(x_min,    0.0,      z_min, z_max).mark(subdomains, MARK_PML_L)
Region(core_width, x_max,  z_min, z_max).mark(subdomains, MARK_PML_R)

V = FunctionSpace(mesh, "CG", 1)
Hy  = TrialFunction(V)
v   = TestFunction(V)

epsr = Function(V)
epsr_array = epsr.vector().get_local()
dof_coords = V.tabulate_dof_coordinates().reshape((-1, 2))

epsr_prism = n_prism**2
epsr_sense = n_sense**2
epsr_metal = n_metal**2
epsr_graph = eps_g_eff

for i, (x,z) in enumerate(dof_coords):
    if x < 0.0:
        base = epsr_prism
    elif x > core_width:
        base = epsr_sense
    else:
        if z <= z_prism_top:
            base = epsr_prism
        elif z <= z_metal_top:
            base = epsr_metal
        elif z <= z_graph_top:
            base = epsr_graph
        else:
            base = epsr_sense
    epsr_array[i] = complex(base).real
epsr.vector()[:] = epsr_array

epsr_imag = Function(V)
arr_im = epsr_imag.vector().get_local()
for i, (x,z) in enumerate(dof_coords):
    if x < 0.0 or x > core_width:
        if x < 0.0:
            xi = abs(x)/t_pml
        else:
            xi = abs(x - core_width)/t_pml
        sigma_max = 50.0
        alpha = sigma_max * (xi**2)
        base_im = alpha
    else:
        if z <= z_prism_top:
            base_im = complex(epsr_prism).imag
        elif z <= z_metal_top:
            base_im = complex(epsr_metal).imag
        elif z <= z_graph_top:
            base_im = complex(epsr_graph).imag
        else:
            base_im = complex(epsr_sense).imag
    arr_im[i] = base_im
epsr_imag.vector()[:] = arr_im

epsr_c = epsr + 1j*epsr_imag

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], x_min, DOLFIN_EPS)
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], x_max, DOLFIN_EPS)
class BottomBoundary(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[1], z_min, DOLFIN_EPS)
class TopBoundary(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[1], z_max, DOLFIN_EPS)

LeftBoundary().mark(boundaries, 1)
RightBoundary().mark(boundaries, 2)
BottomBoundary().mark(boundaries, 3)
TopBoundary().mark(boundaries, 4)

ds_ = Measure("ds", domain=mesh, subdomain_data=boundaries)
dx_ = Measure("dx", domain=mesh, subdomain_data=subdomains)

def Hy_inc_expr(theta_rad):
    s, c = np.sin(theta_rad), np.cos(theta_rad)
    kx = k0 * np.real(n_prism) * s
    kz = k0 * np.real(n_prism) * c
    return Expression('cos(kx*x[0] + kz*x[1]) + i*sin(kx*x[0] + kz*x[1])',
                      degree=4, kx=kx, kz=kz)

def compute_fields(Hy_sol, epsr_complex):
    gradHy = grad(Hy_sol)
    Ex = (1.0/(1j*omega*eps0*epsr_complex)) * gradHy[1]
    Ez = -(1.0/(1j*omega*eps0*epsr_complex)) * gradHy[0]
    return Ex, Ez

Hy_u = Hy
v_u  = v
a_form = (inner(grad(Hy_u), grad(v_u)) - (k0**2)*epsr_c*Hy_u*v_u)*dx

def robin_terms(theta_rad):
    Hy_inc = Hy_inc_expr(theta_rad)
    lhs = -1j*k0*complex(n_prism).real*Hy_u*v_u*ds_(3) -1j*k0*complex(n_sense).real*Hy_u*v_u*ds_(4)
    n = FacetNormal(mesh)
    dHyinc_dn = (Hy_inc.dx(0)*n[0] + Hy_inc.dx(1)*n[1])
    rhs = (dHyinc_dn - 1j*k0*complex(n_prism).real*Hy_inc)*v_u*ds_(3)
    return lhs, rhs, Hy_inc

theta_deg_list = np.linspace(30, 85, 56)
outdir = "out_kubo"
os.makedirs(outdir, exist_ok=True)
results = []
Hy_sol = Function(V)

for theta_deg in theta_deg_list:
    theta = np.deg2rad(theta_deg)
    lhs_robin, rhs_bottom, Hy_inc = robin_terms(theta)
    A = assemble(a_form + lhs_robin)
    b = assemble(rhs_bottom)
    solve(A, Hy_sol.vector(), b, "mumps")

    Ex, Ez = compute_fields(Hy_sol, epsr_c)
    Hy_inc_fun = interpolate(Hy_inc, V)
    dHyinc_dz = Hy_inc.dx(1)
    Ex_inc_fun = project((1.0/(1j*omega*eps0*epsr_prism))*dHyinc_dz, V)

    Sz = -0.5*ufl.re(Ex*ufl.conj(Hy_sol))
    Sz_inc = -0.5*ufl.re(Ex_inc_fun*ufl.conj(Hy_inc_fun))

    P_tot = assemble(Sz*ds_(3))
    P_inc = assemble(Sz_inc*ds_(3))

    R = float(np.real((P_tot - P_inc)/P_inc))
    results.append((theta_deg, max(0.0, R)))

res_idx = int(np.argmin([r for _, r in results]))
theta_res_deg = results[res_idx][0]
R_res = results[res_idx][1]

with open(os.path.join(outdir, "reflectivity.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["theta_deg", "R"])
    for th, rr in results:
        w.writerow([f"{th:.6f}", f"{rr:.8f}"])

print("Resonance angle (deg):", theta_res_deg, "  R_min:", R_res)

try:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([th for th,_ in results], [rr for _,rr in results], '-')
    plt.xlabel("Incident angle θ (deg)")
    plt.ylabel("Reflectivity R")
    plt.title(f"SPR curve (λ={lambda0*1e9:.0f} nm, Au film {t_metal*1e9:.0f} nm, graphene Ef={Ef_eV} eV)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "spr_curve_kubo.png"), dpi=200)
except Exception as e:
    print("Plotting skipped:", e)

theta = np.deg2rad(theta_res_deg)
lhs_robin, rhs_bottom, Hy_inc = robin_terms(theta)
A = assemble(a_form + lhs_robin)
b = assemble(rhs_bottom)
solve(A, Hy_sol.vector(), b, "mumps")

Ex, Ez = compute_fields(Hy_sol, epsr_c)
xdmf = XDMFFile(os.path.join(outdir, "hy_resonance_kubo.xdmf"))
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
Hy_sol.rename("Hy", "Hy")
xdmf.write(Hy_sol)
xdmf.close()

Ex_fun = project(Ex, V); Ex_fun.rename("Ex", "Ex")
xdmf = XDMFFile(os.path.join(outdir, "Efield_resonance_x_kubo.xdmf")); xdmf.write(Ex_fun); xdmf.close()
Ez_fun = project(Ez, V); Ez_fun.rename("Ez", "Ez")
xdmf = XDMFFile(os.path.join(outdir, "Efield_resonance_z_kubo.xdmf")); xdmf.write(Ez_fun); xdmf.close()

print("Done. Outputs in ./", outdir)
