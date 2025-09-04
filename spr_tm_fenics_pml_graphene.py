# spr_tm_fenics_pml_graphene.py
# TM-SPR (Kretschmann) in FEniCS with:
#   - x-PML via complex coordinate stretch s_x(x)=1+i*alpha(x)
#   - graphene monolayer as surface conductivity on the metal/top interface
#   - reflectance R(theta) from Poynting flux, with FWHM and Q
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

set_log_level(30)

# ---------- Helpers ----------
def poynting_Sy_density(ur, ui, inv_eps_r, inv_eps_i, omega, mu0):
    # Sy = (1/(2 ω μ0)) * Im( (1/eps) * ∂y H * conj(H) )
    dyr = ur.dx(1); dyi = ui.dx(1)
    Pr = inv_eps_r*dyr - inv_eps_i*dyi
    Pi = inv_eps_r*dyi + inv_eps_i*dyr
    Sy = (1.0/(2.0*omega*mu0)) * (-Pr*ui + Pi*ur)
    return Sy

def fwhm(x, y):
    ymin = np.min(y); ymax = np.max(y)
    target = ymin + 0.5*(ymax - ymin)
    idx = np.where(np.diff(np.sign(y - target)))[0]
    if len(idx) < 2:
        return None
    def x_at(i):
        x0, x1 = x[i], x[i+1]; y0, y1 = y[i], y[i+1]
        return x0 + (target - y0)*(x1 - x0)/((y1 - y0) + 1e-30)
    xL, xR = x_at(idx[0]), x_at(idx[-1])
    return abs(xR - xL)

# ---------- Constants & setup ----------
eps0 = 8.854187817e-12
mu0  = 4.0e-7*np.pi

lam   = 633e-9
k0    = 2*np.pi/lam
omega = 2*np.pi*3e8/lam

n_prism, n_sense = 1.5, 1.33
n_metal, k_metal = 0.19, 3.59
eps_prism, eps_sense = n_prism**2, n_sense**2
eps_metal = (n_metal + 1j*k_metal)**2

# Graphene (placeholder σ0; swap for Kubo if you have params)
sigma_s = 6.085e-5  # S  (≈ e^2/(4ħ))

t_prism = 1.00e-6
t_metal = 50e-9
t_sense = 1.20e-6
W_core  = 2.4e-6
t_pml   = 0.3e-6
W_total = W_core + 2*t_pml
H_total = t_prism + t_metal + t_sense  # graphene is interface only

nx_core, ny = 240, 220
mesh = RectangleMesh(Point(0.0, 0.0), Point(W_total, H_total),
                     int(nx_core*W_total/W_core), ny)

x, y = SpatialCoordinate(mesh)

# ---------- Cell regions ----------
domains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

class Prism(SubDomain):
    def inside(self, p, on): return p[1] <= t_prism + DOLFIN_EPS
class Metal(SubDomain):
    def inside(self, p, on): return (p[1] > t_prism - DOLFIN_EPS) and (p[1] <= t_prism + t_metal + DOLFIN_EPS)
class Sensing(SubDomain):
    def inside(self, p, on): return p[1] > t_prism + t_metal - DOLFIN_EPS
class PML_Left(SubDomain):
    def inside(self, p, on): return p[0] <= t_pml + DOLFIN_EPS
class PML_Right(SubDomain):
    def inside(self, p, on): return p[0] >= W_total - t_pml - DOLFIN_EPS

PRISM, METAL, SENSE, PMLL, PMLR = 1, 2, 4, 5, 6
Prism().mark(domains, PRISM)
Metal().mark(domains, METAL)
Sensing().mark(domains, SENSE)
PML_Left().mark(domains, PMLL)
PML_Right().mark(domains, PMLR)

for cell in cells(mesh):
    if domains[cell] == 0:
        domains[cell] = SENSE

print("[INFO] Region tags present:", np.unique(domains.array()))
dxm = Measure('dx', domain=mesh, subdomain_data=domains)

# ---------- Graphene interface facets ----------
facet_tags = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
class GrapheneInterface(SubDomain):
    def inside(self, p, on_boundary): return near(p[1], t_prism + t_metal, DOLFIN_EPS*10)
IGRAPH = 10
GrapheneInterface().mark(facet_tags, IGRAPH)
dSg = Measure('dS', domain=mesh, subdomain_data=facet_tags)

# ---------- Materials (DG0) ----------
DG0 = FunctionSpace(mesh, "DG", 0)
eps_r = Function(DG0); eps_i = Function(DG0)

cell2sub = domains.array()
er = eps_r.vector().get_local(); ei = eps_i.vector().get_local()
for cid, tag in enumerate(cell2sub):
    tag = int(tag)
    if tag == PRISM:   val = complex(eps_prism, 0.0)
    elif tag == METAL: val = eps_metal
    else:              val = complex(eps_sense, 0.0)  # SENSE + PML stripes use sensing medium
    er[cid], ei[cid] = np.real(val), np.imag(val)
eps_r.vector()[:] = er; eps_i.vector()[:] = ei

den = project(eps_r*eps_r + eps_i*eps_i, DG0)
inv_eps_r = project( eps_r/den, DG0 )
inv_eps_i = project(-eps_i/den, DG0 )

# ---------- x-PML stretch s_x = 1 + i*alpha(x) ----------
sx_r, sx_i = Function(DG0), Function(DG0)
sxr = sx_r.vector().get_local(); sxi = sx_i.vector().get_local()
alpha_max = 2.0

for c in cells(mesh):
    cid = c.index()
    tag = int(cell2sub[cid])
    X = c.midpoint().x()
    if tag == PMLL:
        xi = (t_pml - X)/t_pml; xi = max(0.0, min(1.0, xi))
        sxi[cid] = alpha_max*xi*xi; sxr[cid] = 1.0
    elif tag == PMLR:
        xi = (X - (W_total - t_pml))/t_pml; xi = max(0.0, min(1.0, xi))
        sxi[cid] = alpha_max*xi*xi; sxr[cid] = 1.0
    else:
        sxi[cid] = 0.0; sxr[cid] = 1.0
sx_r.vector()[:] = sxr; sx_i.vector()[:] = sxi

sx_norm = project(sx_r*sx_r + sx_i*sx_i, DG0)
inv_sx_r = project( sx_r/sx_norm, DG0 )
inv_sx_i = project(-sx_i/sx_norm, DG0 )

Cx_r = project(inv_eps_r*inv_sx_r - inv_eps_i*inv_sx_i, DG0)
Cx_i = project(inv_eps_r*inv_sx_i + inv_eps_i*inv_sx_r, DG0)
Cy_r = project(inv_eps_r*sx_r     - inv_eps_i*sx_i,     DG0)
Cy_i = project(inv_eps_r*sx_i     + inv_eps_i*sx_r,     DG0)

M_r = project(-k0**2 * sx_r, DG0)
M_i = project(-k0**2 * sx_i, DG0)

# ---------- Function spaces ----------
P2 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
ME = FunctionSpace(mesh, MixedElement([P2, P2]))
u = TrialFunction(ME); v = TestFunction(ME)
ur, ui = split(u);  vr, vi = split(v)

uxr = ur.dx(0); uxi = ui.dx(0)
uyr = ur.dx(1); uyi = ui.dx(1)
vxr = vr.dx(0); vxi = vi.dx(0)
vyr = vr.dx(1); vyi = vi.dx(1)

grad_x = ( Cx_r*(uxr*vxr + uxi*vxi) - Cx_i*(uxi*vxr - uxr*vxi) )*dx
grad_y = ( Cy_r*(uyr*vyr + uyi*vyi) - Cy_i*(uyi*vyr - uyr*vyi) )*dx
mass   = ( M_r*(ur*vr + ui*vi) - M_i*(ui*vr - ur*vi) )*dx

alpha_g = Constant(omega*mu0*sigma_s)  # >0
graphene_term = ( -alpha_g*avg(ui)*avg(vr) + alpha_g*avg(ur)*avg(vi) )*dSg(IGRAPH)

a_form = grad_x + grad_y + mass + graphene_term

# ---------- Source (angle-dependent) ----------
y0   = 0.25*t_prism
wsrc = 0.15*t_prism
def rhs_for_angle(theta_deg):
    theta = np.deg2rad(theta_deg)
    kx = k0*n_prism*np.sin(theta); ky = k0*n_prism*np.cos(theta)
    inc_r = Expression('cos(kx*x[0] + ky*(x[1]-y0)) * exp(-pow((x[1]-y0)/w,2))',
                       kx=kx, ky=ky, y0=y0, w=wsrc, degree=2)
    inc_i = Expression('sin(kx*x[0] + ky*(x[1]-y0)) * exp(-pow((x[1]-y0)/w,2))',
                       kx=kx, ky=ky, y0=y0, w=wsrc, degree=2)
    return (inc_r*vr + inc_i*vi)*dxm(PRISM)

# ---------- Flux strip (in prism, below metal) ----------
strip_y1 = t_prism - 0.08e-6
strip_y2 = t_prism - 0.01e-6
flux_mark = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
class FluxStrip(SubDomain):
    def inside(self, p, on): return (strip_y1 <= p[1] <= strip_y2)
FluxStrip().mark(flux_mark, 1)
dxf = Measure('dx', domain=mesh, subdomain_data=flux_mark)

# ---------- Prebuild matrices: stack and reference (no metal, no graphene) ----------
A_stack = assemble(a_form)

# Build reference coefficients (metal -> prism), using same sx fields
eps_r_ref = Function(DG0); eps_i_ref = Function(DG0)
er_ref = eps_r_ref.vector().get_local(); ei_ref = eps_i_ref.vector().get_local()
for cid, tag in enumerate(cell2sub):
    tag = int(tag)
    if tag in (PRISM, METAL):  val = complex(eps_prism, 0.0)   # metal replaced by prism
    else:                      val = complex(eps_sense, 0.0)
    er_ref[cid], ei_ref[cid] = np.real(val), np.imag(val)
eps_r_ref.vector()[:] = er_ref; eps_i_ref.vector()[:] = ei_ref

den_ref = project(eps_r_ref*eps_r_ref + eps_i_ref*eps_i_ref, DG0)
inv_eps_r_ref = project(eps_r_ref/den_ref, DG0)
inv_eps_i_ref = project(-eps_i_ref/den_ref, DG0)

Cx_r_ref = project(inv_eps_r_ref*inv_sx_r - inv_eps_i_ref*inv_sx_i, DG0)
Cx_i_ref = project(inv_eps_r_ref*inv_sx_i + inv_eps_i_ref*inv_sx_r, DG0)
Cy_r_ref = project(inv_eps_r_ref*sx_r     - inv_eps_i_ref*sx_i,     DG0)
Cy_i_ref = project(inv_eps_r_ref*sx_i     + inv_eps_i_ref*sx_r,     DG0)
grad_x_ref = ( Cx_r_ref*(uxr*vxr + uxi*vxi) - Cx_i_ref*(uxi*vxr - uxr*vxi) )*dx
grad_y_ref = ( Cy_r_ref*(uyr*vyr + uyi*vyi) - Cy_i_ref*(uyi*vyr - uyr*vyi) )*dx
mass_ref   = ( M_r*(ur*vr + ui*vi) - M_i*(ui*vr - ur*vi) )*dx
a_ref = grad_x_ref + grad_y_ref + mass_ref
A_ref = assemble(a_ref)

# ---------- Solver wrapper ----------
def solve_and_flux(theta_deg, use_stack=True):
    """
    Solve for a given angle.
    If use_stack=False, solves the reference (no metal, no graphene).
    Returns: (Ur, Ui, Phi_up, Phi_down) where Phi_up = ∫ Sy_up dA, Phi_down = ∫ Sy_down dA
    """
    L = rhs_for_angle(theta_deg)
    U = Function(ME)

    if use_stack:
        try: solve(A_stack, U.vector(), assemble(L), "mumps")
        except: solve(A_stack, U.vector(), assemble(L))
        Ur, Ui = U.split(deepcopy=True)
        Sy = poynting_Sy_density(Ur, Ui, inv_eps_r, inv_eps_i, omega, mu0)
    else:
        try: solve(A_ref, U.vector(), assemble(L), "mumps")
        except: solve(A_ref, U.vector(), assemble(L))
        Ur, Ui = U.split(deepcopy=True)
        Sy = poynting_Sy_density(Ur, Ui, inv_eps_r_ref, inv_eps_i_ref, omega, mu0)

    # Separate upward/downward contributions using UFL conditionals
    Sy_up   = conditional(gt(Sy, 0.0), Sy,   0.0)   # upward only
    Sy_down = conditional(lt(Sy, 0.0), -Sy,  0.0)   # downward magnitude

    Phi_up   = assemble(Sy_up  * dxf(1))
    Phi_down = assemble(Sy_down* dxf(1))
    return Ur, Ui, Phi_up, Phi_down

# ---- Angle sweep with true reflectance R = (upward stack flux) / (downward reference flux) ----
angles_coarse = np.linspace(60.0, 86.0, 14)
R_coarse = np.zeros_like(angles_coarse)

for i, ang in enumerate(angles_coarse):
    _, _, Phi_up_stack, _      = solve_and_flux(ang, True)
    _, _, _,          Phi_dn_ref = solve_and_flux(ang, False)
    R_coarse[i] = Phi_up_stack / (Phi_dn_ref + 1e-30)

theta_guess = float(angles_coarse[np.argmin(R_coarse)])

angles_fine = np.linspace(max(60.0, theta_guess-3.0), min(86.0, theta_guess+3.0), 21)
Rfine = np.zeros_like(angles_fine)
for i, ang in enumerate(angles_fine):
    _, _, Phi_up_stack, _      = solve_and_flux(ang, True)
    _, _, _,          Phi_dn_ref = solve_and_flux(ang, False)
    Rfine[i] = Phi_up_stack / (Phi_dn_ref + 1e-30)

theta_res = float(angles_fine[np.argmin(Rfine)])
print(f"[INFO] Reflectance-based resonance ≈ {theta_res:.2f}°")

# ---------- Plot R(theta), FWHM, Q ----------
plt.figure(figsize=(6,4))
plt.plot(angles_fine, Rfine, 'o-', lw=2, ms=4)
plt.axvline(theta_res, ls='--', color='k', label=f"Resonance ~ {theta_res:.2f}°")
plt.xlabel("Incident angle (deg)")
plt.ylabel("Reflectance R (from Poynting flux)")
plt.title("SPR reflectance (TM, x-PML + graphene)")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig("spr_curve.png", dpi=220)

fw = fwhm(angles_fine, Rfine)
if fw and fw > 0:
    Q = theta_res / fw
    print(f"[INFO] FWHM ≈ {fw:.2f}°, Q ≈ {Q:.2f}")
else:
    print("[WARN] Could not determine FWHM reliably.")

# ---------- Field at resonance ----------
U = Function(ME)
L = rhs_for_angle(theta_res)
try: solve(A_stack, U.vector(), assemble(L), "mumps")
except: solve(A_stack, U.vector(), assemble(L))
Ur, Ui = U.split(deepcopy=True)
Hz_abs = project(sqrt(Ur*Ur + Ui*Ui), FunctionSpace(mesh, "CG", 1))

plt.figure(figsize=(7.5, 2.6))
p = plot(Hz_abs, cmap="viridis")
plt.colorbar(p); plt.title(f"|Hz| at ~{theta_res:.1f}°")
plt.axhline(y=t_prism, color='r', ls='--', lw=0.8, label="Prism/Metal")
plt.axhline(y=t_prism+t_metal, color='orange', ls='--', lw=0.8, label="Metal/Graphene (σₛ)")
plt.legend(loc='upper right', fontsize=7)
plt.tight_layout(); plt.savefig("field_resonance.png", dpi=220)
print("Wrote spr_curve.png and field_resonance.png")
