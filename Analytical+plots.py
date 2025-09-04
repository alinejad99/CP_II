#!/usr/bin/env python
# coding: utf-8

# $Analytical Solution$

# In[5]:


import numpy as np
import matplotlib.pyplot as plt

def spr_reflectivity_analytical_4layer(theta_deg, n_prism, n_metal, n_graphene, d_metal, d_graphene, n_sensing, wavelength):
    """
    Calculates the analytical reflectivity for a 4-layer SPR system:
    Prism | Metal | Graphene | Sensing Medium
    using the full Transfer Matrix Method (TMM) for TM polarization.

    Args:
        theta_deg (float): Incident angle in degrees.
        n_prism (float): Refractive index of the prism.
        n_metal (complex): Complex refractive index of the metal layer.
        n_graphene (complex): Complex refractive index of graphene.
        d_metal (float): Thickness of the metal layer [m].
        d_graphene (float): Thickness of the graphene layer [m].
        n_sensing (float): Refractive index of the sensing medium.
        wavelength (float): Vacuum wavelength of the incident light [m].

    Returns:
        float: Reflectivity (R = |r|^2) at the given angle.
    """
    theta_rad = np.deg2rad(theta_deg)
    k0 = 2 * np.pi / wavelength  # Vacuum wave number

    # Component of the wavevector parallel to the interface (conserved)
    kx = k0 * n_prism * np.sin(theta_rad)

    # Define the layers: [Prism, Metal, Graphene, Sensing]
    n_list = [n_prism, n_metal, n_graphene, n_sensing]
    d_list = [np.inf, d_metal, d_graphene, np.inf]  # Thickness; first and last are semi-infinite

    # Calculate k_z for each layer
    kz_list = []
    for n in n_list:
        kz = np.sqrt((k0 * n)**2 - kx**2 + 0j)
        kz_list.append(kz)

    # Initialize the characteristic matrix of the whole stack as an identity matrix
    M = np.eye(2, dtype=complex)
    
    # Build the characteristic matrix by iterating through the inner layers (1 and 2)
    # Layer 0 (prism) and layer 3 (sensing) are the incident and substrate mediums, not films.
    for i in range(1, len(n_list)-1): # i = 1 (Metal), i = 2 (Graphene)
        n_i = n_list[i]
        kz_i = kz_list[i]
        d_i = d_list[i]
        
        # Phase factor for propagation through the layer
        beta_i = kz_i * d_i
        
        # For TM polarization
        q_i = kz_i / (k0 * n_i**2) # q is related to the surface impedance
        
        # Matrix for interface i-1 to i
        M_i = np.array([
            [np.cos(beta_i), -1j * np.sin(beta_i) / q_i],
            [-1j * q_i * np.sin(beta_i), np.cos(beta_i)]
        ], dtype=complex)
        
        # Multiply the total characteristic matrix by the current layer's matrix
        M = np.dot(M, M_i)

    # Now, M is the characteristic matrix for the stack: Metal + Graphene
    # The Fresnel coefficients for the entire stack are calculated from M
    # q for the incident medium (prism, layer 0) and substrate (sensing, layer 3)
    q0 = kz_list[0] / (k0 * n_list[0]**2)
    q_s = kz_list[3] / (k0 * n_list[3]**2)
    
    # Calculate the total reflection coefficient r
    M11, M12 = M[0, 0], M[0, 1]
    M21, M22 = M[1, 0], M[1, 1]
    
    r = ( (M11 + M12 * q_s) * q0 - (M21 + M22 * q_s) ) / ( (M11 + M12 * q_s) * q0 + (M21 + M22 * q_s) )
    
    # Reflectivity is the squared magnitude of the reflection coefficient
    R = np.abs(r)**2
    return R

# --- Parameters ---
lambda0 = 633e-9 # Wavelength [m]
n_prism = 1.5     # Prism RI

# Gold @ 633nm
n_metal = 0.19 + 3.59j

# GRAPHENE PARAMETERS (From your provided paper: [2] V. Khuong Dien et al., RSC Adv., 2022, 12, 34851)
# For a monolayer, we often use a surface conductivity model.
# A common approximation for the complex refractive index of graphene at 633nm is:
n_graphene = 3.0 - 1.4j # Typical values found in literature for visible range
d_graphene = 0.34e-9      # Thickness of monolayer graphene [m]

n_sensing = 1.33  # Sensing medium RI (water)
d_metal = 50e-9   # Metal thickness [m]

# --- Angle Sweep ---
angles_deg = np.linspace(60, 90, 500)
R_analytical_4layer = np.zeros_like(angles_deg, dtype=float)

for i, angle in enumerate(angles_deg):
    R_analytical_4layer[i] = spr_reflectivity_analytical_4layer(angle, n_prism, n_metal, n_graphene, d_metal, d_graphene, n_sensing, lambda0)

# --- Find Resonance Angle ---
min_index = np.argmin(R_analytical_4layer)
if min_index > 0 and min_index < len(angles_deg)-1:
    x = angles_deg[min_index-1:min_index+2]
    y = R_analytical_4layer[min_index-1:min_index+2]
    parabola_coeffs = np.polyfit(x, y, 2)
    resonance_angle_analytical = -parabola_coeffs[1] / (2 * parabola_coeffs[0])
    min_reflectivity = np.polyval(parabola_coeffs, resonance_angle_analytical)
else:
    resonance_angle_analytical = angles_deg[min_index]
    min_reflectivity = R_analytical_4layer[min_index]

# --- Calculate Performance Parameters (at resonance) ---
# 1. Full Width at Half Maximum (FWHM)
half_max = min_reflectivity + (1 - min_reflectivity) / 2
# Find angles where the curve crosses the half-max value
above_half_max = np.where(R_analytical_4layer > half_max)[0]
# Ensure we are only looking near the dip
left_side = above_half_max[above_half_max < min_index][-1] if np.any(above_half_max < min_index) else 0
right_side = above_half_max[above_half_max > min_index][0] if np.any(above_half_max > min_index) else -1
angle_left = angles_deg[left_side]
angle_right = angles_deg[right_side]
FWHM = angle_right - angle_left

# 2. Sensitivity (S) - You would calculate this by changing n_sensing and seeing how much the resonance angle shifts.
# For now, we'll just print the resonance parameters.
print("=== 4-Layer Analytical Solution (Prism|Au|Graphene|Water) ===")
print(f"Resonance Angle: {resonance_angle_analytical:.4f} degrees")
print(f"Minimum Reflectivity: {min_reflectivity:.6f}")
print(f"FWHM: {FWHM:.4f} degrees")

# --- Plotting ---
plt.figure(figsize=(12, 7))
plt.plot(angles_deg, R_analytical_4layer, 'b-', label='4-Layer: Prism|Au|Graphene|Water', linewidth=1.5)
plt.xlabel('Incident Angle (degrees)')
plt.ylabel('Reflectivity')
plt.title(f'Analytical SPR Curve with Graphene Monolayer\nAu {d_metal*1e9:.0f}nm / Graphene {d_graphene*1e9:.1f}nm / Water | {lambda0*1e9:.0f}nm')
plt.grid(True, alpha=0.3)
plt.legend()

# Mark the resonance angle and FWHM
plt.axvline(x=resonance_angle_analytical, color='r', linestyle='--', linewidth=1, label=f'Resonance = {resonance_angle_analytical:.2f}°')
#plt.axhline(y=half_max, color='g', linestyle=':', linewidth=1, label=f'Half Max ({half_max:.3f})')
#plt.axvline(x=angle_left, color='g', linestyle=':', linewidth=1)
#plt.axvline(x=angle_right, color='g', linestyle=':', linewidth=1, label=f'FWHM = {FWHM:.2f}°')

plt.legend()
plt.savefig('spr_curve_analytical_4layer_with_graphene.pdf', format='pdf', bbox_inches='tight')
plt.show()
# At the end of your analytical code:
np.savez('analytical_results.npz', angles_deg=angles_deg, R_analytical=R_analytical_4layer)


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.tri as mtri

# --- settings ---
TXT_PATH = "field.txt"   # <- your .txt file
# Optionally choose which columns to use (0-based). Leave as None to auto-pick.
XCOL, YCOL, ZCOL = None, None, None

def load_numeric_txt(path):
    """
    Reads a COMSOL-like text file:
      - skips lines starting with '%' or '#'
      - accepts whitespace/commas/semicolons as separators
      - returns an (N, M) float array (M = number of columns)
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('%') or s.startswith('#'):
                continue
            # normalize delimiters to whitespace
            s = s.replace(',', ' ').replace(';', ' ')
            parts = s.split()
            # skip if not at least 2 tokens or if any token is non-numeric
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                continue
            if len(nums) >= 2:
                rows.append(nums)
    if not rows:
        raise ValueError("No numeric rows found after filtering comments.")
    # pad shorter rows to the max length (with NaN), then drop columns that are entirely NaN
    maxlen = max(len(r) for r in rows)
    arr = np.full((len(rows), maxlen), np.nan, dtype=float)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r
    # drop all-NaN columns
    keep = ~np.all(np.isnan(arr), axis=0)
    arr = arr[:, keep]
    # drop rows with any NaN in the first 3 columns we need later
    return arr

def pick_columns(data, xcol=None, ycol=None, zcol=None):
    ncols = data.shape[1]
    if xcol is None: xcol = 0
    if ycol is None: ycol = 1
    if ncols < 2:
        raise ValueError("Need at least 2 numeric columns for plotting.")
    if zcol is not None and zcol >= ncols:
        raise ValueError(f"Z column index {zcol} out of range (columns=0..{ncols-1}).")
    # If a z column exists (by request or by availability), use it
    if zcol is None:
        zcol = 2 if ncols >= 3 else None
    return xcol, ycol, zcol

def plot2d(x, y, out_pdf="field_2d.pdf"):
    plt.figure(figsize=(7,6))
    plt.scatter(x, y, s=8)
    plt.xlabel("X (col0)")
    plt.ylabel("Y (col1)")
    plt.title("2D Scatter (from field.txt)")
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_pdf.replace(".pdf",".png"), dpi=200, bbox_inches="tight")
    plt.show()

def plot3d_scatter_and_surface(x, y, z, base="field_3d"):
    # 3D scatter
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Scatter (from field.txt)")
    plt.tight_layout()
    fig.savefig(base + "_scatter.pdf", bbox_inches="tight")
    fig.savefig(base + "_scatter.png", dpi=200, bbox_inches="tight")
    plt.show()

    # Triangulated surface (works with scattered points)
    tri = mtri.Triangulation(x, y)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(tri, z, linewidth=0.2, antialiased=True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Surface (Triangulated)")
    plt.tight_layout()
    fig.savefig(base + "_surface.pdf", bbox_inches="tight")
    fig.savefig(base + "_surface.png", dpi=200, bbox_inches="tight")
    plt.show()

# ---- run ----
data = load_numeric_txt(TXT_PATH)
xcol, ycol, zcol = pick_columns(data, XCOL, YCOL, ZCOL)

x = data[:, xcol]
y = data[:, ycol]

if zcol is None:
    # Only 2 columns available -> 2D plot
    plot2d(x, y, out_pdf="field_2d.pdf")
else:
    # 3+ columns -> 3D plots
    z = data[:, zcol]
    # Drop any rows with NaN in x/y/z (just in case)
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    x, y, z = x[mask], y[mask], z[mask]
    plot3d_scatter_and_surface(x, y, z, base="field_3d")


# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# --- load COMSOL table (two columns: angle[deg], reflectance) ---
angles, R = [], []
with open('SPR_comsol.txt', 'r', encoding='utf-8') as f:
    for line in f:
        s = line.strip()
        if (not s) or s.startswith('%') or 'Incident angle' in s:
            continue
        parts = s.split()
        if len(parts) >= 2:
            try:
                ang = float(parts[0])
                val = float(parts[-1])
                angles.append(ang); R.append(val)
            except ValueError:
                pass

angles = np.array(angles); R = np.array(R)

# Some COMSOL exports end with an artifact like "90 0" – drop if present
if angles.size and angles[-1] == 90.0 and R[-1] == 0.0:
    angles, R = angles[:-1], R[:-1]

# --- resonance (minimum reflectance) ---
i_min = int(np.argmin(R))
theta_res, R_min = float(angles[i_min]), float(R[i_min])

# --- plot ---
plt.figure(figsize=(10,6))
plt.plot(angles, R, marker='o', linewidth=2, markersize=2.5,
         label='COMSOL (TM, total reflectance)')
plt.axvline(theta_res, linestyle='--', linewidth=1.5,
            label=f'Resonance at {theta_res:.2f}°')
plt.title('SPR Curve — COMSOL (Prism|Au|Graphene|Water, λ=633 nm)')
plt.xlabel('Incident Angle (degrees)')
plt.ylabel('Reflectance R')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('spr_curve_comsol.png', dpi=300)
plt.savefig('spr_curve_comsol.pdf')
print(f"theta_res = {theta_res:.2f} deg, R_min = {R_min:.4f}")


# In[ ]:




