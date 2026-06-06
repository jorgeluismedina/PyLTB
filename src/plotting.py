
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import cast
 
from src.shape_funcs import N_hermite
from src.sections.section_utils import interpolate_section
 
 
# ── Estilo global ──────────────────────────────────────────────────────────
 
_BEAM_COLOR     = '#888888'   # eje / viga no deformada 2D
_DIAGRAM_COLOR  = '#0000ff'   # diagramas de esfuerzos
_DEFORMED_COLOR = '#0000ff'   # deformada estática
_CRIT_COLOR     = '#f00000'   # etiquetas de valores críticos
_MODE_COLORS    = ['#f00000', '#0000ff', '#000000', "#01B701", ]  # v, v', θ, θ'
_BEAM3D_COLOR   = '#0000ff'   # secciones deformadas 3D
_UNDEF_COLOR    = '#888888'   # viga no deformada 3D
 
_FIG_W, _FIG_H = 13, 5
 
 
# ── Helpers internos ───────────────────────────────────────────────────────
 
def critical_indices(values):
    """Índices del máximo y mínimo global (ignora valores ~0)."""
    if np.allclose(values, values[0], atol=1e-12):
        return [len(values) // 2]
    indices = set()
    for fn in (np.argmax, np.argmin):
        idx = fn(values)
        if np.abs(values[idx]) > 1e-12:
            indices.add(idx)
    return sorted(indices)
 
 
def label_offset(y_range, y_val):
    """Offset vertical proporcional al rango para etiquetas críticas."""
    offset = y_range * 0.04
    return offset if y_val >= 0 else -offset
 
 
def get_flange_widths(sec):
    if hasattr(sec, 'bf1'):
        return sec.bf1, sec.bf2
    return sec.bf, sec.bf
 
 
def section_outline(sec, align):
    """Segmentos del perfil I en coordenadas locales (y, z)."""
    bf1, bf2 = get_flange_widths(sec)
    zb = sec.z_from_ref(align, 2)
    zt = sec.z_from_ref(align, 3)

    return [
        np.array([[-bf1 / 2, zt], [bf1 / 2, zt]]),  # ala superior
        np.array([[-bf2 / 2, zb], [bf2 / 2, zb]]),  # ala inferior
        np.array([[0.0, zb], [0.0, zt]]),           # alma
    ]
 
 
def deform_segment(seg, v, theta, zS):
    """Deforma un segmento 2D (y, z) con traslación lateral v y rotación theta."""
    c, s = np.cos(theta), np.sin(theta)

    y = seg[:, 0]
    z = seg[:, 1] - zS   # llevar el punto al pivote

    y_def = y * c - z * s + v
    z_def = y * s + z * c + zS  # volver a la posición original del pivot

    return np.column_stack((y_def, z_def))

def deform_keypoints(kp, v, theta, zS):
    """Deforma puntos 3D [x, y, z] dejando x intacta."""
    out = kp.copy()
    out[:, 1:3] = deform_segment(kp[:, 1:3], v, theta, zS)
    return out
 
 
def section_at(elem, xi):
    """Sección interpolada en xi ∈ [0,1] (uniforme o tapered)."""
    if hasattr(elem, 'section_i'):
        return interpolate_section(elem.section_i, elem.section_j, xi)
    return elem.section
 
 
def interp_mode(elem_ltr_dof, mode, L, xis):
    """
    Interpola v(xi) y theta(xi) con funciones de forma de Hermite.
    Las rotaciones se escalan por L antes de aplicar N_hermite.
    """
    d    = mode[elem_ltr_dof]
    d_v  = np.array([d[0], d[1]*L, d[4], d[5]*L])
    d_th = np.array([d[2], d[3]*L, d[6], d[7]*L])
    N    = N_hermite(xis)   # (4, n_pts)
    return N.T @ d_v, N.T @ d_th
 
 
# ── Funciones de ploteo ────────────────────────────────────────────────────
 
def plot_diagram(model, diagrams, title=""):
    """
    Diagrama de esfuerzos (N, V o M) con etiquetas en puntos críticos.
 
    Parámetros
    ----------
    model    : StabilityModel
    diagrams : salida de StaticSolver.prepare_diagrams — lista de (x, y_norm, vals)
    title    : título del gráfico
    """
    all_x    = np.concatenate([d[0] for d in diagrams])
    all_y    = np.concatenate([d[1] for d in diagrams])
    all_vals = np.concatenate([d[2] for d in diagrams])
    y_range  = all_y.max() - all_y.min() or 1.0
 
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
 
    for e, elem in enumerate(model.elements):
        diag_x, diag_y, _ = diagrams[e]
        ax.plot(elem.coords, np.zeros(2), color=_BEAM_COLOR, lw=1, alpha=0.8)
        ax.plot(diag_x, diag_y, color=_DIAGRAM_COLOR, lw=1)

        ax.plot([diag_x[0],  elem.coords[0]], [diag_y[0],  0], '--',
                color=_DIAGRAM_COLOR, lw=0.5, alpha=0.5)
        
        ax.plot([diag_x[-1], elem.coords[1]], [diag_y[-1], 0], '--',
                color=_DIAGRAM_COLOR, lw=0.5, alpha=0.5)
 
    for idx in critical_indices(all_y):
        offset = label_offset(y_range, all_y[idx])
        ax.text(all_x[idx], all_y[idx] + offset, f'{all_vals[idx]:.3e}',
                color=_CRIT_COLOR, fontsize=8, ha='center', va='center')
 
    #ax.axhline(0, color=_BEAM_COLOR, lw=0.5, alpha=0.5)
    #ax.set_xlabel('x', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.axis('equal')
    ax.grid(True, alpha=0.3, lw=0.5)
    fig.tight_layout()
    return fig, ax
 

def plot_deformed(model, def_shapes, title="Deformed shape"):
    """
    Viga original (gris) y deformada estática (azul).
 
    Parámetros
    ----------
    model      : StabilityModel
    def_shapes : salida de StaticSolver.prepare_diagrams — lista de (X_def, Y_def)
    title      : título del gráfico
    """
    all_x    = np.concatenate([d[0] for d in def_shapes])
    all_z    = np.concatenate([d[1] for d in def_shapes])
    all_xvals = np.concatenate([d[2] for d in def_shapes])
    all_zvals = np.concatenate([d[3] for d in def_shapes])
    y_range  = all_z.max() - all_z.min() or 1.0

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
 
    for e, elem in enumerate(model.elements):
        X_def, Y_def, _, _ = def_shapes[e]
        ax.plot(elem.coords, np.zeros(2), color=_BEAM_COLOR,    lw=1, alpha=0.8)
        ax.plot(X_def, Y_def,             color=_DEFORMED_COLOR, lw=1)

    for idx in critical_indices(all_z):
        offset = label_offset(y_range, all_z[idx])
        ax.text(all_x[idx], all_z[idx] + offset, f'{all_zvals[idx]:.3e}',
                color=_CRIT_COLOR, fontsize=8, ha='center', va='center')
 
    #ax.set_xlabel('x', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.axis('equal')
    ax.grid(True, alpha=0.3, lw=0.5)
    fig.tight_layout()
    return fig, ax



def plot_buckling_modes(model, mu_crs, modes, nmodes=1):
    """
    Diagramas 2D de los modos de pandeo lateral-torsional (v, v', θ, θ').
    Cada componente se normaliza por su propio pico.
 
    Parámetros
    ----------
    model  : StabilityModel
    mu_crs : array de factores de carga crítica
    modes  : (nltr_dofs, nmodes)
    nmodes : número de modos a plotear
    """
    labels = [r'$v$', r"$v'$", r'$\theta$', r"$\theta'$"]
    x      = model.coords
 
    fig, axes = plt.subplots(nmodes, 1, figsize=(_FIG_W, 2.8*nmodes), sharex=True)
    if nmodes == 1:
        axes = [axes]
 
    for i, ax in enumerate(axes):
        mode = modes[:, i]
        dofs = [mode[k::4] for k in range(4)]
        dofs = [d / (np.max(np.abs(d)) or 1.0) for d in dofs]
 
        for d, color, label in zip(dofs, _MODE_COLORS, labels):
            ax.plot(x, d, color=color, lw=1, label=label)
 
        ax.axhline(0, color=_BEAM_COLOR, lw=0.6, alpha=0.5)
        ax.set_ylabel(
            f'Mode {i+1}\n$\\mu_{{cr}}={mu_crs[i]:.2f}$',
            fontsize=9, rotation=0, labelpad=48, va='center'
        )
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(x[0], x[-1])
        ax.grid(True, alpha=0.3, lw=0.5, ls='--')
 
        if i == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
                      ncol=4, frameon=False, fontsize=11)
 
    #axes[-1].set_xlabel('x', fontsize=10)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    return fig, axes 





def draw_axis_arrows(ax, x0, y0, z0, length):
    """Trípode de ejes al estilo CAD en la posición (x0, y0, z0)."""
    axes_def = [
        ([1, 0, 0], '#e74c3c', 'X'),
        ([0, 1, 0], '#27ae60', 'Y'),
        ([0, 0, 1], '#2980b9', 'Z'),
    ]
    for direction, color, label in axes_def:
        dx, dy, dz = [d * length for d in direction]

        ax.quiver(x0, y0, z0, dx, dy, dz,
                  color=color, linewidth=1.1,
                  arrow_length_ratio=0.2, normalize=False)
        
        ax.text(x0 + dx*1.15, y0 + dy*1.15, z0 + dz*1.15,
                label, color=color, fontsize=6, fontweight='bold',
                ha='center', va='center')




def plot_buckling_mode_3d(model, mu_crs, modes, imode=0, scale=1.0, n_sec=5):
    mode   = modes[:, imode]

    peak = np.max(np.abs(mode[0::4]))
    peak = peak if peak > 0 else 1.0
    mode_n = mode * (scale / peak)
 
    fig = plt.figure(figsize=(14, 7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, projection="3d")
 
    # Sin paredes, sin ejes
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane): # type: ignore[union-attr]
        pane.fill = False
        pane.set_edgecolor('none')
    ax.set_axis_off()

    xis = np.linspace(0.0, 1.0, n_sec)

    keypoints_undef_list: list[np.ndarray] = []
    keypoints_def_list: list[np.ndarray] = []

    cs_idx = 6
 
    for elem in model.elements:
        L   = elem.length
        x0  = elem.coords[0]
 
        v_arr, th_arr = interp_mode(elem.ltr_dofs, mode_n, L, xis)
 
        for k, xi in enumerate(xis):
            sec = section_at(elem, xi)
            bf1, bf2 = get_flange_widths(sec)
            align = elem.align

            zG = sec.z_from_ref(align, 0)
            zS = sec.z_from_ref(align, 1)
            zb = sec.z_from_ref(align, 2)
            zt = sec.z_from_ref(align, 3)

            x_k = x0 + xi * L
 
            # Contorno de la sección
            for seg in section_outline(sec, align):
                yz_def = deform_segment(seg, v_arr[k], th_arr[k], zS)

                ax.plot([x_k, x_k], seg[:, 0], seg[:, 1],
                        color=_UNDEF_COLOR, lw=0.3, alpha=1)

                ax.plot([x_k, x_k], yz_def[:, 0], yz_def[:, 1],
                        color=_BEAM3D_COLOR, lw=0.3, alpha=1)
 
            
            # Puntos característicos de la sección
            kp = np.array([
                [x_k, -bf1 / 2, zt],  # top-left
                [x_k,  bf1 / 2, zt],  # top-right
                [x_k, -bf2 / 2, zb],  # bottom-left
                [x_k,  bf2 / 2, zb],  # bottom-right
                [x_k, 0.0, zt],       # centro del ala superior
                [x_k, 0.0, zb],       # centro del ala inferior
                [x_k, 0.0, zG],       # centro de corte
                [x_k, 0.0, zS],       # centro de gravedad
            ])
            
            keypoints_undef_list.append(kp)
            keypoints_def_list.append(deform_keypoints(kp, v_arr[k], th_arr[k], zS))
 
        keypoints_undef = np.stack(keypoints_undef_list, axis=0)
        keypoints_def = np.stack(keypoints_def_list, axis=0)

        
        # Unir los puntos correspondientes a lo largo de la viga
        for j in range(keypoints_undef.shape[1]):
            pts = keypoints_undef[:, j, :]
            color = "purple" if j == 6 else _UNDEF_COLOR
            color = "green" if j == 7 else _UNDEF_COLOR
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=color, lw=0.3, alpha=1)

        for j in range(keypoints_def.shape[1]):
            pts = keypoints_def[:, j, :]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=_BEAM3D_COLOR, lw=0.3, alpha=1)
 
    # Trípode de ejes
    #arrow_len = (model.coords[-1] - model.coords[0]) * 0.05
    arrow_len = 0.06
    ax.set_aspect('equal')  # type: ignore[arg-type]
    draw_axis_arrows(ax, 0.0, 0.0, 0.0, arrow_len)
    
    #ax.margins(x=0.1, y=0.3, z=0.3)
    ax.set_title(
        rf'Mode {imode+1}  —  $\mu_{{cr}} = {mu_crs[imode]:.3f}$',
        fontsize=11, pad=2, y=0.97
    )
    return fig, ax








