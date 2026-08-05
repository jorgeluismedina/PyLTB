
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import cast
 
from pyltb.shape_funcs import N_hermite
from pyltb.sections.section_utils import interpolate_section
 
 
# ── Estilo global ──────────────────────────────────────────────────────────
 
_BEAM_COLOR     = '#888888'   # eje / viga no deformada 2D
_DIAGRAM_COLOR  = '#0000ff'   # diagramas de esfuerzos
_DEFORMED_COLOR = '#0000ff'   # deformada estática
_CRIT_COLOR     = '#f00000'   # etiquetas de valores críticos
_MODE_COLORS    = ['#f00000', '#0000ff', '#000000', "#01B701", ]  # v, v', θ, θ'
_BEAM3D_COLOR   = '#0000ff'   # secciones deformadas 3D
_UNDEF_COLOR    = '#888888'   # viga no deformada 3D
 
_FIG_W, _FIG_H = 14, 5
 
 
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
    offset = y_range * 0.1
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
 
 
# ── Helpers de dibujo por eje ──────────────────────────────────────────────

def _draw_diagram_ax(ax, diagram, title=""):
    x, y, vals = diagram
    y_range = y.max() - y.min() or 1.0
    x_range = x.max()

    ax.plot([0, x_range], [0, 0], color=_BEAM_COLOR, lw=1, alpha=0.8)
    ax.plot(x, y, color=_DIAGRAM_COLOR, lw=1)

    for xs, ys in zip(x, y):
        ax.plot([xs,  xs],  [ys,  0], '--', color=_DIAGRAM_COLOR, lw=0.5, alpha=0.5)

    for idx in critical_indices(y):
        ax.text(x[idx], y[idx] + label_offset(y_range, y[idx]),
                f'{vals[idx]:.3e}', color=_CRIT_COLOR, fontsize=8, ha='center', va='center')

    ax.set_title(title, fontsize=11); ax.axis('equal'); ax.grid(True, alpha=0.3, lw=0.5)


def _draw_deformed_ax(ax, defor, title="Deformed shape"):
    x, y, vals = defor
    y_range = y.max() - y.min() or 1.0
    x_range = x.max()

    ax.plot([0, x_range], [0, 0], color=_BEAM_COLOR, lw=1, alpha=0.8)
    ax.plot(x, y, color=_DEFORMED_COLOR, lw=1)
    
    for idx in critical_indices(y):
        ax.text(x[idx], y[idx] + label_offset(y_range, y[idx]),
                f'{vals[idx]:.3e}', color=_CRIT_COLOR, fontsize=8, ha='center', va='center')
    ax.set_title(title, fontsize=11); ax.axis('equal'); ax.grid(True, alpha=0.3, lw=0.5)


# ── API pública ────────────────────────────────────────────────────────────

def plot_diagrams(diagrams, deformed):
    """Resumen estático en una figura 2×2: N, V, M y deformada."""
    N_diags, V_diags, M_diags = diagrams
    u_defor, w_defor = deformed
    fig, axes = plt.subplots(2, 2, figsize=(_FIG_W, _FIG_H * 1.8))
    _draw_diagram_ax(axes[0, 0], N_diags,    title="Normal (N)")
    _draw_diagram_ax(axes[0, 1], V_diags,    title="Shear (V)")
    _draw_diagram_ax(axes[1, 0], M_diags,    title="Moment (M)")
    _draw_deformed_ax(axes[1, 1],w_defor,   title="Deflection (w)")
    fig.tight_layout()
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


# ── Helpers modos de pandeo ────────────────────────────────────────────────

def _normalize_mode(modes, imode, scale):
    """Extrae y normaliza el modo imode por el pico de v."""
    mode = modes[:, imode]
    peak = np.max(np.abs(mode[0::4])) or 1.0
    return mode * (scale / peak)


def _setup_ax3d(ax):
    """Sin paredes, sin ejes."""
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):  # type: ignore
        pane.fill = False
        pane.set_edgecolor('none')
    ax.set_axis_off()


def _fix_3d_aspect(ax):
    """Reemplaza set_aspect('equal'): ajusta set_box_aspect con los rangos reales de datos."""
    ranges = [lim[1] - lim[0] for lim in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())]
    ax.set_box_aspect(ranges)


def _draw_3d_mode_ax(ax, model, mode_n, xis):
    """Dibuja secciones deformadas y no deformadas sobre el eje 3D dado."""
    kp_colors    = [_UNDEF_COLOR] * 6 + ['purple', 'green']
    kp_undef_list = []
    kp_def_list   = []

    for elem in model.elements:
        L  = elem.length
        x0 = elem.coords[0]
        v_arr, th_arr = interp_mode(elem.ltr_dofs, mode_n, L, xis)

        for k, xi in enumerate(xis):
            sec      = section_at(elem, xi)
            bf1, bf2 = get_flange_widths(sec)
            align    = elem.align
            zG = sec.z_from_ref(align, 0)
            zS = sec.z_from_ref(align, 1)
            zb = sec.z_from_ref(align, 2)
            zt = sec.z_from_ref(align, 3)
            x_k = x0 + xi * L

            for seg in section_outline(sec, align):
                yz_def = deform_segment(seg, v_arr[k], th_arr[k], zS)
                ax.plot([x_k, x_k], seg[:, 0],    seg[:, 1],    color=_UNDEF_COLOR,  lw=0.3)
                ax.plot([x_k, x_k], yz_def[:, 0], yz_def[:, 1], color=_BEAM3D_COLOR, lw=0.3)

            kp = np.array([
                [x_k, -bf1/2, zt], [x_k, bf1/2,  zt],  # alas sup
                [x_k, -bf2/2, zb], [x_k, bf2/2,  zb],  # alas inf
                [x_k,  0.0,   zt], [x_k, 0.0,    zb],  # centros de alas
                [x_k,  0.0,   zG], [x_k, 0.0,    zS],  # G, S
            ])
            kp_undef_list.append(kp)
            kp_def_list.append(deform_keypoints(kp, v_arr[k], th_arr[k], zS))

    kp_undef = np.stack(kp_undef_list)
    kp_def   = np.stack(kp_def_list)

    for j, color in enumerate(kp_colors):
        ax.plot(*kp_undef[:, j, :].T, color=color,         lw=0.3)
    for j in range(kp_def.shape[1]):
        ax.plot(*kp_def[:, j, :].T,   color=_BEAM3D_COLOR, lw=0.3)


def _draw_2d_mode_ax(ax, model, mode, legend_anchor=(0.5, 1.3), legend_fontsize=10):
    """Dibuja las curvas del modo (v, v', θ, θ') normalizadas sobre el eje dado."""
    labels = [r'$v$', r"$v'$", r'$\theta$', r"$\theta'$"]
    x      = model.coords
    dofs   = [mode[k::4] / (np.max(np.abs(mode[k::4])) or 1.0) for k in range(4)]

    for d, color, label in zip(dofs, _MODE_COLORS, labels):
        ax.plot(x, d, color=color, lw=0.7, label=label)

    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(x[0], x[-1])
    #ax.grid(True, alpha=0.3, lw=0.5, ls='--')
    ax.grid(False)
    ax.legend(loc='upper center', bbox_to_anchor=legend_anchor,
              ncol=4, frameon=False, fontsize=legend_fontsize)


# ── API pública ────────────────────────────────────────────────────────────
'''
def plot_buckling_mode(model, mu_crs, modes, imode=0, scale=1.0, n_sec=2):
    """Vista combinada del modo de pandeo: 3D (arriba) y curvas 2D (abajo)."""
    mode_n = _normalize_mode(modes, imode, scale)
    xis    = np.linspace(0.0, 1.0, n_sec)

    fig  = plt.figure(figsize=(_FIG_W, _FIG_H * 1.8))
    gs   = fig.add_gridspec(2, 1, height_ratios=[6, 1], hspace=0.05)
    ax3d = fig.add_subplot(gs[0], projection='3d')
    ax2d = fig.add_subplot(gs[1])

    _setup_ax3d(ax3d)
    _draw_3d_mode_ax(ax3d, model, mode_n, xis)
    _fix_3d_aspect(ax3d)
    draw_axis_arrows(ax3d, 0.0, 0.0, 0.0, 0.06)
    ax3d.set_title(rf'Mode {imode+1}  —  $\mu_{{cr}} = {mu_crs[imode]:.3f}$',
                   fontsize=11, pad=15)

    _draw_2d_mode_ax(ax2d, model, modes[:, imode], legend_fontsize=8)
    ax2d.tick_params(axis='y', labelsize=8)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.07)
    return fig, (ax3d, ax2d)

'''
def plot_buckling_mode(model, mu_crs, modes, imode=0, scale=1.0, n_sec=2, curves=True):
    mode_n = _normalize_mode(modes, imode, scale)
    xis    = np.linspace(0.0, 1.0, n_sec)

    if curves:
        fig = plt.figure(figsize=(_FIG_W, _FIG_H * 1.8))
        gs  = fig.add_gridspec(2, 1, height_ratios=[6, 1], hspace=0.07)
        ax3d = fig.add_subplot(gs[0], projection='3d')
        ax2d = fig.add_subplot(gs[1])
        _draw_2d_mode_ax(ax2d, model, modes[:, imode], legend_fontsize=10)
        ax2d.tick_params(axis='y', labelsize=8)
    else:
        fig  = plt.figure(figsize=(_FIG_W, _FIG_H))
        ax3d = fig.add_subplot(111, projection='3d')
        ax2d = None

    _setup_ax3d(ax3d)
    _draw_3d_mode_ax(ax3d, model, mode_n, xis)
    _fix_3d_aspect(ax3d)
    draw_axis_arrows(ax3d, 0.0, 0.0, 0.0, 0.06)
    ax3d.set_title(rf'Mode {imode+1}  —  $\mu_{{cr}} = {mu_crs[imode]:.3f}$',
                   fontsize=12, pad=15)
  

    fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.07)
    return fig, (ax3d, ax2d)


