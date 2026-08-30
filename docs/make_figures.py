"""Generate the README figures for vmc_torch.

Draws the family of variational ansätze that vmc_torch optimizes with a
single VMC engine:

    (a) tensor network state          -- PEPS, spin / bosonic
    (b) fermionic tensor network      -- same geometry, oriented bonds
    (c) neural tensor network state   -- fPEPS + neural network

Everything is drawn from live quimb tensor networks, so the pictures
stay honest if the geometry changes.

Usage:
    python docs/make_figures.py
"""
import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import quimb.tensor as qtn

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'pics')

# Colourblind-safe (Okabe & Ito), matching quimb's own default palette.
C_TN = '#0072B2'      # blue   -- tensor network site tensors
C_PHYS = '#D55E00'    # orange -- physical indices
C_ARROW = '#009E73'   # green  -- bond orientation (fermionic order)
C_NN = '#E69F00'      # amber  -- neural network


def axonometric(i, j, k, a=22, b=45, p=0.2):
    """Project a 3D lattice coordinate onto the plane.

    The standard quimb layout for 2D tensor networks: site tensors sit in
    one plane (k=1) and their physical indices hang in a plane below
    (k=0), so that no bonds overlap.
    """
    return (
        +i * math.cos(math.pi * a / 180)
        + j * math.cos(math.pi * b / 180) / 2 ** p,
        -i * math.sin(math.pi * a / 180)
        + j * math.sin(math.pi * b / 180) / 2 ** p
        + k,
    )


def _peps_layout(psi, ax, node_scale):
    """Draw a PEPS in axonometric projection; return the drawn positions.

    quimb rescales node coordinates into roughly [-1, 1] when it draws,
    so anything we overlay afterwards has to use the positions it
    actually used -- which ``get='pos'`` reports.
    """
    fix = {psi.site_tag(i, j): axonometric(i, j, 1.0)
           for i, j in psi.gen_site_coos()}
    fix.update({psi.site_ind(i, j): axonometric(i, j, 0.35)
                for i, j in psi.gen_site_coos()})

    draw_opts = dict(
        fix=fix,
        custom_colors=[C_TN] * psi.num_tensors,
        color=tuple(psi.site_tags),
        edge_color='0.25', edge_alpha=0.9, edge_scale=0.9,
        node_scale=node_scale, node_outline_size=1.2,
        node_outline_darkness=0.3,
        show_inds=False, show_tags=False, legend=False,
    )
    pos = psi.draw(get='pos', **draw_opts)
    psi.draw(ax=ax, **draw_opts)

    # positions keyed by site coordinate, in the drawn frame
    sites = {(i, j): pos[list(psi.tag_map[psi.site_tag(i, j)])[0]]
             for i, j in psi.gen_site_coos()}
    phys = {(i, j): pos[psi.site_ind(i, j)]
            for i, j in psi.gen_site_coos()}
    return sites, phys


def panel_tensor_network(ax, Lx=4, Ly=4, D=3):
    """(a) A PEPS drawn in quimb's axonometric publication style."""
    psi = qtn.PEPS.rand(Lx, Ly, D, seed=42)
    _, phys = _peps_layout(psi, ax, node_scale=1.25)

    # Mark the dangling physical indices.
    for (x, y) in phys.values():
        ax.plot([x], [y], 'o', ms=3.4, color=C_PHYS, zorder=5)

    ax.set_title('(a)  tensor network state\nPEPS — spin / bosonic',
                 fontsize=10.5, pad=8)


def _bond_arrow(ax, p_from, p_to, frac=0.30, color=C_ARROW, lw=1.4):
    """Draw a short arrow centred on the bond from p_from to p_to."""
    (x0, y0), (x1, y1) = p_from, p_to
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = (x1 - x0) * frac / 2, (y1 - y0) * frac / 2
    ax.annotate(
        '', xy=(mx + dx, my + dy), xytext=(mx - dx, my - dy),
        arrowprops=dict(arrowstyle='-|>', lw=lw, color=color,
                        shrinkA=0, shrinkB=0, mutation_scale=13),
        zorder=6,
    )


def panel_fermionic_tn(ax, Lx=4, Ly=4, D=3):
    """(b) The same network as (a), with every bond oriented.

    A fermionic tensor network has the identical geometry to a bosonic
    one; what distinguishes it is that every index carries a direction.
    Each bond joins a dual leg to a non-dual leg, and the resulting
    orientation -- together with a fixed site order -- is what fixes the
    anticommutation signs picked up when legs are reordered during a
    contraction.  Arrows here run along the fermionic site order.
    """
    psi = qtn.PEPS.rand(Lx, Ly, D, seed=42)      # same seed as panel (a)
    sites, phys = _peps_layout(psi, ax, node_scale=1.25)

    # Virtual bonds: arrow points along increasing site order, i.e. from
    # the tensor that owns the outgoing (non-dual) leg to its neighbour.
    for i in range(Lx):
        for j in range(Ly):
            for ni, nj in ((i + 1, j), (i, j + 1)):
                if ni < Lx and nj < Ly:
                    _bond_arrow(ax, sites[(i, j)], sites[(ni, nj)])

    # Physical legs are oriented too.
    for coo, p_site in sites.items():
        _bond_arrow(ax, p_site, phys[coo], frac=0.34, lw=1.2)
    for (x, y) in phys.values():
        ax.plot([x], [y], 'o', ms=3.4, color=C_PHYS, zorder=5)

    ax.set_title('(b)  fermionic tensor network\noriented bonds carry the '
                 'fermion signs', fontsize=10.5, pad=8)


def panel_neural(ax, Lx=3, Ly=3, D=3):
    """(c) A neural network feeding corrections into the site tensors."""
    psi = qtn.PEPS.rand(Lx, Ly, D, seed=7)
    sites, phys = _peps_layout(psi, ax, node_scale=1.15)

    for (x, y) in phys.values():
        ax.plot([x], [y], 'o', ms=3.0, color=C_PHYS, zorder=5)

    xs = [p[0] for p in sites.values()]
    ys = [p[1] for p in sites.values()]
    xmid = (min(xs) + max(xs)) / 2
    span = max(xs) - min(xs)
    box_w, box_h = 0.34 * span, 0.075 * span
    box_y = max(ys) + 0.30 * span

    # Corrections fanning out into every site tensor (drawn first, so
    # the box and the network sit on top).
    for (x, y) in sites.values():
        ax.add_patch(FancyArrowPatch(
            (xmid, box_y - box_h), (x, y),
            arrowstyle='-|>', mutation_scale=7, lw=0.7,
            color=C_NN, alpha=0.55, zorder=2,
            connectionstyle='arc3,rad=0.12',
        ))

    ax.add_patch(FancyBboxPatch(
        (xmid - box_w, box_y - box_h), 2 * box_w, 2 * box_h,
        boxstyle='round,pad=0.02', facecolor=C_NN,
        edgecolor='0.2', lw=1.2, alpha=0.95, zorder=6,
    ))
    ax.text(xmid, box_y, r'$f_\phi(x)$', ha='center', va='center',
            fontsize=11, zorder=7)

    # The configuration feeding in from above.
    label_y = box_y + 0.20 * span
    ax.annotate(
        r'configuration $x$',
        xy=(xmid, box_y + box_h), xytext=(xmid, label_y),
        ha='center', va='bottom', fontsize=9,
        arrowprops=dict(arrowstyle='-|>', lw=1.1, color='0.3'),
    )

    # quimb fixes the axes limits to the node bounding box, so the
    # overlay above would be clipped without widening them by hand.
    ymin, _ = ax.get_ylim()
    ax.set_ylim(ymin, label_y + 0.12 * span)

    ax.set_title('(c)  neural tensor network state\n'
                 r'$T(x) = T + \eta\, f_\phi(x)$',
                 fontsize=10.5, pad=8)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    mpl.rcParams['font.family'] = 'DejaVu Sans'

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    panel_tensor_network(axes[0])
    panel_fermionic_tn(axes[1])
    panel_neural(axes[2])

    for ax in axes:
        ax.set_axis_off()

    fig.suptitle(
        'vmc_torch  —  one variational Monte Carlo engine '
        'for a family of ansätze',
        fontsize=12.5, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ('png', 'svg'):
        path = os.path.join(OUTDIR, f'ansatz_family.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print('wrote', path)


if __name__ == '__main__':
    main()
