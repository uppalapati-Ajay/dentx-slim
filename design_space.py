#!/usr/bin/env python3
"""
DentX-Slim design-space exploration (no radiation). Answers: why a 1.5 mm fibre-optic plate, and how robust is the thickness target?
 1. Sweep FOP thickness 1.0 to 2.5 mm: lesion CNR with and without pod firmware (mean of 3 noise realisations, dose 200 quanta/px)
 2. Thickness scenarios: target / realistic / conservative stack-ups vs published thicknesses
All parameters are illustrative; the FOP curve is anchored to one published point (2 mm ~ 100x direct-hit reduction).
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dentx_sensor_sim as sim

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
NAVY, TEAL, CORAL, GOLD, GREY = sim.NAVY, sim.TEAL, sim.CORAL, sim.GOLD, sim.GREY
FIXED = 0.5 + 0.1 + 0.5 + 0.25 + 0.6 + 0.55      # all layers except the FOP (mm) = 2.5

def sweep():
    S = sim.build_all(); T, les, lc, det = S["T"], S["lesion"], S["lc"], S["det"]
    darks, gmap, bad = S["cal"]
    fops = [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5]
    res = {"raw": [], "cor": [], "des": []}
    for f in fops:
        r, c, d = [], [], []
        for _ in range(3):
            raw = det.expose(T, 200, f); cor = sim.correct(raw, darks, gmap, bad); des = sim.despeckle(cor)[0]
            r.append(sim.cnr(raw, les, lc)); c.append(sim.cnr(cor, les, lc)); d.append(sim.cnr(des, les, lc))
        res["raw"].append(np.mean(r)); res["cor"].append(np.mean(c)); res["des"].append(np.mean(d))
        print(f"FOP {f:.1f} mm  thickness {FIXED+f:.1f} mm  CNR raw {res['raw'][-1]:.2f}  corrected {res['cor'][-1]:.2f}  +despeckle {res['des'][-1]:.2f}", flush=True)
    return fops, res

def fig_design_space(fops, res):
    th = [FIXED + f for f in fops]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(th, res["raw"], "o--", color=GREY, label="Raw readout")
    ax.plot(th, res["cor"], "s--", color=CORAL, label="Dark + flat + bad-pixel corrected")
    ax.plot(th, res["des"], "o-", color=TEAL, lw=2.4, label="Corrected + despeckle (pod firmware)")
    ax.axvline(5.7, color=NAVY, ls=":", lw=1); ax.text(5.66, 2.0, "5.7 mm: patent\nCCD package", ha="right", fontsize=7, color=NAVY)
    i = fops.index(1.5); ax.plot([th[i]], [res["des"][i]], "o", ms=13, mfc="none", mec=NAVY, mew=2)
    ax.annotate("DentX-Slim\n1.5 mm plate\n4.0 mm total", (th[i], res["des"][i]), textcoords="offset points", xytext=(-8, -58), fontsize=8, color=NAVY, ha="center",
                arrowprops=dict(arrowstyle="-", color=NAVY))
    ax.set_xlabel("In-mouth thickness (mm) = 2.5 mm fixed layers + fibre-optic plate"); ax.set_ylabel("Caries-lesion CNR (dose 200 quanta/px)")
    ax.set_title("Design space: thickness vs image quality", fontsize=10.5)
    ax.grid(alpha=.25); ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    sec = ax.secondary_xaxis("top", functions=(lambda x: x - FIXED, lambda x: x + FIXED)); sec.set_xlabel("Fibre-optic plate thickness (mm)", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_design_space.png", bbox_inches="tight"); plt.close(fig)

def fig_scenarios():
    layers = ["Front shell", "CsI(Tl)", "Fibre-optic plate", "CMOS die", "Flex + attach", "Rear shell", "Margin"]
    cols = [TEAL, GOLD, "#8FA8C8", NAVY, CORAL, GREY, "#C9D3DD"]
    sc = {"Target": [0.5, 0.10, 1.5, 0.50, 0.25, 0.6, 0.55], "Realistic": [0.6, 0.15, 1.8, 0.50, 0.30, 0.7, 0.50],
          "Conservative": [0.7, 0.20, 2.0, 0.725, 0.30, 0.8, 0.50]}
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    names = list(sc); x = np.arange(len(names))
    for n, name in enumerate(names):
        b = 0
        for v, c in zip(sc[name], cols):
            ax.bar(n, v, bottom=b, color=c, width=0.55, edgecolor="white"); b += v
        ax.text(n, b + 0.08, f"{b:.2f} mm", ha="center", fontsize=9, fontweight="bold", color=NAVY)
    for k, (nm, v) in enumerate([("CCD package\n(patent)", 5.7), ("KaVo ProXam iS", 7.1)]):
        ax.bar(3.3 + k, v, color="#5C6B7A", width=0.55); ax.text(3.3 + k, v + 0.08, f"{v:.1f} mm", ha="center", fontsize=9, fontweight="bold", color=NAVY)
    ax.set_xticks(list(x) + [3.3, 4.3]); ax.set_xticklabels(names + ["CCD package\n(patent)", "KaVo\nProXam iS"], fontsize=8)
    ax.set_ylim(0, 8); ax.set_ylabel("In-mouth thickness (mm)"); ax.set_title("Thickness target under three supplier scenarios", fontsize=10.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, layers)], fontsize=6.8, frameon=False, ncol=2, loc="upper left")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_scenarios.png", bbox_inches="tight"); plt.close(fig)
    for n in names: print(n, round(sum(sc[n]), 2), "mm; vs 5.7:", f"{100*(1-sum(sc[n])/5.7):.0f}%", " vs 7.1:", f"{100*(1-sum(sc[n])/7.1):.0f}%")

if __name__ == "__main__":
    fig_scenarios()
    fops, res = sweep()
    fig_design_space(fops, res)
    print("figures written")
