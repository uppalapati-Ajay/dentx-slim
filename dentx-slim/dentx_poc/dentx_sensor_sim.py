#!/usr/bin/env python3
"""
DentX-Slim  --  proof-of-concept simulation (no X-ray radiation involved)
Author : Sai Venkata Ajay Uppalapati

What this simulates
  1. A synthetic intraoral phantom (2 teeth, pulp, caries lesion, metal filling, bone)
  2. A CMOS + CsI + fibre-optic-plate (FOP) detector model for a Size-2 die
     1300 x 1700 px, 20 um pitch (26 x 34 mm), 14-bit ADC  (Hamamatsu S10831 class)
  3. The pod-side firmware pipeline: dark subtraction, flat-field gain correction,
     bad-pixel repair, direct-hit despeckle
  4. The engineering trade-off that drives the whole design:
        thinner FOP  ->  thinner sensor  ->  more direct-hit speckle  ->  fix it in firmware

All noise / material parameters are ILLUSTRATIVE (documented below), calibrated only to
published anchor points (e.g. a 2 mm FOP cuts direct hits ~100x). The aim is to show the
architecture and the trade-off, not to certify image quality.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Polygon
from scipy.ndimage import gaussian_filter, median_filter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- palette
NAVY, TEAL, CORAL, GOLD, GREY, LIGHT = "#0B2545", "#13A89E", "#E4572E", "#F2A900", "#5C6B7A", "#EAF1F7"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GREY, "axes.labelcolor": NAVY,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.titleweight": "bold",
                     "axes.titlecolor": NAVY, "figure.dpi": 150})

# ---------------------------------------------------------------- sensor constants
H, W = 1700, 1300            # rows (V) x cols (H)  -> 34 x 26 mm at 20 um pitch
PITCH_UM = 20
BITS = 14
FULL = 2 ** BITS - 1
GAIN = 15.0                  # DN per absorbed X-ray quantum (illustrative)
READ_NOISE = 6.0             # DN rms
OFFSET_PIX, OFFSET_COL = 25.0, 10.0   # fixed-pattern offsets (DN rms)
PRNU_PIX, PRNU_COL = 0.010, 0.005     # gain non-uniformity (fraction)
DARK_DN = 12.0               # mean dark level (DN)
HOT_FRAC, DEAD_FRAC = 3e-4, 1e-4
DH_AMP = 600.0               # DN added by one direct hit in silicon
DH_Q = 0.5 * 0.03            # P(photon passes scintillator) * P(absorbed in Si) (illustrative)
FOP_MU_PER_MM = np.log(100) / 2.0   # anchor: 2 mm FOP -> ~100x fewer direct hits (patent US7211817 text)


def fop_reduction(t_mm):
    """Direct-hit reduction factor for a fibre-optic plate of thickness t_mm."""
    return float(np.exp(FOP_MU_PER_MM * t_mm))


# ---------------------------------------------------------------- 1. phantom
def make_phantom():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    A = np.full((H, W), 1.1, np.float32)                     # cheek / gum soft tissue (mu*t)

    # alveolar bone with trabecular texture
    crest = 1000 + 25 * np.sin(xx / 140.0)
    bone = 1 / (1 + np.exp(np.clip(-(yy - crest) / 8.0, -50, 50)))
    tex = gaussian_filter(rng.normal(size=(H, W)).astype(np.float32), 6)
    tex = tex / tex.std()
    A += bone * (1.0 + 0.10 * tex)

    centers = [(430, 330, 800, 1480), (880, 330, 800, 1500)]  # cx, crown top, crown bottom, root bottom
    ground = {}
    for k, (cx, yt, yc, yr) in enumerate(centers):
        hw = np.zeros_like(yy)
        crown = (yy >= yt) & (yy < yc)
        s = np.clip((yy - yt) / (yc - yt), 0, 1)
        hw = np.where(crown, 175 * np.sin(0.5 * np.pi * s) ** 0.35, hw)
        root = (yy >= yc) & (yy <= yr)
        r = np.clip((yy - yc) / (yr - yc), 0, 1)
        hw = np.where(root, 38 + 122 * (1 - r) ** 1.2, hw)
        dx = np.abs(xx - cx)
        inside = (hw > 0) & (dx < hw)
        u = np.where(hw > 0, dx / np.maximum(hw, 1), 1)
        tf = np.sqrt(np.clip(1 - u ** 2, 0, 1))                # elliptical thickness profile
        tooth = 2.0 * tf                                       # dentin
        tooth += np.where(crown, 1.1 * np.clip((u - 0.45) / 0.55, 0, 1) ** 1.2 * tf, 0)  # enamel cap
        pulp_hw = np.where(crown, 60, 14) * (1 - 0.45 * np.clip((yy - yc) / (yr - yc), 0, 1))
        pulp = (dx < pulp_hw) & (yy > yt + 160) & (yy < yr - 120)
        tooth = np.where(pulp, tooth - 1.6 * tf, tooth)
        A += np.where(inside, tooth, 0).astype(np.float32)
        ground[k] = (cx, yt, yc, yr)

    # caries lesion on tooth 0 (proximal, low contrast) and metal filling on tooth 1
    cx, yt, yc, yr = ground[0]
    ly, lx = yt + 330, cx + 118
    lesion = ((xx - lx) / 34.0) ** 2 + ((yy - ly) / 52.0) ** 2 < 1
    A -= 0.16 * lesion
    cx2, yt2, _, _ = ground[1]
    fill = (np.abs(xx - cx2) < 70) & (yy > yt2 + 70) & (yy < yt2 + 190)
    A += 4.5 * fill
    return np.exp(-A).astype(np.float32), lesion, (ly, lx)


# ---------------------------------------------------------------- 2. detector model
class Detector:
    """Per-sensor fixed properties (the things calibration has to remove)."""
    def __init__(self):
        self.off = (rng.normal(0, OFFSET_PIX, (H, W)) + rng.normal(0, OFFSET_COL, (1, W))).astype(np.float32)
        self.dark_rate = DARK_DN + np.abs(rng.normal(0, 3, (H, W))).astype(np.float32)
        hot = rng.random((H, W)) < HOT_FRAC
        self.dark_rate[hot] += rng.uniform(800, 3000, hot.sum())
        self.gain = (1 + rng.normal(0, PRNU_PIX, (H, W)) + rng.normal(0, PRNU_COL, (1, W))).astype(np.float32)
        # smooth vignetting (optics / FOP) so flat-field correction has something real to fix
        yy, xx = np.mgrid[0:H, 0:W]
        self.gain *= (1 - 0.06 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2) * 4).astype(np.float32)
        self.dead = rng.random((H, W)) < DEAD_FRAC
        self.gain[self.dead] *= 0.05

    def expose(self, T, n0, fop_mm):
        """T: transmission map (1 = open beam). n0: quanta per pixel in open beam."""
        q = rng.poisson(n0 * T).astype(np.float32)
        sig = gaussian_filter(q, 0.8)                           # scintillator + FOP spread (MTF)
        sig = sig * GAIN * self.gain
        R = fop_reduction(fop_mm)
        hits = rng.poisson(n0 * T * DH_Q / R).astype(np.float32) * DH_AMP * rng.exponential(1.0, (H, W))
        out = sig + hits + self.dark_rate + self.off + rng.normal(0, READ_NOISE, (H, W))
        return np.clip(np.round(out), 0, FULL).astype(np.float32)

    def dark(self):
        return self.expose(np.zeros((H, W), np.float32), 0, 2.0)


# ---------------------------------------------------------------- 3. pod firmware pipeline
def build_calibration(det, n_frames=16, n0_flat=300):
    darks = np.mean([det.dark() for _ in range(n_frames)], 0)
    flats = np.mean([det.expose(np.ones((H, W), np.float32), n0_flat, 2.0) for _ in range(n_frames)], 0)
    flat_sig = flats - darks
    gain_map = flat_sig / np.median(flat_sig)
    d_std = np.std(darks)
    bad = (darks > np.median(darks) + 8 * d_std) | (gain_map < 0.5)
    return darks, gain_map, bad


def correct(raw, darks, gain_map, bad):
    img = (raw - darks) / np.maximum(gain_map, 0.2)
    med = median_filter(img, 3)
    img = np.where(bad, med, img)
    return img


def despeckle(img, k=4.0):
    """Direct-hit removal: replace positive outliers vs local median using the known noise model."""
    med = median_filter(img, 3)
    sigma = np.sqrt(np.maximum(GAIN * np.maximum(med, 0), 0) + READ_NOISE ** 2) * 1.25
    mask = (img - med) > k * sigma
    return np.where(mask, med, img), mask.mean()


# ---------------------------------------------------------------- metrics
def cnr(img, lesion, center):
    """Lesion contrast / local noise.  Contrast = mean(lesion) - mean(surrounding ring);
    noise = std of the high-pass residual (img - gaussian(img, 3)) in the surrounding box,
    so anatomy does not inflate the noise estimate but speckle / FPN / photon noise do."""
    from scipy.ndimage import binary_dilation
    ly, lx = center
    box = np.zeros_like(lesion)
    box[ly - 90:ly + 90, lx - 60:lx + 60] = True
    ring = box & ~binary_dilation(lesion, iterations=14)
    contrast = abs(img[lesion].mean() - img[ring].mean())
    resid = img - gaussian_filter(img, 3)
    noise = resid[box].std()
    return contrast / noise


def psnr(img, ref):
    a = img / max(np.percentile(ref, 99.5), 1)
    b = ref / max(np.percentile(ref, 99.5), 1)
    mse = np.mean((np.clip(a, 0, 1.5) - np.clip(b, 0, 1.5)) ** 2)
    return 10 * np.log10(1.0 / mse)


def norm_view(img, lo=0.5, hi=99.7, ref=None):
    a, b = np.percentile(img if ref is None else ref, [lo, hi])
    return np.clip((img - a) / (b - a), 0, 1)


def build_all():
    T, lesion, lc = make_phantom()
    det = Detector()
    darks, gmap, bad = build_calibration(det)
    # noise-free reference at the dose used for the main figure
    n0 = 200
    ideal = gaussian_filter(n0 * T, 0.8) * GAIN
    res = {}
    for name, fop in [("2.0 mm FOP", 2.0), ("1.5 mm FOP", 1.5)]:
        raw = det.expose(T, n0, fop)
        cor = correct(raw, darks, gmap, bad)
        des, frac = despeckle(cor)
        res[name] = dict(raw=raw, cor=cor, des=des, frac=frac, fop=fop)
    return dict(T=T, lesion=lesion, lc=lc, det=det, cal=(darks, gmap, bad), ideal=ideal, n0=n0, res=res)


# ---------------------------------------------------------------- figures
def fig_pipeline(S):
    r2, r15 = S["res"]["2.0 mm FOP"], S["res"]["1.5 mm FOP"]
    ly, lx = S["lc"]
    crop = (slice(ly - 260, ly + 260), slice(lx - 330, lx + 220))
    panels = [("Ideal (noise-free)", S["ideal"]), ("Raw readout\n(offsets, gain, hot px)", r2["raw"]),
              ("Dark + flat + bad-pixel\ncorrected, 2.0 mm FOP", r2["cor"]),
              ("1.5 mm FOP corrected\n(direct-hit speckle)", r15["cor"]),
              ("1.5 mm FOP + despeckle\n(DentX-Slim pipeline)", r15["des"])]
    fig, ax = plt.subplots(1, 5, figsize=(13.5, 3.9))
    for i, (a, (t, im)) in enumerate(zip(ax, panels)):
        ref = None if i == 1 else S["ideal"][crop]
        a.imshow(norm_view(im[crop], ref=ref), cmap="gray_r", vmin=0, vmax=1)
        a.set_title(t, fontsize=8.5)
        a.axis("off")
    fig.suptitle("Simulated imaging chain around the caries lesion (open crop of the 1300x1700 frame, dose = 200 quanta/px)",
                 fontsize=9.5, color=NAVY, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def fig_phantom_full(S):
    r = S["res"]["1.5 mm FOP"]
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 5.3))
    ax[0].imshow(norm_view(r["raw"]), cmap="gray_r"); ax[0].set_title("Raw frame from die (1.5 mm FOP)")
    ax[1].imshow(norm_view(r["des"], ref=S["ideal"]), cmap="gray_r"); ax[1].set_title("After pod firmware pipeline")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    ax[0].set_xlabel("26 mm  (1300 px)"); ax[1].set_xlabel("26 mm  (1300 px)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_fullframe.png", bbox_inches="tight"); plt.close(fig)


def sweep(S):
    T, lesion, lc = S["T"], S["lesion"], S["lc"]
    det = S["det"]
    darks, gmap, bad = S["cal"]
    doses = [60, 100, 200, 400, 800]
    out = {k: [] for k in ["raw2", "cor2", "des2", "raw15", "cor15", "des15"]}
    for n0 in doses:
        for fop, tag in [(2.0, "2"), (1.5, "15")]:
            raw = det.expose(T, n0, fop)
            cor = correct(raw, darks, gmap, bad)
            out["raw" + tag].append(cnr(raw, lesion, lc))
            out["cor" + tag].append(cnr(cor, lesion, lc))
            out["des" + tag].append(cnr(despeckle(cor)[0], lesion, lc))
    return doses, out


def fig_sweep(doses, out):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(doses, out["raw2"], "o--", color=GREY, label="Raw, 2.0 mm FOP")
    ax.plot(doses, out["cor2"], "o-", color=NAVY, label="Corrected, 2.0 mm FOP (conventional stack)")
    ax.plot(doses, out["des2"], "o:", color=NAVY, label="Corrected + despeckle, 2.0 mm FOP (fair baseline)")
    ax.plot(doses, out["cor15"], "s--", color=CORAL, label="Corrected, 1.5 mm FOP (no despeckle)")
    ax.plot(doses, out["des15"], "s-", color=TEAL, lw=2.2, label="Corrected + despeckle, 1.5 mm FOP (DentX-Slim)")
    ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(doses); ax.set_xticklabels(doses)
    ax.set_xlabel("Dose proxy: open-beam X-ray quanta per pixel"); ax.set_ylabel("Caries-lesion CNR (per pixel)")
    ax.set_title("Firmware recovers the contrast lost by a thinner FOP", fontsize=10)
    ax.set_ylim(0, 4.6)
    ax.grid(alpha=.25); ax.legend(fontsize=7.2, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_cnr_sweep.png", bbox_inches="tight"); plt.close(fig)


def fig_fop_tradeoff():
    t = np.linspace(0.5, 2.5, 100)
    R = np.array([fop_reduction(x) for x in t])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.semilogy(t, R, color=NAVY, lw=2)
    for tt, lab, c in [(2.0, "2.0 mm (typ. conventional)\n~100x", GREY), (1.5, "1.5 mm (proposed)\n~32x", TEAL),
                       (1.46, "", TEAL)]:
        if lab:
            ax.plot([tt], [fop_reduction(tt)], "o", color=c, ms=8)
            ax.annotate(lab, (tt, fop_reduction(tt)), textcoords="offset points", xytext=(10, -26 if tt == 2.0 else 6), fontsize=7.5, color=c)
    ax.set_xlabel("Fibre-optic-plate thickness (mm)"); ax.set_ylabel("Direct-hit reduction factor (x)")
    ax.set_title("FOP thickness vs direct-hit suppression (exponential model)", fontsize=10)
    ax.grid(alpha=.25, which="both")
    ax.text(0.02, 0.97, "Model anchored to one published data point (2 mm ~ 100x);\nthe curve shape is an engineering estimate to be validated.",
            fontsize=6.8, color=GREY, transform=ax.transAxes, va="top")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_fop_tradeoff.png", bbox_inches="tight"); plt.close(fig)


def fig_thickness():
    layers = [("Front shell (soft-touch)", 0.50, TEAL), ("CsI(Tl) scintillator", 0.10, GOLD),
              ("Fibre-optic plate", 1.50, "#8FA8C8"), ("CMOS die (thinned)", 0.50, NAVY),
              ("Flex + die attach", 0.25, CORAL), ("Rear shield / shell", 0.60, GREY), ("Tolerance margin", 0.55, "#C9D3DD")]
    tot = sum(l[1] for l in layers)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.0, 1.25]})
    # stacked bar
    b = 0
    for name, th, c in layers:
        ax[0].bar(0, th, bottom=b, color=c, width=0.55, edgecolor="white")
        ax[0].text(0.33, b + th / 2, f"{name}  {th:.2f}", va="center", fontsize=7.6, color=NAVY)
        b += th
    ax[0].set_xlim(-0.5, 2.1); ax[0].set_ylim(0, 4.6); ax[0].set_xticks([]); ax[0].set_ylabel("Thickness (mm)")
    ax[0].set_title(f"DentX-Slim stack-up: {tot:.1f} mm design target", fontsize=10)
    for spine in ["top", "right", "bottom"]:
        ax[0].spines[spine].set_visible(False)
    # comparison bars
    names = ["Film\n(size 2)", "PSP plate\n(ScanX)", "DentX-Slim\n(target)", "CCD module\nHamamatsu S10128", "CCD package\n(patent ref.)", "KaVo\nProXam iS"]
    vals = [1.0, 0.4, tot, 5.1, 5.7, 7.1]
    cols = [LIGHT, LIGHT, TEAL, GREY, GREY, GREY]
    ax[1].bar(names, vals, color=cols, edgecolor=NAVY, linewidth=0.6)
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.12, f"{v:.1f}", ha="center", fontsize=8, color=NAVY, fontweight="bold")
    ax[1].set_ylim(0, 8.2); ax[1].set_ylabel("Thickness (mm)")
    ax[1].set_title("Intra-oral thickness: published products vs target", fontsize=10)
    ax[1].tick_params(axis="x", labelsize=7)
    for spine in ["top", "right"]:
        ax[1].spines[spine].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_thickness.png", bbox_inches="tight"); plt.close(fig)
    return tot


def fig_crosssection():
    fig, ax = plt.subplots(2, 1, figsize=(8.6, 4.6))
    # conventional: boxy
    def draw(a, w, th_c, taper, color, title, rad=0):
        a.add_patch(Polygon([[0, 0], [w, 0], [w, th_c - taper], [w - 4, th_c], [4, th_c], [0, th_c - taper]],
                            closed=True, fc=color, ec=NAVY, lw=1.2))
        a.set_xlim(-3, w + 3); a.set_ylim(-1.5, 9); a.set_aspect("equal"); a.axis("off")
        a.set_title(title, fontsize=9.5, loc="left")
        a.annotate("", xy=(w + 1.6, 0), xytext=(w + 1.6, th_c), arrowprops=dict(arrowstyle="<->", color=CORAL))
        a.text(w + 2.0, th_c / 2, f"{th_c:.1f} mm", color=CORAL, fontsize=9, va="center", fontweight="bold")
    draw(ax[0], 44, 7.1, 0.3, "#C9D3DD", "Conventional rigid Size-2 housing (KaVo ProXam iS dims: 44.1 x 30.4 x 7.1 mm)")
    draw(ax[1], 44, 4.0, 1.8, "#BFE8E3", "DentX-Slim: tapered rim, soft front shell, thinned stack (design target)")
    ax[1].text(1.5, 4.4, "tapered\nrim", fontsize=7, color=NAVY, ha="left")
    ax[1].text(40.5, 4.4, "side-exit\ncable", fontsize=7, color=NAVY, ha="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_crosssection.png", bbox_inches="tight"); plt.close(fig)


def fig_product_views():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    a = ax[0]
    a.set_xlim(-4, 74); a.set_ylim(-6, 78); a.set_aspect("equal"); a.axis("off")
    a.set_title("Top view (patient-facing side), same 44 x 30 mm footprint", fontsize=9.5, loc="left")

    def top(y0, rounded, label, dentx):
        r = 5.0 if dentx else 0.8
        a.add_patch(FancyBboxPatch((0, y0), 44, 30, boxstyle=f"round,pad=0,rounding_size={r}",
                                   fc="#BFE8E3" if dentx else "#C9D3DD", ec=NAVY, lw=1.4))
        if dentx:
            a.add_patch(FancyBboxPatch((2, y0 + 2), 40, 26, boxstyle="round,pad=0,rounding_size=3.0",
                                       fc="#7FD1C8", ec=TEAL, lw=1.0))
        a.add_patch(Rectangle((5, y0 + 2), 34, 26, fc="none", ec=CORAL, lw=1.1, ls="--"))
        a.text(22, y0 + 15, "active area\n34 x 26 mm", ha="center", va="center", fontsize=7.5, color=CORAL, fontweight="bold")
        if dentx:
            a.text(22, y0 - 2.6, "rounded corners >= 3 mm,  tapered rim (4.0 to about 2 mm)", ha="center", fontsize=7, color=NAVY)
            # flex tail from lower-right corner + magnetic connector
            a.add_patch(Rectangle((44, y0 + 3), 15, 5, fc="#E8B04A", ec=NAVY, lw=0.8))
            a.add_patch(Rectangle((59, y0 + 1.5), 6, 8, fc=CORAL, ec=NAVY, lw=0.8))
            a.text(62, y0 + 12.2, "magnetic\nbreak-away", ha="center", fontsize=6.4, color=CORAL, fontweight="bold")
            a.text(51.5, y0 - 0.5, "side-exit flex", ha="center", fontsize=6.4, color=NAVY)
        else:
            a.add_patch(Rectangle((44, y0 + 13), 24, 4, fc=GREY, ec=NAVY, lw=0.8))
            a.text(56, y0 + 19.2, "round cable, centre exit", ha="center", fontsize=6.4, color=NAVY)
            a.text(22, y0 - 2.6, "square-ish edges, 0.8 mm radius (illustrative)", ha="center", fontsize=7, color=NAVY)
        a.text(0, y0 + 31.5, label, fontsize=8.5, color=NAVY, fontweight="bold")

    top(42, False, "Conventional rigid Size-2 housing", False)
    top(2, True, "DentX-Slim (design target)", True)

    b = ax[1]
    b.set_xlim(-3, 52); b.set_ylim(-3, 22); b.set_aspect("equal"); b.axis("off")
    b.set_title("Side view (to scale)", fontsize=9.5, loc="left")
    def side(y0, th, taper, color, label):
        b.add_patch(Polygon([[0, y0], [44, y0], [44, y0 + th - taper], [44 - max(taper * 1.1, 0.5), y0 + th],
                             [max(taper * 1.1, 0.5), y0 + th], [0, y0 + th - taper]], closed=True, fc=color, ec=NAVY, lw=1.2))
        b.annotate("", xy=(46.5, y0), xytext=(46.5, y0 + th), arrowprops=dict(arrowstyle="<->", color=CORAL))
        b.text(47.5, y0 + th / 2, f"{th:.1f} mm", color=CORAL, fontsize=9, va="center", fontweight="bold")
        b.text(0, y0 + th + 0.8, label, fontsize=8, color=NAVY, fontweight="bold")
    side(12, 7.1, 0.3, "#C9D3DD", "Conventional (KaVo dims)")
    side(0, 4.0, 1.9, "#BFE8E3", "DentX-Slim (target)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_product_views.png", bbox_inches="tight"); plt.close(fig)


def box(a, x, y, w, h, text, fc, tc="white", fs=8):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="none"))
    a.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold", linespacing=1.25)


def arrow(a, x1, y1, x2, y2, label="", c=NAVY):
    a.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=c, lw=1.4))
    if label:
        a.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.16, label, ha="center", fontsize=6.8, color=c)


def fig_architecture():
    fig, a = plt.subplots(figsize=(11, 4.6))
    a.set_xlim(0, 11); a.set_ylim(0, 4.6); a.axis("off")
    # intraoral zone
    a.add_patch(Rectangle((0.1, 0.35), 3.7, 3.9, fc="#E3F6F4", ec=TEAL, lw=1.5, ls="--"))
    a.text(0.25, 4.0, "INTRA-ORAL HEAD  (no MCU, no battery)", fontsize=8.5, color=TEAL, fontweight="bold")
    box(a, 0.35, 2.85, 3.2, 0.6, "X-ray  ->  CsI(Tl) scintillator", GOLD, NAVY, 8)
    box(a, 0.35, 2.05, 3.2, 0.6, "Fibre-optic plate 1.5 mm\n(blocks direct hits)", "#8FA8C8", NAVY, 7.6)
    box(a, 0.35, 1.15, 3.2, 0.7, "CMOS die 1300x1700, 20 um\n14-bit ADC + LVDS + trigger photodiode", NAVY, "white", 7.4)
    box(a, 0.35, 0.5, 3.2, 0.45, "Flex interposer + decoupling only", CORAL, "white", 7.4)
    for y in (2.85, 2.05, 1.15):
        arrow(a, 1.95, y, 1.95, y - 0.2 if y > 1.15 else y - 0.2)
    # cable
    a.add_patch(FancyBboxPatch((3.95, 1.55), 1.55, 0.5, boxstyle="round,pad=0.02,rounding_size=0.2", fc="#C9D3DD", ec=GREY))
    a.text(4.72, 1.8, "thin flex/micro-coax\nside-exit, 1 m", ha="center", va="center", fontsize=6.6, color=NAVY)
    a.text(4.72, 1.25, "magnetic break-away\nconnector", ha="center", fontsize=6.6, color=CORAL, fontweight="bold")
    arrow(a, 3.55, 1.8, 3.95, 1.8); arrow(a, 5.5, 1.8, 5.85, 1.8)
    # pod
    a.add_patch(Rectangle((5.85, 0.35), 3.1, 3.9, fc="#E8ECF6", ec=NAVY, lw=1.5, ls="--"))
    a.text(6.0, 4.0, "INTERFACE POD (chairside)", fontsize=8.5, color=NAVY, fontweight="bold")
    box(a, 6.05, 2.95, 2.7, 0.55, "LVDS receiver + MCLK/MST drive", GREY, "white", 7.4)
    box(a, 6.05, 2.15, 2.7, 0.6, "Deserialiser (MCU/FPGA)\nX-ray trigger state machine", NAVY, "white", 7.2)
    box(a, 6.05, 1.3, 2.7, 0.65, "SDRAM frame buffer + QSPI flash\ncalibration maps (dark/gain/bad px)", NAVY, "white", 6.9)
    box(a, 6.05, 0.5, 2.7, 0.6, "Dark / flat / despeckle pipeline", TEAL, "white", 7.4)
    # outputs
    box(a, 9.3, 2.6, 1.55, 0.7, "USB-C (HS)\nDICOM-ready host", NAVY, "white", 7.2)
    box(a, 9.3, 1.3, 1.55, 0.7, "Optional Wi-Fi\nmodule (same head)", "#8FA8C8", NAVY, 7.2)
    arrow(a, 8.95, 2.9, 9.3, 2.95); arrow(a, 8.95, 1.7, 9.3, 1.65)
    fig.savefig(f"{OUT}/fig_architecture.png", bbox_inches="tight"); plt.close(fig)


def fig_state_machine():
    fig, a = plt.subplots(figsize=(10.6, 3.2))
    a.set_xlim(0, 10.6); a.set_ylim(0, 3.2); a.axis("off")
    names = [("IDLE / ARMED", "poll trigger photodiode\n(0.357 MHz clock)"), ("DETECT", "photodiode > threshold\nfor N samples"),
             ("INTEGRATE", "global-shutter exposure\nuntil photodiode falls"), ("READ-OUT", "14 MCLK per px\n~1.55 s @ 20 MHz"),
             ("CORRECT", "dark - flat - bad px\n- despeckle (streamed)"), ("TRANSFER", "USB / Wi-Fi to PC\nthen re-arm")]
    x = 0.15
    pos = []
    for i, (n, d) in enumerate(names):
        c = [GREY, GOLD, CORAL, NAVY, TEAL, NAVY][i]
        tc = NAVY if c == GOLD else "white"
        box(a, x, 1.45, 1.55, 0.75, n, c, tc, 8)
        a.text(x + 0.775, 1.1, d, ha="center", va="top", fontsize=6.8, color=NAVY)
        pos.append(x)
        if i:
            arrow(a, pos[i - 1] + 1.55, 1.82, x, 1.82)
        x += 1.75
    a.add_patch(FancyArrowPatch((pos[-1] + 0.8, 2.22), (pos[0] + 0.8, 2.22), connectionstyle="arc3,rad=0.18",
                                arrowstyle="-|>", mutation_scale=11, color=NAVY, lw=1.3))
    a.text(5.3, 2.95, "timeout / false trigger -> back to ARMED (no image stored)", ha="center", fontsize=7.4, color=GREY)
    fig.savefig(f"{OUT}/fig_state_machine.png", bbox_inches="tight"); plt.close(fig)


def fig_timing():
    mclk = np.array([20, 25, 30, 35, 40])
    px = H * W
    t = px * BITS / (mclk * 1e6)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar([str(m) for m in mclk], t, color=[NAVY, "#3A5A85", "#5F82B0", "#8FA8C8", TEAL])
    for i, v in enumerate(t):
        ax.text(i, v + 0.03, f"{v:.2f} s", ha="center", fontsize=8, color=NAVY)
    ax.set_xlabel("Master clock MCLK (MHz)  -  datasheet range 1 to 40 MHz"); ax.set_ylabel("Frame read-out time (s)")
    ax.set_title("Read-out time = pixels x 14 clocks / MCLK", fontsize=10)
    ax.set_ylim(0, 1.9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_timing.png", bbox_inches="tight"); plt.close(fig)
    return px, t


if __name__ == "__main__":
    print("Building phantom, detector and calibration ...")
    S = build_all()
    for k, r in S["res"].items():
        lc = S["lc"]
        print(f"{k}: raw CNR={cnr(r['raw'], S['lesion'], lc):.2f}  corrected CNR={cnr(r['cor'], S['lesion'], lc):.2f}  "
              f"despeckled CNR={cnr(r['des'], S['lesion'], lc):.2f}  PSNR raw/cor/des = "
              f"{psnr(r['raw'], S['ideal']):.1f}/{psnr(r['cor'], S['ideal']):.1f}/{psnr(r['des'], S['ideal']):.1f} dB  "
              f"despeckle flagged {100 * r['frac']:.2f}% px")
    fig_pipeline(S); fig_phantom_full(S)
    doses, out = sweep(S)
    fig_sweep(doses, out)
    print("dose sweep (quanta/px):", doses)
    for k, v in out.items():
        print(" ", k, [round(x, 2) for x in v])
    fig_fop_tradeoff()
    tot = fig_thickness(); print("stack-up total (mm):", round(tot, 2))
    fig_crosssection(); fig_product_views(); fig_architecture(); fig_state_machine()
    px, t = fig_timing()
    print("pixels:", px, " bits/frame (Mbit):", round(px * BITS / 1e6, 2), " readout (s):", [round(x, 2) for x in t])
    print("Figures written to", OUT)
