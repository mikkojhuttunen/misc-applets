"""New Fig. 2 and Fig. 3 artifacts: copies of the current pages (step-10 spec cases) with the step-11 robustness
panels added below the main figure, drawn by one small SVG renderer (separate <text> elements for sub/superscripts,
so svg2pdf places them correctly) with PDF, SVG and CSV export.

python scripts/build_robust_artifacts.py FIG2_SRC.html FIG3_SRC.html OUT_DIR
  FIG2_SRC / FIG3_SRC: the saved HTML of the published Fig. 2 / Fig. 3 artifacts (output of build_fig2_artifact.py and
  build_fig3_artifact.py). Writes OUT_DIR/fig2_robust.html and OUT_DIR/fig3_robust.html.
Fig. 2 (g)-(i): static fringe, both dithers against the wavelength-dither depth (600 nm) and against film thickness
(20 pm), angle dither alone against film thickness. Fig. 3 (c)-(d): NH3 limit at 1 s against film thickness, both
dithers and angle dither alone. Regular cell vs chaotic median and quartiles of 30 layouts.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robustness_data import build  # noqa: E402

RENDER = r"""
  /* ---------------------------------------------------------------- robustness panels (QUIET step 11) */
  const ROB = JSON.parse(document.getElementById("robdata").textContent);
  const ROB_FONT = `Arial, "Liberation Sans", Arimo, Helvetica, sans-serif`;
  const robCtx = document.createElement("canvas").getContext("2d");
  const robTw = (s, size) => { robCtx.font = `${size}px ${ROB_FONT}`; return robCtx.measureText(s).width; };
  let robLast = null;
  function robSvg(panels, Wcm, Hcm, fs, legend) {
    const W = Wcm / 2.54 * 72, H = Hcm / 2.54 * 72, n = panels.length, o = [];
    const f2 = v => String(Math.round(v * 100) / 100), esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
    /* text as runs [[string, 0 | 1 (superscript) | -1 (subscript)]]; every run is its own <text> at an explicit x */
    const TR = (x, y, runs, a = {}) => {
      const size = a.fs || fs, sz = r => r[1] ? size * 0.7 : size, w = runs.map(r => robTw(r[0], sz(r))), tot = w.reduce((p, q) => p + q, 0);
      let cx = x - (a.anchor === "middle" ? tot / 2 : a.anchor === "end" ? tot : 0);
      runs.forEach((r, i) => {
        const dy = r[1] > 0 ? -0.38 * size : r[1] < 0 ? 0.2 * size : 0;
        o.push(`<text x="${f2(cx)}" y="${f2(y + dy)}" font-size="${f2(sz(r))}" fill="${a.fill || "#000"}"${a.rot ? ` transform="rotate(-90 ${f2(x)} ${f2(y)})"` : ""} xml:space="preserve">${esc(r[0])}</text>`);
        cx += w[i];
      });
    };
    const T = (x, y, s, a) => TR(x, y, [[s, 0]], a);
    /* dashed horizontal line as solid segments (svg2pdf loses the stroke colour of some dashed paths) */
    const dashH = (xa, xb, y, col, lw, on = 4, off = 2.5) => { let d = ""; for (let x = xa; x < xb - 1e-6; x += on + off) d += `M${f2(x)} ${f2(y)}H${f2(Math.min(x + on, xb))}`; o.push(`<path d="${d}" stroke="${col}" stroke-width="${lw}" fill="none"/>`); };
    const cw = W / n;
    panels.forEach((p, i) => {
      const x0 = i * cw + (p.ylab ? 3.6 * fs + 6 : 2.6 * fs + 6), x1 = (i + 1) * cw - 0.8 * fs, y0 = (legend ? 1.9 : 0.7) * fs, y1 = H - 2.8 * fs;
      const tx = v => (p.xlog ? Math.log10(v) : v), ty = v => (p.ylog ? Math.log10(v) : v);
      const X = v => x0 + (tx(v) - tx(p.xr[0])) / (tx(p.xr[1]) - tx(p.xr[0])) * (x1 - x0);
      const Y = v => y1 - (ty(Math.min(Math.max(v, p.yr[0]), p.yr[1])) - ty(p.yr[0])) / (ty(p.yr[1]) - ty(p.yr[0])) * (y1 - y0);
      const P = (xs, ys) => xs.map((x, j) => `${j ? "L" : "M"}${f2(X(x))} ${f2(Y(ys[j]))}`).join("");
      if (p.vspan) o.push(`<rect x="${f2(X(p.vspan[0]))}" y="${f2(y0)}" width="${f2(X(p.vspan[1]) - X(p.vspan[0]))}" height="${f2(y1 - y0)}" fill="#000000" fill-opacity="0.07"/>`);
      for (const b of p.bands || []) o.push(`<path d="${P(b.x, b.hi)}L${P(b.x.slice().reverse(), b.lo.slice().reverse()).slice(1)}Z" fill="${b.col}" fill-opacity="${b.a}" stroke="none"/>`);
      for (const h of p.hlines || []) dashH(x0, x1, Y(h.y), h.col, 1);
      for (const s of p.series || []) {
        o.push(`<path d="${P(s.x, s.y)}" fill="none" stroke="${s.col}" stroke-width="${s.lw}" stroke-linejoin="round" stroke-linecap="round"${s.dash ? ` stroke-dasharray="${s.dash}"` : ""}/>`);
        if (s.mk) s.x.forEach((x, j) => o.push(`<circle cx="${f2(X(x))}" cy="${f2(Y(s.y[j]))}" r="${s.mk}" fill="${s.col}" stroke="${s.col}" stroke-width="0.3"/>`));
      }
      o.push(`<rect x="${f2(x0)}" y="${f2(y0)}" width="${f2(x1 - x0)}" height="${f2(y1 - y0)}" fill="none" stroke="#000" stroke-width="0.8"/>`);
      for (const [v, lab] of p.xticks) { const x = X(v); o.push(`<path d="M${f2(x)} ${f2(y1)}v3.5" stroke="#000" stroke-width="0.8"/>`); T(x, y1 + 3.5 + 0.95 * fs, lab, { anchor: "middle" }); }
      if (p.ylog) for (let k = Math.ceil(Math.log10(p.yr[0]) - 1e-9); k <= Math.floor(Math.log10(p.yr[1]) + 1e-9); k++) {
        const y = Y(10 ** k); o.push(`<path d="M${f2(x0)} ${f2(y)}h-3.5" stroke="#000" stroke-width="0.8"/>`);
        TR(x0 - 5.5, y + 0.35 * fs, [["10", 0], [k < 0 ? "−" + -k : String(k), 1]], { anchor: "end" });
      }
      TR((x0 + x1) / 2, H - 0.45 * fs, p.xlab, { anchor: "middle" });
      if (p.ylab) TR(x0 - 2.45 * fs - 6, (y0 + y1) / 2, p.ylab, { anchor: "middle", rot: true });
      T(x0 + 0.4 * fs, y0 + 1.1 * fs, p.title);
    });
    if (legend) {                                       /* one legend row above the panels, centred */
      const lf = fs * 0.9, ws = legend.map(g => 2.0 * fs + robTw(g.label, lf)), gap = 1.4 * fs, tot = ws.reduce((a, b) => a + b, 0) + gap * (legend.length - 1);
      let lx = (W - tot) / 2; const ly = 1.05 * fs;
      legend.forEach((g, j) => {
        if (g.band) o.push(`<rect x="${f2(lx)}" y="${f2(ly - 0.62 * fs)}" width="${f2(1.6 * fs)}" height="${f2(0.75 * fs)}" fill="${g.col}" fill-opacity="${g.a}"/>`);
        else if (g.dash) dashH(lx, lx + 1.6 * fs, ly - 0.3 * fs, g.col, g.lw, 3, 2);
        else o.push(`<path d="M${f2(lx)} ${f2(ly - 0.3 * fs)}h${f2(1.6 * fs)}" stroke="${g.col}" stroke-width="${g.lw}"/>`);
        T(lx + 2.0 * fs, ly, g.label, { fs: lf });
        lx += ws[j] + gap;
      });
    }
    return { svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${f2(W)} ${f2(H)}" font-family='${ROB_FONT}' role="img" aria-label="${esc(ROB_CFG.aria)}"><rect width="${f2(W)}" height="${f2(H)}" fill="#ffffff"/>${o.join("")}</svg>`, W, H };
  }
  const robSize = () => ({ w: +$("robW").value, h: +$("robH").value, fs: +$("robFs").value });
  function robDraw() {
    const { w, h, fs } = robSize(), r = robSvg(ROB_CFG.panels(), w, h, fs, ROB_CFG.legend());
    robLast = r; $("robSheet").innerHTML = r.svg;
  }
  async function robPdf() {
    const st = $("robStatus"); st.textContent = "Writing the PDF…";
    try {
      await ROB_CFG.loadPdfLibs();
      const r = robLast, host = document.createElement("div");
      host.style.cssText = `position:absolute;left:-99999px;top:0;width:${r.W}px;height:${r.H}px;`;
      host.innerHTML = r.svg.replace("<svg ", `<svg width="${r.W}" height="${r.H}" `);
      document.body.appendChild(host);
      const svg = host.firstElementChild;
      const doc = new window.jspdf.jsPDF({ unit: "pt", format: [r.W, r.H], orientation: r.W >= r.H ? "landscape" : "portrait", compress: true });
      ROB_CFG.addFont(doc); svg.setAttribute("font-family", ROB_CFG.pdfFont);
      try { if (typeof doc.svg === "function") await doc.svg(svg, { x: 0, y: 0, width: r.W, height: r.H }); else await window.svg2pdf.svg2pdf(svg, doc, { x: 0, y: 0, width: r.W, height: r.H }); }
      finally { host.remove(); }
      await ROB_CFG.save(`${ROB_CFG.name}_${robSize().w}x${robSize().h}cm.pdf`, doc.output("blob"), st);
    } catch (e) { st.textContent = `PDF failed: ${e && e.message || e}`; }
  }
  $("robPdf").onclick = robPdf;
  $("robSvgBtn").onclick = () => ROB_CFG.save(`${ROB_CFG.name}_${robSize().w}x${robSize().h}cm.svg`,
    new Blob(['<?xml version="1.0" encoding="UTF-8"?>\n' + robLast.svg.replace("<svg ", `<svg width="${robSize().w}cm" height="${robSize().h}cm" `)], { type: "image/svg+xml" }), $("robStatus"));
  $("robCsv").onclick = () => ROB_CFG.save(`${ROB_CFG.name}_data.csv`, new Blob([ROB_CFG.csv()], { type: "text/csv" }), $("robStatus"));
  for (const id of ["robW", "robH", "robFs"]) $(id).addEventListener("input", robDraw);
  $("robTbl").innerHTML = ROB_CFG.table();
  robDraw();
"""

ROB_HTML = """
      <div class="panel" id="robPanel">
        <h2>{h2}</h2>
        <div class="sheetwrap"><div class="sheet" id="robSheet"></div></div>
        <div class="btns">
          <label class="sel mono" for="robW">width (cm) <input id="robW" type="number" min="8" max="28" step="0.1" value="16.8" style="width:5em"></label>
          <label class="sel mono" for="robH">height (cm) <input id="robH" type="number" min="3" max="15" step="0.1" value="{hdef}" style="width:5em"></label>
          <label class="sel mono" for="robFs">font (pt) <input id="robFs" type="number" min="6" max="14" step="0.5" value="10" style="width:4.5em"></label>
          <button type="button" id="robPdf">Download PDF</button><button type="button" id="robSvgBtn">Download SVG</button><button type="button" id="robCsv">Download data (CSV)</button>
          <span class="status" id="robStatus" role="status"></span>
        </div>
        <p class="hint">{hint}</p>
        <div class="tbl"><table class="vals" id="robTbl"></table></div>
      </div>"""

TABLE_JS = r"""
  const robE = v => (v >= 1e4 || v < 0.01 ? v.toExponential(1) : +v.toPrecision(2)), robX = v => v.toExponential(1);
"""

FIG2_CFG = r"""
  const ROB_CFG = {
    name: "fig2_robustness", aria: "Robustness of the static fringe: regular vs chaotic CMPC against wavelength-dither depth and film thickness",
    pdfFont: "LiberationSans", addFont: doc => { doc.addFileToVFS("LiberationSans.ttf", FONT_B64); doc.addFont("LiberationSans.ttf", "LiberationSans", "normal"); doc.setFont("LiberationSans", "normal"); },
    loadPdfLibs: async () => {
      await loadJsPdf();
      if (!window.svg2pdf) await new Promise((ok, bad) => { const s = document.createElement("script"); s.src = "https://cdn.jsdelivr.net/npm/svg2pdf.js@2.2.3/dist/svg2pdf.umd.min.js"; s.onload = ok; s.onerror = () => bad(new Error("could not load svg2pdf")); document.head.appendChild(s); });
    },
    save: async (name, blob, st) => { await save(name, blob); st.textContent = $("saveStatus").textContent; },
    panels() {
      const R = $("colCR").value, C = $("colCC").value, TG = "#d62728", t = ROB.films, F = ROB.fringe, c6 = ROB.curve600;
      const film = [["SiN", 0], ["x", -1], [" film thickness (nm)", 0]], xt = [[500, "500"], [600, "600"], [700, "700"]];
      const pair = (k, yr, title, extra = {}) => Object.assign({ title, xr: [490, 710], xticks: xt, xlab: film, ylog: true, yr,
        bands: [{ x: t, lo: F[k].q1, hi: F[k].q3, col: C, a: 0.22 }], hlines: [{ y: ROB.target, col: TG }],
        series: [{ x: t, y: F[k].reg, col: R, lw: 1.4, mk: 1.8 }, { x: t, y: F[k].med, col: C, lw: 1.4, mk: 1.8 }] }, extra);
      return [
        { title: "(g) 600 nm film", xlog: true, xr: [5, 60], xticks: [[5, "5"], [10, "10"], [20, "20"], [50, "50"]], xlab: [["Wavelength dither (pm)", 0]],
          ylog: true, yr: [5e-9, 2e-6], ylab: [["Static fringe ΔP/P", 0]], vspan: ROB.window_pm,
          bands: [{ x: c6.dl_pm, lo: c6.q1, hi: c6.q3, col: C, a: 0.22 }], hlines: [{ y: ROB.target, col: TG }],
          series: [{ x: c6.dl_pm, y: c6.reg, col: R, lw: 1.4 }, { x: c6.dl_pm, y: c6.med, col: C, lw: 1.4 }] },
        pair("both", [5e-9, 2e-6], "(h) both dithers, 20 pm"),
        pair("angle", [3e-8, 2e-5], "(i) angle dither only"),
      ];
    },
    legend() { const R = $("colCR").value, C = $("colCC").value;
      return [{ label: "regular CMPC", col: R, lw: 1.6 }, { label: "chaotic CMPC, median", col: C, lw: 1.6 }, { label: "quartiles, 30 layouts", col: C, band: 1, a: 0.22 }, { label: "target 5×10⁻⁷", col: "#d62728", lw: 1, dash: "4 2.5" }]; },
    csv() {
      const F = ROB.fringe, rows = ["film_nm,n_eff,gamma," + ["both", "angle", "spectral", "none"].flatMap(k => ["reg", "med", "q1", "q3"].map(s => `${k}_${s}`)).join(",")];
      ROB.films.forEach((t, i) => rows.push([t, ROB.n_eff[i], ROB.gamma[i], ...["both", "angle", "spectral", "none"].flatMap(k => ["reg", "med", "q1", "q3"].map(s => F[k][s][i]))].join(",")));
      rows.push("", "dither_pm_600nm,both_reg,both_med,both_q1,both_q3");
      ROB.curve600.dl_pm.forEach((d, i) => rows.push([d, ROB.curve600.reg[i], ROB.curve600.med[i], ROB.curve600.q1[i], ROB.curve600.q3[i]].join(",")));
      return rows.join("\n") + "\n";
    },
    table() {
      const W = ROB.window, rd = W.regular, ct = W.chaotic_typical, iq = W.chaotic_q1_q3, em = W.chaotic_ensemble_median, pc = v => `${Math.round(100 * v)} %`;
      let h = `<thead><tr><th>device, over ${ROB.films.length} films × ${ROB.window_pm[0].toFixed(1)}–${ROB.window_pm[1].toFixed(1)} pm</th><th>fringe median</th><th>90th pct</th><th>worst</th><th>worst / best</th><th>points < 1e-7</th><th>NH₃ 1 s median</th><th>NH₃ worst</th></tr></thead><tbody>`;
      h += `<tr><td>regular cell</td><td>${robX(rd.median)}</td><td>${robX(rd.p90)}</td><td>${robX(rd.max)}</td><td>${Math.round(rd.max_over_min)}×</td><td>${pc(rd["share_below_1e-7"])}</td><td>${Math.round(rd.nh3_median)} ppb</td><td>${Math.round(rd.nh3_max)} ppb</td></tr>`;
      h += `<tr><td>chaotic, typical layout</td><td>${robX(ct.median)}</td><td>${robX(ct.p90)}</td><td>${robX(ct.max)}</td><td>${Math.round(ct.max_over_min)}×</td><td>${pc(ct["share_below_1e-7"])}</td><td>${Math.round(ct.nh3_median)} ppb</td><td>${Math.round(ct.nh3_max)} ppb</td></tr>`;
      h += `<tr><td>chaotic, layout quartiles</td><td>${robX(iq.median[0])}–${robX(iq.median[1])}</td><td>${robX(iq.p90[0])}–${robX(iq.p90[1])}</td><td>${robX(iq.max[0])}–${robX(iq.max[1])}</td><td>${Math.round(iq.max_over_min[0])}–${Math.round(iq.max_over_min[1])}×</td><td>${pc(iq["share_below_1e-7"][0])}–${pc(iq["share_below_1e-7"][1])}</td><td>${Math.round(iq.nh3_median[0])}–${Math.round(iq.nh3_median[1])} ppb</td><td>${Math.round(iq.nh3_max[0])}–${Math.round(iq.nh3_max[1])} ppb</td></tr>`;
      h += `<tr><td>chaotic, ensemble median</td><td>${robX(em.median)}</td><td>${robX(em.p90)}</td><td>${robX(em.max)}</td><td>${Math.round(em.max_over_min)}×</td><td>${pc(em["share_below_1e-7"])}</td><td></td><td></td></tr></tbody>`;
      h += `</table><table class="vals"><thead><tr><th>film</th><th>n_eff</th><th>both: regular</th><th>chaotic [quartiles]</th><th>angle only: regular</th><th>chaotic [quartiles]</th><th>wavelength only: regular</th><th>chaotic</th></tr></thead><tbody>`;
      const F = ROB.fringe;
      ROB.films.forEach((t, i) => { h += `<tr><td>${t} nm</td><td>${ROB.n_eff[i].toFixed(3)}</td><td>${robX(F.both.reg[i])}</td><td>${robX(F.both.med[i])} [${robX(F.both.q1[i])}–${robX(F.both.q3[i])}]</td><td>${robX(F.angle.reg[i])}</td><td>${robX(F.angle.med[i])} [${robX(F.angle.q1[i])}–${robX(F.angle.q3[i])}]</td><td>${robX(F.spectral.reg[i])}</td><td>${robX(F.spectral.med[i])}</td></tr>`; });
      return h + "</tbody>";
    },
  };
"""

FIG3_CFG = r"""
  const ROB_CFG = {
    name: "fig3_robustness", aria: "NH3 limit at 1 s against SiNx film thickness, regular vs chaotic CMPC, both dithers and angle dither only",
    pdfFont: "LibSans", addFont: doc => { doc.addFileToVFS("LibSans-Regular.ttf", FONTS.r); doc.addFont("LibSans-Regular.ttf", "LibSans", "normal"); doc.setFont("LibSans", "normal"); },
    loadPdfLibs: async () => { if (!(window.jspdf && window.jspdf.jsPDF)) throw new Error("PDF library did not load; check the connection and reload"); },
    save: async (name, blob, st) => {
      if (!downloads) { st.textContent = "Downloads are not available in this view; open the artifact in claude.ai."; return; }
      try { await downloads.save({ filename: name, data: blob }); st.textContent = `Saved ${name}.`; }
      catch (e) { const c = e && e.code; st.textContent = c === "declined" ? "Download cancelled." : c === "rate_limited" ? "A save dialog is already open; try again in a moment." : "Downloads are not available in this view."; }
    },
    panels() {
      const R = "#B892DC", C = "#5B2A86", t = ROB.films, N = ROB.nh3;
      const film = [["SiN", 0], ["x", -1], [" film thickness (nm)", 0]], xt = [[500, "500"], [600, "600"], [700, "700"]];
      const pan = (k, title, extra = {}) => Object.assign({ title, xr: [490, 710], xticks: xt, xlab: film, ylog: true, yr: [10, 1e4],
        bands: [{ x: t, lo: N[k].q1, hi: N[k].q3, col: C, a: 0.22 }],
        series: [{ x: t, y: N[k].reg, col: R, lw: 1.4, mk: 1.8 }, { x: t, y: N[k].med, col: C, lw: 1.4, mk: 1.8 }] }, extra);
      return [
        pan("both", "(c) both dithers", { ylab: [["NH", 0], ["3", -1], [" limit at 1 s (ppb)", 0]] }),
        pan("angle", "(d) angle dither only"),
      ];
    },
    legend: () => [{ label: "regular CMPC", col: "#B892DC", lw: 1.6 }, { label: "chaotic CMPC, median", col: "#5B2A86", lw: 1.6 }, { label: "quartiles, 30 layouts", col: "#5B2A86", band: 1, a: 0.22 }],
    csv() {
      const N = ROB.nh3, rows = ["film_nm,n_eff,gamma," + ["both", "angle"].flatMap(k => ["reg", "med", "q1", "q3"].map(s => `nh3_ppb_${k}_${s}`)).join(",")];
      ROB.films.forEach((t, i) => rows.push([t, ROB.n_eff[i], ROB.gamma[i], ...["both", "angle"].flatMap(k => ["reg", "med", "q1", "q3"].map(s => N[k][s][i]))].join(",")));
      return rows.join("\n") + "\n";
    },
    table() {
      const N = ROB.nh3, r0 = v => Math.round(v);
      let h = `<thead><tr><th>film</th><th>n_eff</th><th>Γ</th><th>both: regular</th><th>chaotic median [quartiles]</th><th>angle only: regular</th><th>chaotic median [quartiles]</th></tr></thead><tbody>`;
      ROB.films.forEach((t, i) => { h += `<tr><td>${t} nm</td><td>${ROB.n_eff[i].toFixed(3)}</td><td>${ROB.gamma[i].toFixed(2)}</td><td>${r0(N.both.reg[i])} ppb</td><td>${r0(N.both.med[i])} [${r0(N.both.q1[i])}–${r0(N.both.q3[i])}] ppb</td><td>${r0(N.angle.reg[i])} ppb</td><td>${r0(N.angle.med[i])} [${r0(N.angle.q1[i])}–${r0(N.angle.q3[i])}] ppb</td></tr>`; });
      return h + "</tbody>";
    },
  };
"""


def main(f2src, f3src, outdir):
    data = json.dumps(build())
    # ---- Fig. 2
    s = open(f2src, encoding='utf-8').read()

    def rep(src, a, b):
        assert src.count(a) == 1, (src.count(a), a[:90])
        return src.replace(a, b)
    s = rep(s, '<title>Fig. 2 MPC raytracing analysis</title>', '<title>Fig. 2 raytracing and robustness</title>')
    s = rep(s, '<h1>Multipass Chaos Figure: ideal vs target vs minimum spec (30-layout ensembles)</h1>',
            '<h1>Multipass Chaos Figure: spec ensembles and robustness (QUIET steps 10–11)</h1>')
    s = rep(s, '<div class="sheetwrap"><div class="sheet" id="sheet"></div></div>',
            '<div class="sheetwrap"><div class="sheet" id="sheet"></div></div>' + ROB_HTML.format(
                h2='Fig. 2 (g)–(i): robustness of the fringe suppression, regular vs chaotic CMPC (QUIET step 11)', hdef='6.0',
                hint='Circle CMPC, target spec (designed R(χ), 0.1 dB/cm), SiNx films 500–700 nm (TM₀ n_eff 1.53–1.74 and Γ at 1.55 µm), ±2 mrad angle and ±20 pm '
                     'wavelength dither, ε = 10⁻⁶; regular cell vs median and quartiles of 30 inclusion layouts. (g) static fringe with both dithers against the '
                     'wavelength-dither depth for the 600 nm film (grey: the ±20 % window of the tables); (h) both dithers at 20 pm against film thickness; '
                     '(i) angle dither alone. Dashed: target 5×10⁻⁷. The regular cell sends its light along one path, so its residual oscillates with optical '
                     'path × dither depth; the chaotic cell has no such resonance, and with the angle dither alone only the chaotic cell meets the target.'))
    s = rep(s, '<script id="engine">', f'<script id="robdata" type="application/json">{data}</script>\n<script id="engine">')
    s = rep(s, '  /* settings file: the rail and engine options of the last run, the figure as drawn, and the bar values */',
            TABLE_JS + FIG2_CFG + RENDER + '\n  /* settings file: the rail and engine options of the last run, the figure as drawn, and the bar values */')
    s = rep(s, '$("colGrey").onclick = () => { $("colHR").value = "#9a9a9a"; $("colCR").value = "#9a9a9a"; draw(); };',
            '$("colGrey").onclick = () => { $("colHR").value = "#9a9a9a"; $("colCR").value = "#9a9a9a"; draw(); robDraw(); };\n'
            '  for (const id of ["colCR", "colCC"]) $(id).addEventListener("input", () => robDraw());')
    s = s.replace('.status { color: var(--muted); font: 12px var(--f-data); }',
                  '.status { color: var(--muted); font: 12px var(--f-data); }\n.sel { display: flex; gap: 6px; align-items: center; color: var(--muted); font-size: 12.5px; }\n#robSheet { min-width: 560px; }', 1)
    open(os.path.join(outdir, 'fig2_robust.html'), 'w', encoding='utf-8').write(s)
    # ---- Fig. 3
    s = open(f3src, encoding='utf-8').read()
    s = rep(s, '<title>Fig. 3 HITRAN spectra + noise</title>', '<title>Fig. 3 spectra, noise and robustness</title>')
    s = rep(s, '<h1>Fig. 3 spectra and noise: ideal vs target vs minimum spec</h1>',
            '<h1>Fig. 3 spectra and noise: spec cases and robustness (QUIET steps 10–11)</h1>')
    s = rep(s, '<h2>Ensemble analysis at T = 1 s (NH₃ line, 21 layouts)</h2>', '<h2>Ensemble analysis at T = 1 s (NH₃ line, 30 layouts)</h2>')
    s = rep(s, '<div class="sheetwrap"><div class="sheet" id="sheet"></div></div>',
            '<div class="sheetwrap"><div class="sheet" id="sheet"></div></div>' + ROB_HTML.format(
                h2='Fig. 3 (c)–(d): NH₃ limit at 1 s against membrane thickness, regular vs chaotic CMPC (QUIET step 11)', hdef='6.0',
                hint='NH₃ 1531.7 nm line, 1 mW, shot noise and the fringe residual after 1 s (drift 0.1 rad/s), circle CMPC at the target spec (designed R(χ), '
                     '0.1 dB/cm), SiNx films 500–700 nm with their own n_eff and Γ; regular cell vs median and quartiles of 30 inclusion layouts. (c) both dithers '
                     '(±2 mrad, ±20 pm); (d) the on-chip angle dither alone. Only the chaotic cell keeps the limit within one decade for every film with the '
                     'angle dither alone.'))
    s = rep(s, '<script id="fonts" type="application/json">', f'<script id="robdata" type="application/json">{data}</script>\n<script id="fonts" type="application/json">')
    s = rep(s, '  $("savePdf").onclick = () => savePdf(false);', TABLE_JS + FIG3_CFG + RENDER + '\n  $("savePdf").onclick = () => savePdf(false);')
    s = s.replace('.status { color: var(--muted); font: 12px var(--f-data); }',
                  '.status { color: var(--muted); font: 12px var(--f-data); }\n.sel { display: flex; gap: 6px; align-items: center; color: var(--muted); font-size: 12.5px; }\n#robSheet { min-width: 560px; }', 1)
    open(os.path.join(outdir, 'fig3_robust.html'), 'w', encoding='utf-8').write(s)
    print('ok')


if __name__ == '__main__':
    main(*sys.argv[1:4])
