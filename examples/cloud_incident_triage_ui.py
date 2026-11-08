"""Interactive Web UI for the Multi-Domain Industrial Decision Router (`http://127.0.0.1:7865`).

Uses `gemmarouter==0.1.0` installed from PyPI (`site-packages`) across 6 distinct industrial
use-cases (Cloud SRE, Smart Factory PLC, Corporate Treasury, Electrical Grid, Aviation MRO,
and Pharma BioLab), styled with a Forest Green, Warm Beige, and Royal Purple palette and
separate structured Candidate Option cards.
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gemmarouter
from examples.cloud_incident_triage import (
    IndustrialUseCase,
    compute_parameter_summary,
    evaluate_usecase,
    get_benchmark_usecases,
    prewarm_all_usecases,
    verify_permutation_invariance,
    verify_pypi_provenance,
)

UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GemmaRouter PyPI v0.1.1 — Multi-Domain Industrial Decision Router</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@500;700&family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    @layer base, theme, components;

    @layer theme {
      :root {
        --font-sans: "Roboto", system-ui, -apple-system, sans-serif;
        --font-mono: "Roboto Mono", ui-monospace, monospace;
        --beige-canvas: #f2ece1;
        --beige-surface: #faf7f0;
        --beige-card: #f4efe4;
        --beige-subtle: #e7dfd0;
        --beige-border: #9e927d;
        --forest-deep: #133826;
        --forest-main: #1b4d3e;
        --forest-tint: #dceee5;
        --forest-border: #2d6a4f;
        --purple-deep: #46226e;
        --purple-main: #5c318c;
        --purple-tint: #eee6f8;
        --purple-border: #7a4eb0;
        --text-main: #1d241f;
        --text-muted: #525953;
        --warn-bg: #fbe6e4;
        --warn-fg: #9e1c20;
        --warn-border: #c95b5e;
      }
    }

    @layer base {
      body, div, section, header, h1, h2, p, input, textarea, button, table, th, td, caption {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        padding: 1.35rem;
        font-family: var(--font-sans);
        background: var(--beige-canvas);
        color: var(--text-main);
        line-height: 1.5;
        overflow-wrap: break-word;
      }
      button, input, textarea {
        font-family: inherit;
      }
      code {
        font-family: var(--font-mono);
      }
      h1 { margin: 0 0 0.5rem; font-size: 1.32rem; color: var(--forest-deep); letter-spacing: -0.01em; text-wrap: balance; }
      h2 {
        margin: 0 0 0.85rem;
        font-size: 1.04rem;
        color: var(--forest-deep);
        display: flex;
        align-items: center;
        gap: 0.45rem;
        border-block-end: 2px solid var(--beige-subtle);
        padding-block-end: 0.45rem;
        text-wrap: balance;
      }
    }

    @layer components {
      .container { max-inline-size: 1280px; margin-inline: auto; display: grid; gap: 1.1rem; }
      .hero-card {
        background: var(--forest-tint);
        border: 1.5px solid var(--forest-border);
        border-inline-start: 6px solid var(--forest-main);
        border-radius: 12px;
        padding: 1.15rem 1.4rem;
        color: var(--forest-deep);
        box-shadow: 0 2px 8px rgba(29, 36, 31, 0.06);
      }
      .card {
        background: var(--beige-surface);
        border: 1.5px solid var(--beige-border);
        border-radius: 12px;
        padding: 1.15rem 1.35rem;
        box-shadow: 0 2px 8px rgba(29, 36, 31, 0.06);
      }
      .meta-bar { display: flex; flex-wrap: wrap; gap: 0.5rem; font-size: 0.81rem; }
      .pill {
        background: var(--beige-surface);
        border: 1.5px solid var(--forest-border);
        color: var(--forest-deep);
        font-weight: 600;
        border-radius: 999px;
        padding: 0.22rem 0.72rem;
        font-family: var(--font-mono);
      }
      .pill-purple {
        background: var(--purple-tint);
        border: 1px solid var(--purple-border);
        color: var(--purple-deep);
        font-weight: 600;
      }
      .grid-2 { display: grid; grid-template-columns: 1.22fr 1fr; gap: 1.1rem; align-items: start; }
      @media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }
      .preset-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(235px, 1fr)); gap: 0.5rem; margin-block-end: 0.9rem; }
      button {
        cursor: pointer;
        border: 1.5px solid var(--beige-border);
        background: var(--beige-card);
        color: var(--text-main);
        border-radius: 8px;
        padding: 0.48rem 0.8rem;
        font-size: 0.82rem;
        font-weight: 600;
        text-align: start;
        transition: background 0.15s ease, border-color 0.15s ease, transform 0.1s ease;
      }
      button:hover:not(:disabled) {
        background: var(--purple-tint);
        border-color: var(--purple-border);
        color: var(--purple-deep);
      }
      button.active-preset {
        background: var(--purple-tint);
        border: 2px solid var(--purple-border);
        color: var(--purple-deep);
        font-weight: 700;
      }
      button.primary {
        background: var(--forest-tint);
        color: var(--forest-deep);
        border: 2px solid var(--forest-border);
        font-weight: 700;
        text-align: center;
      }
      button.primary:hover:not(:disabled) {
        background: #c5e4d4;
        border-color: var(--forest-deep);
        color: var(--forest-deep);
      }
      button.secondary-purple {
        background: var(--purple-tint);
        color: var(--purple-deep);
        border: 1.5px solid var(--purple-border);
        text-align: center;
      }
      button.secondary-purple:hover:not(:disabled) {
        background: #e2d4f4;
        border-color: var(--purple-deep);
        color: var(--purple-deep);
      }
      button.ghost-btn {
        background: var(--purple-tint);
        border: 1.5px dashed var(--purple-border);
        color: var(--purple-deep);
        text-align: center;
        padding: 0.38rem 0.75rem;
      }
      .placeholder-box {
        padding: 1.2rem;
        border: 2px dashed var(--purple-border);
        border-radius: 10px;
        background: var(--purple-tint);
        color: var(--purple-deep);
        font-size: 0.88rem;
      }
      textarea {
        field-sizing: content;
        height: auto;
        width: 100%;
        min-block-size: 2.6lh;
        max-block-size: 8lh;
        padding: 0.65rem 0.75rem;
        border: 1.5px solid var(--forest-border);
        border-radius: 8px;
        background: #ffffff;
        color: var(--text-main);
        font: inherit;
        font-size: 0.88rem;
        margin-block-end: 0.85rem;
      }
      .field-label {
        font-size: 0.84rem;
        font-weight: 700;
        color: var(--forest-deep);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-block-end: 0.38rem;
      }
      .options-stack {
        display: grid;
        gap: 0.55rem;
        margin-block-end: 0.75rem;
      }
      .option-card {
        display: grid;
        grid-template-columns: auto 210px 1fr auto;
        gap: 0.55rem;
        align-items: center;
        background: var(--beige-card);
        border: 1.5px solid var(--beige-border);
        border-inline-start: 5px solid var(--purple-main);
        border-radius: 8px;
        padding: 0.5rem 0.65rem;
      }
      @media (max-width: 720px) {
        .option-card { grid-template-columns: 1fr; }
      }
      .opt-index {
        background: var(--forest-tint);
        color: var(--forest-deep);
        border: 1px solid var(--forest-border);
        font-family: var(--font-mono);
        font-size: 0.76rem;
        font-weight: 700;
        padding: 0.18rem 0.48rem;
        border-radius: 6px;
      }
      .opt-key-input {
        font-family: var(--font-mono);
        font-size: 0.81rem;
        font-weight: 700;
        color: var(--purple-deep);
        background: var(--purple-tint);
        border: 1.5px solid var(--purple-border);
        border-radius: 6px;
        padding: 0.36rem 0.55rem;
        width: 100%;
      }
      .opt-desc-input {
        font-size: 0.84rem;
        color: var(--text-main);
        background: #ffffff;
        border: 1.5px solid var(--beige-border);
        border-radius: 6px;
        padding: 0.36rem 0.6rem;
        width: 100%;
      }
      .opt-remove-btn {
        background: transparent;
        border: 1px solid var(--beige-border);
        color: var(--text-muted);
        padding: 0.22rem 0.5rem;
        border-radius: 6px;
        font-size: 0.8rem;
        text-align: center;
      }
      .controls-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.65rem;
        margin-block-start: 0.75rem;
        padding-block-start: 0.75rem;
        border-block-start: 1.5px solid var(--beige-subtle);
      }
      .badge-ok {
        background: var(--forest-tint);
        color: var(--forest-deep);
        border: 1.5px solid var(--forest-border);
        padding: 0.24rem 0.65rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
      }
      .badge-abstain {
        background: var(--warn-bg);
        color: var(--warn-fg);
        border: 1.5px solid var(--warn-border);
        padding: 0.24rem 0.65rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
      }
      .badge-clarify {
        background: var(--purple-tint);
        color: var(--purple-deep);
        border: 1.5px solid var(--purple-border);
        padding: 0.24rem 0.65rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
      }
      .result-panel {
        background: var(--beige-card);
        border: 1.5px solid var(--forest-border);
        border-radius: 10px;
        padding: 0.95rem 1.05rem;
      }
      .prob-card {
        background: var(--beige-surface);
        border: 1px solid var(--beige-border);
        border-radius: 7px;
        padding: 0.45rem 0.65rem;
        margin-block: 0.38rem;
      }
      .prob-row {
        display: grid;
        grid-template-columns: 185px 1fr 62px;
        align-items: center;
        gap: 0.55rem;
        font-size: 0.82rem;
      }
      .bar-bg { background: var(--beige-subtle); height: 11px; border-radius: 99px; overflow: hidden; }
      .bar-fill {
        background: linear-gradient(90deg, var(--forest-main), var(--purple-main));
        height: 100%;
      }
      .table-wrapper {
        overflow: auto;
        max-inline-size: 100%;
        max-block-size: min(520px, 80vh);
        container-type: inline-size;
        border: 1.5px solid var(--beige-border);
        border-radius: 8px;
      }
      table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.83rem; }
      caption {
        text-align: start;
        padding: 0.6rem 0.8rem;
        font-weight: 700;
        color: var(--forest-deep);
        background: var(--beige-subtle);
        border-block-end: 1.5px solid var(--beige-border);
      }
      th, td { padding: 0.55rem 0.7rem; border-block-end: 1px solid var(--beige-subtle); text-align: start; }
      thead { position: sticky; inset-block-start: 0; z-index: 1; background: var(--forest-tint); color: var(--forest-deep); }
      tbody tr:nth-child(even) { background: var(--beige-card); }
      tbody th {
        position: sticky;
        inset-inline-start: 0;
        background: var(--purple-tint);
        color: var(--purple-deep);
        font-family: var(--font-mono);
        border-inline-end: 1.5px solid var(--purple-border);
      }
      @container (width < 600px) {
        table, thead, tbody, tr, th, td { display: block; }
        thead { display: none; }
        tr { margin-block-end: 1rem; border: 1.5px solid var(--beige-border); }
        tbody th { position: sticky; inset-block-start: 0; z-index: 2; }
        table {
          --label-2: "Industrial Domain";
          --label-3: "Expected";
          --label-4: "Actual Decision";
          --label-5: "Confidence";
          --label-6: "P(null)";
          --label-7: "Cache Hit";
          --label-8: "Latency";
          --label-9: "Verdict";
        }
        td:nth-child(2)::before { content: var(--label-2) ": "; font-weight: 700; }
        td:nth-child(3)::before { content: var(--label-3) ": "; font-weight: 700; }
        td:nth-child(4)::before { content: var(--label-4) ": "; font-weight: 700; }
        td:nth-child(5)::before { content: var(--label-5) ": "; font-weight: 700; }
        td:nth-child(6)::before { content: var(--label-6) ": "; font-weight: 700; }
        td:nth-child(7)::before { content: var(--label-7) ": "; font-weight: 700; }
        td:nth-child(8)::before { content: var(--label-8) ": "; font-weight: 700; }
        td:nth-child(9)::before { content: var(--label-9) ": "; font-weight: 700; }
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <header class="hero-card">
      <h1>GemmaRouter PyPI (<code>gemmarouter==0.1.1</code>) — Multi-Domain Industrial Decision Router</h1>
      <div class="meta-bar" id="meta-bar">Loading PyPI package provenance...</div>
    </header>

    <div class="grid-2">
      <section class="card">
        <h2>1. Select Industrial Use-Case, Prompt &amp; Separate Candidate Options</h2>
        <div class="preset-list" id="preset-buttons"></div>

        <label class="field-label" for="prompt-input">
          <span>Incoming Industrial Alert / Prompt</span>
          <span class="pill pill-purple" id="active-domain-pill">SRE-DB-101</span>
        </label>
        <textarea id="prompt-input" spellcheck="false" placeholder="Enter an industrial command or alert..."></textarea>

        <div class="field-label">
          <span>Candidate Options for This Use-Case (Separate Cards)</span>
          <button type="button" class="ghost-btn" id="add-opt-btn">+ Add Option Card</button>
        </div>
        <div class="options-stack" id="options-stack"></div>

        <div class="controls-row">
          <button type="button" class="primary" id="run-single-btn">▶ Dispatch Prompt</button>
          <button type="button" class="secondary-purple" id="run-batch-btn">⚡ Run All 7 Industrial Use-Cases</button>
          <label style="font-size:0.83rem;font-weight:600;color:var(--forest-deep);">Null Threshold:
            <input type="number" id="threshold-input" value="0.58" min="0.05" max="0.99" step="0.01" style="width:68px;padding:0.28rem;border:1.5px solid var(--forest-border);border-radius:6px;background:#fff;">
          </label>
          <label style="font-size:0.83rem;font-weight:600;color:var(--purple-deep);">Margin Gate:
            <input type="number" id="margin-input" value="0.40" min="0.00" max="0.95" step="0.05" style="width:68px;padding:0.28rem;border:1.5px solid var(--purple-border);border-radius:6px;background:#fff;">
          </label>
        </div>
      </section>

      <section class="card">
        <h2>2. Live Dispatch &amp; Circuit-Breaker Result</h2>
        <div id="single-result" class="placeholder-box">Select a use-case preset on the left, then click <b>▶ Dispatch Prompt</b> to run live inference.</div>
      </section>
    </div>

    <section class="card">
      <h2>3. Multi-Domain Batch Verification Audit Log (7 Distinct Industrial Use-Cases)</h2>
      <div id="batch-placeholder" class="placeholder-box">No benchmark results yet. Click <b>⚡ Run All 7 Industrial Use-Cases</b> to execute the verification suite live.</div>
      <div id="batch-table-wrap" tabindex="0" role="region" aria-labelledby="audit-table-caption" class="table-wrapper" style="display:none;">
        <table>
          <caption id="audit-table-caption">Multi-Domain Industrial Decision Router Results</caption>
          <thead>
            <tr>
              <th>Use-Case ID</th>
              <th>Industrial Domain</th>
              <th>Expected</th>
              <th>Actual Decision</th>
              <th>Confidence</th>
              <th>P(null)</th>
              <th>Cache Hit</th>
              <th>Latency</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody id="audit-tbody"></tbody>
        </table>
      </div>
    </section>
  </div>

  <script>
    let selectedUsecaseId = "SRE-DB-101";
    let selectedDomainTitle = "Cloud DB Failover (SRE)";

    function esc(s) {
      const d = document.createElement("div");
      d.textContent = String(s ?? "");
      return d.innerHTML;
    }

    function renderOptionCards(optionsObj) {
      const stack = document.getElementById("options-stack");
      stack.innerHTML = "";
      Object.entries(optionsObj).forEach(([k, v], idx) => {
        appendOptionCard(k, v, idx + 1);
      });
    }

    function renumberOptionCards() {
      document.querySelectorAll("#options-stack .option-card").forEach((card, i) => {
        const badge = card.querySelector(".opt-index");
        if (badge) badge.textContent = `#${i + 1}`;
      });
    }

    function appendOptionCard(keyText = "", descText = "", indexNum = 1) {
      const stack = document.getElementById("options-stack");
      const row = document.createElement("div");
      row.className = "option-card";
      row.innerHTML = `
        <span class="opt-index">#${indexNum}</span>
        <input type="text" class="opt-key-input" spellcheck="false" placeholder="option_key" value="${esc(keyText)}" aria-label="Option key">
        <input type="text" class="opt-desc-input" spellcheck="false" placeholder="Describe what this option does..." value="${esc(descText)}" aria-label="Option description">
        <button type="button" class="opt-remove-btn" title="Remove option">✕</button>
      `;
      row.querySelector(".opt-key-input").oninput = onCustomEdit;
      row.querySelector(".opt-desc-input").oninput = onCustomEdit;
      row.querySelector(".opt-remove-btn").onclick = () => {
        row.remove();
        renumberOptionCards();
        onCustomEdit();
      };
      stack.appendChild(row);
    }

    function collectOptionsFromCards() {
      const out = {};
      document.querySelectorAll("#options-stack .option-card").forEach(card => {
        const k = card.querySelector(".opt-key-input").value.trim();
        const v = card.querySelector(".opt-desc-input").value.trim();
        if (k) out[k] = v || k;
      });
      return out;
    }

    function showSinglePlaceholder(ucId, domainTitle, optCount) {
      document.getElementById("active-domain-pill").textContent = `${ucId} · ${domainTitle}`;
      const box = document.getElementById("single-result");
      box.className = "placeholder-box";
      box.innerHTML = `Loaded <code>${esc(ucId)}</code> — <b>${esc(domainTitle)}</b> with <code>${optCount}</code> separate candidate option cards. Click <b>▶ Dispatch Prompt</b> to execute live model inference.`;
    }

    async function loadInit() {
      const r = await fetch("/api/info");
      const data = await r.json();
      const p = data.provenance;
      const m = data.parameters;
      document.getElementById("meta-bar").innerHTML = [
        `<span class="pill">PyPI: ${esc(p.package_name)}==${esc(p.version)}</span>`,
        `<span class="pill">site-packages: ${esc(p.installed_in_site_packages)}</span>`,
        `<span class="pill">Device: ${esc(data.device)} (${esc(data.dtype)})</span>`,
        `<span class="pill">Params: ${(m.total_parameters / 1e6).toFixed(2)}M (${(m.decision_head_parameters / 1e6).toFixed(2)}M head)</span>`,
        `<span class="pill">Prewarmed Option Vectors: ${esc(data.prewarmed_tools)}</span>`
      ].join("");

      const btnWrap = document.getElementById("preset-buttons");
      btnWrap.innerHTML = "";
      data.usecases.forEach((uc, idx) => {
        const b = document.createElement("button");
        b.textContent = `${uc.usecase_id} · ${uc.domain_title}`;
        b.onclick = () => {
          selectedUsecaseId = uc.usecase_id;
          selectedDomainTitle = uc.domain_title;
          document.querySelectorAll("#preset-buttons button").forEach(el => el.classList.remove("active-preset"));
          b.classList.add("active-preset");
          document.getElementById("prompt-input").value = uc.prompt_text;
          renderOptionCards(uc.options);
          showSinglePlaceholder(uc.usecase_id, uc.domain_title, Object.keys(uc.options).length);
        };
        btnWrap.appendChild(b);
        if (idx === 0) {
          b.classList.add("active-preset");
          selectedUsecaseId = uc.usecase_id;
          selectedDomainTitle = uc.domain_title;
          document.getElementById("prompt-input").value = uc.prompt_text;
          renderOptionCards(uc.options);
          showSinglePlaceholder(uc.usecase_id, uc.domain_title, Object.keys(uc.options).length);
        }
      });
    }

    async function runSingle() {
      const prompt = document.getElementById("prompt-input").value.trim();
      const options = collectOptionsFromCards();
      const threshold = parseFloat(document.getElementById("threshold-input").value) || 0.58;
      const margin_threshold = parseFloat(document.getElementById("margin-input").value);
      const effMargin = Number.isFinite(margin_threshold) ? margin_threshold : 0.40;
      const box = document.getElementById("single-result");
      box.className = "placeholder-box";
      box.textContent = "Routing prompt across candidate option cards via gemmarouter...";
      const r = await fetch("/api/dispatch", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          usecase_id: selectedUsecaseId,
          domain_title: selectedDomainTitle,
          prompt_text: prompt,
          options,
          threshold,
          margin_threshold: effMargin
        })
      });
      const d = await r.json();
      if (d.error) {
        box.textContent = `Error: ${d.error}`;
        return;
      }
      box.className = "result-panel";
      let badge = `<span class="badge-ok"><span aria-hidden="true">✔</span> DISPATCHED → ${esc(d.route.decision)}</span>`;
      if (d.route.needs_clarification) {
        badge = `<span class="badge-clarify"><span aria-hidden="true">❓</span> CLARIFICATION NEEDED (Twin Collision)</span>`;
      } else if (d.route.abstain) {
        badge = `<span class="badge-abstain"><span aria-hidden="true">⛔</span> ABSTAINED (decision = None)</span>`;
      }
      const isOut = d.route.abstain && !d.route.needs_clarification;
      const probsHtml = isOut
        ? ""
        : Object.entries(d.route.probabilities)
            .sort((a, b) => b[1] - a[1])
            .map(([k, v]) => `
              <div class="prob-card">
                <div class="prob-row">
                  <code style="color:var(--purple-deep);font-weight:700;">${esc(k)}</code>
                  <div class="bar-bg"><div class="bar-fill" style="width:${(v * 100).toFixed(1)}%"></div></div>
                  <span style="text-align:end;font-weight:700;color:var(--forest-deep);">${(v * 100).toFixed(1)}%</span>
                </div>
              </div>`).join("");
      const topOptMeta = d.route.abstain
        ? `Margin: <code>${d.route.margin.toFixed(4)}</code> (min ${d.route.margin_threshold}) | `
        : `Selected Option: <code>${esc(d.route.decision)}</code> (${(d.route.confidence * 100).toFixed(2)}%, margin ${d.route.margin.toFixed(4)}) | `;
      box.innerHTML = `
        <div style="margin-block-end:0.55rem;">${badge}</div>
        <div style="font-size:0.84rem;margin-block-end:0.55rem;font-family:var(--font-mono);color:var(--forest-deep);font-weight:600;">${esc(d.dispatch_output)}</div>
        <div style="font-size:0.82rem;color:var(--text-muted);margin-block-end:0.65rem;">
          ${topOptMeta}<code>P(null) = ${d.route.null_probability.toFixed(4)}</code> (thresh ${d.route.threshold}) |
          Cache Hit: <code>${d.route.option_cache_hit}</code> |
          Latency: <code>${d.route.latency_ms} ms</code> |
          Perm Drift: <code>${d.permutation.max_probability_diff}</code>
        </div>
        ${probsHtml ? `<div>${probsHtml}</div>` : ""}
      `;
    }

    function renderBatchTable(records) {
      document.getElementById("batch-placeholder").style.display = "none";
      document.getElementById("batch-table-wrap").style.display = "block";
      const tbody = document.getElementById("audit-tbody");
      tbody.innerHTML = records.map(r => {
        const actLabel = r.needs_clarification
          ? `CLARIFY (${(r.clarify_options || []).join(" vs ")})`
          : (r.actual_decision || "ABSTAIN (None)");
        return `
        <tr>
          <th>${esc(r.usecase_id)}</th>
          <td>${esc(r.domain_title)}</td>
          <td><code>${esc(r.expected_decision || "ABSTAIN (None)")}</code></td>
          <td><code>${esc(actLabel)}</code></td>
          <td>${r.abstain ? "—" : (r.confidence * 100).toFixed(2) + "%"}</td>
          <td>${r.null_probability.toFixed(4)}</td>
          <td>${r.option_cache_hit ? "Yes" : "No"}</td>
          <td>${r.latency_ms.toFixed(1)} ms</td>
          <td>${r.passed ? '<span class="badge-ok">✔ PASS</span>' : '<span class="badge-abstain">⛔ FAIL</span>'}</td>
        </tr>`;
      }).join("");
    }

    async function runBatch() {
      const ph = document.getElementById("batch-placeholder");
      ph.style.display = "block";
      ph.textContent = "Running all 7 industrial use-cases via gemmarouter...";
      document.getElementById("batch-table-wrap").style.display = "none";
      const r = await fetch("/api/batch", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({})
      });
      const d = await r.json();
      renderBatchTable(d.records);
    }

    function onCustomEdit() {
      selectedUsecaseId = "CUSTOM";
      selectedDomainTitle = "Custom Scenario";
      document.querySelectorAll("#preset-buttons button").forEach(el => el.classList.remove("active-preset"));
      const optCount = Object.keys(collectOptionsFromCards()).length;
      showSinglePlaceholder("CUSTOM", "Custom Scenario", optCount);
    }

    document.getElementById("prompt-input").oninput = onCustomEdit;
    document.getElementById("add-opt-btn").onclick = () => {
      const nextNum = document.querySelectorAll("#options-stack .option-card").length + 1;
      appendOptionCard(`custom_option_${nextNum}`, "Describe candidate action...", nextNum);
      onCustomEdit();
    };
    document.getElementById("run-single-btn").onclick = () => runSingle();
    document.getElementById("run-batch-btn").onclick = () => runBatch();
    loadInit();
  </script>
</body>
</html>
"""


class TriageGatewayState:
    """Thread-safe holder for the loaded PyPI DecisionRouter and multi-domain industrial use-cases."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.provenance = verify_pypi_provenance()
        self.router = gemmarouter.load()
        self.parameters = compute_parameter_summary(self.router)
        self.usecases: list[IndustrialUseCase] = get_benchmark_usecases()
        self.prewarmed_tools = prewarm_all_usecases(self.router, self.usecases)

    def get_info_payload(self) -> dict[str, Any]:
        """Build the initial metadata payload without running inference until requested."""
        return {
            "provenance": self.provenance,
            "device": str(self.router.device),
            "dtype": str(next(self.router.model.parameters()).dtype),
            "parameters": self.parameters,
            "prewarmed_tools": self.prewarmed_tools,
            "usecases": [asdict(uc) for uc in self.usecases],
        }

    def run_batch_payload(self) -> dict[str, Any]:
        """Execute all 6 industrial use-cases on demand when the user clicks the batch button."""
        with self.lock:
            records = [asdict(evaluate_usecase(self.router, uc)) for uc in self.usecases]
        return {"records": records}

    def dispatch_single(
        self,
        usecase_id: str,
        domain_title: str,
        prompt_text: str,
        options: dict[str, str],
        threshold: float,
        margin_threshold: float = 0.40,
    ) -> dict[str, Any]:
        """Route a single prompt against its domain option set and run a permutation check."""
        with self.lock:
            res = self.router.route(
                prompt_text,
                options=options,
                threshold=threshold,
                margin_threshold=margin_threshold,
            )
            perm = verify_permutation_invariance(self.router, prompt_text, options, threshold=threshold)
        if not res.abstain and res.decision is not None:
            msg = f"[EXECUTED {res.decision}] Dispatched action for {usecase_id} ({domain_title})."
        elif res.needs_clarification and res.clarification_prompt:
            msg = f"[CLARIFICATION NEEDED] {res.clarification_prompt}"
        else:
            msg = "[CIRCUIT-BREAKER ABSTAINED] Blocked execution; escalated to human operator."
        return {"route": res.to_dict(), "dispatch_output": msg, "permutation": perm}


class _GatewayHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Multi-Domain Industrial Decision Router UI."""

    state: TriageGatewayState

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        """Serialize and write a JSON HTTP response with no-store cache headers."""
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        """Serve `/api/info` JSON or the main dashboard HTML with no-store cache headers."""
        if self.path.startswith("/api/info"):
            self._send_json(200, self.state.get_info_payload())
            return
        html = UI_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def do_POST(self) -> None:  # noqa: N802
        """Handle `/api/dispatch` and `/api/batch` POST requests."""
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if self.path == "/api/dispatch":
            try:
                payload = self.state.dispatch_single(
                    str(body.get("usecase_id", "CUSTOM")),
                    str(body.get("domain_title", "Custom Scenario")),
                    str(body.get("prompt_text", "")),
                    dict(body.get("options") or {}),
                    float(body.get("threshold", 0.58)),
                    float(body.get("margin_threshold", 0.40)),
                )
                self._send_json(200, payload)
            except Exception as exc:  # noqa: BLE001
                self._send_json(400, {"error": str(exc)})
            return
        if self.path == "/api/batch":
            self._send_json(200, self.state.run_batch_payload())
            return
        self._send_json(404, {"error": "Not found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence default stderr access logs."""
        return


def create_gateway_server(host: str = "127.0.0.1", port: int = 7865) -> ThreadingHTTPServer:
    """Create the HTTP server serving the Multi-Domain Industrial Decision Router UI."""
    _GatewayHTTPHandler.state = TriageGatewayState()
    return ThreadingHTTPServer((host, port), _GatewayHTTPHandler)


if __name__ == "__main__":
    srv = create_gateway_server("127.0.0.1", 7865)
    print("Multi-Domain Industrial Decision Router UI running at http://127.0.0.1:7865")
    srv.serve_forever()
