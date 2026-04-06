// ── Config ────────────────────────────────────────────────────────────────────
const API_BASE = "";

const COLOR = {
  cat0:  "#6B7280",   // grey   — Infra (0.0)
  cat1:  "#F59E0B",   // amber  — Risque (1.0)
  cat15: "#EF4444",   // red    — Interaction (1.5)
  cat2:  "#10B981",   // green  — Corrigé (2.0)
  brand: "#E2001A",
};

// ── State ─────────────────────────────────────────────────────────────────────
let lastResult = null;
let pieChart   = null;

// ── View switcher ─────────────────────────────────────────────────────────────
function showView(id) {
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.getElementById(id).classList.add("active");
}

// ── File entry point ──────────────────────────────────────────────────────────
function handleFile(input) {
  const file = input.files[0];
  if (!file) return;
  runAnalysis(file);
}

// ── Main analysis flow ────────────────────────────────────────────────────────
async function runAnalysis(file) {
  showView("view-loading");
  try {
    const [_, data] = await Promise.all([
      animateSteps(),
      callAPI(file),
    ]);
    lastResult = data;
    renderResults(data);
    showView("view-results");
  } catch (err) {
    showView("view-upload");
    document.getElementById("file-input").value = "";
    alert("Erreur : " + err.message + "\n\nVérifiez que le serveur Python tourne sur " + API_BASE);
  }
}

// ── API call ──────────────────────────────────────────────────────────────────
async function callAPI(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

// ── Loading animation ─────────────────────────────────────────────────────────
async function animateSteps() {
  const steps = [
    "Lecture du fichier Excel",
    "Nettoyage & normalisation du texte",
    "Couche 1 — Détection risque humain",
    "Couche 2 — Détection d'action",
    "Couche 3 — Validation correction",
    "Génération du classement",
  ];
  const fill = document.getElementById("progress-fill");
  const list = document.getElementById("steps-list");
  list.innerHTML = steps.map((s, i) =>
    `<div class="step" id="step-${i}"><span class="step-icon">⏳</span>${s}</div>`
  ).join("");

  for (let i = 0; i < steps.length; i++) {
    await wait(480);
    fill.style.width = `${Math.round(((i + 1) / steps.length) * 100)}%`;
    const el = document.getElementById(`step-${i}`);
    el.classList.add("done");
    el.querySelector(".step-icon").textContent = "✅";
  }
  await wait(300);
}

function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
  const { total, dist, observers, month } = data;

  // ── Month label
  document.getElementById("chart-month-label").textContent =
    `Points prédits · ${month || "Nouveau mois"}`;

  // ── Leaderboard
  // Backend now returns: name, total, rank (min-rank), total_obs, cat0/1/15/2 (obs counts)
  // Sorted by rank ascending (rank 1 = highest points). Ties share the same rank number.
  const maxPts = observers.length > 0 ? observers[0].total : 1;

  const medals    = { 1: "🥇", 2: "🥈", 3: "🥉" };
  const rowCls    = { 1: "row-gold", 2: "row-silver", 3: "row-bronze" };

  document.getElementById("leaderboard-table").innerHTML = `
    <table class="lb-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Observateur</th>
          <th>Obs.</th>
          <th style="color:${COLOR.cat0}">Infra<br><small>0.0</small></th>
          <th style="color:${COLOR.cat1}">Risque<br><small>1.0</small></th>
          <th style="color:${COLOR.cat15}">Interact.<br><small>1.5</small></th>
          <th style="color:${COLOR.cat2}">Corrigé<br><small>2.0</small></th>
          <th>Pts Prédit</th>
          <th style="width:90px"></th>
        </tr>
      </thead>
      <tbody>
        ${observers.map((o) => {
          const rankCell = medals[o.rank]
            ? `<span title="Rang ${o.rank}">${medals[o.rank]}</span>`
            : `<span class="rank-num">${o.rank}</span>`;
          const rowClass = rowCls[o.rank] || "";
          const barPct   = maxPts > 0 ? Math.round((o.total / maxPts) * 100) : 0;

          return `
            <tr class="${rowClass}">
              <td class="rank-cell">${rankCell}</td>
              <td class="name-cell">${o.name}</td>
              <td class="obs-cell">${o.total_obs}</td>
              <td class="num-cell">
                <span class="cat-badge" style="background:${COLOR.cat0}">${o.cat0}</span>
              </td>
              <td class="num-cell">
                <span class="cat-badge" style="background:${COLOR.cat1}">${o.cat1}</span>
              </td>
              <td class="num-cell">
                <span class="cat-badge" style="background:${COLOR.cat15}">${o.cat15}</span>
              </td>
              <td class="num-cell">
                <span class="cat-badge" style="background:${COLOR.cat2}">${o.cat2}</span>
              </td>
              <td class="pts-cell">${parseFloat(o.total).toFixed(1)}</td>
              <td>
                <div class="lb-bar-track">
                  <div class="lb-bar-fill" style="width:${barPct}%"></div>
                </div>
              </td>
            </tr>
          `;
        }).join("")}
      </tbody>
    </table>
  `;

  // ── Pie chart
  const pieTotal = dist.cat0 + dist.cat1 + dist.cat15 + dist.cat2;

  if (pieChart) pieChart.destroy();
  pieChart = new Chart(document.getElementById("chart-pie").getContext("2d"), {
    type: "doughnut",
    data: {
      labels: ["Infra (0.0)", "Risque (1.0)", "Interaction (1.5)", "Corrigé (2.0)"],
      datasets: [{
        data: [dist.cat0, dist.cat1, dist.cat15, dist.cat2],
        backgroundColor: [COLOR.cat0, COLOR.cat1, COLOR.cat15, COLOR.cat2],
        borderWidth: 2,
        borderColor: "#fff",
        hoverOffset: 10,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "62%",
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const val = ctx.raw;
              const pct = pieTotal > 0 ? ((val / pieTotal) * 100).toFixed(1) : 0;
              return ` ${val} obs · ${pct}%`;
            }
          }
        }
      },
    },
  });

  // ── Custom pie legend
  const legendItems = [
    { label: "Infra (0.0)",        color: COLOR.cat0,  count: dist.cat0  },
    { label: "Risque (1.0)",       color: COLOR.cat1,  count: dist.cat1  },
    { label: "Interaction (1.5)",  color: COLOR.cat15, count: dist.cat15 },
    { label: "Corrigé (2.0)",      color: COLOR.cat2,  count: dist.cat2  },
  ];

  document.getElementById("pie-legend").innerHTML = legendItems.map(item => {
    const pct = pieTotal > 0 ? ((item.count / pieTotal) * 100).toFixed(1) : "0.0";
    return `
      <div class="pie-legend-item">
        <div class="pie-legend-left">
          <div class="pie-dot" style="background:${item.color}"></div>
          <span class="pie-label">${item.label}</span>
        </div>
        <div>
          <span class="pie-count">${item.count}</span>
          <span class="pie-pct">${pct}%</span>
        </div>
      </div>
    `;
  }).join("");

  // ── Download button
  document.getElementById("btn-download").onclick = () => downloadExcel(data);
}

// ── Excel download ────────────────────────────────────────────────────────────
function downloadExcel(data) {
  if (!data?.excel_b64) return;
  const bytes = atob(data.excel_b64);
  const arr   = new Uint8Array(bytes.length).map((_, i) => bytes.charCodeAt(i));
  const blob  = new Blob([arr], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
  const url   = URL.createObjectURL(blob);
  const a     = Object.assign(document.createElement("a"), { href: url, download: data.filename || "BBS_Predit.xlsx" });
  a.click();
  URL.revokeObjectURL(url);
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetView() {
  lastResult = null;
  document.getElementById("file-input").value = "";
  document.getElementById("progress-fill").style.width = "0%";
  showView("view-upload");
}
