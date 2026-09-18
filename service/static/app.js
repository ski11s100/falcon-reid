/* Клиент пульта оператора.
 *
 * Тонкий клиент по разделу 6 ТЗ: вся логика на сервере, здесь только сбор
 * ввода, вызовы API и отрисовка. Никаких внешних библиотек: контейнер
 * работает без доступа в интернет (раздел 8 ТЗ).
 *
 * Сценарий оператора:
 *   1. перетащить кадр с камеры в окно (или выбрать файл, или Ctrl+V);
 *   2. обвести машину мышью прямо на кадре;
 *   3. «Найти» — справа появляются кандидаты, на схеме города у камер
 *      всплывают их снимки; кандидат открывается в окне сравнения с картой
 *      внимания Grad-CAM; результат выгружается в CSV или JSON.
 */

"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  dataUrl: null,
  fileName: null,
  width: 0,
  height: 0,
  lastSearch: null,     // ответ /api/search вместе с тем, что искали
  explainCache: new Map(),
};

const scene = window.CityScene ? new CityScene($("map"), $("ticker")) : null;

scene?.start();

/* ---------- Связь с сервером ---------- */

function apiKey() {
  try { return sessionStorage.getItem("falcon-api-key"); } catch { return null; }
}

// Ключ API нужен, только если сервер запущен с FALCON_API_KEY. Тогда на
// первый отказ 401 интерфейс спрашивает ключ и повторяет запрос.
function askApiKey() {
  const dialog = $("key-dialog");
  $("key-input").value = "";
  dialog.showModal();
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => {
      const key = $("key-input").value.trim();
      try { if (key) sessionStorage.setItem("falcon-api-key", key); } catch { /* не критично */ }
      resolve(Boolean(key));
    }, { once: true });
  });
}

async function call(method, path, body, retried = false) {
  const headers = { "Content-Type": "application/json" };
  const key = apiKey();
  if (key) headers["X-API-Key"] = key;
  const response = await fetch(path, {
    method, headers, body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (response.status === 401 && !retried && await askApiKey()) {
    return call(method, path, body, true);
  }
  let data = null;
  try { data = await response.json(); } catch { /* пустой ответ */ }
  if (!response.ok) {
    const detail = data && data.detail;
    throw new Error(typeof detail === "string" ? detail : `Ошибка сервиса (${response.status})`);
  }
  return data;
}

function toast(message) {
  const box = $("toast");
  box.textContent = message;
  box.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { box.hidden = true; }, 4000);
}

/* ---------- Состояние сервиса ---------- */

function setSignal(text, tone) {
  const signal = $("st-status");
  signal.className = `signal signal--${tone}`;
  signal.querySelector(".signal__text").textContent = text;
}

async function refreshStatus() {
  try {
    const health = await (await fetch("/api/health")).json();
    const ok = health.status === "ok";
    setSignal(ok ? "готов" : "модель не загружена", ok ? "ok" : "alert");
    $("st-model").textContent = health.model;
    $("st-device").textContent = health.device === "cuda" ? "GPU" : health.device;
    $("m-gallery").textContent = health.gallery_size.toLocaleString("ru-RU");
    $("m-threshold").textContent = health.threshold_calibrated ? "задан" : "нет";
    if (health.embedding_dim) $("m-dim").textContent = health.embedding_dim.toLocaleString("ru-RU");
  } catch {
    setSignal("сервис недоступен", "alert");
  }
}

/* ---------- Кадр: файл, перетаскивание, буфер обмена ---------- */

function readAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error(`не читается: ${file.name}`));
    reader.readAsDataURL(file);
  });
}

async function loadFrame(file) {
  if (!file || !file.type.startsWith("image/")) {
    toast("Нужен снимок в формате JPEG, PNG или WEBP");
    return;
  }
  const dataUrl = await readAsDataUrl(file);
  const image = $("frame-image");
  image.onload = () => {
    state.dataUrl = dataUrl;
    state.fileName = file.name || "кадр из буфера";
    state.width = image.naturalWidth;
    state.height = image.naturalHeight;
    $("drop-empty").hidden = true;
    $("frame").hidden = false;
    $("query-tools").hidden = false;
    $("drop").classList.add("drop--loaded");
    setBBox(null);
    $("btn-search").disabled = false;
    $("btn-register").disabled = false;
    scene?.setQuery(true);
  };
  image.src = dataUrl;
}

function resetFrame() {
  state.dataUrl = null;
  $("frame-image").removeAttribute("src");
  $("frame").hidden = true;
  $("query-tools").hidden = true;
  $("drop-empty").hidden = false;
  $("drop").classList.remove("drop--loaded");
  setBBox(null);
  $("btn-search").disabled = true;
  $("btn-register").disabled = true;
  $("file").value = "";
  scene?.setQuery(false);
}

$("file").addEventListener("change", (event) => loadFrame(event.target.files[0]));
$("frame-clear").addEventListener("click", resetFrame);

const drop = $("drop");
["dragenter", "dragover"].forEach((type) => drop.addEventListener(type, (event) => {
  event.preventDefault();
  drop.classList.add("drop--over");
}));
["dragleave", "drop"].forEach((type) => drop.addEventListener(type, () => {
  drop.classList.remove("drop--over");
}));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  loadFrame(event.dataTransfer.files[0]);
});
drop.addEventListener("keydown", (event) => {
  if ((event.key === "Enter" || event.key === " ") && !state.dataUrl) $("file").click();
});

document.addEventListener("paste", (event) => {
  if (event.target instanceof HTMLInputElement) return;
  const item = [...(event.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) loadFrame(item.getAsFile());
});

/* ---------- Рамка: рисуется мышью или вводится вручную ---------- */

const bboxInputs = ["bx", "by", "bw", "bh"];

function readBBox() {
  const values = bboxInputs.map((id) => $(id).value.trim());
  if (values.some((value) => value === "")) return null;
  const [x, y, w, h] = values.map(Number);
  if ([x, y, w, h].some((n) => !Number.isFinite(n)) || w <= 1 || h <= 1) return null;
  return { x, y, w, h };
}

function setBBox(box) {
  bboxInputs.forEach((id, i) => {
    $(id).value = box ? Math.round([box.x, box.y, box.w, box.h][i]) : "";
  });
  drawBBox(box);
}

function drawBBox(box) {
  const element = $("frame-box");
  const image = $("frame-image");
  $("frame-prompt").hidden = Boolean(box);
  if (!box || !state.width || !image.clientWidth) { element.hidden = true; return; }
  const scale = image.clientWidth / state.width;
  Object.assign(element.style, {
    left: `${box.x * scale}px`, top: `${box.y * scale}px`,
    width: `${Math.max(2, box.w * scale)}px`, height: `${Math.max(2, box.h * scale)}px`,
  });
  element.hidden = false;
}

function toImagePoint(event) {
  const rect = $("frame-image").getBoundingClientRect();
  const scale = state.width / rect.width;
  return {
    x: Math.min(state.width, Math.max(0, (event.clientX - rect.left) * scale)),
    y: Math.min(state.height, Math.max(0, (event.clientY - rect.top) * scale)),
  };
}

function boxBetween(a, b) {
  return { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y),
           w: Math.abs(b.x - a.x), h: Math.abs(b.y - a.y) };
}

let dragStart = null;
const frame = $("frame");
frame.addEventListener("pointerdown", (event) => {
  if (!state.dataUrl) return;
  dragStart = toImagePoint(event);
  frame.setPointerCapture(event.pointerId);
});
frame.addEventListener("pointermove", (event) => {
  if (dragStart) drawBBox(boxBetween(dragStart, toImagePoint(event)));
});
frame.addEventListener("pointerup", (event) => {
  if (!dragStart) return;
  const box = boxBetween(dragStart, toImagePoint(event));
  dragStart = null;
  // Щелчок без протяжки — не рамка: оставляем прежнюю.
  if (box.w < 12 || box.h < 12) { drawBBox(readBBox()); return; }
  setBBox(box);
});

bboxInputs.forEach((id) => $(id).addEventListener("input", () => drawBBox(readBBox())));
$("box-clear").addEventListener("click", () => setBBox(null));
window.addEventListener("resize", () => drawBBox(readBBox()));

function payload(extra = {}) {
  const body = { image_base64: state.dataUrl, ...extra };
  const bbox = readBBox();
  if (bbox) body.bbox = bbox;
  return body;
}

/* Кроп запроса для окна сравнения: вырезается в браузере из исходного кадра. */
function queryCropUrl() {
  const bbox = readBBox();
  if (!bbox) return state.dataUrl;
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(bbox.w));
  canvas.height = Math.max(1, Math.round(bbox.h));
  canvas.getContext("2d").drawImage($("frame-image"), bbox.x, bbox.y, bbox.w, bbox.h,
    0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.92);
}

/* ---------- Поиск и регистрация ---------- */

function busy(id, on, label) {
  const button = $(id);
  button.disabled = on;
  if (label) button.textContent = label;
}

function resultCaption(data) {
  if (!data.candidates.length) return "Галерея пуста — искать не среди чего";
  const top = data.candidates[0];
  const above = data.threshold === null ? 0
    : data.candidates.filter((c) => c.score >= data.threshold).length;
  if (data.accepted) {
    return `Найдено совпадений выше порога: ${above} · лучшее — ${top.vehicle_id || top.image_id}, сходство ${top.score.toFixed(2)}`;
  }
  return `Уверенного совпадения нет — сервис отказался от ответа (лучшее сходство ${top.score.toFixed(2)})`;
}

async function search() {
  if (!state.dataUrl) return;
  busy("btn-search", true, "Поиск…");
  scene?.setScanning(true);
  const started = performance.now();
  try {
    const request = payload({ top_k: 10 });
    const data = await call("POST", "/api/search", request);
    // Волна по камерам должна успеть стать заметной, даже если сервер ответил
    // за десятки миллисекунд: иначе оператор не увидит, что поиск был.
    await new Promise((resolve) => setTimeout(resolve, Math.max(0, 700 - (performance.now() - started))));
    state.lastSearch = { data, request, fileName: state.fileName, cropUrl: queryCropUrl(), at: new Date() };
    state.explainCache.clear();
    renderSearch(data);
    $("m-last").textContent = `${Math.round(data.elapsed_ms)} мс`;
    scene?.setScanning(false);
    scene?.showResults(data.candidates, data.threshold, resultCaption(data));
  } catch (error) {
    scene?.setScanning(false);
    renderError(error.message);
  } finally {
    busy("btn-search", false, "Найти");
  }
}

$("btn-search").addEventListener("click", search);
document.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && state.dataUrl) search();
});

$("btn-register").addEventListener("click", async () => {
  busy("btn-register", true, "…");
  try {
    const vehicleId = $("vehicle").value.trim();
    const data = await call("POST", "/api/gallery/register",
      payload({ image_id: `ui-${Date.now()}`, vehicle_id: vehicleId || null }));
    toast(`Добавлено в галерею: ${data.vehicle_id || "без ID"} · всего снимков ${data.gallery_size}`);
    refreshStatus();
  } catch (error) {
    toast(error.message);
  } finally {
    busy("btn-register", false, "Добавить");
  }
});

/* ---------- Пакетная загрузка галереи ---------- */

function vehicleFromName(name) {
  // ТС-881_камера-10.jpg -> ТС-881: так названы снимки, выгруженные из
  // датасета, и это избавляет от ручного ввода идентификатора для каждого.
  const base = name.replace(/\.[^.]+$/, "");
  const cut = base.indexOf("_");
  return cut > 0 ? base.slice(0, cut) : base;
}

function renderBatch(done, total, message, errors = []) {
  const percent = total ? Math.round((done / total) * 100) : 0;
  $("batch-status").innerHTML = `
    <div class="batch-progress">
      <div>${escapeHtml(message)} · ${done} из ${total}</div>
      <div class="batch-progress__bar"><span class="batch-progress__fill" style="width:${percent}%"></span></div>
      ${errors.length ? `<ul class="batch-progress__list">${errors.slice(0, 5)
        .map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul>` : ""}
    </div>`;
}

// Выбор файлов сразу запускает загрузку: лишняя кнопка «Загрузить» — лишний шаг.
$("batch").addEventListener("change", async (event) => {
  const files = Array.from(event.target.files || []);
  if (!files.length) return;
  const errors = [];
  let registered = 0;

  // Порциями по 20: один запрос на сотни снимков упёрся бы в лимит размера
  // тела, а по одному снова превратился бы в минуты ожидания.
  const CHUNK = 20;
  for (let start = 0; start < files.length; start += CHUNK) {
    renderBatch(registered, files.length, "Загружаю галерею", errors);
    const items = [];
    for (const file of files.slice(start, start + CHUNK)) {
      try {
        items.push({ image_id: file.name, vehicle_id: vehicleFromName(file.name),
                     image_base64: await readAsDataUrl(file) });
      } catch (error) {
        errors.push(error.message);
      }
    }
    if (!items.length) continue;
    try {
      const result = await call("POST", "/api/gallery/register-batch", { items });
      registered += result.registered;
      (result.failed || []).forEach((f) => errors.push(`${f.image_id}: ${f.error}`));
    } catch (error) {
      errors.push(error.message);
    }
  }
  renderBatch(registered, files.length, errors.length ? `Готово, замечаний: ${errors.length}` : "Готово", errors);
  event.target.value = "";
  refreshStatus();
});

/* ---------- Результат ---------- */

const VERDICTS = {
  "совпадение": { css: "accept", title: "Совпадение найдено" },
  "требуется проверка": { css: "review", title: "Требуется проверка" },
  "совпадений нет": { css: "refuse", title: "Совпадений нет" },
};

function isOver(candidate, threshold) {
  return threshold !== null && candidate.score >= threshold;
}

/* Кольцо цифрового отпечатка кандидата: проекция эмбеддинга на 128
 * направлений (см. fingerprint() в service/app.py). У снимков одной машины
 * кольца похожи. Лёгкое сглаживание делает форму читаемой. */
function ringSvg(values, colour) {
  if (!values || values.length < 8) return '<span class="candidate__ring"></span>';
  const n = values.length;
  const smooth = values.map((_, i) =>
    [-2, -1, 0, 1, 2].reduce((sum, k) => sum + values[(i + k + n) % n] * Math.exp(-(k * k) / 2), 0));
  const sorted = smooth.slice().sort((a, b) => a - b);
  const low = sorted[Math.floor(n * 0.04)];
  const high = sorted[Math.ceil(n * 0.96) - 1];
  const span = high - low || 1;
  const size = 38, c = size / 2, inner = 7, reach = 10;
  let path = "";
  smooth.forEach((value, i) => {
    const v = Math.min(1, Math.max(0, (value - low) / span));
    const a = (i / n) * Math.PI * 2 - Math.PI / 2;
    const r2 = inner + 1.5 + reach * v;
    path += `M${(c + Math.cos(a) * inner).toFixed(1)} ${(c + Math.sin(a) * inner).toFixed(1)}`
          + `L${(c + Math.cos(a) * r2).toFixed(1)} ${(c + Math.sin(a) * r2).toFixed(1)}`;
  });
  return `<svg class="candidate__ring" viewBox="0 0 ${size} ${size}" aria-hidden="true">`
       + `<title>Цифровой отпечаток</title><path d="${path}" stroke="${colour}" stroke-width="1" fill="none"/></svg>`;
}

function renderSearch(data) {
  const view = VERDICTS[data.verdict] || VERDICTS["совпадений нет"];
  const threshold = data.threshold;

  $("verdict").className = `verdict verdict--${view.css}`;
  $("verdict").innerHTML = `
    <span class="verdict__dot" aria-hidden="true"></span>
    <div>
      <p class="verdict__title">${view.title}</p>
      <p class="verdict__note">${escapeHtml(data.refusal_reason || "Лучший кандидат выше порога уверенности")}</p>
    </div>`;
  $("result-meta").textContent = `${Math.round(data.elapsed_ms)} мс · рамка ${data.quality.width}×${data.quality.height}`
    + (threshold !== null ? ` · порог ${threshold.toFixed(3)}` : "") + " · нажмите на кандидата, чтобы сравнить";

  $("candidates").innerHTML = data.candidates.map((candidate, index) => {
    const over = isOver(candidate, threshold);
    const width = Math.max(0, Math.min(100, candidate.score * 100));
    const vehicle = escapeHtml(candidate.vehicle_id || "без ID");
    const thumb = candidate.thumbnail
      ? `<img class="candidate__thumb" src="${candidate.thumbnail}" alt="">`
      : `<span class="candidate__thumb">нет снимка</span>`;
    return `
      <li><button type="button" class="candidate ${over ? "candidate--over" : ""}"
          data-index="${index}" style="animation-delay:${index * 30}ms"
          aria-label="Кандидат ${index + 1}: ${vehicle}, сходство ${candidate.score.toFixed(3)}">
        <span class="candidate__rank">${index + 1}</span>
        ${thumb}
        <span class="candidate__info">
          <span class="candidate__vehicle">${vehicle}</span>
          <span class="candidate__id">${escapeHtml(candidate.image_id)}</span>
          <span class="bar"><span class="bar__fill" style="width:${width}%"></span></span>
        </span>
        ${ringSvg(candidate.fingerprint, over ? "#34c98b" : "#7cc4ff")}
        <span class="candidate__score">${candidate.score.toFixed(3)}</span>
      </button></li>`;
  }).join("") || '<li class="result-meta">Галерея пуста: загрузите снимки пачкой слева</li>';

  $("warnings").innerHTML = data.quality.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("");
  document.querySelector(".export").hidden = !data.candidates.length;
  showResultBody();
}

function showResultBody() {
  $("result-empty").hidden = true;
  $("result-body").hidden = false;
}

function renderError(message) {
  $("verdict").className = "verdict verdict--error";
  $("verdict").innerHTML = `
    <span class="verdict__dot" aria-hidden="true"></span>
    <div><p class="verdict__title">Ошибка</p><p class="verdict__note">${escapeHtml(message)}</p></div>`;
  $("result-meta").textContent = "";
  $("candidates").innerHTML = "";
  $("warnings").innerHTML = "";
  document.querySelector(".export").hidden = true;
  showResultBody();
  scene?.setCaption(`Ошибка: ${message}`);
}

$("candidates").addEventListener("click", (event) => {
  const button = event.target.closest(".candidate");
  if (button) openCompare(Number(button.dataset.index));
});

/* ---------- Сравнение с картой внимания ---------- */

function openCompare(index) {
  const search = state.lastSearch;
  const candidate = search?.data.candidates[index];
  if (!candidate) return;
  const threshold = search.data.threshold;
  const over = isOver(candidate, threshold);

  $("compare-title").textContent = `Кандидат ${index + 1}: ${candidate.vehicle_id || "без ID"}`;
  $("compare-meta").textContent = `Сходство ${candidate.score.toFixed(3)}` + (threshold !== null
    ? (over ? ` — выше порога ${threshold.toFixed(3)}, совпадение` : ` — ниже порога ${threshold.toFixed(3)}`)
    : "");
  $("compare-query").src = search.cropUrl;
  $("compare-candidate").src = candidate.thumbnail || "";
  $("compare-candidate-caption").textContent = candidate.image_id;
  $("compare-heat").checked = false;
  $("compare-explain").textContent = "Включите «куда смотрела модель», чтобы увидеть области запроса, "
    + "на которые модель опиралась, сравнивая его с этим кандидатом (Grad-CAM).";
  $("compare").dataset.index = index;
  $("compare").showModal();
}

$("compare-heat").addEventListener("change", async (event) => {
  const search = state.lastSearch;
  const index = Number($("compare").dataset.index);
  const candidate = search?.data.candidates[index];
  if (!candidate) return;
  if (!event.target.checked) { $("compare-query").src = search.cropUrl; return; }

  try {
    let explain = state.explainCache.get(candidate.image_id);
    if (!explain) {
      $("compare-explain").textContent = "Считаю карту внимания…";
      const { top_k, ...request } = search.request;
      explain = await call("POST", "/api/explain",
        { ...request, gallery_id: candidate.image_id, show_plate_region: true });
      state.explainCache.set(candidate.image_id, explain);
    }
    if (Number($("compare").dataset.index) !== index || !event.target.checked) return;
    $("compare-query").src = `data:image/png;base64,${explain.overlay_png_base64}`;
    const check = explain.plate_check;
    $("compare-explain").innerHTML = "Красным — области, сильнее всего повлиявшие на сходство с кандидатом."
      + (check ? ` Проверка номера: ${escapeHtml(check.verdict)}.` : "");
  } catch (error) {
    event.target.checked = false;
    $("compare-explain").textContent = error.message;
  }
});

/* ---------- Экспорт ---------- */

function download(name, type, content) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportStem() {
  const at = state.lastSearch.at.toISOString().replace(/[:.]/g, "-").slice(0, 19);
  return `falcon-search-${at}`;
}

function csvCell(value) {
  const text = value === null || value === undefined ? "" : String(value);
  // Защита от CSV-инъекций: ячейка, начинающаяся с =, +, -, @, в Excel
  // исполнилась бы как формула. Идентификаторы задаёт внешний клиент.
  const safe = /^[=+\-@\t\r]/.test(text) ? `'${text}` : text;
  return /[";\n,]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe;
}

$("export-csv").addEventListener("click", () => {
  const { data, fileName } = state.lastSearch;
  const lines = [["rank", "gallery_image_id", "vehicle_id", "score", "above_threshold"].join(",")];
  data.candidates.forEach((c, i) => lines.push([
    i + 1, c.image_id, c.vehicle_id, c.score.toFixed(6), isOver(c, data.threshold) ? 1 : 0,
  ].map(csvCell).join(",")));
  const header = `# query=${csvCell(fileName)}; verdict=${data.verdict}; threshold=${data.threshold ?? ""}\n`;
  download(`${exportStem()}.csv`, "text/csv;charset=utf-8", "﻿" + header + lines.join("\n") + "\n");
});

$("export-json").addEventListener("click", () => {
  const { data, request, fileName, at } = state.lastSearch;
  const report = {
    query: { file: fileName, bbox: request.bbox || null, searched_at: at.toISOString() },
    verdict: data.verdict,
    accepted: data.accepted,
    threshold: data.threshold,
    refusal_reason: data.refusal_reason,
    quality: data.quality,
    elapsed_ms: data.elapsed_ms,
    candidates: data.candidates.map((c, i) => ({
      rank: i + 1, image_id: c.image_id, vehicle_id: c.vehicle_id, score: c.score,
      above_threshold: isOver(c, data.threshold),
    })),
  };
  download(`${exportStem()}.json`, "application/json", JSON.stringify(report, null, 2));
});

/* ---------- Прочее ---------- */

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = String(value);
  return div.innerHTML;
}

refreshStatus();
setInterval(refreshStatus, 15000);
