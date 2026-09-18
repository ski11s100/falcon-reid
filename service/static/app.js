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
  view: "vehicles",     // «по машинам» или «все снимки»
  history: [],          // недавние запросы этого сеанса
};

const scene = window.CityScene ? new CityScene($("map"), $("ticker")) : null;

scene?.start();
// Щелчок по снимку кандидата на схеме открывает то же окно сравнения.
if (scene) scene.onPick = (index) => openCompare(index);

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
    $("st-model").textContent = health.model_summary || health.model;
    $("st-model").title = health.model;
    $("st-device").textContent = health.device === "cuda" ? "GPU" : health.device.toUpperCase();
    $("m-gallery").textContent = health.gallery_size.toLocaleString("ru-RU");
    $("m-threshold").textContent = health.threshold != null ? formatThreshold(health.threshold) : "не задан";
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
    state.lastSearch = { data, request, fileName: state.fileName, cropUrl: queryCropUrl(), at: new Date(),
                         frame: { dataUrl: state.dataUrl, width: state.width, height: state.height } };
    state.explainCache.clear();
    remember(state.lastSearch);
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
    loadGallery();
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
  loadGallery();
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

function thumbHtml(candidate, className = "candidate__thumb") {
  return candidate.thumbnail
    ? `<img class="${className}" src="${candidate.thumbnail}" alt="">`
    : `<span class="${className}">нет снимка</span>`;
}

/* Шкала уверенности: где лежат кандидаты относительно порога. Делает
 * решение «совпадение или отказ» видимым, а не только словом в вердикте. */
function renderConfidence(data) {
  const threshold = data.threshold;
  const position = (score) => `${Math.max(0, Math.min(100, score * 100)).toFixed(1)}%`;
  const dots = data.candidates.map((c, index) => `
    <button type="button" class="confidence__dot ${isOver(c, threshold) ? "confidence__dot--over" : ""}
      ${index === 0 ? "confidence__dot--top" : ""}" style="left:${position(c.score)}" data-index="${index}"
      title="${escapeHtml(c.vehicle_id || c.image_id)} · ${c.score.toFixed(3)}"
      aria-label="Кандидат ${index + 1}, сходство ${c.score.toFixed(3)}"></button>`).join("");
  $("confidence").innerHTML = `
    <div class="confidence__track">
      ${threshold !== null ? `<span class="confidence__zone" style="left:${position(threshold)}"></span>
        <span class="confidence__threshold" style="left:${position(threshold)}">
          <span>порог ${threshold.toFixed(2)}</span></span>` : ""}
      ${dots}
    </div>
    <div class="confidence__scale"><span>0</span><span>0.25</span><span>0.5</span><span>0.75</span><span>1 · сходство</span></div>`;
}

/* Кандидаты одной машины собираются в одну карточку: оператору важно, какая
 * это машина и насколько уверенно, а не пять одинаковых строк подряд. */
function groupByVehicle(candidates) {
  const groups = new Map();
  candidates.forEach((candidate, index) => {
    const key = candidate.vehicle_id || `снимок:${candidate.image_id}`;
    if (!groups.has(key)) groups.set(key, { vehicle: candidate.vehicle_id, items: [] });
    groups.get(key).items.push({ candidate, index });
  });
  return [...groups.values()];
}

function renderShots(data) {
  const threshold = data.threshold;
  return data.candidates.map((candidate, index) => {
    const over = isOver(candidate, threshold);
    const width = Math.max(0, Math.min(100, candidate.score * 100));
    const vehicle = escapeHtml(candidate.vehicle_id || "без ID");
    return `
      <li><button type="button" class="candidate ${over ? "candidate--over" : ""}"
          data-index="${index}" style="animation-delay:${index * 30}ms"
          aria-label="Кандидат ${index + 1}: ${vehicle}, сходство ${candidate.score.toFixed(3)}">
        <span class="candidate__rank">${index + 1}</span>
        ${thumbHtml(candidate)}
        <span class="candidate__info">
          <span class="candidate__vehicle">${vehicle}</span>
          <span class="candidate__id">${escapeHtml(candidate.image_id)}</span>
          <span class="bar"><span class="bar__fill" style="width:${width}%"></span></span>
        </span>
        ${ringSvg(candidate.fingerprint, over ? "#34c98b" : "#7cc4ff")}
        <span class="candidate__score">${candidate.score.toFixed(3)}</span>
      </button></li>`;
  }).join("");
}

function renderVehicles(data) {
  const threshold = data.threshold;
  return groupByVehicle(data.candidates).map((group, position) => {
    const best = group.items[0];
    const over = group.items.filter(({ candidate }) => isOver(candidate, threshold)).length;
    const width = Math.max(0, Math.min(100, best.candidate.score * 100));
    const shots = group.items.map(({ candidate, index }) => `
      <button type="button" class="shot ${isOver(candidate, threshold) ? "shot--over" : ""}" data-index="${index}"
        aria-label="Снимок ${index + 1}, сходство ${candidate.score.toFixed(3)}">
        ${thumbHtml(candidate, "shot__image")}
        <span class="shot__score">${candidate.score.toFixed(2)}</span>
      </button>`).join("");
    const count = group.items.length;
    return `
      <li class="group ${over ? "group--over" : ""}" style="animation-delay:${position * 40}ms">
        <div class="group__head">
          <span class="group__rank">${position + 1}</span>
          <div class="group__title">
            <span class="group__vehicle">${escapeHtml(group.vehicle || "без ID")}</span>
            <span class="group__meta">${count} ${plural(count, "снимок", "снимка", "снимков")} в десятке${
              threshold !== null ? ` · выше порога: ${over}` : ""}</span>
          </div>
          ${ringSvg(best.candidate.fingerprint, over ? "#34c98b" : "#7cc4ff")}
          <span class="group__score" title="Лучшее сходство">${best.candidate.score.toFixed(3)}</span>
        </div>
        <span class="bar"><span class="bar__fill" style="width:${width}%"></span></span>
        <div class="group__shots">${shots}</div>
      </li>`;
  }).join("");
}

function plural(n, one, few, many) {
  const mod10 = n % 10, mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
  return many;
}

function renderCandidates() {
  const data = state.lastSearch?.data;
  if (!data) return;
  const list = $("candidates");
  list.className = `candidates candidates--${state.view}`;
  list.innerHTML = data.candidates.length
    ? (state.view === "vehicles" ? renderVehicles(data) : renderShots(data))
    : '<li class="result-meta">Галерея пуста: загрузите снимки пачкой слева</li>';
  document.querySelectorAll(".segmented__option").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.view === state.view));
  });
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
    + (threshold !== null ? ` · порог ${formatThreshold(threshold)}` : "") + " · нажмите на снимок, чтобы сравнить";
  renderConfidence(data);
  renderCandidates();
  $("warnings").innerHTML = data.quality.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("");
  document.querySelector(".result-tools").hidden = !data.candidates.length;
  $("confidence").hidden = !data.candidates.length;
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
  $("confidence").hidden = true;
  document.querySelector(".result-tools").hidden = true;
  showResultBody();
  scene?.setCaption(`Ошибка: ${message}`);
}

// Любой элемент с data-index в результате — снимок кандидата: открыть сравнение.
["candidates", "confidence"].forEach((id) => $(id).addEventListener("click", (event) => {
  const target = event.target.closest("[data-index]");
  if (target) openCompare(Number(target.dataset.index));
}));

// Наведение на кандидата подсвечивает его снимок на схеме города.
$("candidates").addEventListener("mouseover", (event) => {
  const target = event.target.closest("[data-index]");
  scene?.setHighlight(target ? Number(target.dataset.index) : null);
});
$("candidates").addEventListener("mouseleave", () => scene?.setHighlight(null));

document.querySelectorAll(".segmented__option").forEach((button) => {
  button.addEventListener("click", () => {
    state.view = button.dataset.view;
    try { localStorage.setItem("falcon-view", state.view); } catch { /* не критично */ }
    renderCandidates();
  });
});
try { state.view = localStorage.getItem("falcon-view") || state.view; } catch { /* по умолчанию */ }

/* ---------- Сравнение с картой внимания ---------- */

function openCompare(index) {
  const search = state.lastSearch;
  const candidate = search?.data.candidates[index];
  if (!candidate) return;
  const threshold = search.data.threshold;
  const over = isOver(candidate, threshold);

  const total = search.data.candidates.length;
  $("compare-title").textContent = `Кандидат ${index + 1}: ${candidate.vehicle_id || "без ID"}`;
  $("compare-counter").textContent = `${index + 1} из ${total}`;
  $("compare-prev").disabled = index === 0;
  $("compare-next").disabled = index === total - 1;
  $("compare-meta").textContent = `Сходство ${candidate.score.toFixed(3)}` + (threshold !== null
    ? (over ? ` — выше порога ${formatThreshold(threshold)}, совпадение` : ` — ниже порога ${formatThreshold(threshold)}`)
    : "");
  $("compare-query").src = search.cropUrl;
  $("compare-candidate").src = candidate.thumbnail || "";
  $("compare-candidate-caption").textContent = candidate.image_id;
  $("compare-explain").textContent = "Включите «куда смотрела модель», чтобы увидеть области запроса, "
    + "на которые модель опиралась, сравнивая его с этим кандидатом (Grad-CAM).";
  $("compare").dataset.index = index;
  if (!$("compare").open) {
    $("compare-heat").checked = false;
    $("compare").showModal();
  } else if ($("compare-heat").checked) {
    // Карта внимания остаётся включённой и при листании кандидатов.
    $("compare-heat").dispatchEvent(new Event("change"));
  }
}

function stepCompare(delta) {
  const total = state.lastSearch?.data.candidates.length || 0;
  const next = Number($("compare").dataset.index) + delta;
  if (next >= 0 && next < total) openCompare(next);
}

$("compare-prev").addEventListener("click", () => stepCompare(-1));
$("compare-next").addEventListener("click", () => stepCompare(1));
// Esc закрываем сами: встроенное закрытие модального окна Chrome пропускает,
// если окно открыто без свежего действия пользователя (правило CloseWatcher).
document.querySelectorAll("dialog").forEach((dialog) => {
  dialog.addEventListener("keydown", (event) => {
    if (event.key === "Escape") { event.preventDefault(); dialog.close(); }
  });
});

$("compare").addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") { event.preventDefault(); stepCompare(-1); }
  if (event.key === "ArrowRight") { event.preventDefault(); stepCompare(1); }
  // H и Р — одна клавиша в латинской и русской раскладке.
  if (["h", "H", "р", "Р"].includes(event.key)) {
    $("compare-heat").checked = !$("compare-heat").checked;
    $("compare-heat").dispatchEvent(new Event("change"));
  }
});

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

/* ---------- Недавние запросы ---------- */

function remember(search) {
  state.history = [search, ...state.history.filter((item) => item !== search)].slice(0, 6);
  renderHistory();
}

function renderHistory() {
  $("history-section").hidden = !state.history.length;
  $("history").innerHTML = state.history.map((item, index) => {
    const view = VERDICTS[item.data.verdict] || VERDICTS["совпадений нет"];
    const top = item.data.candidates[0];
    const time = item.at.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
    return `
      <button type="button" class="history__item ${item === state.lastSearch ? "history__item--active" : ""}"
          data-history="${index}" title="${escapeHtml(item.fileName || "")}">
        <img class="history__image" src="${item.cropUrl}" alt="">
        <span class="history__text">
          <span class="history__verdict history__verdict--${view.css}">${view.title}</span>
          <span class="history__meta">${time}${top ? ` · ${escapeHtml(top.vehicle_id || "без ID")} ${top.score.toFixed(2)}` : ""}</span>
        </span>
      </button>`;
  }).join("");
}

// Возврат к прошлому запросу: кадр, рамка, результат и схема — как были.
$("history").addEventListener("click", (event) => {
  const target = event.target.closest("[data-history]");
  if (!target) return;
  const item = state.history[Number(target.dataset.history)];
  const image = $("frame-image");
  image.onload = () => {
    Object.assign(state, { dataUrl: item.frame.dataUrl, width: item.frame.width,
                           height: item.frame.height, fileName: item.fileName });
    $("drop-empty").hidden = true;
    $("frame").hidden = false;
    $("query-tools").hidden = false;
    $("drop").classList.add("drop--loaded");
    setBBox(item.request.bbox || null);
    $("btn-search").disabled = false;
    $("btn-register").disabled = false;
    state.lastSearch = item;
    state.explainCache.clear();
    renderSearch(item.data);
    renderHistory();
    scene?.setQuery(true);
    scene?.showResults(item.data.candidates, item.data.threshold, resultCaption(item.data));
  };
  image.src = item.frame.dataUrl;
});

/* ---------- Обзор галереи ---------- */

async function loadGallery() {
  try {
    const data = await call("GET", "/api/gallery?limit=60");
    const vehicles = data.vehicles || [];
    $("gallery-count").textContent = `${vehicles.length} ${plural(vehicles.length, "машина", "машины", "машин")} · `
      + `${data.size} ${plural(data.size, "снимок", "снимка", "снимков")}`;
    $("gallery-grid").innerHTML = vehicles.length ? vehicles.map((v) => `
      <button type="button" class="gallery-card" data-vehicle="${escapeHtml(v.vehicle_id)}"
          title="Добавлять новые кадры к ${escapeHtml(v.vehicle_id)}">
        ${v.thumbnail ? `<img class="gallery-card__image" src="${v.thumbnail}" alt="">`
                      : '<span class="gallery-card__image"></span>'}
        <span class="gallery-card__id">${escapeHtml(v.vehicle_id)}</span>
        <span class="gallery-card__count">${v.shots}</span>
      </button>`).join("")
      : '<p class="section-hint">Пока пусто. Загрузите снимки пачкой: ID машины берётся из имени файла.</p>';
  } catch (error) {
    $("gallery-grid").innerHTML = `<p class="section-hint">${escapeHtml(error.message)}</p>`;
  }
}

$("gallery-grid").addEventListener("click", (event) => {
  const card = event.target.closest("[data-vehicle]");
  if (!card) return;
  $("vehicle").value = card.dataset.vehicle;
  $("vehicle").focus();
  toast(state.dataUrl ? `Текущий кадр будет добавлен к ${card.dataset.vehicle} — нажмите «Добавить»`
                      : `Загрузите кадр, и его можно будет добавить к ${card.dataset.vehicle}`);
});

/* ---------- Прочее ---------- */

// Порог показываем как есть (0.4925), а не округлённым до 0.492.
function formatThreshold(value) {
  return String(Number(value.toFixed(4)));
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = String(value);
  return div.innerHTML;
}

refreshStatus();
loadGallery();
setInterval(refreshStatus, 15000);
