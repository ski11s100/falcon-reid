/* Клиент интерфейса оператора.
 *
 * Тонкий клиент по разделу 6 ТЗ: вся логика на сервере, здесь только сбор
 * ввода, вызовы API и отрисовка. Никаких внешних библиотек — контейнер
 * работает без доступа в интернет (раздел 8 ТЗ).
 */

const $ = (id) => document.getElementById(id);

const state = {
  dataUrl: null,
  naturalWidth: 0,
  naturalHeight: 0,
  threshold: null,
};

/* ---------- Состояние сервиса ---------- */

async function refreshStatus() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();

    const ok = health.status === "ok";
    $("st-status").textContent = ok ? "готов" : "модель не загружена";
    $("st-status").className = "status__value " + (ok ? "status__value--ok" : "status__value--alert");
    $("st-model").textContent = health.model;
    $("st-device").textContent = health.device;
    $("st-gallery").textContent = `${health.gallery_size} наблюдений · ${health.storage}`;

    state.threshold = health.threshold_calibrated;
    $("st-threshold").textContent = health.threshold_calibrated ? "откалиброван" : "не откалиброван";
    $("st-threshold").className =
      "status__value " + (health.threshold_calibrated ? "status__value--ok" : "status__value--alert");
  } catch {
    $("st-status").textContent = "сервис недоступен";
    $("st-status").className = "status__value status__value--alert";
  }
}

/* ---------- Загрузка кадра ---------- */

$("file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    state.dataUrl = reader.result;
    const image = new Image();
    image.onload = () => {
      state.naturalWidth = image.naturalWidth;
      state.naturalHeight = image.naturalHeight;
      renderPreview();
      $("btn-search").disabled = false;
      $("btn-register").disabled = false;
    };
    image.src = reader.result;
  };
  reader.readAsDataURL(file);
});

["bx", "by", "bw", "bh"].forEach((id) => $(id).addEventListener("input", renderPreview));

function readBBox() {
  const values = ["bx", "by", "bw", "bh"].map((id) => $(id).value.trim());
  if (values.some((value) => value === "")) return null;
  const [x, y, w, h] = values.map(Number);
  if ([x, y, w, h].some((n) => !Number.isFinite(n)) || w <= 1 || h <= 1) return null;
  return { x, y, w, h };
}

function renderPreview() {
  const preview = $("preview");
  if (!state.dataUrl) {
    preview.innerHTML = '<p class="preview__empty">Кадр не загружен</p>';
    return;
  }

  preview.innerHTML = "";
  const img = document.createElement("img");
  img.alt = "Загруженный кадр";

  const bbox = readBBox();
  // Рамку можно позиционировать только после того, как изображение разложилось
  // в вёрстке: до этого его ширина равна нулю, и рамка схлопнулась бы в точку,
  // а её затемняющая подложка закрыла бы весь кадр.
  const drawBox = () => {
    if (!bbox || !state.naturalWidth) return;
    const rect = img.getBoundingClientRect();
    const host = preview.getBoundingClientRect();
    if (rect.width < 1) return;

    preview.querySelector(".preview__box")?.remove();
    const scale = rect.width / state.naturalWidth;
    const box = document.createElement("div");
    box.className = "preview__box";
    box.style.left = `${rect.left - host.left + bbox.x * scale}px`;
    box.style.top = `${rect.top - host.top + bbox.y * scale}px`;
    box.style.width = `${Math.max(2, bbox.w * scale)}px`;
    box.style.height = `${Math.max(2, bbox.h * scale)}px`;
    preview.appendChild(box);
  };

  img.addEventListener("load", () => requestAnimationFrame(drawBox));
  img.src = state.dataUrl;
  preview.appendChild(img);
  if (img.complete) requestAnimationFrame(drawBox);
}

// Кадр масштабируется вместе с окном, поэтому рамку нужно пересчитывать.
window.addEventListener("resize", () => renderPreview());

/* ---------- Вызовы API ---------- */

function payload(extra = {}) {
  const body = { image_base64: state.dataUrl, ...extra };
  const bbox = readBBox();
  if (bbox) body.bbox = bbox;
  return body;
}

async function call(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Ошибка сервиса");
  return data;
}

$("btn-search").addEventListener("click", async () => {
  setBusy(true, "Поиск…");
  try {
    renderSearch(await call("/api/search", payload({ top_k: 10 })));
  } catch (error) {
    renderError(error.message);
  } finally {
    setBusy(false, "Найти");
  }
});

$("btn-register").addEventListener("click", async () => {
  const vehicleId = $("vehicle").value.trim();
  const imageId = `ui-${Date.now()}`;
  setBusy(true, "…", "btn-register");
  try {
    const result = await call("/api/gallery/register",
      payload({ image_id: imageId, vehicle_id: vehicleId || null }));
    renderRegistered(result);
    refreshStatus();
  } catch (error) {
    renderError(error.message);
  } finally {
    setBusy(false, "В галерею", "btn-register");
  }
});

function setBusy(busy, label, id = "btn-search") {
  const button = $(id);
  button.disabled = busy;
  button.textContent = label;
}

/* ---------- Отрисовка ---------- */

const VERDICTS = {
  "совпадение": { css: "accept", sign: "=", title: "Совпадение найдено" },
  "требуется проверка": { css: "review", sign: "?", title: "Требуется проверка" },
  "совпадений нет": { css: "refuse", sign: "—", title: "Совпадений нет" },
};

function renderSearch(data) {
  const view = VERDICTS[data.verdict] || VERDICTS["совпадений нет"];
  const threshold = data.threshold;

  const candidates = data.candidates.map((candidate, index) => {
    const over = threshold !== null && candidate.score >= threshold;
    const width = Math.max(0, Math.min(100, candidate.score * 100));
    return `
      <li class="candidate ${over ? "candidate--over" : ""}">
        <span class="candidate__rank">${index + 1}</span>
        <span class="candidate__id">${escapeHtml(candidate.image_id)}
          <span class="candidate__vehicle">${escapeHtml(candidate.vehicle_id || "ТС не размечено")}</span>
        </span>
        <span class="candidate__score">${candidate.score.toFixed(4)}</span>
        <span class="bar"><span class="bar__fill" style="width:${width}%"></span></span>
      </li>`;
  }).join("");

  const warnings = data.quality.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("");

  $("result").innerHTML = `
    <div class="verdict verdict--${view.css}">
      <div class="verdict__sign">${view.sign}</div>
      <div class="verdict__text">
        <p class="verdict__title">${view.title}</p>
        <p class="verdict__note">${escapeHtml(data.refusal_reason || "Верхний кандидат прошёл порог уверенности")}</p>
      </div>
    </div>
    <p class="hint">Обработано за ${data.elapsed_ms.toFixed(1)} мс · кроп
      ${data.quality.width}×${data.quality.height} пикс.${
        threshold !== null ? ` · порог ${threshold.toFixed(3)}` : ""}</p>
    ${candidates ? `<ul class="candidates">${candidates}</ul>`
                 : '<div class="empty-state">Галерея пуста</div>'}
    ${warnings ? `<ul class="warnings">${warnings}</ul>` : ""}`;
}

function renderRegistered(data) {
  $("result").innerHTML = `
    <div class="verdict verdict--accept">
      <div class="verdict__sign">+</div>
      <div class="verdict__text">
        <p class="verdict__title">Добавлено в галерею</p>
        <p class="verdict__note">${escapeHtml(data.image_id)} ·
          ${escapeHtml(data.vehicle_id || "без идентификатора ТС")} ·
          вектор ${data.embedding_dim} измерений</p>
      </div>
    </div>
    <p class="hint">Всего в галерее: ${data.gallery_size} наблюдений.
      Обработано за ${data.elapsed_ms.toFixed(1)} мс.</p>
    ${data.quality.warnings.length
      ? `<ul class="warnings">${data.quality.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>`
      : ""}`;
}

function renderError(message) {
  $("result").innerHTML = `
    <div class="verdict verdict--refuse">
      <div class="verdict__sign">!</div>
      <div class="verdict__text">
        <p class="verdict__title">Ошибка</p>
        <p class="verdict__note">${escapeHtml(message)}</p>
      </div>
    </div>`;
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = String(value);
  return div.innerHTML;
}

refreshStatus();
setInterval(refreshStatus, 15000);
