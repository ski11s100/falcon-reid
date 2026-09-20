/* Живая схема города — фон пульта оператора.
 *
 * Показывает задачу без слов: карта города сверху, по улицам едут машины, на
 * перекрёстках стоят камеры с конусами обзора. Камера, заметившая машину,
 * обводит её рамкой. Одна машина — цель: её узнают на разных камерах по
 * внешнему виду, и между камерами тянется линия маршрута. Это и есть повторная
 * идентификация (ReID): та же машина на другой камере, без номера.
 *
 * Схема отвечает на действия оператора:
 *   setQuery(true)   — кадр загружен: цель выделена, остальное приглушено;
 *   setScanning(on)  — идёт поиск: от цели по камерам бежит волна;
 *   showResults(...) — у камер всплывают настоящие снимки найденных кандидатов
 *                      со сходством: зелёные выше порога, серые ниже.
 * Камеры на схеме условные: сервис не знает, где установлены камеры (их нет в
 * данных по условию задачи), поэтому снимки раскладываются по ближайшим к цели.
 *
 * Рисуется на canvas в реальном времени: видеофайл не нужен, всё работает без
 * интернета. Цикл останавливается, когда первый экран не виден или вкладка
 * скрыта. При prefers-reduced-motion рисуется один неподвижный кадр.
 */

(function () {
  "use strict";

  // Ночной город: асфальт, светлые тротуары по краю кварталов, тёмные дворы.
  const C = {
    asphalt: "#0e1522",
    sidewalk: "#1a2437",
    yard: "#0a0f18",
    park: "#0c1f18",
    path: "rgba(160, 176, 150, 0.14)",
    tree: "#123326",
    treeLight: "#1f5540",
    stall: "rgba(214, 224, 240, 0.13)",
    marking: "rgba(214, 224, 240, 0.2)",
    crosswalk: "rgba(214, 224, 240, 0.17)",
    lamp: "255, 196, 120",
    edge: "rgba(140, 186, 255, 0.14)",
    target: "#7cc4ff",
    cone: "124, 196, 255",
    camera: "#f5b53d",
    text: "#dfe6f3",
  };

  // Фасады: левая грань в тени, правая освещена, крыша с градиентом.
  const PALETTES = [
    { left: "#0b111c", right: "#141e30", top: ["#1c2740", "#121a2a"] },
    { left: "#0b1316", right: "#142229", top: ["#1b2c34", "#111c22"] },
    { left: "#111119", right: "#1b1c29", top: ["#252640", "#17182a"] },
    { left: "#0e131c", right: "#18202d", top: ["#212b3d", "#151c29"] },
  ];
  const ROOF = { left: "#161d2a", right: "#222c3d", top: ["#2b3547", "#202938"] };

  // Неподвижный город рисуется в двух темах, и каждый кадр смешивает их по
  // времени суток: днём светлые фасады с отражениями в стёклах и тени от
  // домов, ночью горящие окна, фонари и свет фар.
  const THEMES = {
    night: {
      night: true, asphalt: C.asphalt, sidewalk: C.sidewalk, yard: C.yard, park: C.park, path: C.path,
      tree: C.tree, treeLight: C.treeLight, stall: C.stall, marking: C.marking,
      crosswalk: C.crosswalk, edge: C.edge, palettes: PALETTES, roof: ROOF,
      haze: "rgba(7, 11, 19, 0.62)", vignette: "rgba(5, 8, 14, 0.6)",
    },
    day: {
      night: false, asphalt: "#384150", sidewalk: "#5f697c", yard: "#46535c", park: "#3a6547",
      path: "rgba(225, 214, 180, 0.32)", tree: "#2e6440", treeLight: "#5a9c67",
      stall: "rgba(255, 255, 255, 0.3)", marking: "rgba(255, 255, 255, 0.45)",
      crosswalk: "rgba(255, 255, 255, 0.5)", edge: "rgba(255, 255, 255, 0.16)",
      palettes: [
        { left: "#515c6f", right: "#7f8ba0", top: ["#aab4c5", "#8f9bb0"] },
        { left: "#4c5f64", right: "#778f94", top: ["#a1b6b9", "#88a0a3"] },
        { left: "#5b5966", right: "#85828f", top: ["#a9a6b3", "#908d9b"] },
        { left: "#5f5a51", right: "#8e877b", top: ["#b3ab9e", "#9b9386"] },
      ],
      roof: { left: "#646d7e", right: "#919bad", top: ["#b8c0ce", "#a0a9ba"] },
      haze: "rgba(178, 198, 224, 0.34)", vignette: "rgba(18, 26, 38, 0.38)",
    },
  };

  // Сутки на схеме: день, закат, ночь, рассвет — 48 секунд на круг.
  const DAY = { length: 48, start: 6, keys: [[0, 9], [18, 18], [24, 21], [42, 29], [48, 33]] };
  const smoothstep = (x) => x * x * (3 - 2 * x);

  // Машины разных цветов, как на настоящей улице; изредка жёлтое такси.
  const CAR_COLOURS = ["#e9edf3", "#c7cfdb", "#8d97a9", "#454f62", "#262c3b",
                       "#3f6aa3", "#a8413c", "#e0b13f", "#4c8a67", "#d9dde4"];

  const inset = ([u0, v0, u1, v1], d) => [u0 + d, v0 + d, u1 - d, v1 - d];
  const centroid = (points) => [points.reduce((a, p) => a + p[0], 0) / points.length,
                                points.reduce((a, p) => a + p[1], 0) / points.length];

  // Карточка найденного снимка у камеры: миниатюра и подпись.
  const CARD = { w: 86, h: 66 };

  // Детерминированный генератор: город одинаковый при каждой загрузке.
  function random(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  class CityScene {
    constructor(canvas, caption) {
      this.canvas = canvas;
      this.caption = caption;
      this.context = canvas.getContext("2d");
      this.reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      this.running = false;
      this.visible = true;
      this.enabled = false;
      this.time = 0;
      this.scan = 0;
      this.scanning = false;
      this.locked = false;
      this.query = false;
      this.results = [];

      this.frame = this.frame.bind(this);
      this.hitboxes = [];
      this.highlight = null;
      const hit = (event) => {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left, y = event.clientY - rect.top;
        return this.hitboxes.find((b) => x >= b.left && x <= b.right && y >= b.top && y <= b.bottom);
      };
      canvas.addEventListener("click", (event) => {
        const box = hit(event);
        if (box && this.onPick) this.onPick(box.index);
      });
      canvas.addEventListener("mousemove", (event) => {
        canvas.style.cursor = hit(event) ? "pointer" : "";
      });
      // Перестройка города дорогая (дома с окнами в двух темах), поэтому при
      // перетаскивании края окна она откладывается до конца изменения.
      new ResizeObserver(() => {
        clearTimeout(this.resizeTimer);
        this.resizeTimer = setTimeout(() => this.resize(), this.layers ? 100 : 0);
      }).observe(canvas);
      new IntersectionObserver((entries) => {
        this.visible = entries[0].isIntersecting;
        this.visible ? this.resume() : this.pause();
      }).observe(canvas);
      document.addEventListener("visibilitychange", () => (document.hidden ? this.pause() : this.resume()));
    }

    /* ---------- Управление ---------- */

    start() { this.enabled = true; this.resize(); this.resume(); }
    stop() { this.enabled = false; this.pause(); }
    pause() { this.running = false; }

    resume() {
      if (!this.enabled || !this.visible || document.hidden || this.running || !this.width) return;
      if (this.reducedMotion) { this.draw(); return; }
      this.running = true;
      this.last = performance.now();
      requestAnimationFrame(this.frame);
    }

    /* Панели поверх карты, в пикселях холста. Снимки кандидатов раскладываются
     * только по камерам, где карточка снимка не уйдёт под панель. */
    setObstacles(rects) {
      this.obstacles = rects;
      const left = Math.max(0, ...rects.filter((r) => r.left < this.width / 2).map((r) => r.right));
      const right = Math.min(this.width, ...rects.filter((r) => r.left >= this.width / 2).map((r) => r.left));
      this.safeLeft = left;
      this.safeRight = right;
    }

    /* Где поместится карточка снимка у камеры: над ней, под ней или нигде.
     * Нельзя закрывать панели, края холста и подпись внизу по центру. */
    placement(u, v) {
      const [x, y] = this.project(u, v);
      const ticker = { left: this.width / 2 - 330, right: this.width / 2 + 330, top: this.height - 44 };
      const fits = (top) => {
        const card = { left: x - CARD.w / 2, right: x + CARD.w / 2, top, bottom: top + CARD.h };
        if (card.left < 6 || card.right > this.width - 6 || card.top < 6 || card.bottom > this.height - 6) {
          return false;
        }
        if (card.bottom > ticker.top && card.right > ticker.left && card.left < ticker.right) return false;
        return !(this.obstacles || []).some((r) =>
          card.left < r.right && card.right > r.left && card.top < r.bottom && card.bottom > r.top);
      };
      if (fits(y - CARD.h - 16)) return "above";
      if (fits(y + 12)) return "below";
      return null;
    }

    inView(u, v) { return this.placement(u, v) !== null; }

    /* Подсветка снимка кандидата при наведении на него в списке. */
    setHighlight(index) {
      this.highlight = index;
      if (this.reducedMotion) this.draw();
    }

    /* Цель переносится на перекрёсток в центре видимой части карты. */
    centreTarget() {
      if (!this.xs) return;   // город ещё не построен: холст нулевого размера
      const want = [((this.safeLeft ?? 0) + (this.safeRight ?? this.width)) / 2, this.height * 0.58];
      let best = null;
      let bestDistance = Infinity;
      for (let i = 1; i < this.xs.length - 1; i++) {
        for (let j = 1; j < this.ys.length - 1; j++) {
          const [x, y] = this.project(this.xs[i], this.ys[j]);
          const distance = Math.hypot(x - want[0], y - want[1]);
          if (distance < bestDistance) { bestDistance = distance; best = [i, j]; }
        }
      }
      if (!best) return;
      const car = this.target;
      car.ti = best[0]; car.tj = best[1]; car.t = 1;
      car.u = this.xs[best[0]]; car.v = this.ys[best[1]];
      this.pickNext(car, this.rand);
      this.trail = [];
      this.route = [];
    }

    /* Кадр загружен: цель на схеме — «машина с кадра». */
    setQuery(active) {
      if (active && !this.query) this.centreTarget();
      this.query = active;
      this.results = [];
      this.locked = active;
      this.setCaption(active
        ? "Цель — машина с загруженного кадра. Обведите её и нажмите «Найти»"
        : "Камеры города узнают одну и ту же машину по внешнему виду");
    }

    /* Во время поиска все камеры «опрашиваются»: от цели бежит волна. */
    setScanning(on) {
      this.scanning = on;
      if (on) {
        this.results = [];
        this.locked = true;
        this.setCaption("Ищем эту машину по всем снимкам галереи…");
      }
    }

    /* Найденные снимки раскладываются по камерам, ближайшим к цели. */
    showResults(candidates, threshold, caption) {
      // Холст ещё нулевого размера (скрытая вкладка): покажем после постройки.
      if (!this.cameras) { this.pending = [candidates, threshold, caption]; return; }
      this.pending = null;
      const target = this.target;
      const byDistance = this.cameras.slice().sort((a, b) =>
        Math.hypot(a.u - target.u, a.v - target.v) - Math.hypot(b.u - target.u, b.v - target.v));
      // Лучше показать меньше снимков, чем спрятать их под панелями: полный
      // список кандидатов всё равно в панели результата.
      const onScreen = byDistance.filter((c) => this.inView(c.u, c.v));
      const cameras = onScreen.length ? onScreen : byDistance;
      this.results = candidates.slice(0, Math.min(6, cameras.length)).map((candidate, index) => {
        const image = new Image();
        if (candidate.thumbnail) image.src = candidate.thumbnail;
        const camera = cameras[index % cameras.length];
        return {
          camera,
          index,
          placement: this.placement(camera.u, camera.v) || "above",
          image,
          label: (candidate.vehicle_id || "без ID") + " · " + candidate.score.toFixed(2),
          over: threshold !== null && candidate.score >= threshold,
          born: this.time + index * 0.18,
        };
      });
      this.results.forEach((r) => { r.camera.flash = 1; });
      this.locked = true;
      this.setCaption(caption);
      if (this.reducedMotion) this.draw();
    }

    setCaption(text) { if (this.caption) this.caption.textContent = text; }

    /* ---------- Город ---------- */

    resize() {
      const rect = this.canvas.getBoundingClientRect();
      if (rect.width < 1 || rect.height < 1) return;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      this.ratio = ratio;
      this.width = rect.width;
      this.height = rect.height;
      this.canvas.width = Math.round(rect.width * ratio);
      this.canvas.height = Math.round(rect.height * ratio);
      this.context.setTransform(ratio, 0, 0, ratio, 0, 0);

      // Центр схемы — между панелями запроса и результата.
      const narrow = rect.width < 860;
      this.cx = narrow ? rect.width * 0.5 : rect.width * 0.5 - 20;
      this.cy = rect.height * 0.5;
      this.zoom = narrow ? 0.7 : Math.max(0.85, Math.min(1.3, rect.width / 1350));
      this.angle = -0.62;           // поворот карты: улицы идут по диагонали
      this.squash = 0.56;           // сжатие по вертикали: вид сверху под углом
      this.build();
      this.renderStatic();
      if (this.query) this.centreTarget();
      if (this.pending) this.showResults(...this.pending);
      if (!this.running) this.draw();
    }

    project(u, v) {
      const cos = Math.cos(this.angle);
      const sin = Math.sin(this.angle);
      const x = (u * cos - v * sin) * this.zoom;
      const y = (u * sin + v * cos) * this.zoom * this.squash;
      return [this.cx + x, this.cy + y];
    }

    build() {
      const rand = random(20260918);
      const reach = Math.hypot(this.width, this.height / this.squash) / this.zoom * 0.62;
      const lines = (start) => {
        const out = [start];
        while (out[out.length - 1] < reach) out.push(out[out.length - 1] + 110 + rand() * 70);
        return out;
      };
      const half = lines(0);
      this.xs = [...half.slice(1).map((v) => -v).reverse(), ...half];
      this.ys = this.xs.map((v) => v * 0.92);
      this.road = 38;

      // Кварталы: парки, парковки, остальные застроены домами разной высоты.
      this.buildings = [];
      this.parks = [];
      this.parking = [];
      this.lots = [];
      for (let i = 0; i < this.xs.length - 1; i++) {
        for (let j = 0; j < this.ys.length - 1; j++) {
          const u0 = this.xs[i] + this.road / 2;
          const u1 = this.xs[i + 1] - this.road / 2;
          const v0 = this.ys[j] + this.road / 2;
          const v1 = this.ys[j + 1] - this.road / 2;
          const kind = rand();
          if (kind < 0.12) { this.parks.push(this.makePark([u0, v0, u1, v1], rand)); continue; }
          if (kind < 0.2) { this.parking.push(this.makeParking([u0, v0, u1, v1], rand)); continue; }
          this.lots.push([u0, v0, u1, v1]);
          const cols = 1 + Math.floor(rand() * 3);
          const rows = 1 + Math.floor(rand() * 2);
          const gap = 7;
          for (let a = 0; a < cols; a++) {
            for (let b = 0; b < rows; b++) {
              const bu0 = u0 + (u1 - u0) * a / cols + (a ? gap / 2 : 0);
              const bu1 = u0 + (u1 - u0) * (a + 1) / cols - (a < cols - 1 ? gap / 2 : 0);
              const bv0 = v0 + (v1 - v0) * b / rows + (b ? gap / 2 : 0);
              const bv1 = v0 + (v1 - v0) * (b + 1) / rows - (b < rows - 1 ? gap / 2 : 0);
              const centre = Math.hypot((bu0 + bu1) / 2, (bv0 + bv1) / 2);
              const height = (3 + rand() * 7 + Math.max(0, 12 - centre / 45) * rand()) * 0.8;
              this.buildings.push(this.makeBuilding([bu0, bv0, bu1, bv1], height, rand));
            }
          }
        }
      }
      // Порядок отрисовки «от дальнего к ближнему»: ближние дома перекрывают дальние.
      this.buildings.forEach((b) => {
        b.depth = this.project((b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2)[1];
      });
      this.buildings.sort((a, b) => a.depth - b.depth);

      // Камеры на перекрёстках ближе к центру, каждая смотрит вдоль улицы.
      const nodes = [];
      this.xs.forEach((u, i) => this.ys.forEach((v, j) => nodes.push({ i, j, u, v })));
      const central = nodes.filter((n) => Math.hypot(n.u, n.v) < reach * 0.55);
      const picked = central.sort(() => rand() - 0.5).slice(0, 16);
      this.cameras = picked.map((n, index) => {
        const heading = [0, Math.PI / 2, Math.PI, -Math.PI / 2][Math.floor(rand() * 4)];
        return { id: index + 1, u: n.u + 13, v: n.v - 13, heading,
                 sweep: rand() * Math.PI * 2, flash: 0 };
      });

      // Фонари вдоль улиц, на тротуарах по обе стороны.
      this.lamps = [];
      const side = this.road / 2 + 1.5;
      for (let i = 0; i < this.xs.length; i++) {
        for (let v = this.ys[0]; v < this.ys[this.ys.length - 1]; v += 64) {
          this.lamps.push({ u: this.xs[i] + (i % 2 ? side : -side), v: v + 30 });
        }
      }
      for (let j = 0; j < this.ys.length; j++) {
        for (let u = this.xs[0]; u < this.xs[this.xs.length - 1]; u += 64) {
          this.lamps.push({ u: u + 30, v: this.ys[j] + (j % 2 ? side : -side) });
        }
      }

      // Машины ездят по сетке улиц; у каждой своя скорость.
      this.cars = Array.from({ length: 58 }, (_, k) => this.newCar(rand, k === 0));
      this.target = this.cars[0];
      this.trail = [];
      this.links = [];
      this.marks = [];
      this.route = [];
      this.seen = new Map();
      this.rand = rand;
    }

    makeBuilding(box, height, rand) {
      const roof = [];
      const [u0, v0, u1, v1] = box;
      for (let k = Math.floor(rand() * 3); k > 0; k--) {
        const size = 3 + rand() * 4;
        const u = u0 + 3 + rand() * Math.max(0, u1 - u0 - size - 6);
        const v = v0 + 3 + rand() * Math.max(0, v1 - v0 - size - 6);
        roof.push({ box: [u, v, u + size, v + size * (0.6 + rand() * 0.6)], height: 0.8 + rand() * 1.4 });
      }
      return {
        box, height, roof,
        palette: Math.floor(rand() * PALETTES.length),
        windows: { seed: Math.floor(rand() * 1e9), lit: 0.16 + rand() * 0.34 },
        antenna: height > 10 ? 1 + rand() * 6 : 0,
      };
    }

    makePark(box, rand) {
      const [u0, v0, u1, v1] = inset(box, 8);
      const trees = [];
      const count = Math.max(4, Math.floor((u1 - u0) * (v1 - v0) / 520));
      for (let k = 0; k < count; k++) {
        trees.push({ u: u0 + rand() * (u1 - u0), v: v0 + rand() * (v1 - v0), r: 4 + rand() * 4 });
      }
      trees.sort((a, b) => (a.u + a.v) - (b.u + b.v));
      return { box, trees };
    }

    makeParking(box, rand) {
      const [u0, v0, u1, v1] = inset(box, 8);
      const slots = [];
      const depth = 11, width = 6.5;
      for (let v = v0; v + depth <= v1; v += depth + 9) {
        for (let u = u0; u + width <= u1; u += width) {
          const taken = rand() < 0.62;
          slots.push({ box: [u, v, u + width, v + depth],
                       colour: taken ? CAR_COLOURS[Math.floor(rand() * CAR_COLOURS.length)] : null });
        }
      }
      return { box, slots };
    }

    newCar(rand, isTarget) {
      const i = 1 + Math.floor(rand() * (this.xs.length - 2));
      const j = 1 + Math.floor(rand() * (this.ys.length - 2));
      const car = { i, j, ti: i, tj: j, t: 1, speed: isTarget ? 62 : 38 + rand() * 46,
                    target: isTarget, u: this.xs[i], v: this.ys[j], dir: 0,
                    colour: CAR_COLOURS[Math.floor(rand() * CAR_COLOURS.length)] };
      this.pickNext(car, rand);
      return car;
    }

    pickNext(car, rand) {
      const options = [[1, 0], [-1, 0], [0, 1], [0, -1]].filter(([di, dj]) => {
        const ni = car.ti + di;
        const nj = car.tj + dj;
        const back = ni === car.i && nj === car.j && car.t >= 1 && (car.i !== car.ti || car.j !== car.tj);
        return ni >= 0 && nj >= 0 && ni < this.xs.length && nj < this.ys.length && !back;
      });
      // Цель держится центра карты, чтобы её маршрут был виден.
      let choice = options[Math.floor(rand() * options.length)];
      if (car.target) {
        options.sort((a, b) => Math.hypot(this.xs[car.ti + a[0]], this.ys[car.tj + a[1]])
                             - Math.hypot(this.xs[car.ti + b[0]], this.ys[car.tj + b[1]]));
        choice = options[Math.floor(rand() * Math.min(2, options.length))];
      }
      car.i = car.ti; car.j = car.tj;
      car.ti = car.i + choice[0]; car.tj = car.j + choice[1];
      car.t = 0;
    }

    /* ---------- Кадр ---------- */

    frame(now) {
      if (!this.running) return;
      const dt = Math.min(0.05, (now - this.last) / 1000);
      this.last = now;
      this.time += dt;
      this.update(dt);
      this.draw();
      requestAnimationFrame(this.frame);
    }

    update(dt) {
      this.scan += ((this.scanning ? 1 : 0) - this.scan) * (1 - Math.exp(-dt * 6));

      for (const car of this.cars) {
        const u0 = this.xs[car.i], v0 = this.ys[car.j];
        const u1 = this.xs[car.ti], v1 = this.ys[car.tj];
        const length = Math.hypot(u1 - u0, v1 - v0) || 1;
        car.t += car.speed * dt / length;
        if (car.t >= 1) { this.pickNext(car, this.rand); continue; }
        const du = (u1 - u0) / length, dv = (v1 - v0) / length;
        // Правостороннее движение: машина смещена к правому краю улицы.
        car.u = u0 + (u1 - u0) * car.t - dv * 7;
        car.v = v0 + (v1 - v0) * car.t + du * 7;
        car.dir = Math.atan2(dv, du);
      }

      if (this.target) {
        this.trail.push([this.target.u, this.target.v]);
        if (this.trail.length > 220) this.trail.shift();
      }

      for (const camera of this.cameras) {
        camera.flash = Math.max(0, camera.flash - dt * 1.6);
        const look = camera.heading + Math.sin(this.time * 0.5 + camera.sweep) * 0.35;
        camera.look = look;
        for (const car of this.cars) {
          const du = car.u - camera.u, dv = car.v - camera.v;
          const distance = Math.hypot(du, dv);
          if (distance > 120) continue;
          let off = Math.atan2(dv, du) - look;
          off = Math.atan2(Math.sin(off), Math.cos(off));
          if (Math.abs(off) > 0.5) continue;
          const key = `${camera.id}:${this.cars.indexOf(car)}`;
          if ((this.seen.get(key) || -99) > this.time - 4) continue;
          this.seen.set(key, this.time);
          this.detect(camera, car);
        }
      }
      this.marks = this.marks.filter((m) => m.until > this.time);
      this.links = this.links.filter((l) => l.until > this.time);
    }

    detect(camera, car) {
      camera.flash = 1;
      if (!car.target) {
        this.marks.push({ car, until: this.time + 0.9, target: false });
        return;
      }
      this.marks.push({ car, until: this.time + 2.6, target: true });
      const previous = this.route[this.route.length - 1];
      if (previous && previous.id !== camera.id) {
        this.links.push({ from: previous, to: camera, until: this.time + 9 });
      }
      this.route.push(camera);
      if (this.route.length > 4) this.route.shift();
      if (!this.locked) {
        const path = this.route.map((c) => c.id).join(" → ");
        this.setCaption(`Та же машина на камере ${camera.id} · сходство ${score.toFixed(2)} · маршрут: ${path}`);
      }
    }

    /* ---------- Отрисовка ----------
     *
     * Неподвижный город (улицы, дворы, парки, дома с окнами, фонари) рисуется
     * один раз на два внеэкранных слоя: «земля» под машинами и «застройка» над
     * ними. Дом, выросший вверх от своего основания, закрывает только то, что
     * на экране выше основания, то есть позади него, поэтому один слой
     * застройки поверх машин даёт верное перекрытие. В каждом кадре рисуются
     * только машины, конусы камер, сами камеры и результаты поиска:
     * детализация города не стоит ничего.
     */

    layer(paint) {
      const canvas = document.createElement("canvas");
      canvas.width = this.canvas.width;
      canvas.height = this.canvas.height;
      const g = canvas.getContext("2d");
      g.setTransform(this.ratio, 0, 0, this.ratio, 0, 0);
      paint(g);
      return canvas;
    }

    renderStatic() {
      this.layers = {};
      for (const [name, theme] of Object.entries(THEMES)) {
        this.layers[name] = {
          ground: this.layer((g) => this.paintGround(g, theme)),
          city: this.layer((g) => { this.paintBuildings(g, theme); this.paintAtmosphere(g, theme); }),
        };
      }
    }

    /* Доля ночи: 0 — день, 1 — ночь, между ними закат и рассвет. */
    nightness() {
      if (this.reducedMotion) return 0;
      const t = (this.time + DAY.start) % DAY.length;
      if (t < 18) return 0;
      if (t < 24) return smoothstep((t - 18) / 6);
      if (t < 42) return 1;
      return 1 - smoothstep((t - 42) / 6);
    }

    /* Сколько сейчас «часов» в городе — дробное число для стрелок. */
    hours() {
      const t = (this.time + DAY.start) % DAY.length;
      const k = DAY.keys.findIndex(([at], i) => t < DAY.keys[i + 1]?.[0]);
      const [t0, h0] = DAY.keys[k], [t1, h1] = DAY.keys[k + 1];
      return (h0 + (h1 - h0) * (t - t0) / (t1 - t0)) % 24;
    }

    /* Слой дня под слоем ночи, прозрачность ночного — по времени суток. */
    blend(g, key, night) {
      g.drawImage(this.layers.day[key], 0, 0, this.width, this.height);
      if (night > 0.001) {
        g.globalAlpha = night;
        g.drawImage(this.layers.night[key], 0, 0, this.width, this.height);
        g.globalAlpha = 1;
      }
    }

    draw() {
      const g = this.context;
      if (!this.width || !this.layers) return;
      const night = this.nightness();
      this.night = night;
      g.clearRect(0, 0, this.width, this.height);
      this.blend(g, "ground", night);
      this.drawCones(g);
      this.drawTrail(g);
      this.drawCars(g);
      this.blend(g, "city", night);
      // Закат и рассвет: тёплый свет поверх города.
      const warm = 4 * night * (1 - night);
      if (warm > 0.01) {
        g.globalCompositeOperation = "soft-light";
        g.fillStyle = `rgba(255, 138, 64, ${0.38 * warm})`;
        g.fillRect(0, 0, this.width, this.height);
        g.globalCompositeOperation = "source-over";
      }
      this.drawClock(g, night);
      this.drawBeacons(g);
      this.drawCameras(g);
      this.drawLinks(g);
      this.drawMarks(g);
      this.drawTarget(g);
      this.drawResults(g);
    }

    polygon(g, points) {
      g.beginPath();
      points.forEach(([x, y], k) => (k ? g.lineTo(x, y) : g.moveTo(x, y)));
      g.closePath();
    }

    boxPoints([u0, v0, u1, v1], lift = 0) {
      return [[u0, v0], [u1, v0], [u1, v1], [u0, v1]].map(([u, v]) => {
        const [x, y] = this.project(u, v);
        return [x, y - lift];
      });
    }

    ellipseAt(g, u, v, radius, lift = 0) {
      const [x, y] = this.project(u, v);
      g.beginPath();
      g.ellipse(x, y - lift, radius * this.zoom, radius * this.zoom * this.squash, 0, 0, Math.PI * 2);
    }

    /* ---------- Земля: улицы, тротуары, дворы, парки, парковки, фонари ---------- */

    paintGround(g, T) {
      g.fillStyle = T.asphalt;
      g.fillRect(0, 0, this.width, this.height);

      // Квартал: тротуар по краю, внутри двор.
      for (const lot of [...this.lots, ...this.parks.map((p) => p.box), ...this.parking.map((p) => p.box)]) {
        g.fillStyle = T.sidewalk;
        this.polygon(g, this.boxPoints(lot));
        g.fill();
        g.fillStyle = T.yard;
        this.polygon(g, this.boxPoints(inset(lot, 4)));
        g.fill();
      }

      // Осевая разметка между перекрёстками и «зебры» у каждого перекрёстка.
      const half = this.road / 2;
      g.strokeStyle = T.marking;
      g.lineWidth = 1;
      g.setLineDash([6, 8]);
      g.beginPath();
      for (let i = 0; i < this.xs.length; i++) {
        for (let j = 0; j < this.ys.length - 1; j++) {
          g.moveTo(...this.project(this.xs[i], this.ys[j] + half + 7));
          g.lineTo(...this.project(this.xs[i], this.ys[j + 1] - half - 7));
        }
      }
      for (let j = 0; j < this.ys.length; j++) {
        for (let i = 0; i < this.xs.length - 1; i++) {
          g.moveTo(...this.project(this.xs[i] + half + 7, this.ys[j]));
          g.lineTo(...this.project(this.xs[i + 1] - half - 7, this.ys[j]));
        }
      }
      g.stroke();
      g.setLineDash([]);

      g.fillStyle = T.crosswalk;
      for (const u of this.xs) {
        for (const v of this.ys) {
          for (let s = -half + 3; s < half - 3; s += 5) {
            // Полосы поперёк проезжей части с четырёх сторон перекрёстка.
            this.polygon(g, this.boxPoints([u + s, v - half - 6, u + s + 2.4, v - half - 1])); g.fill();
            this.polygon(g, this.boxPoints([u + s, v + half + 1, u + s + 2.4, v + half + 6])); g.fill();
            this.polygon(g, this.boxPoints([u - half - 6, v + s, u - half - 1, v + s + 2.4])); g.fill();
            this.polygon(g, this.boxPoints([u + half + 1, v + s, u + half + 6, v + s + 2.4])); g.fill();
          }
        }
      }

      // Парки: газон, дорожка по диагонали, деревья с тенью и бликом.
      for (const park of this.parks) {
        g.fillStyle = T.park;
        this.polygon(g, this.boxPoints(inset(park.box, 4)));
        g.fill();
        const [u0, v0, u1, v1] = inset(park.box, 8);
        g.strokeStyle = T.path;
        g.lineWidth = 2 * this.zoom;
        g.beginPath();
        g.moveTo(...this.project(u0, v0));
        g.lineTo(...this.project(u1, v1));
        g.stroke();
        for (const tree of park.trees) {
          g.fillStyle = "rgba(0, 0, 0, 0.35)";
          this.ellipseAt(g, tree.u + 2, tree.v + 2, tree.r);
          g.fill();
          g.fillStyle = T.tree;
          this.ellipseAt(g, tree.u, tree.v, tree.r, 3 * this.zoom);
          g.fill();
          g.fillStyle = T.treeLight;
          this.ellipseAt(g, tree.u - tree.r * 0.3, tree.v - tree.r * 0.3, tree.r * 0.5, 3.6 * this.zoom);
          g.fill();
        }
      }

      // Парковки: разметка мест и стоящие машины.
      for (const lot of this.parking) {
        g.strokeStyle = T.stall;
        g.lineWidth = 1;
        for (const slot of lot.slots) {
          this.polygon(g, this.boxPoints(slot.box));
          g.stroke();
          if (slot.colour) {
            g.fillStyle = slot.colour;
            this.polygon(g, this.boxPoints(inset(slot.box, 1.4)));
            g.fill();
            g.fillStyle = "rgba(8, 12, 20, 0.55)";
            const [a, b, c, d] = slot.box;
            this.polygon(g, this.boxPoints([a + 1.8, b + (d - b) * 0.3, c - 1.8, b + (d - b) * 0.62]));
            g.fill();
          }
        }
      }

      if (!T.night) { this.paintShadows(g); return; }

      // Фонари вдоль улиц: тёплые пятна света на асфальте.
      g.globalCompositeOperation = "lighter";
      for (const lamp of this.lamps) {
        const [x, y] = this.project(lamp.u, lamp.v);
        const radius = 26 * this.zoom;
        const glow = g.createRadialGradient(x, y, 0, x, y, radius);
        glow.addColorStop(0, `rgba(${C.lamp}, 0.16)`);
        glow.addColorStop(1, `rgba(${C.lamp}, 0)`);
        g.fillStyle = glow;
        g.beginPath();
        g.ellipse(x, y, radius, radius * this.squash, 0, 0, Math.PI * 2);
        g.fill();
      }
      g.globalCompositeOperation = "source-over";
      g.fillStyle = `rgba(${C.lamp}, 0.9)`;
      for (const lamp of this.lamps) {
        const [x, y] = this.project(lamp.u, lamp.v);
        g.fillRect(x - 0.8, y - 0.8, 1.6, 1.6);
      }
    }

    /* Днём дома отбрасывают тени: солнце справа сверху, тень влево вниз. Тени
     * рисуются сплошным цветом на отдельном холсте и накладываются одной
     * прозрачностью, чтобы перекрытия не темнели сильнее. */
    paintShadows(g) {
      const lift = this.zoom * 1.15;
      const shadows = this.layer((s) => {
        s.fillStyle = "#000";
        for (const b of this.buildings) {
          const height = b.height * lift;
          const base = this.boxPoints(b.box);
          const off = base.map(([x, y]) => [x - height * 0.9, y + height * 0.28]);
          for (let k = 0; k < 4; k++) {
            const n = (k + 1) % 4;
            this.polygon(s, [base[k], base[n], off[n], off[k]]);
            s.fill();
          }
          this.polygon(s, off);
          s.fill();
        }
      });
      g.globalAlpha = 0.26;
      g.drawImage(shadows, 0, 0, this.width, this.height);
      g.globalAlpha = 1;
    }

    /* ---------- Застройка: грани с окнами, крыши, техника, антенны ---------- */

    paintBuildings(g, T) {
      const lift = this.zoom * 1.15;
      if (T.night) this.beacons = [];
      for (const b of this.buildings) {
        const height = b.height * lift;
        this.paintBox(g, b.box, 0, height, T.palettes[b.palette], b.windows, T);

        // Парапет: край крыши чуть выше её поля.
        const top = this.boxPoints(b.box, height);
        const [cx, cy] = centroid(top);
        const inner = top.map(([x, y]) => [cx + (x - cx) * 0.84, cy + (y - cy) * 0.84]);
        g.fillStyle = "rgba(0, 0, 0, 0.22)";
        this.polygon(g, inner);
        g.fill();

        for (const unit of b.roof) {
          this.paintBox(g, unit.box, height, height + unit.height * lift, T.roof, null, T);
        }
        if (b.antenna) {
          g.strokeStyle = "rgba(160, 176, 200, 0.55)";
          g.lineWidth = 1;
          g.beginPath();
          g.moveTo(cx, cy);
          g.lineTo(cx, cy - 11 * this.zoom);
          g.stroke();
          if (T.night) this.beacons.push({ x: cx, y: cy - 11 * this.zoom, phase: b.antenna });
        }
      }
    }

    /* Параллелепипед: видимые боковые грани (левая темнее, правая светлее),
     * окна на гранях и крыша с градиентом. */
    paintBox(g, box, baseLift, topLift, palette, windows, T) {
      const base = this.boxPoints(box, baseLift);
      const top = this.boxPoints(box, topLift);
      const [cx, cy] = centroid(base);
      for (let k = 0; k < 4; k++) {
        const n = (k + 1) % 4;
        const mx = (base[k][0] + base[n][0]) / 2;
        const my = (base[k][1] + base[n][1]) / 2;
        if (my < cy - 0.01) continue;           // грань смотрит от зрителя
        const left = mx < cx;
        g.fillStyle = left ? palette.left : palette.right;
        this.polygon(g, [base[k], base[n], top[n], top[k]]);
        g.fill();
        if (windows) this.paintWindows(g, base[k], base[n], topLift - baseLift, windows, k, left, T);
      }
      const shade = g.createLinearGradient(top[0][0], top[0][1], top[2][0], top[2][1]);
      shade.addColorStop(0, palette.top[0]);
      shade.addColorStop(1, palette.top[1]);
      g.fillStyle = shade;
      this.polygon(g, top);
      g.fill();
      g.strokeStyle = T.edge;
      g.lineWidth = 0.8;
      g.stroke();
    }

    paintWindows(g, a, b, height, windows, face, left, T) {
      const width = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const cols = Math.floor(width / 5.2);
      const rows = Math.floor(height / 4.4);
      if (cols < 1 || rows < 1) return;
      const rand = random(windows.seed + face * 7919);
      const at = (s, t) => [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s - height * t];
      for (let c = 0; c < cols; c++) {
        for (let r = 0; r < rows; r++) {
          const s0 = (c + 0.24) / cols, s1 = (c + 0.76) / cols;
          const t0 = (r + 0.3) / rows, t1 = (r + 0.72) / rows;
          const roll = rand();
          if (!T.night) {
            // Днём окна — тёмное стекло, часть отражает небо.
            g.fillStyle = roll < 0.22 ? "rgba(214, 232, 250, 0.5)" : "rgba(26, 38, 58, 0.55)";
          } else if (roll < windows.lit) {
            const warm = rand() < 0.72;
            const alpha = (left ? 0.55 : 0.75) * (0.6 + rand() * 0.4);
            g.fillStyle = warm ? `rgba(255, 204, 128, ${alpha})` : `rgba(160, 210, 255, ${alpha * 0.85})`;
          } else {
            g.fillStyle = "rgba(120, 150, 200, 0.07)";
          }
          this.polygon(g, [at(s0, t0), at(s1, t0), at(s1, t1), at(s0, t1)]);
          g.fill();
        }
      }
    }

    /* Дымка вдали (верх схемы) и виньетка по краям: у города появляется глубина. */
    paintAtmosphere(g, T) {
      const haze = g.createLinearGradient(0, 0, 0, this.height * 0.5);
      haze.addColorStop(0, T.haze);
      haze.addColorStop(1, T.haze.replace(/[\d.]+\)$/, "0)"));
      g.fillStyle = haze;
      g.fillRect(0, 0, this.width, this.height);
      const r = Math.max(this.width, this.height) * 0.62;
      const vignette = g.createRadialGradient(this.width / 2, this.height * 0.6, r * 0.45,
                                              this.width / 2, this.height * 0.6, r);
      vignette.addColorStop(0, T.vignette.replace(/[\d.]+\)$/, "0)"));
      vignette.addColorStop(1, T.vignette);
      g.fillStyle = vignette;
      g.fillRect(0, 0, this.width, this.height);
    }

    /* ---------- Движущееся ---------- */

    /* Часы на схеме — аналоговые: стрелки идут вместе с суточным циклом,
     * ободок теплеет днём и синеет ночью. Цифр нет: подписи на схеме мешают. */
    drawClock(g, night) {
      const hours = this.hours();
      const accent = night > 0.5 ? "rgba(185, 215, 255, 0.85)" : "rgba(255, 205, 122, 0.9)";
      const r = 16;
      g.save();
      g.translate(this.width - r - 20, r + 16);

      g.fillStyle = "rgba(9, 13, 21, 0.66)";
      g.beginPath();
      g.arc(0, 0, r, 0, Math.PI * 2);
      g.fill();
      g.strokeStyle = accent;
      g.lineWidth = 1.3;
      g.stroke();

      for (let i = 0; i < 12; i++) {
        const a = (i / 12) * Math.PI * 2;
        const long = i % 3 === 0;
        g.strokeStyle = long ? "rgba(223, 230, 243, 0.8)" : "rgba(223, 230, 243, 0.3)";
        g.lineWidth = long ? 1.4 : 1;
        const inner = r - (long ? 5 : 3);
        g.beginPath();
        g.moveTo(Math.sin(a) * inner, -Math.cos(a) * inner);
        g.lineTo(Math.sin(a) * (r - 1.6), -Math.cos(a) * (r - 1.6));
        g.stroke();
      }

      const hand = (angle, length, width, colour) => {
        g.strokeStyle = colour;
        g.lineWidth = width;
        g.lineCap = "round";
        g.beginPath();
        g.moveTo(-Math.sin(angle) * 2.5, Math.cos(angle) * 2.5);
        g.lineTo(Math.sin(angle) * length, -Math.cos(angle) * length);
        g.stroke();
      };
      hand(((hours % 12) / 12) * Math.PI * 2, r * 0.52, 2.6, "#e7ecf5");
      hand(((hours % 1) * 60 / 60) * Math.PI * 2, r * 0.8, 1.5, "#c3cbdb");
      g.fillStyle = accent;
      g.beginPath();
      g.arc(0, 0, 2, 0, Math.PI * 2);
      g.fill();
      g.restore();
    }

    drawBeacons(g) {
      if (this.night < 0.3) return;
      for (const beacon of this.beacons || []) {
        const on = Math.sin(this.time * 2.4 + beacon.phase) > 0.55;
        if (!on) continue;
        const glow = g.createRadialGradient(beacon.x, beacon.y, 0, beacon.x, beacon.y, 3.2);
        glow.addColorStop(0, "rgba(255, 90, 80, 0.85)");
        glow.addColorStop(1, "rgba(255, 90, 80, 0)");
        g.fillStyle = glow;
        g.fillRect(beacon.x - 3.2, beacon.y - 3.2, 6.4, 6.4);
      }
    }

    drawCones(g) {
      for (const camera of this.cameras) {
        const [x, y] = this.project(camera.u, camera.v);
        const reach = 120;
        const look = camera.look ?? camera.heading;
        const points = [[x, y]];
        for (let k = -4; k <= 4; k++) {
          const a = look + k * 0.125;
          points.push(this.project(camera.u + Math.cos(a) * reach, camera.v + Math.sin(a) * reach));
        }
        const away = Math.hypot(camera.u - this.target.u, camera.v - this.target.v);
        const wave = this.scan * Math.max(0, Math.sin(this.time * 6 - away / 60));
        const alpha = (0.08 + camera.flash * 0.16 + wave * 0.18) * (0.65 + 0.35 * this.night);
        const fill = g.createRadialGradient(x, y, 0, x, y, reach * this.zoom);
        fill.addColorStop(0, `rgba(${C.cone}, ${alpha * 2.2})`);
        fill.addColorStop(1, `rgba(${C.cone}, 0)`);
        g.fillStyle = fill;
        this.polygon(g, points);
        g.fill();
        // Линия развёртки — камера «ведёт» взглядом вдоль улицы.
        const [ex, ey] = this.project(camera.u + Math.cos(look) * reach * 0.9,
                                      camera.v + Math.sin(look) * reach * 0.9);
        const line = g.createLinearGradient(x, y, ex, ey);
        line.addColorStop(0, `rgba(${C.cone}, ${0.35 + camera.flash * 0.4})`);
        line.addColorStop(1, `rgba(${C.cone}, 0)`);
        g.strokeStyle = line;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(x, y);
        g.lineTo(ex, ey);
        g.stroke();
      }
    }

    drawTrail(g) {
      if (this.trail.length < 2) return;
      g.lineWidth = 2.2;
      g.lineCap = "round";
      for (let k = 1; k < this.trail.length; k++) {
        g.strokeStyle = `rgba(124, 196, 255, ${(k / this.trail.length) * 0.55})`;
        g.beginPath();
        g.moveTo(...this.project(...this.trail[k - 1]));
        g.lineTo(...this.project(...this.trail[k]));
        g.stroke();
      }
    }

    /* Машина сверху: тень, кузов, лобовое и заднее стёкла, крыша, фары со
     * световым конусом и стоп-сигналы. */
    drawCars(g) {
      const busy = this.query || this.results.length;
      const width = 5.8 * Math.max(0.85, this.zoom);
      for (const car of this.cars) {
        const [x, y] = this.project(car.u, car.v);
        const [hx, hy] = this.project(car.u + Math.cos(car.dir) * 6, car.v + Math.sin(car.dir) * 6);
        const angle = Math.atan2(hy - y, hx - x);
        const length = Math.hypot(hx - x, hy - y) * 1.6 + 8;
        const dim = busy && !car.target;
        g.save();
        g.translate(x, y);
        g.rotate(angle);

        g.globalCompositeOperation = "lighter";
        const beam = g.createLinearGradient(length / 2, 0, length / 2 + 24, 0);
        beam.addColorStop(0, `rgba(255, 232, 176, ${(dim ? 0.07 : 0.2) * this.night})`);
        beam.addColorStop(1, "rgba(255, 232, 176, 0)");
        g.fillStyle = beam;
        g.beginPath();
        g.moveTo(length / 2, -width * 0.32);
        g.lineTo(length / 2 + 24, -width * 1.25);
        g.lineTo(length / 2 + 24, width * 1.25);
        g.lineTo(length / 2, width * 0.32);
        g.closePath();
        g.fill();
        g.globalCompositeOperation = "source-over";

        g.globalAlpha = dim ? 0.42 : 1;
        g.fillStyle = "rgba(0, 0, 0, 0.45)";
        g.beginPath();
        g.roundRect(-length / 2 + 1, -width / 2 + 1.6, length, width, 2.4);
        g.fill();
        if (car.target) { g.shadowColor = C.target; g.shadowBlur = 14; }
        g.fillStyle = car.target ? C.target : car.colour;
        g.beginPath();
        g.roundRect(-length / 2, -width / 2, length, width, 2.4);
        g.fill();
        g.shadowBlur = 0;
        g.fillStyle = "rgba(9, 14, 24, 0.82)";              // стёкла
        g.beginPath();
        g.roundRect(length * 0.06, -width * 0.38, length * 0.16, width * 0.76, 1);
        g.roundRect(-length * 0.33, -width * 0.36, length * 0.12, width * 0.72, 1);
        g.fill();
        g.fillStyle = "rgba(255, 255, 255, 0.14)";          // блик на крыше
        g.fillRect(-length * 0.2, -width * 0.3, length * 0.25, width * 0.24);
        g.fillStyle = "rgba(255, 246, 222, 0.95)";          // фары
        g.fillRect(length / 2 - 1.2, -width / 2 + 0.6, 1.2, 1.5);
        g.fillRect(length / 2 - 1.2, width / 2 - 2.1, 1.2, 1.5);
        g.fillStyle = "rgba(255, 64, 56, 0.95)";            // стоп-сигналы
        g.fillRect(-length / 2, -width / 2 + 0.6, 1.1, 1.5);
        g.fillRect(-length / 2, width / 2 - 2.1, 1.1, 1.5);
        g.restore();
      }
      g.globalAlpha = 1;
    }

    /* Камера: столб, корпус с объективом по направлению обзора и индикатор
     * записи. Сработав, камера вспыхивает объективом — без подписи: номер
     * камеры виден в карточке результата, на схеме он только мешает. */
    drawCameras(g) {
      const pole = 15 * Math.max(0.85, this.zoom);
      for (const camera of this.cameras) {
        const [x, y] = this.project(camera.u, camera.v);
        const topY = y - pole;
        const look = camera.look ?? camera.heading;
        const [lx, ly] = this.project(camera.u + Math.cos(look) * 8, camera.v + Math.sin(look) * 8);
        const angle = Math.atan2(ly - y, lx - x);
        const active = camera.flash > 0.05;

        g.fillStyle = "rgba(0, 0, 0, 0.4)";
        g.beginPath();
        g.ellipse(x + 1, y + 0.5, 3.4, 1.7, 0, 0, Math.PI * 2);
        g.fill();
        g.strokeStyle = "#5a6982";
        g.lineWidth = 1.6;
        g.beginPath();
        g.moveTo(x, y);
        g.lineTo(x, topY);
        g.stroke();

        g.save();
        g.translate(x, topY);
        g.rotate(angle);
        g.fillStyle = "#b9c3d2";                             // козырёк
        g.fillRect(-2.6, -3.4, 12, 1.5);
        g.fillStyle = "#e3e8f0";                             // корпус
        g.beginPath();
        g.roundRect(-2.2, -2.4, 10.4, 5, 1.6);
        g.fill();
        if (active) { g.shadowColor = C.target; g.shadowBlur = 12 * camera.flash; }
        g.fillStyle = active ? C.target : "#18202e";         // объектив
        g.beginPath();
        g.arc(8.4, 0.1, 2, 0, Math.PI * 2);
        g.fill();
        g.shadowBlur = 0;
        g.restore();

        // Индикатор записи: мигает красным, при срабатывании горит голубым.
        const blink = Math.sin(this.time * 3 + camera.sweep) > 0.3;
        g.fillStyle = active ? C.target : blink ? "#ff4d42" : "rgba(255, 77, 66, 0.25)";
        g.beginPath();
        g.arc(x - 2.6, topY + 3.2, 1.1, 0, Math.PI * 2);
        g.fill();

      }
    }

    drawLinks(g) {
      for (const link of this.links) {
        const life = Math.min(1, (link.until - this.time) / 3);
        const pole = 15 * Math.max(0.85, this.zoom);
        const [x1, y1] = this.project(link.from.u, link.from.v);
        const [x2, y2] = this.project(link.to.u, link.to.v);
        const mx = (x1 + x2) / 2, my = Math.min(y1, y2) - 44;
        g.strokeStyle = `rgba(124, 196, 255, ${0.8 * life})`;
        g.lineWidth = 1.5;
        g.setLineDash([5, 5]);
        g.lineDashOffset = -this.time * 18;                   // линия «бежит» от камеры к камере
        g.beginPath();
        g.moveTo(x1, y1 - pole);
        g.quadraticCurveTo(mx, my, x2, y2 - pole);
        g.stroke();
        g.setLineDash([]);
        g.lineDashOffset = 0;
      }
    }

    drawMarks(g) {
      for (const mark of this.marks) {
        const [x, y] = this.project(mark.car.u, mark.car.v);
        const size = mark.target ? 13 : 9;
        const fade = Math.min(1, (mark.until - this.time) * 3);
        g.strokeStyle = mark.target ? `rgba(124, 196, 255, ${fade})` : `rgba(223, 230, 243, ${0.55 * fade})`;
        g.lineWidth = mark.target ? 1.6 : 1;
        // Уголки рамки — как у разметки в системах видеонаблюдения.
        const c = size * 0.45;
        g.beginPath();
        for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
          const px = x + sx * size, py = y + sy * size * 0.8;
          g.moveTo(px - sx * c, py);
          g.lineTo(px, py);
          g.lineTo(px, py - sy * c);
        }
        g.stroke();
      }
    }

    drawTarget(g) {
      if (!this.query && !this.results.length) return;
      const [x, y] = this.project(this.target.u, this.target.v);
      // Во время поиска от цели расходятся волны, как у радара.
      if (this.scan > 0.02) {
        for (let k = 0; k < 3; k++) {
          const phase = ((this.time * 0.9 + k / 3) % 1);
          const r = phase * 150 * this.zoom;
          g.strokeStyle = `rgba(124, 196, 255, ${this.scan * (1 - phase) * 0.55})`;
          g.lineWidth = 1.4;
          g.beginPath();
          g.ellipse(x, y, r, r * this.squash, 0, 0, Math.PI * 2);
          g.stroke();
        }
      }
      const pulse = 14 + 5 * Math.sin(this.time * 4);
      g.strokeStyle = "rgba(124, 196, 255, 0.85)";
      g.lineWidth = 1.5;
      g.beginPath();
      g.ellipse(x, y, pulse, pulse * 0.62, 0, 0, Math.PI * 2);
      g.stroke();
      // Перекрестие вместо подписи: понятно без слов.
      g.strokeStyle = "rgba(124, 196, 255, 0.55)";
      g.lineWidth = 1;
      g.beginPath();
      for (const [dx, dy] of [[-1, 0], [1, 0]]) {
        g.moveTo(x + dx * (pulse + 4), y);
        g.lineTo(x + dx * (pulse + 11), y);
      }
      for (const [dx, dy] of [[0, -1], [0, 1]]) {
        g.moveTo(x, y + dy * (pulse * 0.62 + 4));
        g.lineTo(x, y + dy * (pulse * 0.62 + 9));
      }
      g.stroke();
    }

    /* Карточки найденных снимков у камер и маршрут между совпадениями. */
    drawResults(g) {
      if (!this.results.length) return;
      const shown = this.results.filter((r) => r.born <= this.time);
      const over = shown.filter((r) => r.over);

      g.setLineDash([6, 6]);
      g.lineWidth = 1.6;
      g.strokeStyle = "rgba(52, 201, 139, 0.8)";
      const [tx, ty] = this.project(this.target.u, this.target.v);
      g.beginPath();
      g.moveTo(tx, ty);
      over.forEach((r) => {
        const [x, y] = this.project(r.camera.u, r.camera.v);
        g.lineTo(x, y - 6);
      });
      g.stroke();
      g.setLineDash([]);

      g.font = "700 10px Manrope, 'Segoe UI', sans-serif";
      this.hitboxes = [];
      for (const r of shown) {
        const [x, y] = this.project(r.camera.u, r.camera.v);
        const grow = Math.min(1, (this.time - r.born) * 4);
        const left = x - CARD.w / 2;
        const top = r.placement === "below" ? y + 12 : y - CARD.h - 16;
        const lit = this.highlight === r.index;
        g.globalAlpha = grow;
        g.fillStyle = "rgba(10, 14, 22, 0.94)";
        g.strokeStyle = lit ? "#7cc4ff" : r.over ? "rgba(52, 201, 139, 0.9)" : "rgba(223, 230, 243, 0.35)";
        g.lineWidth = lit ? 2.4 : r.over ? 1.6 : 1;
        if (lit) { g.shadowColor = "#7cc4ff"; g.shadowBlur = 16; }
        g.beginPath();
        g.roundRect(left, top, CARD.w, CARD.h, 7);
        g.fill();
        g.stroke();
        g.shadowBlur = 0;
        if (r.image.complete && r.image.naturalWidth) {
          g.save();
          g.beginPath();
          g.roundRect(left + 3, top + 3, CARD.w - 6, CARD.h - 20, 5);
          g.clip();
          g.drawImage(r.image, left + 3, top + 3, CARD.w - 6, CARD.h - 20);
          g.restore();
        }
        g.fillStyle = r.over ? "#34c98b" : "#a3aab8";
        g.fillText(r.label, left + 5, top + CARD.h - 6, CARD.w - 10);
        g.beginPath();
        g.moveTo(x, r.placement === "below" ? top : top + CARD.h);
        g.lineTo(x, r.placement === "below" ? y + 2 : y - 8);
        g.stroke();
        g.globalAlpha = 1;
        this.hitboxes.push({ left, top, right: left + CARD.w, bottom: top + CARD.h, index: r.index });
      }
    }
  }

  window.CityScene = CityScene;
})();
