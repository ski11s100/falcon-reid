/* Живая схема города — фон пульта оператора.
 *
 * Показывает задачу без слов: карта города сверху, по улицам едут машины, на
 * перекрёстках стоят камеры с конусами обзора. Камера, заметившая машину,
 * обводит её рамкой. Одна машина — цель: её узнают на разных камерах по
 * внешнему виду, и между камерами тянется линия маршрута. Это и есть повторная
 * идентификация (ReID): та же машина на другой камере, без номера.
 *
 * Схема отвечает на действия оператора:
 *   setQuery(true)   — кадр загружен: цель подписана, остальное приглушено;
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

  // Ночной город: улицы светлее кварталов, чтобы читались как дороги.
  const C = {
    ground: "#161e2d",
    lot: "#0a0e16",
    park: "#0c1915",
    top: "#121a29",
    topLight: "#18223a",
    side: "#0b1019",
    edge: "rgba(124, 196, 255, 0.10)",
    dash: "rgba(255, 255, 255, 0.07)",
    car: "#c3cbdb",
    target: "#7cc4ff",
    cone: "124, 196, 255",
    camera: "#f5b53d",
    text: "#dfe6f3",
  };

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
      new ResizeObserver(() => this.resize()).observe(canvas);
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
      this.road = 32;

      // Кварталы: часть — парки, остальные застроены домами разной высоты.
      this.buildings = [];
      this.parks = [];
      this.lots = [];
      for (let i = 0; i < this.xs.length - 1; i++) {
        for (let j = 0; j < this.ys.length - 1; j++) {
          const u0 = this.xs[i] + this.road / 2;
          const u1 = this.xs[i + 1] - this.road / 2;
          const v0 = this.ys[j] + this.road / 2;
          const v1 = this.ys[j + 1] - this.road / 2;
          if (rand() < 0.12) { this.parks.push([u0, v0, u1, v1]); continue; }
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
              const height = 3 + rand() * 7 + Math.max(0, 12 - centre / 45) * rand();
              this.buildings.push({ box: [bu0, bv0, bu1, bv1], height });
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

      // Машины ездят по сетке улиц; у каждой своя скорость.
      this.cars = Array.from({ length: 46 }, (_, k) => this.newCar(rand, k === 0));
      this.target = this.cars[0];
      this.trail = [];
      this.links = [];
      this.marks = [];
      this.route = [];
      this.seen = new Map();
      this.rand = rand;
    }

    newCar(rand, isTarget) {
      const i = 1 + Math.floor(rand() * (this.xs.length - 2));
      const j = 1 + Math.floor(rand() * (this.ys.length - 2));
      const car = { i, j, ti: i, tj: j, t: 1, speed: isTarget ? 62 : 38 + rand() * 46,
                    target: isTarget, u: this.xs[i], v: this.ys[j], dir: 0 };
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
      const score = 0.82 + this.rand() * 0.14;
      this.marks.push({ car, until: this.time + 2.6, target: true, camera: camera.id, score });
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

    draw() {
      const g = this.context;
      if (!this.width) return;
      g.clearRect(0, 0, this.width, this.height);
      g.fillStyle = C.ground;
      g.fillRect(0, 0, this.width, this.height);

      this.drawGround(g);
      this.drawCones(g);
      this.drawTrail(g);
      this.drawCars(g);
      this.drawBuildings(g);
      this.drawCameras(g);
      this.drawLinks(g);
      this.drawMarks(g);
      this.drawTargetLabel(g);
      this.drawResults(g);
    }

    drawTargetLabel(g) {
      if (!this.query && !this.results.length) return;
      const [x, y] = this.project(this.target.u, this.target.v);
      const pulse = 14 + 6 * Math.sin(this.time * 4);
      g.strokeStyle = "rgba(124, 196, 255, 0.8)";
      g.lineWidth = 1.5;
      g.beginPath();
      g.ellipse(x, y, pulse, pulse * 0.62, 0, 0, Math.PI * 2);
      g.stroke();
      g.font = "700 11px Manrope, 'Segoe UI', sans-serif";
      g.fillStyle = "rgba(124, 196, 255, 1)";
      g.fillText("ЦЕЛЬ", x - 14, y - pulse - 6);
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

    drawGround(g) {
      // Разметка по осям улиц.
      g.strokeStyle = C.dash;
      g.lineWidth = 1;
      g.setLineDash([6, 10]);
      g.beginPath();
      const u0 = this.xs[0], u1 = this.xs[this.xs.length - 1];
      const v0 = this.ys[0], v1 = this.ys[this.ys.length - 1];
      for (const u of this.xs) { g.moveTo(...this.project(u, v0)); g.lineTo(...this.project(u, v1)); }
      for (const v of this.ys) { g.moveTo(...this.project(u0, v)); g.lineTo(...this.project(u1, v)); }
      g.stroke();
      g.setLineDash([]);
      g.fillStyle = C.lot;
      for (const lot of this.lots) { this.polygon(g, this.boxPoints(lot)); g.fill(); }
      g.fillStyle = C.park;
      for (const park of this.parks) { this.polygon(g, this.boxPoints(park)); g.fill(); }
    }

    drawBuildings(g) {
      const lift = this.zoom * 1.15;
      for (const b of this.buildings) {
        const base = this.boxPoints(b.box);
        const top = this.boxPoints(b.box, b.height * lift);
        g.fillStyle = C.side;
        for (let k = 0; k < 4; k++) {
          const n = (k + 1) % 4;
          this.polygon(g, [base[k], base[n], top[n], top[k]]);
          g.fill();
        }
        const shade = g.createLinearGradient(top[0][0], top[0][1], top[2][0], top[2][1]);
        shade.addColorStop(0, C.topLight);
        shade.addColorStop(1, C.top);
        g.fillStyle = shade;
        this.polygon(g, top);
        g.fill();
        g.strokeStyle = C.edge;
        g.lineWidth = 1;
        g.stroke();
      }
    }

    drawCones(g) {
      for (const camera of this.cameras) {
        const [x, y] = this.project(camera.u, camera.v);
        const reach = 120;
        const points = [[x, y]];
        for (let k = -4; k <= 4; k++) {
          const a = (camera.look ?? camera.heading) + k * 0.125;
          points.push(this.project(camera.u + Math.cos(a) * reach, camera.v + Math.sin(a) * reach));
        }
        const away = Math.hypot(camera.u - this.target.u, camera.v - this.target.v);
        const wave = this.scan * Math.max(0, Math.sin(this.time * 6 - away / 60));
        const alpha = 0.09 + camera.flash * 0.16 + wave * 0.16;
        const fill = g.createRadialGradient(x, y, 0, x, y, reach * this.zoom);
        fill.addColorStop(0, `rgba(${C.cone}, ${alpha * 2.2})`);
        fill.addColorStop(1, `rgba(${C.cone}, 0)`);
        g.fillStyle = fill;
        this.polygon(g, points);
        g.fill();
      }
    }

    drawTrail(g) {
      if (this.trail.length < 2) return;
      g.lineWidth = 2;
      g.lineCap = "round";
      for (let k = 1; k < this.trail.length; k++) {
        g.strokeStyle = `rgba(124, 196, 255, ${(k / this.trail.length) * 0.5})`;
        g.beginPath();
        g.moveTo(...this.project(...this.trail[k - 1]));
        g.lineTo(...this.project(...this.trail[k]));
        g.stroke();
      }
    }

    drawCars(g) {
      for (const car of this.cars) {
        const [x, y] = this.project(car.u, car.v);
        const [hx, hy] = this.project(car.u + Math.cos(car.dir) * 6, car.v + Math.sin(car.dir) * 6);
        const angle = Math.atan2(hy - y, hx - x);
        const length = Math.hypot(hx - x, hy - y) * 1.5 + 7;
        g.save();
        g.translate(x, y);
        g.rotate(angle);
        if (car.target) {
          g.shadowColor = C.target;
          g.shadowBlur = 12;
        }
        g.fillStyle = car.target ? C.target
          : (this.query || this.results.length ? "rgba(174, 184, 204, 0.45)" : C.car);
        g.beginPath();
        g.roundRect(-length / 2, -3, length, 6, 2);
        g.fill();
        g.shadowBlur = 0;
        g.fillStyle = "rgba(255, 244, 214, 0.95)";   // фары
        g.fillRect(length / 2 - 1.5, -2.6, 1.5, 1.6);
        g.fillRect(length / 2 - 1.5, 1, 1.5, 1.6);
        g.fillStyle = "rgba(255, 80, 70, 0.8)";      // стоп-сигналы
        g.fillRect(-length / 2, -2.6, 1.2, 1.6);
        g.fillRect(-length / 2, 1, 1.2, 1.6);
        g.restore();
      }
    }

    drawCameras(g) {
      g.font = "600 10px Manrope, 'Segoe UI', sans-serif";
      for (const camera of this.cameras) {
        const [x, y] = this.project(camera.u, camera.v);
        g.fillStyle = camera.flash > 0.05 ? C.target : C.camera;
        g.beginPath();
        g.arc(x, y - 6, 3.2, 0, Math.PI * 2);
        g.fill();
        g.strokeStyle = "rgba(245, 181, 61, 0.45)";
        g.beginPath();
        g.moveTo(x, y - 3);
        g.lineTo(x, y);
        g.stroke();
        g.fillStyle = "rgba(223, 230, 243, 0.55)";
        g.fillText(`К${camera.id}`, x + 6, y - 8);
      }
    }

    drawLinks(g) {
      for (const link of this.links) {
        const life = Math.min(1, (link.until - this.time) / 3);
        const [x1, y1] = this.project(link.from.u, link.from.v);
        const [x2, y2] = this.project(link.to.u, link.to.v);
        const mx = (x1 + x2) / 2, my = Math.min(y1, y2) - 40;
        g.strokeStyle = `rgba(124, 196, 255, ${0.75 * life})`;
        g.lineWidth = 1.4;
        g.setLineDash([5, 5]);
        g.beginPath();
        g.moveTo(x1, y1 - 6);
        g.quadraticCurveTo(mx, my, x2, y2 - 6);
        g.stroke();
        g.setLineDash([]);
      }
    }

    drawMarks(g) {
      g.font = "600 11px Manrope, 'Segoe UI', sans-serif";
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
        if (mark.target) {
          const label = `К${mark.camera} · ${mark.score.toFixed(2)}`;
          const width = g.measureText(label).width + 12;
          g.fillStyle = `rgba(13, 20, 34, ${0.85 * fade})`;
          g.fillRect(x + size + 4, y - size - 6, width, 18);
          g.fillStyle = `rgba(124, 196, 255, ${fade})`;
          g.fillText(label, x + size + 10, y - size + 7);
        }
      }
    }
  }

  window.CityScene = CityScene;
})();
