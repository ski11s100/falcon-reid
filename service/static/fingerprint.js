/* Живой цифровой отпечаток — анимация первого экрана.
 *
 * Модель превращает машину в вектор из 4096 признаков. Сервер проецирует его
 * на 128 фиксированных направлений (см. fingerprint() в service/app.py), и здесь
 * они рисуются кольцом из 128 лучей. У каждой машины кольцо своё, у снимков
 * одной машины кольца похожи — это и есть «визуальный цифровой отпечаток» из
 * названия задачи, показанный буквально.
 *
 * Пока поиска не было, кольцо перебирает отпечатки случайных машин. Во время
 * поиска по нему идёт сканирующая волна, а по ответу сервера оно перетекает в
 * настоящий отпечаток запроса. Фон — расходящиеся лучи света.
 *
 * Рисуется на canvas в реальном времени: видеофайл не нужен, всё работает без
 * интернета. Цикл останавливается, когда первый экран не виден, вкладка скрыта
 * или выбран другой вариант оформления. При prefers-reduced-motion кадр
 * рисуется только при изменениях.
 */

(function () {
  "use strict";

  const GROUPS = 128;
  const BLUE = [79, 140, 255];
  const CYAN = [124, 196, 255];

  /* Подготовка к показу: лёгкое круговое сглаживание и растяжка по 4-му и
   * 96-му процентилям. Сервер отдаёт проекцию эмбеддинга, значения которой
   * независимы от соседей, и без сглаживания кольцо выглядит шумом. Сглаживание
   * sigma=1 оставляет узнаваемость: ближайшее кольцо — та же машина в 90%
   * случаев против 97% без него (замер в service/app.py, fingerprint). */
  function stretch(values) {
    const n = values.length;
    const kernel = [-3, -2, -1, 0, 1, 2, 3].map((k) => [k, Math.exp(-(k * k) / 2)]);
    const norm = kernel.reduce((sum, [, w]) => sum + w, 0);
    const smooth = values.map((_, i) =>
      kernel.reduce((sum, [k, w]) => sum + w * values[(i + k + n) % n], 0) / norm);
    const sorted = smooth.slice().sort((a, b) => a - b);
    const low = sorted[Math.floor(n * 0.04)];
    const high = sorted[Math.ceil(n * 0.96) - 1];
    const span = high - low || 1;
    return smooth.map((v) => Math.min(1, Math.max(0, (v - low) / span)));
  }

  /* Правдоподобный отпечаток для демонстрации: гладкая основа плюс несколько
   * пиков. Настоящие отпечатки приходят с сервера. */
  function randomFingerprint() {
    const values = new Array(GROUPS).fill(0).map(() => 0.25 + Math.random() * 0.25);
    const peaks = 7 + Math.floor(Math.random() * 6);
    for (let p = 0; p < peaks; p++) {
      const centre = Math.random() * GROUPS;
      const width = 1.5 + Math.random() * 3.5;
      const height = 0.35 + Math.random() * 0.55;
      for (let i = 0; i < GROUPS; i++) {
        let d = Math.abs(i - centre);
        d = Math.min(d, GROUPS - d);
        values[i] += height * Math.exp(-(d * d) / (2 * width * width));
      }
    }
    return stretch(values);
  }

  function rgba(rgb, alpha) {
    return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
  }

  function mix(a, b, t) {
    return [0, 1, 2].map((k) => Math.round(a[k] + (b[k] - a[k]) * t));
  }

  class FingerprintScene {
    constructor(canvas, caption) {
      this.canvas = canvas;
      this.caption = caption;
      this.context = canvas.getContext("2d");
      this.reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      this.current = randomFingerprint();
      this.target = randomFingerprint();
      this.locked = false;          // true, пока показан настоящий отпечаток
      this.scan = 0;                // 0..1, сила сканирующей волны
      this.scanning = false;
      this.nextMorph = 0;
      this.time = 0;
      this.last = 0;
      this.running = false;
      this.visible = true;
      this.particles = [];

      this.resize = this.resize.bind(this);
      this.frame = this.frame.bind(this);

      new ResizeObserver(this.resize).observe(canvas);
      new IntersectionObserver((entries) => {
        this.visible = entries[0].isIntersecting;
        this.visible ? this.resume() : this.pause();
      }).observe(canvas);
      document.addEventListener("visibilitychange", () => {
        document.hidden ? this.pause() : this.resume();
      });
    }

    /* ---------- Управление ---------- */

    start() {
      this.enabled = true;
      this.resize();
      this.resume();
    }

    stop() {
      this.enabled = false;
      this.pause();
    }

    resume() {
      if (!this.enabled || !this.visible || document.hidden || this.running) return;
      if (this.reducedMotion) { this.draw(0); return; }
      this.running = true;
      this.last = performance.now();
      requestAnimationFrame(this.frame);
    }

    pause() {
      this.running = false;
    }

    setScanning(on) {
      this.scanning = on;
      if (on) this.setCaption("Снимаем отпечаток…");
      if (this.reducedMotion) this.draw(0);
    }

    showFingerprint(values, caption) {
      if (!values || values.length !== GROUPS) return;
      this.target = stretch(values);
      this.locked = true;
      this.setCaption(caption);
      if (this.reducedMotion) { this.current = this.target.slice(); this.draw(0); }
    }

    setCaption(text) {
      if (this.caption) this.caption.textContent = text;
    }

    /* ---------- Геометрия ---------- */

    resize() {
      const rect = this.canvas.getBoundingClientRect();
      if (rect.width < 1 || rect.height < 1) return;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      this.width = rect.width;
      this.height = rect.height;
      this.canvas.width = Math.round(rect.width * ratio);
      this.canvas.height = Math.round(rect.height * ratio);
      this.context.setTransform(ratio, 0, 0, ratio, 0, 0);

      const narrow = rect.width < 900;
      this.cx = narrow ? rect.width / 2 : rect.width * 0.72;
      this.cy = narrow ? 160 : rect.height * 0.48;
      this.radius = narrow ? 70 : Math.max(90, Math.min(rect.height * 0.19, rect.width * 0.12, 150));
      this.reach = Math.hypot(Math.max(this.cx, rect.width - this.cx),
                              Math.max(this.cy, rect.height - this.cy));
      this.renderRays(ratio);
      this.seedParticles();
      if (this.reducedMotion) this.draw(0);
    }

    /* Лучи рисуются один раз в отдельный слой и потом только поворачиваются. */
    renderRays(ratio) {
      const size = Math.ceil(this.reach * 2);
      const layer = document.createElement("canvas");
      layer.width = Math.round(size * ratio);
      layer.height = Math.round(size * ratio);
      const g = layer.getContext("2d");
      g.setTransform(ratio, 0, 0, ratio, 0, 0);
      const c = size / 2;
      const inner = this.radius * 1.25;

      const fade = g.createRadialGradient(c, c, inner, c, c, c);
      fade.addColorStop(0, rgba(CYAN, 0));
      fade.addColorStop(0.06, rgba(CYAN, 0.55));
      fade.addColorStop(0.3, rgba(BLUE, 0.22));
      fade.addColorStop(1, rgba(BLUE, 0));

      const rays = 120;
      for (let i = 0; i < rays; i++) {
        const angle = (i / rays) * Math.PI * 2 + (Math.random() - 0.5) * 0.02;
        const strong = i % 6 === 0;
        g.strokeStyle = fade;
        g.globalAlpha = strong ? 0.95 : 0.25 + Math.random() * 0.35;
        g.lineWidth = strong ? 2.2 : 0.8 + Math.random() * 0.9;
        g.beginPath();
        g.moveTo(c + Math.cos(angle) * inner, c + Math.sin(angle) * inner);
        g.lineTo(c + Math.cos(angle) * c, c + Math.sin(angle) * c);
        g.stroke();
      }
      this.rays = layer;
      this.raysSize = size;
    }

    seedParticles() {
      this.particles = Array.from({ length: 80 }, () => this.newParticle(true));
    }

    newParticle(anywhere) {
      const outer = Math.min(this.reach, this.radius * 5.5);
      return {
        angle: Math.random() * Math.PI * 2,
        r: anywhere ? this.radius * 1.3 + Math.random() * (outer - this.radius * 1.3) : outer,
        speed: 18 + Math.random() * 40,
        size: 0.6 + Math.random() * 1.3,
      };
    }

    /* ---------- Кадр ---------- */

    frame(now) {
      if (!this.running) return;
      const dt = Math.min(0.05, (now - this.last) / 1000);
      this.last = now;
      this.time += dt;
      this.update(dt);
      this.draw(dt);
      requestAnimationFrame(this.frame);
    }

    update(dt) {
      // Пока настоящего отпечатка нет, кольцо перебирает демонстрационные.
      if (!this.locked && this.time > this.nextMorph) {
        this.target = randomFingerprint();
        this.nextMorph = this.time + 3.6;
      }
      const ease = 1 - Math.exp(-dt * 3.2);
      for (let i = 0; i < GROUPS; i++) {
        this.current[i] += (this.target[i] - this.current[i]) * ease;
      }
      this.scan += ((this.scanning ? 1 : 0) - this.scan) * (1 - Math.exp(-dt * 6));

      const pull = 1 + this.scan * 3;
      for (const p of this.particles) {
        p.r -= p.speed * pull * dt;
        if (p.r < this.radius * 1.28) Object.assign(p, this.newParticle(false));
      }
    }

    draw() {
      const g = this.context;
      const { cx, cy, radius: R } = this;
      if (!this.width) return;
      g.clearRect(0, 0, this.width, this.height);

      // Лучи: медленное вращение и едва заметное дыхание.
      g.save();
      g.globalAlpha = 0.8 + 0.2 * Math.sin(this.time * 0.7) + this.scan * 0.2;
      g.translate(cx, cy);
      g.rotate(this.time * 0.012);
      g.drawImage(this.rays, -this.raysSize / 2, -this.raysSize / 2, this.raysSize, this.raysSize);
      g.restore();

      // Свечение за кольцом.
      const halo = g.createRadialGradient(cx, cy, R * 0.2, cx, cy, R * 2.4);
      halo.addColorStop(0, rgba(BLUE, 0.28 + this.scan * 0.15));
      halo.addColorStop(1, rgba(BLUE, 0));
      g.fillStyle = halo;
      g.beginPath();
      g.arc(cx, cy, R * 2.4, 0, Math.PI * 2);
      g.fill();

      // Частицы стекаются к кольцу: кадр превращается в признаки.
      for (const p of this.particles) {
        const k = Math.min(1, (p.r - R * 1.28) / (R * 1.5));
        g.fillStyle = rgba(CYAN, 0.15 + 0.55 * (1 - k));
        g.beginPath();
        g.arc(cx + Math.cos(p.angle) * p.r, cy + Math.sin(p.angle) * p.r, p.size, 0, Math.PI * 2);
        g.fill();
      }

      // Опорная окружность.
      g.strokeStyle = rgba(CYAN, 0.22);
      g.lineWidth = 1;
      g.beginPath();
      g.arc(cx, cy, R, 0, Math.PI * 2);
      g.stroke();

      // Кольцо отпечатка: 128 лучей, длина — энергия группы признаков.
      const spin = this.time * 0.05;
      const sweep = (this.time * 5) % (Math.PI * 2);
      for (let pass = 0; pass < 2; pass++) {
        for (let i = 0; i < GROUPS; i++) {
          const angle = (i / GROUPS) * Math.PI * 2 - Math.PI / 2 + spin;
          let v = this.current[i];
          let boost = 0;
          if (this.scan > 0.01) {
            let d = Math.abs(((angle - sweep) % (Math.PI * 2) + Math.PI * 3) % (Math.PI * 2) - Math.PI);
            boost = this.scan * Math.max(0, 1 - d / 0.5);
            v = Math.min(1, v + boost * 0.5);
          }
          const length = R * (0.1 + 0.6 * v);
          const cos = Math.cos(angle);
          const sin = Math.sin(angle);
          const colour = mix(BLUE, CYAN, Math.min(1, v * 0.8 + boost));
          g.strokeStyle = pass === 0 ? rgba(colour, 0.16) : rgba(colour, 0.55 + 0.45 * v);
          g.lineWidth = pass === 0 ? 5 : 1.8;
          g.lineCap = "round";
          g.beginPath();
          g.moveTo(cx + cos * (R + 3), cy + sin * (R + 3));
          g.lineTo(cx + cos * (R + 3 + length), cy + sin * (R + 3 + length));
          g.stroke();
        }
      }

      // Внутренние короткие штрихи — тот же профиль в миниатюре.
      g.strokeStyle = rgba(CYAN, 0.3);
      g.lineWidth = 1;
      g.beginPath();
      for (let i = 0; i < GROUPS; i += 2) {
        const angle = (i / GROUPS) * Math.PI * 2 - Math.PI / 2 + spin;
        const inner = R * (0.9 - 0.14 * this.current[i]);
        g.moveTo(cx + Math.cos(angle) * R * 0.92, cy + Math.sin(angle) * R * 0.92);
        g.lineTo(cx + Math.cos(angle) * inner, cy + Math.sin(angle) * inner);
      }
      g.stroke();

      // Шестигранник в центре — фирменный мотив, вращается навстречу кольцу.
      const hex = R * 0.46;
      g.save();
      g.translate(cx, cy);
      g.rotate(-this.time * 0.1);
      const fill = g.createLinearGradient(-hex, -hex, hex, hex);
      fill.addColorStop(0, rgba(CYAN, 0.22));
      fill.addColorStop(1, rgba(BLUE, 0.05));
      g.fillStyle = fill;
      g.strokeStyle = rgba(CYAN, 0.55);
      g.lineWidth = 1.2;
      g.beginPath();
      for (let k = 0; k < 6; k++) {
        const a = (k / 6) * Math.PI * 2;
        const x = Math.cos(a) * hex;
        const y = Math.sin(a) * hex;
        k ? g.lineTo(x, y) : g.moveTo(x, y);
      }
      g.closePath();
      g.fill();
      g.stroke();
      g.restore();
    }
  }

  /* Маленькое кольцо отпечатка для строки кандидата, в SVG. */
  function ringSvg(values, colour) {
    if (!values || values.length !== GROUPS) return '<span class="candidate__ring"></span>';
    const v = stretch(values);
    const size = 46;
    const c = size / 2;
    const inner = 9;
    const reach = 12;
    let path = "";
    for (let i = 0; i < GROUPS; i++) {
      const angle = (i / GROUPS) * Math.PI * 2 - Math.PI / 2;
      const r2 = inner + 1.5 + reach * v[i];
      path += `M${(c + Math.cos(angle) * inner).toFixed(2)} ${(c + Math.sin(angle) * inner).toFixed(2)}`
            + `L${(c + Math.cos(angle) * r2).toFixed(2)} ${(c + Math.sin(angle) * r2).toFixed(2)}`;
    }
    return `<svg class="candidate__ring" viewBox="0 0 ${size} ${size}" aria-hidden="true">`
         + `<path d="${path}" stroke="${colour}" stroke-width="1.1" stroke-linecap="round" fill="none"/></svg>`;
  }

  window.FingerprintScene = FingerprintScene;
  window.ringSvg = ringSvg;
})();
