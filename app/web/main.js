// The two-array show (S12-C). Left array: the curated spikes (NIX Good). Right array: the detections of a
// −Kσ threshold on ŷ, the model's output decoded from 125 Hz tokens. Everything is read from the bundle that
// `python -m app.export` wrote (show/): this file only plays it back and draws it.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- the bundle (the recording named after # in the address)
const recordings = await fetch('show/recordings.json').then((r) => r.json());
const REC = recordings.includes(location.hash.slice(1)) ? location.hash.slice(1) : recordings[0];
const load = (file) => fetch(`show/${REC}/${file}`).then((r) => r.arrayBuffer());
const meta = JSON.parse(new TextDecoder().decode(await load('meta.json')));
const EV = new Int32Array(await load('events.bin'));        // [sample, electrode, kind, flag] × n, sorted by sample
const LAT = new Float32Array(await load('latent.bin'));     // [electrode, token, 2]
const { hz: HZ, samples: T, electrodes: N, frame: FRAME } = meta;
const NIX = '#56b4e9', YHAT = '#e69f00', XHAT = '#cc79a7', GREY = '#8b8880', WIN = 2 * HZ;

// ---------------------------------------------------------------- the 3D scene
const renderer = new THREE.WebGLRenderer({ canvas: $('scene'), antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color('#0a0a0a');
scene.fog = new THREE.Fog('#0a0a0a', 14, 34);
scene.add(new THREE.HemisphereLight('#ffffff', '#000000', 1.2));
const sun = new THREE.DirectionalLight('#ffffff', 2);
sun.position.set(3, 8, 5);
scene.add(sun);
const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
camera.position.set(0, 9, 19);
const composer = new EffectComposer(renderer);              // bloom: colours brighter than white glow
composer.addPass(new RenderPass(scene, camera));
composer.addPass(new UnrealBloomPass(new THREE.Vector2(), 0.9, 0.5, 0.7));
composer.addPass(new OutputPass());

// A Utah array: 10 × 10 shanks without the corners, 400 µm apart, 1.5 mm long (1 unit = 1 mm). Channel order.
const PITCH = 0.4, H = 1.5, GAP = 3;
const CELLS = [...Array(100).keys()].filter((c) => ![0, 9, 90, 99].includes(c));
const xz = (i) => [(CELLS[i] % 10 - 4.5) * PITCH, (Math.floor(CELLS[i] / 10) - 4.5) * PITCH];

function utah(side, color) {
  const group = new THREE.Group(), pins = new THREE.Group(), m = new THREE.Matrix4();
  const plate = new THREE.Mesh(new THREE.BoxGeometry(4.2, 0.2, 4.2), new THREE.MeshStandardMaterial({ color: '#1a1a1a', roughness: 0.4 }));
  const shanks = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.012, 0.05, H, 8).translate(0, H / 2, 0), new THREE.MeshStandardMaterial({ roughness: 0.4 }), N);
  const tips = new THREE.InstancedMesh(new THREE.SphereGeometry(0.05), new THREE.MeshBasicMaterial(), N);
  const boxes = new THREE.InstancedMesh(new THREE.BoxGeometry(PITCH, H, PITCH).translate(0, H / 2, 0), new THREE.MeshBasicMaterial({ visible: false }), N);
  for (let i = 0; i < N; i++) {
    const [x, z] = xz(i);
    shanks.setMatrixAt(i, m.makeTranslation(x, 0, z));
    boxes.setMatrixAt(i, m);                                // invisible, only for clicking
    tips.setMatrixAt(i, m.makeTranslation(x, H, z));
  }
  plate.position.y = -0.1;
  group.position.x = side * GAP;
  pins.add(shanks, tips, boxes);
  group.add(plate, pins);
  scene.add(group);
  return { pins, shanks, tips, boxes, color: new THREE.Color(color), flash: new Float32Array(N), miss: new Float32Array(N), rate: new Float32Array(N) };
}
const A = [utah(-1, NIX), utah(1, YHAT)];                   // A[0]: curated spikes, A[1]: detections on ŷ

// Threads: when the model finds a curated spike, a line lights up between the two tips of that electrode.
const blueToOrange = Array.from({ length: 40 }, (_, j) => new THREE.Color(NIX).lerp(new THREE.Color(YHAT), j / 39).multiplyScalar(2.5).toArray()).flat();
const threadShape = Array.from({ length: N }, (_, i) => {
  const [x, z] = xz(i), curve = new THREE.QuadraticBezierCurve3(new THREE.Vector3(x - GAP, H, z), new THREE.Vector3(x, H + 2.4, z), new THREE.Vector3(x + GAP, H, z));
  return new THREE.BufferGeometry().setFromPoints(curve.getPoints(39)).setAttribute('color', new THREE.Float32BufferAttribute(blueToOrange, 3));
});
const threads = Array.from({ length: 32 }, () => new THREE.Line(threadShape[0], new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0, blending: THREE.AdditiveBlending })));
scene.add(...threads);
let nextThread = 0;

// ---------------------------------------------------------------- playback and the S10 counters
// ref: curated spikes · fy, fx: of those, found on ŷ and on bp(x) · dy, dx: detections · ay, ax: detections that
// match a sorted spike (NIX Good or Bad; the others are false alarms)
const zero = () => ({ ref: 0, fy: 0, fx: 0, dy: 0, ay: 0, dx: 0, ax: 0 });
let t = 0, next = 0, playing = false, speed = 0.25, all = zero(), per = Array.from({ length: N }, zero);

function count(c, kind, flag) {
  if (kind === 0) { c.ref++; c.fy += flag & 1; c.fx += flag >> 1; }
  else if (kind === 1) { c.dy++; c.ay += flag; }
  else { c.dx++; c.ax += flag; }
}

// Play every event before sample t1. -> an electrode where ŷ found a curated spike (−1 if none), for the thread.
function advance(t1, silent) {
  let found = -1;
  for (; next < EV.length && EV[next] < t1; next += 4) {
    const i = EV[next + 1], kind = EV[next + 2], flag = EV[next + 3];
    count(all, kind, flag);
    count(per[i], kind, flag);
    if (silent || kind === 2) continue;
    A[kind].rate[i]++;
    if (kind === 0 || flag) A[kind].flash[i] = 1; else A[kind].miss[i] = 1;      // a false alarm flashes white
    if (kind === 0 && flag & 1) found = i;
  }
  t = t1;
  return found;
}

function seek(t1) {                                         // replay the counters from the start, without the lights
  next = 0; all = zero(); per = Array.from({ length: N }, zero);
  advance(Math.max(0, Math.min(T, t1)), true);
}

function play(on) {
  if (on && t >= T) seek(0);
  playing = on;
  $('play').textContent = on ? 'pause' : 'play';
}

// ---------------------------------------------------------------- the numbers on screen
const percent = (a, b) => b ? `${(100 * a / b).toFixed(1)}%` : '–';
const clock = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`;

function numbers() {
  $('time').textContent = `${clock(t / HZ)} / ${clock(T / HZ)}`;
  $('played').style.width = `${100 * t / T}%`;
  $('nRef').textContent = all.ref.toLocaleString('en');
  $('recY').textContent = percent(all.fy, all.ref);
  $('statY').textContent = `of the curated spikes · ${percent(all.ay, all.dy)} of its detections match a sorted spike`;
  $('statX').textContent = `the same threshold on the raw input: ${percent(all.fx, all.ref)} · ${percent(all.ax, all.dx)}`;
  if (sel >= 0) {
    const c = per[sel];
    $('elStat').textContent = `${c.ref} curated spikes · ŷ finds ${percent(c.fy, c.ref)}, ${percent(c.ay, c.dy)} of its detections match a sorted spike · raw input: ${percent(c.fx, c.ref)}, ${percent(c.ax, c.dx)}`;
  }
}

// The timeline: spikes per pixel of the whole recording, curated above the axis and ŷ detections below.
function timeline() {
  const c = $('hist'), g = c.getContext('2d'), w = c.width = c.clientWidth, h = c.height = c.clientHeight;
  const bins = [new Float32Array(w), new Float32Array(w)];
  for (let p = 0; p < EV.length; p += 4) if (EV[p + 2] < 2) bins[EV[p + 2]][Math.floor(EV[p] / T * w)]++;
  const max = Math.max(...bins[0], ...bins[1]);
  for (let x = 0; x < w; x++) {
    g.fillStyle = NIX; g.fillRect(x, h / 2, 1, -0.45 * h * bins[0][x] / max);
    g.fillStyle = YHAT; g.fillRect(x, h / 2, 1, 0.45 * h * bins[1][x] / max);
  }
}

// ---------------------------------------------------------------- one electrode: its signals in the drawer
let sel = -1, SIG = null, mine = [];                        // SIG: int16 [4, T] = x, x̂, bp(x), ŷ · mine: its events

async function select(i) {
  sel = i; SIG = null;
  $('drawer').classList.toggle('open', i >= 0);
  if (i < 0) return;
  $('elName').textContent = `Channel ${i + 1}`;
  mine = [];
  for (let p = 0; p < EV.length; p += 4) if (EV[p + 1] === i && EV[p + 2] < 2) mine.push([EV[p], EV[p + 2], EV[p + 3]]);
  const signals = new Int16Array(await load(`sig/${i}.bin`));
  if (sel === i) SIG = signals;
}

function traces() {                                         // the last 2 s, three rows on one time axis
  const c = $('traces'), g = c.getContext('2d'), d = devicePixelRatio, w = c.clientWidth, h = c.clientHeight / 3;
  c.width = w * d; c.height = 3 * h * d;
  g.scale(d, d); g.font = '11px Plex'; g.fillStyle = GREY; g.textAlign = 'right';
  const t0 = t - WIN, X = (s) => (s - t0) / WIN * w, thr = meta.k * meta.sigma_bp[sel];
  const Y = (row, v, lo, hi) => (row + 0.12 + 0.68 * Math.min(1, Math.max(0, (hi - v) / (hi - lo)))) * h;   // 12–80 % of the row

  const signal = (row, k, lo, hi, color, width) => {        // min and max of every pixel column
    if (!SIG) return;
    g.strokeStyle = color; g.lineWidth = width; g.beginPath();
    for (let x = 0; x < w; x++) {
      let a = Infinity, b = -Infinity;
      for (let s = Math.max(0, Math.floor(t0 + x * WIN / w)), end = Math.floor(t0 + (x + 1) * WIN / w); s < end; s++) {
        const v = SIG[k * T + s];
        if (v < a) a = v;
        if (v > b) b = v;
      }
      if (a <= b) { g.moveTo(x, Y(row, b / meta.scale, lo, hi)); g.lineTo(x, Y(row, a / meta.scale, lo, hi) + 1); }
    }
    g.stroke();
  };
  g.fillText('input x (grey) · reconstruction x̂ from the tokens (pink)', w, 11);
  signal(0, 0, -4, 4, GREY, 3);
  signal(0, 1, -4, 4, XHAT, 1);
  g.fillText('spike band of x (grey) · ŷ (orange) · threshold −Kσ · ticks: curated spikes, then detections on ŷ', w, h + 11);
  signal(1, 2, -3 * thr, 1.5 * thr, GREY, 1);
  signal(1, 3, -3 * thr, 1.5 * thr, YHAT, 1);
  g.fillStyle = GREY; g.fillRect(0, Y(1, -thr, -3 * thr, 1.5 * thr), w, 0.5);
  for (const [s, kind, flag] of mine) {
    if (s < t0 || s > t) continue;
    g.fillStyle = kind === 0 ? NIX : flag ? YHAT : '#fff';
    g.globalAlpha = kind === 0 && !(flag & 1) ? 0.35 : 1;   // a curated spike the model missed is dim
    g.fillRect(X(s) - 1, (kind === 0 ? 1.84 : 1.93) * h, 2, 0.06 * h);
  }
  g.globalAlpha = 1; g.fillStyle = GREY;
  g.fillText('tokens z · 2 principal components · one value every 8 ms', w, 2 * h + 11);
  for (const pc of [0, 1]) {
    g.strokeStyle = pc ? GREY : '#ece9e2'; g.lineWidth = 1; g.beginPath();
    for (let j = Math.max(0, Math.floor(t0 / FRAME)); j < t / FRAME; j++) {
      const y = Y(2, LAT[(sel * T / FRAME + j) * 2 + pc], -5, 5);
      g.lineTo(X(j * FRAME), y); g.lineTo(X((j + 1) * FRAME), y);
    }
    g.stroke();
  }
}

// ---------------------------------------------------------------- controls
let begun = -1, mouse = [0, 0];
const ray = new THREE.Raycaster(), time = new THREE.Clock();

function pick(e) {                                          // the electrode under the pointer (−1 if none)
  ray.setFromCamera(new THREE.Vector2(e.clientX / innerWidth * 2 - 1, 1 - e.clientY / innerHeight * 2), camera);
  return ray.intersectObjects(A.map((a) => a.boxes))[0]?.instanceId ?? -1;
}
function start() {
  if (begun >= 0) return;
  begun = time.elapsedTime;
  document.body.classList.remove('intro');
  setTimeout(() => play(true), 2500);
}
$('start').onclick = start;
$('play').onclick = () => play(!playing);
$('close').onclick = () => select(-1);
$('speed').onclick = () => { speed = { 1: 0.25, 0.25: 0.05, 0.05: 1 }[speed]; $('speed').textContent = `${speed}×`; };
$('scene').onclick = (e) => { const i = pick(e); if (i >= 0 && begun >= 0) select(i); };
$('scene').onpointermove = (e) => {
  mouse = [e.clientX / innerWidth * 2 - 1, e.clientY / innerHeight * 2 - 1];
  $('scene').style.cursor = pick(e) >= 0 ? 'pointer' : '';
};
$('timeline').onpointerdown = $('timeline').onpointermove = (e) => {
  if (e.buttons) seek((e.clientX - $('timeline').getBoundingClientRect().left) / $('timeline').clientWidth * T);
};
onkeydown = (e) => {
  if (e.code === 'Space') { e.preventDefault(); begun < 0 ? start() : play(!playing); }
  if (e.code === 'ArrowRight') seek(t + 5 * HZ);
  if (e.code === 'ArrowLeft') seek(t - 5 * HZ);
  if (e.code === 'KeyF') document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  if (e.code === 'Escape') select(-1);
};
onresize = () => {
  renderer.setSize(innerWidth, innerHeight, false);
  composer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  timeline();
};
onresize();

// ---------------------------------------------------------------- every frame
const colour = new THREE.Color(), grey = new THREE.Color('#77746e'), white = new THREE.Color('#ffffff');
const look = new THREE.Vector3(0, 0.3, 0), v = new THREE.Vector3();
const ease = (x) => 1 - (1 - Math.min(1, Math.max(0, x))) ** 3;

function frame() {
  requestAnimationFrame(frame);
  const dt = Math.min(time.getDelta(), 0.1), now = time.elapsedTime;

  if (playing) {
    const i = advance(Math.min(T, t + dt * speed * HZ));
    if (i >= 0) {                                           // at most one new thread per frame
      const thread = threads[nextThread++ % threads.length];
      thread.geometry = threadShape[i];
      thread.material.opacity = 1;
    }
    if (t >= T) play(false);
  }
  for (const thread of threads) thread.material.opacity *= 0.9;

  // tips: a flash per event (decays in ~60 ms) over a glow that follows the firing rate (~0.5 s)
  A.forEach((a, side) => {
    a.pins.scale.y = Math.max(0.001, ease(begun < 0 ? 0 : (now - begun) / 1.6 - side * 0.5));   // the shanks rise at the start
    for (let i = 0; i < N; i++) {
      a.flash[i] *= Math.exp(-dt / 0.06); a.miss[i] *= Math.exp(-dt / 0.06); a.rate[i] *= Math.exp(-dt / 0.5);
      a.tips.setColorAt(i, colour.copy(a.color).multiplyScalar(0.06 + 0.05 * a.rate[i] + 4 * a.flash[i]).addScalar(3 * a.miss[i]));
      a.shanks.setColorAt(i, i === sel ? white : colour.copy(grey).lerp(a.color, Math.min(0.8, a.rate[i] / 12)));
    }
    a.tips.instanceColor.needsUpdate = a.shanks.instanceColor.needsUpdate = true;
  });

  // camera: arrives after the intro, sways slowly, follows the pointer a little, rises when the drawer is open
  if (begun >= 0) camera.position.lerp(v.set(mouse[0] * 1.2 + Math.sin(now * 0.15) * 0.8, 5.4 - mouse[1] * 0.6, sel >= 0 ? 15 : 10.8), 0.03);
  look.y += ((sel >= 0 ? -2.6 : 0.9) - look.y) * 0.05;
  camera.lookAt(look);

  numbers();
  if (sel >= 0) traces();
  composer.render();
}
$('rec').innerHTML = `patient ${meta.patient}, never seen in training · ` + recordings.map((r) => `<a href="#${r}" class="${r === REC ? 'on' : ''}">${r.split('_').slice(1, 4).join('-')}</a>`).join(' ');
onhashchange = () => location.reload();
$('start').textContent = 'press space to start →';
$('start').disabled = false;
frame();
