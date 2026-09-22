import type { PointerField } from './fluid-field';
import { invertTrailTone } from './ink-trail';

/** Warp luminance before ASCII conversion; never distort the canvas or HTML type. */
export function paintMediaSource(out: Uint8ClampedArray, source: Uint8ClampedArray, width: number, height: number, flow: PointerField, refraction = .014) {
  if (!(flow.hasRefraction ?? flow.active)) { out.set(source); return; }
  const field = [0, 0, 0];
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    flow.sample(x / width, y / height, field);
    // Bound refraction relative to the scene: large field impulses must not fold
    // the silhouette back on itself and create detached duplicate contours.
    const sx = Math.max(0, Math.min(width - 1, x - Math.tanh(field[0] / .08) * width * refraction));
    const sy = Math.max(0, Math.min(height - 1, y - Math.tanh(field[1] / .08) * height * refraction));
    const ix = Math.floor(sx), iy = Math.floor(sy), fx = sx - ix, fy = sy - iy;
    const x1 = Math.min(width - 1, ix + 1), y1 = Math.min(height - 1, iy + 1);
    const i = (y * width + x) * 4;
    const a = (iy * width + ix) * 4, b = (iy * width + x1) * 4;
    const c = (y1 * width + ix) * 4, d = (y1 * width + x1) * 4;
    for (let channel = 0; channel < 3; channel++) {
      const value = (source[a + channel] * (1 - fx) + source[b + channel] * fx) * (1 - fy)
        + (source[c + channel] * (1 - fx) + source[d + channel] * fx) * fy;
      out[i + channel] = flow.invertsDensity ? invertTrailTone(value / 255, field[2] / 3) * 255 : value * (flow.densityTrail ? 1 : 1 + field[2] * .3) + (flow.densityTrail ? field[2] * 72 : 0);
    }
    out[i + 3] = 255;
  }
}

// A fixed shadow curve reveals dark colored scenery without per-frame
// normalization (which can pump brightness as the subject moves).
const HERO_SHADOW_GAIN = Float64Array.from({ length: 256 }, (_, value) =>
  value ? Math.pow(value / 255, .64) * 255 / value : 1);

/** Preserve the whole scene and its hue in worker and poster/fallback paths. */
export function gradeHeroMedia(pixels: Uint8ClampedArray, grade: Float64Array) {
  for (let i = 0; i < pixels.length; i += 4) {
    const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
    const luminance = Math.round(r * .299 + g * .587 + b * .114);
    // Scale channels together, limiting gain before a channel clips. Never
    // identify/remove a background color, or invent marks in true black.
    const gain = Math.min(HERO_SHADOW_GAIN[luminance], 255 / Math.max(1, r, g, b));
    const exposure = gain * .97 * grade[i / 4];
    pixels[i] *= exposure; pixels[i + 1] *= exposure; pixels[i + 2] *= exposure;
    pixels[i + 3] = 255;
  }
}

/** Warm brand ink with luminance preserved; black pixels fully replace old frames. */
export function tintMediaSource(pixels: Uint8ClampedArray, width: number, height: number) {
  for (let i = 0; i < width * height * 4; i += 4) {
    const value = pixels[i] < 10 ? 0 : Math.min(255, pixels[i] * 1.45);
    pixels[i] = value; pixels[i + 1] = value * (185 / 232); pixels[i + 2] = 0;
    pixels[i + 3] = 255;
  }
}

/** Opaque sampling avoids source-over accumulation in engine 1.0.118's sampler. */
export function flattenSourceOnBlack(pixels: Uint8ClampedArray) {
  for (let i = 0; i < pixels.length; i += 4) {
    const alpha = pixels[i + 3];
    if (alpha === 255) continue;
    const opacity = alpha / 255;
    pixels[i] *= opacity; pixels[i + 1] *= opacity; pixels[i + 2] *= opacity;
    pixels[i + 3] = 255;
  }
}

/** Fixed scene geometry is computed on resize, not on every decoded video frame. */
export function createHeroVignette(width: number, height: number, aspect: number): Float64Array {
  const mask = new Float64Array(width * height);
  const centerY = aspect < 1 ? .50 : .34;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const distance = Math.hypot((x / width - .5) / .53, (y / height - centerY) / (aspect < 1 ? .25 : .29));
    const edge = Math.max(0, Math.min(1, (1 - distance) / .38));
    mask[y * width + x] = edge * edge * (3 - 2 * edge);
  }
  return mask;
}

/** Full-bleed media keeps its edges; only the text zone receives tonal protection. */
export function createFullSceneGrade(width: number, height: number): Float64Array {
  const mask = new Float64Array(width * height);
  for (let y = 0; y < height; y++) {
    const progress = Math.max(0, Math.min(1, (y / height - .46) / .43));
    const shade = 1 - .15 * progress * progress * (3 - 2 * progress);
    mask.fill(shade, y * width, (y + 1) * width);
  }
  return mask;
}


// Fixed tonal response, not adaptive exposure: give midtones more negative space.
const HERO_DENSITY_GAIN = Float64Array.from({ length: 256 }, (_, value) =>
  value ? Math.pow(value / 255, 1.65) * 255 / value : 1);
export function shapeHeroDensity(pixels: Uint8ClampedArray) {
  for (let i = 0; i < pixels.length; i += 4) {
    const tone = Math.round(pixels[i] * .299 + pixels[i + 1] * .587 + pixels[i + 2] * .114);
    const gain = HERO_DENSITY_GAIN[tone];
    pixels[i] *= gain; pixels[i + 1] *= gain; pixels[i + 2] *= gain;
  }
}

/** Fixed light emphasis for the hands film; retain the complete courtyard and
 * grade worker frames and poster frames identically, without adaptive exposure. */
export function createGestureGrade(width: number, height: number): Float64Array {
  const grade = createFullSceneGrade(width, height);
  for (let y = 0; y < height; y++) {
    const distance = (y / height - .59) / .19;
    const emphasis = .76 + .5 * Math.exp(-Math.pow(distance, 4));
    for (let x = 0; x < width; x++) grade[y * width + x] *= emphasis;
  }
  return grade;
}
