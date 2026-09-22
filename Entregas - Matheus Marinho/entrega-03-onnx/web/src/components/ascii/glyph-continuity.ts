import type { AsciiFrame } from 'asciify-engine';

/** Fixed per-cell thresholds break contour bands without animated noise. */
export function glyphThreshold(x: number, y: number) {
  const value = Math.imul(x + 1, 374761393) ^ Math.imul(y + 1, 668265263);
  const hash = Math.imul(value ^ (value >>> 13), 1274126177);
  return ((hash ^ (hash >>> 16)) >>> 0) / 4294967295 - .5;
}

/** Display-rate tonal continuity. One glyph per cell; no overlapping frames. */
export class GlyphContinuity {
  private target?: AsciiFrame;
  private rendered?: AsciiFrame;
  private state = new Float32Array(0);
  private ranks = new Uint16Array(0);
  private thresholds = new Float32Array(0);
  private outputs: AsciiFrame[] = [];
  private chars: string[] = [];
  private columns = 0;
  private rows = 0;
  private turn = 0;
  active = false;

  setTarget(frame: AsciiFrame, charset: string) {
    const cols = frame[0]?.length ?? 0, rows = frame.length;
    if (!cols || !rows) { this.clear(); return; }
    const chars = [...new Set(Array.from(charset))];
    const reset = cols !== this.columns || rows !== this.rows || chars.join('') !== this.chars.join('');
    this.target = frame;
    if (reset) {
      this.columns = cols; this.rows = rows; this.chars = chars;
      this.state = new Float32Array(cols * rows * 4);
      this.ranks = new Uint16Array(cols * rows);
      this.thresholds = new Float32Array(cols * rows);
      this.outputs = [0, 1].map(() => frame.map(row => row.map(c => ({ ...c }))));
      for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
        const c = frame[y][x], i = y * cols + x;
        this.state.set([c.r, c.g, c.b, c.lum ?? c.r * .299 + c.g * .587 + c.b * .114], i * 4);
        this.thresholds[i] = glyphThreshold(x, y) * .65;
      }
      this.step(0, true);
    }
    this.active = true;
  }

  step(dt: number, snap = false): AsciiFrame {
    if (!this.target) return [];
    if (!this.active && !snap && this.rendered) return this.rendered;
    const blend = snap ? 1 : -Math.expm1(-Math.max(0, Math.min(.1, dt)) / .045);
    const output = this.outputs[this.turn++ % 2], levels = Math.max(1, this.chars.length - 1);
    let unsettled = false;
    for (let y = 0; y < this.rows; y++) for (let x = 0; x < this.columns; x++) {
      const target = this.target[y][x], cell = output[y][x], i = y * this.columns + x, p = i * 4;
      const tone = target.lum ?? target.r * .299 + target.g * .587 + target.b * .114;
      for (let k = 0; k < 4; k++) {
        const value = k === 0 ? target.r : k === 1 ? target.g : k === 2 ? target.b : tone;
        const delta = value - this.state[p + k];
        this.state[p + k] = Math.abs(delta) < .3 || snap ? value : this.state[p + k] + delta * blend;
        if (Math.abs(value - this.state[p + k]) >= .3) unsettled = true;
        else this.state[p + k] = value;
      }
      const position = this.state[p + 3] / 255 * levels + this.thresholds[i];
      const previous = this.ranks[i];
      let rank = previous;
      if (snap || position > previous + .62 || position < previous - .62) rank = Math.max(0, Math.min(levels, Math.round(position)));
      if (this.state[p + 3] < .3 || target.a < 10) rank = 0;
      this.ranks[i] = rank;
      cell.char = this.chars[rank] ?? ' ';
      cell.r = this.state[p]; cell.g = this.state[p + 1]; cell.b = this.state[p + 2];
      cell.a = target.a; cell.lum = this.state[p + 3];
    }
    this.active = unsettled;
    this.rendered = output;
    return output;
  }

  clear() {
    this.target = undefined; this.rendered = undefined; this.outputs = []; this.state = new Float32Array(0);
    this.ranks = new Uint16Array(0); this.thresholds = new Float32Array(0);
    this.columns = this.rows = 0; this.chars = []; this.active = false;
  }
}

/** Measure once per palette/font load, not per video or pointer frame. */
export function createDensityPalette() {
  const canvas = document.createElement('canvas'); canvas.width = 40; canvas.height = 48;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  let previous = '', palette = '';
  return {
    get(charset: string) {
      if (charset === previous) return palette;
      ctx.font = '28px "JetBrains Mono", monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      const marks = [...new Set(Array.from(charset))].filter(char => char !== ' ').map(char => {
        ctx.clearRect(0, 0, 40, 48); ctx.fillText(char, 20, 24);
        const pixels = ctx.getImageData(0, 0, 40, 48).data;
        let coverage = 0; for (let i = 3; i < pixels.length; i += 4) coverage += pixels[i];
        return { char, coverage };
      });
      previous = charset; palette = ' ' + marks.sort((a, b) => a.coverage - b.coverage).map(mark => mark.char).join('');
      return palette;
    },
    reset() { previous = ''; },
  };
}
