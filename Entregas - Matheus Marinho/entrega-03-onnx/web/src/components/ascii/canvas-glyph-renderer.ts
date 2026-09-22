import { createTextMaskPainter, type TextMaskFrame } from './text-mask';
import type { AsciiFrame, AsciiOptions } from 'asciify-engine';
import { prepareGlyphAtlas, type GlyphAtlasPlan } from './glyph-renderer';

/** Match the hero's glyph spacing when WebGL is unavailable. */
export function createCanvasGlyphRenderer(scale = 1) {
  const maskPainter=createTextMaskPainter();
  const atlas = document.createElement('canvas'), ink = document.createElement('canvas');
  const atlasCtx = atlas.getContext('2d')!, inkCtx = ink.getContext('2d')!;
  let plan: GlyphAtlasPlan | null = null, colors: ImageData | null = null;
  return {
    render(ctx: CanvasRenderingContext2D, frame: AsciiFrame, options: AsciiOptions, width: number, height: number, mask?: TextMaskFrame) {
      if (!frame.length || !frame[0].length) return;
      const rows = frame.length, cols = frame[0].length, ratio = ctx.canvas.width / width;
      const next = prepareGlyphAtlas(plan, options, width, height, ratio, cols, rows, 4096, 0, scale);
      if (next !== plan) {
        atlas.width = next.atlasWidth; atlas.height = next.atlasHeight;
        atlasCtx.font = `${next.fontSize * ratio}px "JetBrains Mono", monospace`;
        atlasCtx.textAlign = 'center'; atlasCtx.textBaseline = 'middle'; atlasCtx.fillStyle = '#fff';
        next.glyphs.forEach((char, i) => atlasCtx.fillText(char, (i % next.atlasColumns) * next.tileWidth + next.padding + next.cellWidth * ratio / 2, Math.floor(i / next.atlasColumns) * next.tileHeight + next.padding + next.cellHeight * ratio / 2));
        ink.width = cols; ink.height = rows; colors = inkCtx.createImageData(cols, rows);
      }
      plan = next;
      ctx.clearRect(0, 0, width, height);
      const accent = options.colorMode === 'accent' ? [1, 3, 5].map(start => parseInt(options.accentColor.slice(start, start + 2), 16)) : null;
      for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
        const cell = frame[y][x], i = (y * cols + x) * 4, glyph = next.indices.get(cell.char) ?? 0;
        const gray = cell.r * .299 + cell.g * .587 + cell.b * .114;
        colors!.data[i] = accent?.[0] ?? gray;
        colors!.data[i + 1] = accent?.[1] ?? gray;
        colors!.data[i + 2] = accent?.[2] ?? gray;
        colors!.data[i + 3] = 255;
        if (!glyph || cell.a < 10) continue;
        ctx.globalAlpha = cell.a / 255;
        ctx.drawImage(atlas, (glyph % next.atlasColumns) * next.tileWidth + next.padding, Math.floor(glyph / next.atlasColumns) * next.tileHeight + next.padding, next.cellWidth * ratio, next.cellHeight * ratio, x * next.cellWidth, y * next.cellHeight, next.cellWidth, next.cellHeight);
      }
      ctx.globalAlpha = 1; inkCtx.putImageData(colors!, 0, 0);
      const smoothing = ctx.imageSmoothingEnabled; ctx.imageSmoothingEnabled = false;
      ctx.globalCompositeOperation = 'source-in'; ctx.drawImage(ink, 0, 0, width, height);
      ctx.globalCompositeOperation = 'source-over'; ctx.imageSmoothingEnabled = smoothing; maskPainter.paint(ctx,mask,width,height);
    },
    reset() { plan = null; },
    destroy() { maskPainter.destroy(); plan = null; colors = null; atlas.width = atlas.height = ink.width = ink.height = 1; },
  };
}
