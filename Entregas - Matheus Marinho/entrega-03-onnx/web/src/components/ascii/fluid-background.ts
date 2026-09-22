import type { SurfaceFinish } from './surface-finish';
import type { TextMaskFrame } from './text-mask';
import { type AsciiFrame, DEFAULT_OPTIONS, imageToAsciiFrame, renderFrameToCanvas } from 'asciify-engine';
import { FluidField, paintLiquidSource, type PointerField } from './fluid-field';
import { createGlyphRenderer } from './glyph-renderer';
import { WaterSurface, type HeroHover } from './water-surface';
import { flattenSourceOnBlack, tintMediaSource, paintMediaSource } from './media-source';
import { GlyphContinuity, createDensityPalette } from './glyph-continuity';
import { createCanvasGlyphRenderer } from './canvas-glyph-renderer';

/** App-side background. Call destroy() on unmount. No media assets required. */
export function mountFluidBackground(host: HTMLElement, target: HTMLCanvasElement, config: {
  texture?: HTMLCanvasElement;
  isPaused?: () => boolean;
  onError?: () => void;
  paintSource?: typeof paintLiquidSource;
  onMotion?: (playing: boolean) => void;
  fps?: number;
  /** When present, idle rendering follows decoded media frames. Hover remains independent. */
  sourceVersion?: () => number;
  colorMode?: 'accent' | 'fullcolor' | 'grayscale';
  accentColor?: string;
  fontSize?: number | ((width: number) => number);
  maxGlyphs?: number;
  /** Stable, display-rate character changes for decoded video. */
  smoothGlyphs?: boolean;
  glyphScale?: number;
  /** Optical finish around a sharp center; 0 disables it. */
  peripheralLens?: number;
  getFinish?: () => SurfaceFinish;
  getTextMask?: () => TextMaskFrame | undefined;
  getHover?: () => HeroHover;
  getHoverRadius?: () => number;
  getEdgeSafe?: () => boolean;
  getCharset?: () => string;
  getStyle?: () => import('asciify-engine/studio').StudioSettings['style'];
  fieldDecay?: number;
  continuousWake?: boolean;
  waterSurface?: boolean;
  previewY?: (aspect: number) => number;
} = {}) {
    const ctx = target.getContext('2d');
    const textureCanvas = config.texture, textureCtx = textureCanvas?.getContext('2d');
    if (!ctx) { config.onError?.(); return; }
    const source = document.createElement('canvas'), sourceCtx = source.getContext('2d', { willReadFrequently: true });
    if (!sourceCtx) { config.onError?.(); return; }
    const gpuCanvas = document.createElement('canvas');
    let studio: ReturnType<typeof import('./hero-studio-renderer').createHeroStudioRenderer> | undefined;
    let studioLoading = false;
    let gpuFailed = false;
    let onGpuFailure = () => {};
    const gpu = createGlyphRenderer(gpuCanvas, () => { gpuFailed = true; onGpuFailure(); }, config.glyphScale, config.smoothGlyphs ? 40 : 0, config.peripheralLens);
    const continuity = config.smoothGlyphs ? new GlyphContinuity() : null;
    const palette = config.smoothGlyphs ? createDensityPalette() : null;
    const canvasGlyphs = config.smoothGlyphs && (config.colorMode === 'grayscale' || (config.colorMode === 'accent' && /^#[\da-f]{6}$/i.test(config.accentColor ?? '#e8b900'))) ? createCanvasGlyphRenderer(config.glyphScale) : null;
    // Keep water rendering on the GPU. Copying WebGL into a 2D canvas each hover
    // frame forces a costly cross-context synchronization on some browsers.
    const directGpu = !!(config.waterSurface && gpu);
    const previousVisibility = target.style.visibility;
    const previousOpacity = target.style.opacity;
    if (directGpu) {
      gpuCanvas.className = target.className;
      gpuCanvas.setAttribute('aria-hidden', 'true');
      gpuCanvas.dataset.renderer = 'gpu';
      host.append(gpuCanvas); target.style.visibility = 'hidden';
    }
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    let width = 1, height = 1, aspect = 1, field: PointerField = new FluidField();
    let cachedFrame: AsciiFrame | undefined;
    let displayFrame: AsciiFrame | undefined, continuityAt = 0;
    let fallbackSource = new Uint8ClampedArray(4);
    let pixels = sourceCtx.createImageData(1, 1);
    let raf = 0, visible = false, disposed = false, broken = false, dirty = true, finishDirty = false;
    let pointerSamples: { x: number; y: number; time: number }[] = [];
    let pointerLeft = false;
    let last = 0, painted = -Infinity, time = 12, renderedVersion: number | undefined;
    const options = { ...DEFAULT_OPTIONS, fontSize: 7, charAspect: .58, charset: ' .:+-=xICA$FY#@',
      colorMode: config.colorMode ?? 'fullcolor', accentColor: config.accentColor ?? '#e8b900', normalize: false, contrast: .1, hoverStrength: 0, chromaKey: null };
    const resting = () => config.isPaused?.() || reduced.matches;
    const tick = (now: number) => {
      raf = 0;
      if (disposed || broken || !visible || document.hidden) return;
      const requestedStyle = config.getStyle?.() ?? 'ascii';
      if(requestedStyle === 'ascii' && studio){studio.destroy();studio=undefined;studioLoading=false;}
      if(requestedStyle !== 'ascii' && !studio && !studioLoading) {
        studioLoading = true;
        void import('./hero-studio-renderer').then(module => {
          if(disposed) return;
          studio = module.createHeroStudioRenderer(target); dirty = true; wake();
        }).catch(() => { if(!disposed) config.onError?.(); });
      }
      const alternate = requestedStyle !== 'ascii' && !!studio;
      const dt = last ? Math.min(.05, (now - last) / 1000) : 1 / 60; last = now;
      if (!resting()) time += dt;
      if (field instanceof WaterSurface) {
        const hover = config.getHover?.() ?? 'trail';
        field.configure(hover, ['contour', 'dissolve', 'silk', 'vortex'].includes(hover) ? .7 : .55, config.getHoverRadius?.() ?? (hover === 'trail' ? .45 : .2), config.getEdgeSafe?.() ?? false);
      }
      if (pointerSamples.length) {
        const rect = host.getBoundingClientRect();
        if (rect.width && rect.height) for (const point of pointerSamples) {
          field.move((point.x - rect.left) / rect.width, (point.y - rect.top) / rect.height, point.time);
          if(alternate) studio?.pointer((point.x-rect.left)/rect.width,(point.y-rect.top)/rect.height,point.time);
        }
        pointerSamples = [];
      }
      if(pointerLeft) { field.leave();studio?.leave();pointerLeft=false; }
      const wasMoving = field.active;
      field.step(dt);
      // Reuse the media frame between decoder updates; input still renders at display refresh.
      const version = config.sourceVersion?.();
      const interval = 1000 / (config.fps ?? 30);
      const sourceChanged = config.sourceVersion ? version !== renderedVersion : now - painted >= interval - .5;
      const continuityDue = !alternate && !!continuity?.active && now - continuityAt >= 1000 / 24;
      if (dirty || finishDirty || wasMoving || field.active || sourceChanged || continuityDue || (!alternate && gpu?.transitioning) || (alternate && studio?.active)) {
        try {
          const water = field instanceof WaterSurface ? field : undefined;
          // Cursor wakes reuse the converted grid. Only decoder
          // updates/settings changes need source sampling, conversion and packing.
          if (!water || !gpu || gpuFailed || dirty || sourceChanged || !cachedFrame) {
            (config.paintSource ?? paintLiquidSource)(pixels.data, source.width, source.height, aspect, time, field);
            if (!config.paintSource) tintMediaSource(pixels.data, source.width, source.height);
            if (water && (!gpu || gpuFailed)) {
              fallbackSource.set(pixels.data);
              paintMediaSource(pixels.data, fallbackSource, source.width, source.height, field, .018);
            }
            flattenSourceOnBlack(pixels.data);
            sourceCtx.putImageData(pixels, 0, 0);
            if(!alternate) {
            const charset = config.getCharset?.() ?? options.charset;
            options.charset = palette?.get(charset) ?? charset;
            cachedFrame = imageToAsciiFrame(source, options, width, height).frame;
            continuity?.setTarget(cachedFrame, options.charset);
            }
          }
          if(!alternate) {
          if (continuity && (dirty || sourceChanged || continuityDue || !displayFrame)) {
            displayFrame = continuity.step(continuityAt ? (now - continuityAt) / 1000 : 1 / 24, resting());
            continuityAt = now;
          }
          const frame = continuity ? displayFrame! : cachedFrame!;
          if(directGpu && !gpuFailed){gpuCanvas.style.display='';target.style.visibility='hidden';}
          ctx.setTransform(target.width/width,0,0,target.height/height,0,0);
          if (gpu && !gpuFailed) {
            try { gpu.render(frame, options, width, height, water?.refraction, resting(), config.getTextMask?.(), config.getFinish?.()); }
            catch { gpuFailed = true; gpu.destroy(); }
            if (!gpuFailed) {
              if (!directGpu) {
                ctx.clearRect(0, 0, width, height);
                ctx.drawImage(gpuCanvas, 0, 0, width, height);
              }
              target.dataset.renderer = 'gpu';
            }
          }
          if (!gpu || gpuFailed) {
            if (directGpu) { gpuCanvas.style.display = 'none'; target.style.visibility = previousVisibility; }
            if (canvasGlyphs) canvasGlyphs.render(ctx, frame, options, width, height, config.getTextMask?.());
            else renderFrameToCanvas(ctx, frame, options, width, height, time, null);
            target.dataset.renderer = 'canvas';
            target.style.opacity = '1';
          }
          } else {
            gpuCanvas.style.display='none';target.style.visibility=previousVisibility;target.style.opacity='1';
            studio!.render(source,time,width,height,requestedStyle,config.getHover?.()??'trail',config.getHoverRadius?.()??.45,config.getEdgeSafe?.()??false,config.getFinish?.()??{},config.getTextMask?.(),!!resting(),dirty || sourceChanged);
            target.dataset.renderer='studio';
          }
          target.dataset.style = alternate ? requestedStyle : 'ascii';
          textureCtx?.drawImage(!alternate && directGpu && !gpuFailed ? gpuCanvas : target, 0, 0);
          painted = dirty || !Number.isFinite(painted) ? now : now - (now - painted) % interval;
          dirty = false; finishDirty = false; renderedVersion = version;
        } catch { broken = true; config.onError?.(); }
      }
      if (!resting() && !broken) raf = requestAnimationFrame(tick);
    };
    const wake = () => { if (!raf && visible && !disposed && !broken && !document.hidden) raf = requestAnimationFrame(tick); };
    onGpuFailure = () => { dirty = true; wake(); };
    const sync = () => { config.onMotion?.(visible && !document.hidden && !resting()); last = 0; dirty = true; if (resting() || document.hidden) { field.clear(); pointerSamples = []; } cancelAnimationFrame(raf); raf = 0; wake(); };
    const resize = new ResizeObserver(() => {
      const rect = host.getBoundingClientRect(); width = Math.max(1, Math.round(rect.width)); height = Math.max(1, Math.round(rect.height)); aspect = width / height;
      // Bound both draw work and backing-store memory on very large displays.
      const fontSize = typeof config.fontSize === 'function' ? config.fontSize(width) : config.fontSize;
      options.fontSize = Math.max(fontSize ?? 7, Math.sqrt(width * height * options.charAspect / (config.maxGlyphs ?? 26000)));
      const dpr = Math.min(1.5, devicePixelRatio || 1, Math.sqrt(4_000_000 / (width * height)));
      target.width = Math.round(width * dpr); target.height = Math.round(height * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (gpu && !gpuFailed) {
        try { gpu.resize(width, height, dpr); } catch { gpuFailed = true; gpu.destroy(); }
      }
      if (textureCanvas) { textureCanvas.width = target.width; textureCanvas.height = target.height; }
      const detail = Math.max(192, Math.min(384, Math.ceil(Math.max(width, height * options.charAspect) / (options.fontSize * options.charSpacing))));
      source.width = Math.max(1, Math.round(aspect >= 1 ? detail : detail * aspect)); source.height = Math.max(1, Math.round(aspect >= 1 ? detail / aspect : detail));
      pixels = sourceCtx.createImageData(source.width, source.height); fallbackSource = new Uint8ClampedArray(pixels.data.length);
      field = config.waterSurface ? new WaterSurface(aspect) : new FluidField(aspect, config.fieldDecay, config.continuousWake);
      cachedFrame = undefined; displayFrame = undefined; continuityAt = 0; continuity?.clear(); dirty = true; wake();
    });
    const observer = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting && entry.intersectionRatio > .05; if (visible) sync(); else { config.onMotion?.(false); cancelAnimationFrame(raf); raf = 0; field.clear(); pointerSamples = []; } }, { rootMargin: '-56px 0px 0px 0px', threshold: .05 });
    const move = (event: PointerEvent) => {
      if (resting()) return;
      if (event.target instanceof Element && event.target.closest('button, a, input, select, summary, textarea')) { leave(); return; }
      pointerLeft=false;
      pointerSamples.push({ x: event.clientX, y: event.clientY, time: event.timeStamp });
      if (pointerSamples.length > 6) pointerSamples.shift();
      wake();
    };
    const leave = () => { pointerLeft=true;wake(); };
    const preview = () => {
      if (resting()) return;
      for (let i = 0; i < 10; i++) {
        const x=.3+i*.04,y=(config.previewY?.(aspect)??.57)+Math.sin(i*.4)*.08;
        field.move(x,y);if(config.getStyle?.()!=='ascii')studio?.pointer(x,y,performance.now()+i*16);
      }
      studio?.leave();
      field.leave(); wake();
    };
    const fontReady = () => { palette?.reset(); canvasGlyphs?.reset(); if (!disposed) sync(); };
    document.fonts?.addEventListener('loadingdone', fontReady);
    void document.fonts?.ready.then(fontReady);
    resize.observe(host); observer.observe(host);
    host.addEventListener('pointermove', move); host.addEventListener('pointerdown', move);
    host.addEventListener('pointerleave', leave); host.addEventListener('pointercancel', leave);
    document.addEventListener('visibilitychange', sync); reduced.addEventListener('change', sync);
    const destroy = () => { disposed = true; studio?.destroy(); continuity?.clear(); canvasGlyphs?.destroy(); gpu?.destroy(); config.onMotion?.(false); cancelAnimationFrame(raf); resize.disconnect(); observer.disconnect();
      host.removeEventListener('pointermove', move); host.removeEventListener('pointerdown', move); host.removeEventListener('pointerleave', leave); host.removeEventListener('pointercancel', leave);
      document.fonts?.removeEventListener('loadingdone', fontReady);
      document.removeEventListener('visibilitychange', sync); reduced.removeEventListener('change', sync);
      if (directGpu) { gpuCanvas.remove(); target.style.visibility = previousVisibility; }
      target.style.opacity = previousOpacity;
    };
    const redraw = () => { finishDirty = true; wake(); };
    return { sync, preview, redraw, destroy };
}
