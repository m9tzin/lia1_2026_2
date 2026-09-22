import { createStudioRenderer, type StudioSettings } from 'asciify-engine/studio';
import type { SurfaceFinish } from './surface-finish';
import type { TextMaskFrame } from './text-mask';
import type { HeroHover } from './water-surface';
/** Loaded only after a non-ASCII style is requested. Shares the hero's decoded media and RAF. */
export function createHeroStudioRenderer(target: HTMLCanvasElement) {
 const artwork=document.createElement('canvas');
 const renderer=createStudioRenderer(artwork,{}, {maxDimension:2560,maxCells:18000});
 const ctx=target.getContext('2d')!;
 let key='';
 return {
  render(source:HTMLCanvasElement,time:number,width:number,_height:number,style:StudioSettings['style'],hover:HeroHover,radius:number,edgeSafe:boolean,finish:SurfaceFinish,mask?:TextMaskFrame,paused=false,sourceChanged=true) {
   const strength=finish.edgeEffect??.95;
   const settings={style,cellSize:6,colorMode:'accent',ink:'#888888',backdrop:{mode:'transparent'},hover:{effect:paused?'none':hover,radius,strength:.55,edgeSafe},dither:{scale:3,palette:'mono',colors:['#080808','#888888']},effects:{prism:finish.filter==='prism'?strength:0,scanlines:finish.filter==='etch'?strength*.45:0,glitch:finish.filter==='signal'?strength*.35:0}};
   const next=JSON.stringify(settings);if(next!==key){key=next;renderer.configure(settings);}
   renderer.setPixelRatio(target.width/width);
   if(sourceChanged) renderer.invalidate(); // Cursor-only frames reuse the sampled source.
   renderer.render(source,time,target.width,target.height);
   ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,target.width,target.height);
   ctx.drawImage(artwork,0,0,target.width,target.height);
   // Filled styles must not paint over the real HTML headline.
   if(mask){ctx.globalCompositeOperation='destination-out';ctx.drawImage(mask.canvas,0,0,target.width,target.height);ctx.globalCompositeOperation='source-over';}
  },
  pointer:renderer.pointer,leave:renderer.leave,
  get active(){return renderer.active;},
  destroy:()=>{renderer.destroy();artwork.width=artwork.height=1;},
 };
}
