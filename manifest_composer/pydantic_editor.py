"""
pydantic_editor.py — windowed Pydantic model editor.
Usage: from pydantic_editor import up_server; up_server(MyModel, agent_manifest_mode=True)
Requires: pip install fastapi uvicorn pydantic
"""
import json, threading, webbrowser
from typing import Any, Type
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Placeholders replaced at runtime: __SCHEMA__  __MODEL_NAME__  __MODEL_LOWER__ __AGENT_MANIFEST_MODE__ __ACC__ __ACC2__ __ACC3__
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no"/>
<title>Pydantic Editor</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#080b10;--bg2:#0c1018;--sur:#0f1420;--sur2:#141923;--sur3:#1a2130;
  --bor:#1e2a3a;--bor2:#253040;--acc:__ACC__;--acc2:__ACC2__;--acc3:__ACC3__;
  --tx:#c8d8e8;--tx2:#8899aa;--tx3:#566273;
  --ok:#00c896;--err:#ff5252;--warn:#ffab40;
  --r:6px;--r2:8px;--mono:'JetBrains Mono',monospace;--sans:'Inter',sans-serif;
  --tbar:50px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font-family:var(--sans);font-size:13px;
  height:100vh;overflow:hidden;user-select:none;
  background-image:radial-gradient(ellipse 80% 60% at 15% 10%,rgba(230,168,0,.035) 0%,transparent 55%),
    radial-gradient(ellipse 60% 80% at 85% 90%,rgba(255,107,53,.03) 0%,transparent 55%);}
@media (max-width: 780px){
  body{font-size:14px;}
}

/* ── App layout ── */
#app{display:flex;height:calc(100vh - var(--tbar));overflow:hidden;position:relative;}

/* ── Shared panel header ── */
.p-hdr{height:38px;padding:0 14px;display:flex;align-items:center;gap:10px;
  border-bottom:1px solid var(--bor);flex-shrink:0;background:var(--sur2);}
.p-lbl{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--tx2);}
.p-hdr-spacer{flex:1;}
.p-hdr-btn{display:inline-flex;align-items:center;gap:3px;padding:3px 8px;
  background:transparent;color:var(--tx2);border:1px solid var(--bor2);border-radius:var(--r);
  font-size:10px;font-weight:500;cursor:pointer;font-family:var(--sans);transition:all .15s;}
.p-hdr-btn:hover{color:var(--tx);background:var(--sur3);}
.p-collapse-btn{width:28px;height:24px;display:flex;align-items:center;justify-content:center;
  background:var(--sur3);border:1px solid var(--bor2);color:var(--tx2);border-radius:var(--r);cursor:pointer;font-size:14px;flex-shrink:0;}
.p-collapse-btn:hover{color:var(--tx);border-color:var(--acc);}

/* ── Editor panel ── */
#editor-panel{width:380px;min-width:180px;border-right:1px solid var(--bor);
  display:flex;flex-direction:column;flex-shrink:0;position:relative;background:var(--sur);
  transition:transform .22s ease, width .22s ease, opacity .22s ease; z-index:50;}
#editor-panel.collapsed{transform:translateX(-100%); width:0 !important; min-width:0; opacity:0; pointer-events:none; border-right:none;}
#editor-scroll{flex:1;overflow-y:auto;padding:10px;}
#e-resize{position:absolute;right:-3px;top:0;bottom:0;width:6px;cursor:col-resize;
  z-index:50;background:transparent;transition:background .15s;}
#e-resize:hover,#e-resize.active{background:rgba(230,168,0,.3);}

/* Agent tabs */
#agent-tabs-wrap{display:none; border-bottom:1px solid var(--bor); background:var(--sur2); flex-shrink:0;}
#agent-tabs-wrap.on{display:block;}
#agent-tabs{display:flex;overflow-x:auto;gap:4px;padding:6px 8px;scrollbar-width:none; touch-action: pan-x;}
#agent-tabs::-webkit-scrollbar{display:none;}
.agent-tab{flex-shrink:0;padding:5px 12px;border:1px solid var(--bor2);border-radius:20px;
  background:var(--sur3);color:var(--tx2);font-family:var(--mono);font-size:11px;cursor:pointer;transition:all .15s;
  max-width:170px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; position:relative;}
.agent-tab:hover{color:var(--tx);border-color:var(--acc2);}
.agent-tab.active{background:rgba(230,168,0,.12); color:var(--acc); border-color:rgba(230,168,0,.38);}
.agent-tab .tab-crumb{display:block;font-size:9px;color:var(--tx3);opacity:.9;}
.agent-tab.active .tab-crumb{color:var(--acc2);opacity:.85;}

/* ── Desktop ── */
#desktop-viewport{flex:1;position:relative;overflow:hidden;background:var(--bg);
/* infinite dots - positioned via JS */
  background-image: radial-gradient(circle, #1e2a3a 0 7%, transparent 7%);
  background-size: 24px 24px;
  touch-action: none;
}
#desktop{position:absolute;left:0;top:0;width:100%;height:100%;
  transform-origin:0 0; background: transparent;}
#desktop.panning{cursor:grabbing;}
#desktop-hint{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  text-align:center;pointer-events:none;color:var(--tx3);font-size:12px;line-height:2.2;z-index:0;}
#desktop-hint.hidden{display:none;}
#wires-svg{position:absolute;inset:0;width:100%;height:100%;
  pointer-events:none;z-index:1;overflow:visible;}
#zoom-hud{position:absolute;left:10px;bottom:10px;z-index:200;display:flex;align-items:center;gap:4px;
  background:rgba(15,20,32,.85);border:1px solid var(--bor2);border-radius:var(--r);padding:4px;
  backdrop-filter:blur(6px);}
.zoom-btn{width:22px;height:22px;display:flex;align-items:center;justify-content:center;
  background:var(--sur3);border:1px solid var(--bor2);color:var(--tx2);border-radius:4px;
  cursor:pointer;font-size:13px;font-family:var(--mono);transition:all .15s; user-select:none;}
.zoom-btn:hover{color:var(--tx);background:var(--sur2);}
.zoom-pct{font-family:var(--mono);font-size:10px;color:var(--tx3);width:38px;text-align:center;}
#dz-target{position:absolute;width:1px;height:1px;pointer-events:none;opacity:0;z-index:999;
  border:2px dashed var(--acc);border-radius:var(--r2);background:rgba(230,168,0,.06);
  transition:opacity .12s;}
#dz-target.show{opacity:1;}
#canvas-dim{position:absolute;inset:0;background:rgba(0,0,0,.45);z-index:40;display:none;pointer-events:auto;}
#canvas-dim.on{display:block;}

/* ── Output panel ── */
#o-resize{width:6px;cursor:col-resize;flex-shrink:0;background:transparent;
  border-left:1px solid var(--bor);transition:background .15s;position:relative; z-index:50;}
#o-resize:hover,#o-resize.active{background:rgba(230,168,0,.3);}
#output-panel{width:360px;min-width:180px;display:flex;flex-direction:column;
  flex-shrink:0;background:var(--sur); transition:transform .22s ease, width .22s ease, opacity .22s ease; z-index:50;}
#output-panel.collapsed{transform:translateX(100%); width:0 !important; min-width:0; opacity:0; pointer-events:none;}
#output-scroll{flex:1;overflow-y:auto;padding:14px;background:var(--bg2);}
#output-foot{padding:10px 12px;border-top:1px solid var(--bor);display:flex;gap:8px;
  background:var(--sur2);flex-shrink:0;}
#output-foot .btn-g{flex:1;justify-content:center;}

/* --- Output panel and Editor panel */
#editor-panel.resizing, #output-panel.resizing{transition:none !important;}

/* Mobile overlay panels */
@media (max-width: 780px){
  #editor-panel, #output-panel{
    position:absolute; top:0; bottom:var(--tbar); height:auto; width:92vw !important; max-width:420px;
    box-shadow: 8px 0 40px rgba(0,0,0,.6); z-index:200;
  }
  #editor-panel{left:0; border-right:1px solid var(--bor);}
  #output-panel{right:0; left:auto; border-left:1px solid var(--bor);}
  #e-resize, #o-resize{display:none;}
  #editor-panel.collapsed{transform:translateX(-105%);}
  #output-panel.collapsed{transform:translateX(105%);}
}

.fob{position:absolute; top:0; z-index:120; display:none; padding:7px 14px; background:rgba(15,20,32,.92); border:1px solid var(--bor2); color:var(--tx2);
  font-size:11px; cursor:pointer; backdrop-filter:blur(6px);}
.fob.show{display:block;}
.fob:hover{color:var(--tx); border-color:var(--acc);}
#open-editor-btn{left:0; border-radius:0 0 8px 0; border-left:none; border-top:none;}
#open-output-btn{right:0; border-radius:0 0 0 8px; border-right:none; border-top:none;}

/* ── Taskbar ── */
#taskbar{position:fixed;bottom:0;left:0;right:0;height:var(--tbar);
  background:rgba(8,11,16,.94);border-top:1px solid var(--bor);backdrop-filter:blur(18px);
  display:flex;align-items:center;padding:0 14px;gap:8px;z-index:9000; overflow-x:auto;}
.tb-brand{font-family:var(--mono);font-size:11px;font-weight:500;color:var(--acc);
  letter-spacing:.1em;text-transform:uppercase;padding-right:14px;border-right:1px solid var(--bor2);
  white-space:nowrap;flex-shrink:0;}
.tb-badge{font-family:var(--mono);font-size:11px;color:var(--acc2);
  background:rgba(255,107,53,.09);border:1px solid rgba(255,107,53,.24);
  padding:2px 10px;border-radius:20px;white-space:nowrap;flex-shrink:0;}
.tb-sep{width:1px;height:22px;background:var(--bor2);flex-shrink:0;}
.tb-right{margin-left:auto;display:flex;gap:8px;align-items:center;flex-shrink:0;}
.btn-val{padding:7px 18px;background:linear-gradient(135deg,var(--acc2),var(--acc));
  color:#151008;border:none;border-radius:var(--r);font-size:12px;font-weight:600;
  cursor:pointer;font-family:var(--sans);transition:opacity .15s,transform .1s;}
.btn-val:hover{opacity:.88;}.btn-val:active{transform:scale(.97);}
.btn-g{display:inline-flex;align-items:center;gap:5px;padding:6px 11px;
  background:transparent;color:var(--tx2);border:1px solid var(--bor2);border-radius:var(--r);
  font-size:11px;font-weight:500;cursor:pointer;font-family:var(--sans);transition:all .15s;}
.btn-g:hover{color:var(--tx);border-color:var(--bor2);background:var(--sur3);}

/* ── Model sections (editor) ── */
.m-sec{border:1px solid var(--bor);border-radius:var(--r);margin-bottom:8px;overflow:hidden;
  transition:border-color .15s;}
.m-sec.open{border-color:var(--bor2);}
.m-sec-hdr{padding:8px 10px;background:var(--sur2);display:flex;align-items:center;
  gap:7px;cursor:pointer;transition:background .15s;border-left:3px solid var(--acc2);}
.m-sec-hdr:hover{background:var(--sur3);}
.m-sec-hdr[draggable="true"]{cursor:grab;}
.m-sec-hdr.dragging{opacity:.4;}
.m-arr{width:14px;height:14px;color:var(--tx3);transition:transform .22s;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;}
.m-sec.open .m-arr{transform:rotate(90deg);}
.m-icon{width:18px;height:18px;background:rgba(230,168,0,.08);border:1px solid rgba(230,168,0,.20);
  border-radius:4px;display:flex;align-items:center;justify-content:center;flex-shrink:0; color:var(--acc);}
.m-name{font-family:var(--mono);font-size:11.5px;font-weight:500;color:var(--acc);flex:1;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.m-cnt{font-size:10px;color:var(--tx3);background:var(--sur3);padding:1px 6px;
  border-radius:20px;font-family:var(--mono);flex-shrink:0;}
.m-desc-collapsed{padding:5px 10px 6px 32px;font-size:10.5px;color:var(--tx3);
  background:var(--sur2);border-bottom:1px solid var(--bor);font-style:italic;}
.m-sec-body{display:block; max-height:0; opacity:0; overflow:hidden;
  background:var(--bg2); border-top:1px solid var(--bor);
  transition:max-height .28s ease, opacity .22s ease;}
.m-sec.open .m-sec-body{max-height:5000px; opacity:1;}
.m-desc-expanded{padding:7px 12px;font-size:10.5px;color:var(--tx3);
  border-bottom:1px solid var(--bor);font-style:italic;}
.pop-btn{width:20px;height:20px;background:rgba(230,168,0,.07);
  border:1px solid rgba(230,168,0,.22);color:var(--acc);border-radius:4px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
  transition:background .15s;margin-left:2px; font-size:13px;}
.pop-btn:hover{background:rgba(230,168,0,.2);}
.nav-parent-btn{width:20px;height:20px;background:rgba(255,107,53,.08);border:1px solid rgba(255,107,53,.22);
  color:var(--acc2);border-radius:4px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:11px;}
.nav-parent-btn:hover{background:rgba(255,107,53,.18);}
.nav-find-btn{width:20px;height:20px;background:rgba(0,200,150,.06);border:1px solid rgba(0,200,150,.18);
  color:var(--acc3);border-radius:4px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:10px;}
.nav-find-btn:hover{background:rgba(0,200,150,.16);}

/* flash */
.flash-sec{box-shadow:0 0 0 2px var(--acc) !important;}
.flash-field{box-shadow:inset 0 0 0 2px var(--acc) !important; background:rgba(230,168,0,.06) !important; border-radius:4px;}
.flash-list-item{outline:2px solid var(--acc3) !important; outline-offset:1px; border-radius:6px;}

/* ── Field rows ── */
.f-row{display:flex;align-items:flex-start;padding:7px 10px;gap:8px;
  border-bottom:1px solid var(--bor);transition:background .1s;}
.f-row:last-child{border-bottom:none;}
.f-row:hover{background:rgba(255,255,255,.018);}
.f-meta{flex:1;min-width:0;padding-top:2px;}
.f-nm{font-family:var(--mono);font-size:11px;color:var(--tx);
  display:flex;align-items:center;gap:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.f-req{color:var(--warn);font-size:10px;flex-shrink:0;}
.f-tp{font-family:var(--mono);font-size:10px;color:var(--acc2);margin-top:2px;opacity:.72;}
.f-ds{font-size:10px;color:var(--tx3);margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.f-ctl{flex-shrink:0;display:flex;align-items:center;gap:5px;flex-wrap:wrap;max-width:170px;justify-content:flex-end;}

/* ── Form controls ── */
.fi{background:var(--sur3);border:1px solid var(--bor2);color:var(--tx);border-radius:var(--r);
  padding:4px 7px;font-size:11.5px;font-family:var(--mono);width:110px;
  transition:border-color .15s,box-shadow .15s;outline:none;}
.fi:focus{border-color:var(--acc);box-shadow:0 0 0 2px rgba(230,168,0,.13);}
.fi.wide{width:130px;}
.fi:disabled{opacity:.45;cursor:not-allowed;}
.fsel{background:var(--sur3);border:1px solid var(--bor2);color:var(--tx);border-radius:var(--r);
  padding:4px 7px;font-size:11.5px;max-width:150px;cursor:pointer;outline:none;
  appearance:none; -webkit-appearance:none;
  background-image: linear-gradient(45deg, transparent 50%, var(--tx3) 50%), linear-gradient(135deg, var(--tx3) 50%, transparent 50%);
  background-position: calc(100% - 12px) calc(50% - 2px), calc(100% - 7px) calc(50% - 2px);
  background-size:5px 5px, 5px 5px; background-repeat:no-repeat; padding-right:22px;}
.fsel:focus{border-color:var(--acc);}
.fsel:disabled{opacity:.45;cursor:not-allowed;}
.ftog{width:34px;height:18px;background:var(--sur3);border:1px solid var(--bor2);
  border-radius:20px;position:relative;cursor:pointer;
  transition:background .15s,border-color .15s;flex-shrink:0;}
.ftog.on{background:var(--acc);border-color:var(--acc);}
.ftog::after{content:'';position:absolute;top:2px;left:2px;width:12px;height:12px;
  background:var(--tx3);border-radius:50%;transition:left .15s,background .15s;}
.ftog.on::after{left:18px;background:#1a1200;}
.ref-chip{display:inline-flex;align-items:center;gap:3px;background:rgba(255,107,53,.08);
  border:1px solid rgba(255,107,53,.25);color:var(--acc2);font-family:var(--mono);
  font-size:10px;padding:2px 8px;border-radius:20px;cursor:pointer;transition:background .15s;
  white-space:nowrap;}
.ref-chip:hover{background:rgba(255,107,53,.18);}
.ref-chip.disabled{opacity:.4;cursor:default;}
.ref-chip.disabled:hover{background:rgba(255,107,53,.08);}
.ref-chip.agent-chip{background:rgba(0,200,150,.07); border-color:rgba(0,200,150,.25); color:var(--acc3);}

.ml-edit-btn{width:22px;height:22px;background:var(--sur3);border:1px solid var(--bor2);color:var(--tx2);
  border-radius:4px;cursor:pointer;font-size:11px;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.ml-edit-btn:hover{color:var(--acc); border-color:var(--acc);}

/* ── List controls ── */
.list-wrap{width:100%;}
.list-items{display:flex;flex-direction:column;gap:4px;margin-bottom:5px;}
.list-item{display:flex;align-items:center;gap:5px;}
.list-item .fi{flex:1;width:auto;min-width:0;}
.list-item .ref-chip{flex:1;}
.li-del{width:20px;height:20px;background:rgba(255,82,82,.1);
  border:1px solid rgba(255,82,82,.2);color:var(--err);border-radius:4px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;font-size:15px;line-height:1;
  transition:background .15s;flex-shrink:0;}
.li-del:hover{background:rgba(255,82,82,.22);}
.li-add{display:inline-flex;align-items:center;gap:4px;font-size:10.5px;color:var(--acc3);
  cursor:pointer;background:none;border:none;font-family:var(--sans);padding:2px 0;
  transition:opacity .15s;}
.li-add:hover{opacity:.72;}
.null-tog-row{display:flex;align-items:center;gap:6px;width:100%;}
.null-tog-row .fi,.null-tog-row .fsel{flex:1;width:auto;min-width:0;}

/* ── Dict control ── */
.kv-item{display:flex;align-items:center;gap:4px;margin-bottom:4px;}
.kv-item .fi{width:60px;flex:1;}

/* ── Desktop windows ── */
.dwin{position:absolute;display:flex;flex-direction:column;background:var(--sur);
  border:1px solid var(--bor);border-radius:var(--r2);
  box-shadow:0 16px 48px rgba(0,0,0,.6),0 0 0 1px var(--bor);
  min-width:240px;min-height:100px;z-index:10; touch-action: none;
  transition: box-shadow .18s, border-color .18s;}
.dwin.dwin-focused{border-color:rgba(230,168,0,.35);
  box-shadow:0 16px 48px rgba(0,0,0,.7),0 0 0 1px rgba(230,168,0,.22),0 0 24px rgba(230,168,0,.07);}
.dwin.dwin-root{border-color:rgba(0,200,150,.35);}
.dwin.dwin-root .dwin-hdr{border-left-color:var(--acc3);}
.dwin.dwin-root .dwin-name{color:var(--acc3);}
.dwin-hdr{height:34px;padding:0 10px;background:var(--sur2);
  border-bottom:1px solid var(--bor);border-radius:var(--r2) var(--r2) 0 0;
  display:flex;align-items:center;gap:7px;cursor:move;flex-shrink:0;
  border-left:3px solid var(--acc); touch-action: none;}
.dwin-name{font-family:var(--mono);font-size:11.5px;font-weight:500;color:var(--acc);
  flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.dwin-ctrls{display:flex;gap:4px;flex-shrink:0;}
.dwc{width:15px;height:15px;border-radius:3px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;font-size:9px;
  transition:all .15s;flex-shrink:0;}
.dwc-m{background:rgba(254,188,46,.1);color:#febc2e;border:1px solid rgba(254,188,46,.25);}
.dwc-m:hover{background:rgba(254,188,46,.25);}
.dwc-x{background:rgba(255,95,87,.1);color:#ff5f57;border:1px solid rgba(255,95,87,.2);}
.dwc-x:hover{background:rgba(255,95,87,.25);}
.dwin-body{flex:1;overflow-y:auto;background:var(--bg2); overscroll-behavior: contain; -webkit-overflow-scrolling:touch; touch-action: pan-y;
  max-height:2000px; opacity:1; transition: max-height .26s ease, opacity .2s ease;}
.dwin.dwin-min .dwin-body{max-height:0 !important; opacity:0; overflow:hidden;}
.dwin.dwin-min{height:auto !important;min-height:0;}
.dwin.dwin-min .win-rsz{display:none;}
.dwin.dwin-min .dwin-hdr{border-radius:var(--r2);border-bottom:none;}
.win-rsz{position:absolute;right:0;bottom:0;width:16px;height:16px;cursor:nwse-resize;z-index:5; touch-action:none;}
.win-rsz::after{content:'';position:absolute;right:3px;bottom:3px;width:7px;height:7px;
  border-right:2px solid var(--bor2);border-bottom:2px solid var(--bor2);border-radius:1px;}

/* ── JSON output ── */
#vmsg{padding:7px 10px;border-radius:var(--r);font-size:11px;font-family:var(--mono);
  margin-bottom:10px;display:none;white-space:pre-wrap;line-height:1.5;}
#vmsg.ok{background:rgba(0,200,150,.07);color:var(--ok);border:1px solid rgba(0,200,150,.2);}
#vmsg.err{background:rgba(255,82,82,.07);color:var(--err);border:1px solid rgba(255,82,82,.2);}
#jout{font-family:var(--mono);font-size:11.5px;color:var(--acc3);line-height:1.7;white-space:pre;}

/* ── Multiline editor / generic modal ── */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.56);backdrop-filter:blur(4px);z-index:9998;display:none;align-items:center;justify-content:center;}
.modal-overlay.on{display:flex;}
.modal-box{background:var(--sur);border:1px solid var(--bor2);border-radius:10px;box-shadow:0 24px 80px rgba(0,0,0,.7);
  width:min(680px,92vw); display:flex;flex-direction:column; position:relative; overflow:hidden;}
#ml-box{height:min(520px,78vh); min-width:300px; min-height:220px; resize:both;}
@media (max-width:600px){
  #ml-box{width:96vw; height:72vh; resize:none;}
}
.modal-hdr{display:flex;align-items:center;gap:8px;padding:10px 12px;border-bottom:1px solid var(--bor);background:var(--sur2);flex-shrink:0;}
.modal-title{font-family:var(--mono);font-size:11px;color:var(--acc2);flex:1;}
.modal-close{background:transparent;border:1px solid var(--bor2);color:var(--tx2);border-radius:6px;width:24px;height:24px;cursor:pointer;}
.modal-close:hover{color:var(--tx); background:var(--sur3);}
#ml-text{flex:1;width:100%;background:var(--bg2);color:var(--tx);border:none;outline:none;padding:12px;font-family:var(--mono);font-size:12.5px;resize:none;line-height:1.6;}
.modal-foot{display:flex;justify-content:flex-end;gap:8px;padding:10px 12px;border-top:1px solid var(--bor);background:var(--sur2);}
#ml-resize-handle{position:absolute;right:4px;bottom:4px;width:14px;height:14px;cursor:nwse-resize;opacity:.6;}
#ml-resize-handle::after{content:'';position:absolute;right:2px;bottom:2px;width:8px;height:8px;border-right:2px solid var(--tx3);border-bottom:2px solid var(--tx3);border-radius:1px;}

/* child picker */
#child-picker-box{width:min(520px,92vw); max-height:70vh;}
#child-picker-list{padding:10px; overflow-y:auto; max-height:52vh; display:flex; flex-direction:column; gap:6px;}
.child-picker-item{padding:9px 12px; border:1px solid var(--bor2); border-radius:6px; background:var(--sur3); cursor:pointer; font-family:var(--mono); font-size:11.5px; color:var(--tx); transition:all .12s;}
.child-picker-item:hover{background:var(--sur2); border-color:var(--acc); color:var(--acc);}
.child-picker-item small{color:var(--tx3); font-family:var(--sans); margin-left:6px;}

/* ── Toast ── */
#toast{position:fixed;bottom:calc(var(--tbar) + 14px);right:20px;background:var(--ok);color:#05100c;
  padding:7px 14px;border-radius:var(--r);font-size:12px;font-weight:600;
  display:flex;align-items:center;gap:6px;transform:translateY(80px);opacity:0;
  transition:all .22s ease;pointer-events:none;z-index:9999;}
#toast.show{transform:translateY(0);opacity:1;}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--bor2);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--tx3);}
</style>
</head>
<body>
<div id="app">

  <div id="editor-panel">
    <div class="p-hdr">
      <svg width="10" height="10" viewBox="0 0 10 10" fill="none" style="flex-shrink:0"><polygon points="5,1 9,3.5 9,6.5 5,9 1,6.5 1,3.5" stroke="var(--acc2)" stroke-width="1.3"/></svg>
      <span class="p-lbl">Editor</span>
      <div class="p-hdr-spacer"></div>
      <button class="p-hdr-btn" id="expand-all-btn" title="Expand all sections">&#x2913; All</button>
      <button class="p-hdr-btn" id="collapse-all-btn" title="Collapse all sections">&#x2911; All</button>
      <button class="p-collapse-btn" id="editor-collapse-btn" title="Collapse Editor">⟨</button>
    </div>
    <div id="agent-tabs-wrap"><div id="agent-tabs"></div></div>
    <div id="editor-scroll"></div>
    <div id="e-resize"></div>
  </div>

  <div id="desktop-viewport">
    <div id="canvas-dim"></div>
    <button class="fob" id="open-editor-btn">☰ Editor</button>
    <button class="fob" id="open-output-btn">{} JSON</button>
    <div id="desktop">
      <svg id="wires-svg"></svg>
      <div id="desktop-hint">
        &#x2190; Click &#x2197; or drag any model section here<br>
        <span style="font-size:10px;color:var(--tx3)">Connections between related models will appear as wires</span>
      </div>
    </div>
    <div id="dz-target"></div>
    <div id="zoom-hud">
      <div class="zoom-btn" id="zoom-out-btn" title="Zoom out (-)">−</div>
      <div class="zoom-pct" id="zoom-pct">100%</div>
      <div class="zoom-btn" id="zoom-in-btn" title="Zoom in (+)">+</div>
      <div class="zoom-btn" id="zoom-reset-btn" title="Reset view / Ctrl+Space">&#x2302;</div>
    </div>
  </div>

  <div id="o-resize"></div>

  <div id="output-panel">
    <div class="p-hdr">
      <button class="p-collapse-btn" id="output-collapse-btn" title="Collapse JSON">⟩</button>
      <svg width="10" height="10" viewBox="0 0 10 10" fill="none" style="flex-shrink:0"><rect x="1" y="1" width="8" height="8" rx="1.5" stroke="var(--acc3)" stroke-width="1.3"/><line x1="3" y1="3.5" x2="7" y2="3.5" stroke="var(--acc3)" stroke-width="1"/><line x1="3" y1="5.5" x2="7" y2="5.5" stroke="var(--acc3)" stroke-width="1"/><line x1="3" y1="7.5" x2="5" y2="7.5" stroke="var(--acc3)" stroke-width="1"/></svg>
      <span class="p-lbl">JSON Output</span>
    </div>
    <div id="output-scroll">
      <div id="vmsg"></div>
      <pre id="jout" style="color:var(--tx3)">Edit fields or click Validate...</pre>
    </div>
    <div id="output-foot">
      <button class="btn-g" onclick="copyJson()">⎘ Copy</button>
      <button class="btn-g" onclick="downloadJson()">↓ Download</button>
    </div>
  </div>
</div>

<div id="taskbar">
  <span class="tb-brand">⬡ PyCanvas</span>
  <span class="tb-badge" id="tb-badge"></span>
  <div class="tb-sep"></div>
  <button class="btn-g" id="find-child-btn" title="Find children">↓ Child</button>
  <button class="btn-g" id="find-parent-btn" title="Find parent">↑ Parent</button>
  <div class="tb-sep"></div>
  <div class="tb-right">
    <button class="btn-g" id="desc-btn" onclick="cycleDescMode()" title="Toggle description display">Desc: Off</button>
    <div class="tb-sep"></div>
    <button class="btn-g" onclick="resetAll()">↺ Reset</button>
    <button class="btn-g" onclick="document.getElementById('file-input').click()">↑ Load JSON</button>
    <button class="btn-val" onclick="validateAndShow()">✓ Validate</button>
  </div>
</div>

<input type="file" id="file-input" accept=".json" style="display:none" onchange="handleFileLoad(event)"/>
<div id="toast"><span id="toast-msg"></span></div>

<!-- Multiline editor -->
<div id="ml-overlay" class="modal-overlay">
  <div id="ml-box" class="modal-box">
    <div class="modal-hdr" id="ml-hdr"><span class="modal-title" id="ml-title">Edit text</span><button class="modal-close" id="ml-close">×</button></div>
    <textarea id="ml-text" placeholder="Multiline…"></textarea>
    <div class="modal-foot">
      <button class="btn-g" id="ml-cancel">Cancel</button>
      <button class="btn-val" id="ml-save">Save</button>
    </div>
    <div id="ml-resize-handle"></div>
  </div>
</div>

<!-- Child picker -->
<div id="child-picker-overlay" class="modal-overlay">
  <div id="child-picker-box" class="modal-box">
    <div class="modal-hdr"><span class="modal-title" id="child-picker-title">Find child</span><button class="modal-close" id="child-picker-close">×</button></div>
    <div id="child-picker-list"></div>
    <div class="modal-foot"><button class="btn-g" id="child-picker-cancel">Close</button></div>
  </div>
</div>

<script>
'use strict';

// ── Constants ──────────────────────────────────────────────────────────────
var SCHEMA = __SCHEMA__;
var MODEL_NAME = __MODEL_NAME__;
var MODEL_LOWER = __MODEL_LOWER__;
var AGENT_MANIFEST_MODE = __AGENT_MANIFEST_MODE__;

// If the root model is self-referential
var ROOT_SCHEMA = SCHEMA;
if (SCHEMA && SCHEMA['$ref']) {
  var __rootDefName = SCHEMA['$ref'].replace('#/$defs/', '');
  ROOT_SCHEMA = (SCHEMA['$defs'] || {})[__rootDefName] || SCHEMA;
}

document.getElementById('tb-badge').textContent = MODEL_NAME + (AGENT_MANIFEST_MODE ? ' · Agent mode' : '');
document.title = MODEL_NAME + ' — Pydantic Editor';

var DESC_MODE = 'none';
var INST = {};
var IC = 0;
var ROOT_ID = null;
var zTop = 10;
var AGENT_COUNTER = 0;
var ACTIVE_AGENT_TAB = null;

function isAgentModel(name){
  return AGENT_MANIFEST_MODE && (name === MODEL_NAME || name === '__root__');
}

// ── Schema helpers ─────────────────────────────────────────────────────────

function isEnumDef(name) {
  var d = (SCHEMA['$defs'] || {})[name];
  return !!(d && d['enum'] !== undefined);
}
function resolveSchema(name) {
  if (name === '__root__') return ROOT_SCHEMA;
  return (SCHEMA['$defs'] || {})[name] || {};
}
function derefSchema(ref) {
  var n = ref.replace('#/$defs/', '');
  var d = (SCHEMA['$defs'] || {})[n] || {};
  if (d['$ref']) return derefSchema(d['$ref']);
  return d;
}
function refName(ref) {
  if (!ref) return null;
  var n = ref.replace('#/$defs/', '');
  return isEnumDef(n) ? null : n;
}
function getUnionRefs(f) {
  if (!f) return null;
  var opts = f.oneOf || f.anyOf;
  if (!opts) return null;
  var refs = [];
  var nullable = false;
  for (var i = 0; i < opts.length; i++) {
    var o = opts[i];
    if (o.type === 'null') { nullable = true; continue; }
    if (o['$ref']) {
      var n = refName(o['$ref']);
      if (n) { refs.push(n); continue; }
    }
    return null;
  }
  if (refs.length >= 2) return { refs: refs, nullable: nullable };
  return null;
}
function getDirectRef(f) {
  if (!f || !f['$ref']) return null;
  return refName(f['$ref']);
}
function getOptionalRef(f) {
  if (!f || !f.anyOf) return null;
  if (getUnionRefs(f)) return null;
  var refs = f.anyOf.filter(function(o){ return o['$ref']; });
  var nulls = f.anyOf.filter(function(o){ return o.type === 'null'; });
  if (refs.length === 1 && nulls.length >= 1) return refName(refs[0]['$ref']);
  return null;
}
function getArrayModelRef(f) {
  if (!f) return null;
  if (f.type === 'array' && f.items && f.items['$ref']) return refName(f.items['$ref']);
  if (f.anyOf) {
    for (var i = 0; i < f.anyOf.length; i++) {
      var o = f.anyOf[i];
      if (o.type === 'array' && o.items && o.items['$ref']) return refName(o.items['$ref']);
    }
  }
  return null;
}
function isOptArrayRef(f) {
  if (!f || !f.anyOf) return false;
  return f.anyOf.some(function(o){ return o.type === 'null'; }) && !!getArrayModelRef(f);
}
function isPrimArray(f) {
  if (!f) return false;
  if (f.type === 'array' && f.items && !f.items['$ref']) return true;
  return false;
}
function isOptPrimArray(f) {
  if (!f || !f.anyOf) return false;
  if (getArrayModelRef(f)) return false;
  var hasNull = f.anyOf.some(function(o){ return o.type === 'null'; });
  var hasArr = f.anyOf.some(function(o){ return o.type === 'array' && (!o.items || !o.items['$ref']); });
  return hasNull && hasArr;
}
function isOptPrim(f) {
  if (!f || !f.anyOf) return false;
  if (getOptionalRef(f)) return false;
  if (getUnionRefs(f)) return false;
  var nulls = f.anyOf.filter(function(o){ return o.type === 'null'; });
  var nonNull = f.anyOf.filter(function(o){ return o.type && o.type !== 'null'; });
  return nulls.length >= 1 && nonNull.length >= 1;
}
function optPrimType(f) {
  if (!f) return 'string';
  if (f.anyOf) {
    var nn = f.anyOf.filter(function(o){ return o.type && o.type !== 'null'; });
    if (nn.length) return nn[0].type;
  }
  return f.type || 'string';
}
function getEnumRef(f) {
  if (!f) return null;
  if (f['$ref']) {
    var n = f['$ref'].replace('#/$defs/', '');
    return isEnumDef(n) ? n : null;
  }
  if (f.anyOf) {
    for (var i = 0; i < f.anyOf.length; i++) {
      var o = f.anyOf[i];
      if (o['$ref']) {
        var n2 = o['$ref'].replace('#/$defs/', '');
        if (isEnumDef(n2)) return n2;
      }
    }
  }
  return null;
}
function getInlineEnum(f){
  if (!f) return null;
  if (Array.isArray(f.enum)) return f.enum;
  if (f.anyOf){
    var enums = [];
    var hasNull = false;
    for (var i=0;i<f.anyOf.length;i++){
      var o = f.anyOf[i];
      if (o.type === 'null'){ hasNull=true; continue; }
      if (Array.isArray(o.enum)){ enums = enums.concat(o.enum); }
      else if (o.const !== undefined){ enums.push(o.const); }
    }
    if (enums.length) return {values: enums, nullable: hasNull};
  }
  if (f.const !== undefined) return [f.const];
  return null;
}
function isOptEnum(f) {
  if (!f || !f.anyOf) return false;
  var hasEnumRef = f.anyOf.some(function(o){ return o['$ref'] && isEnumDef(o['$ref'].replace('#/$defs/','')); });
  var hasNull = f.anyOf.some(function(o){ return o.type === 'null'; });
  return hasEnumRef && hasNull;
}
function isDictField(f) {
  return !!(f && f.type === 'object' && !f.properties);
}
function getDictValueModelRef(f){
  if (!isDictField(f)) return null;
  var ap = f.additionalProperties;
  if (ap && ap['$ref']) return refName(ap['$ref']);
  return null;
}
function fieldTypeName(f) {
  if (!f) return 'any';
  var ur = getUnionRefs(f);
  if (ur) return ur.refs.join(' | ') + (ur.nullable ? ' | None' : '');
  var amr = getArrayModelRef(f);
  if (amr) return isOptArrayRef(f) ? 'List[' + amr + '] | None' : 'List[' + amr + ']';
  var dr = getDirectRef(f);
  if (dr) return dr;
  var oref = getOptionalRef(f);
  if (oref) return oref + ' | None';
  var er = getEnumRef(f);
  if (er) return isOptEnum(f) ? er + ' | None' : er;
  var ie = getInlineEnum(f);
  if (ie){
    var vals = Array.isArray(ie) ? ie : (ie.values || []);
    if (vals.length) return 'Literal[' + vals.map(function(v){return JSON.stringify(v)}).join(', ') + ']';
  }
  if (isPrimArray(f)) return ((f.items && f.items.type) || 'any') + '[]';
  if (isOptPrimArray(f)) return 'array | None';
  if (isOptPrim(f)) return optPrimType(f) + ' | None';
  if (isDictField(f)) {
    var dmr = getDictValueModelRef(f);
    return dmr ? 'dict[string, ' + dmr + ']' : 'dict[string, any]';
  }
  return f.type || 'any';
}
function coerce(v, schema) {
  if (!schema) return v;
  var t = schema.type;
  if (!t && schema.anyOf) { var nn = schema.anyOf.find(function(o){ return o.type && o.type !== 'null'; }); if (nn) t = nn.type; }
  if (t === 'integer') { var n = parseInt(v, 10); return isNaN(n) ? null : n; }
  if (t === 'number')  { var n2 = parseFloat(v);  return isNaN(n2) ? null : n2; }
  if (t === 'boolean') return !!v;
  return v;
}

// ── Instance management ────────────────────────────────────────────────────

function mkId() { return 'i' + (++IC); }

function newInst(modelName, parentId, parentField, parentIndex) {
  var id = mkId();
  var model = resolveSchema(modelName);
  var props = model.properties || {};
  var data = {};
  for (var fn in props) {
    var fs = props[fn];
    var er = getEnumRef(fs);
    if (er) {
      var ed = (SCHEMA['$defs'] || {})[er] || {};
      data[fn] = fs['default'] !== undefined ? fs['default'] : (ed['enum'] && ed['enum'][0]);
      continue;
    }
    var ie = getInlineEnum(fs);
    if (ie){
      var vals = Array.isArray(ie) ? ie : (ie.values || []);
      if (vals.length) data[fn] = fs['default'] !== undefined ? fs['default'] : vals[0];
      continue;
    }
    if (fs['default'] !== undefined && !getDirectRef(fs) && !getArrayModelRef(fs) && !getOptionalRef(fs) && !getUnionRefs(fs)) {
      data[fn] = fs['default'];
    }
    if (fs.type === 'boolean') {
      data[fn] = fs['default'] !== undefined ? fs['default'] : false;
    }
  }
  var inst = {
    id: id, modelName: modelName,
    parentId: parentId || null, parentField: parentField || null,
    parentIndex: parentIndex != null ? parentIndex : null,
    location: 'editor',
    enabled: true,
    listState: {},
    primListState: {},
    primListCache: {},
    unionChoice: {},
    data: data,
    pos: { x: 0, y: 0 },
    size: { w: 320, h: 380 },
    minimized: false
  };
  if (isAgentModel(modelName)){
    inst.agentTabIndex = ++AGENT_COUNTER;
  }
  INST[id] = inst;
  return id;
}

function spawnChildren(id) {
  var inst = INST[id];
  var model = resolveSchema(inst.modelName);
  var props = model.properties || {};
  for (var fn in props) {
    var fs = props[fn];
    var dr = getDirectRef(fs);
    var oref = getOptionalRef(fs);
    var ur = getUnionRefs(fs);
    if (dr) {
      if (isAgentModel(dr)) continue;
      var cid = newInst(dr, id, fn, null);
      spawnChildren(cid);
    } else if (oref) {
      if (isAgentModel(oref)) continue;
      var cid2 = newInst(oref, id, fn, null);
      INST[cid2].enabled = false;
    } else if (ur) {
      var realRefs = ur.refs.filter(function(r){ return !isAgentModel(r); });
      if (realRefs.length === 0) continue;
      inst.unionChoice[fn] = realRefs[0];
      ur.refs.forEach(function(refName, idx){
        if (isAgentModel(refName)) return;
        var cidU = newInst(refName, id, fn, null);
        INST[cidU].enabled = (refName === realRefs[0]);
        INST[cidU].unionOptionFor = fn;
      });
    }
  }
}

function ensureChildrenSpawned(id) {
  var inst = INST[id];
  if (!inst) return;
  var model = resolveSchema(inst.modelName);
  var props = model.properties || {};
  for (var fn in props) {
    var fs = props[fn];
    var dr = getDirectRef(fs);
    var oref = getOptionalRef(fs);
    var ur = getUnionRefs(fs);
    if (dr && !isAgentModel(dr)) {
      var child = findChild(id, fn);
      if (!child) {
        var cid = newInst(dr, id, fn, null);
        spawnChildren(cid);
      } else {
        ensureChildrenSpawned(child.id);
      }
    } else if (oref && !isAgentModel(oref)) {
      var child2 = findChild(id, fn);
      if (!child2) {
        var cid2 = newInst(oref, id, fn, null);
        INST[cid2].enabled = false;
      }
    } else if (ur) {
      var existing = findUnionChildren(id, fn);
      if (!existing.length) {
        var realRefs = ur.refs.filter(function(r){ return !isAgentModel(r); });
        if (realRefs.length){
          inst.unionChoice[fn] = realRefs[0];
          realRefs.forEach(function(refName, idx){
            var cidU = newInst(refName, id, fn, null);
            INST[cidU].enabled = (idx === 0);
            INST[cidU].unionOptionFor = fn;
          });
        }
      }
    }
  }
}

function findChild(parentId, field) {
  var keys = Object.keys(INST);
  for (var i = 0; i < keys.length; i++) {
    var inst = INST[keys[i]];
    if (inst.parentId === parentId && inst.parentField === field && inst.parentIndex === null && !inst.unionOptionFor) return inst;
  }
  var listKids = findListChildren(parentId, field);
  return listKids.length ? listKids[0] : null;
}
function findUnionChildren(parentId, field) {
  var result = [];
  var keys = Object.keys(INST);
  for (var i = 0; i < keys.length; i++) {
    var inst = INST[keys[i]];
    if (inst.parentId === parentId && inst.parentField === field && inst.unionOptionFor === field) result.push(inst);
  }
  return result;
}
function findListChildren(parentId, field) {
  var result = [];
  var keys = Object.keys(INST);
  for (var i = 0; i < keys.length; i++) {
    var inst = INST[keys[i]];
    if (inst.parentId === parentId && inst.parentField === field && inst.parentIndex !== null) result.push(inst);
  }
  result.sort(function(a, b){ return a.parentIndex - b.parentIndex; });
  return result;
}

function findDescendant(parentId, predicate){
  var stack = [parentId];
  var visited = {};
  while(stack.length){
    var pid = stack.pop();
    if(visited[pid]) continue;
    visited[pid]=true;
    var kids = Object.values(INST).filter(function(i){ return i.parentId === pid; });
    for(var k=0;k<kids.length;k++){
      var kid = kids[k];
      if(predicate(kid)) return kid;
      stack.push(kid.id);
    }
  }
  return null;
}

function deleteInstTree(id) {
  if (!INST[id]) return;
  var inst = INST[id];
  var parentId = inst.parentId;
  var parentField = inst.parentField;
  var parentIndex = inst.parentIndex;

  var toDelete = [id];
  var i = 0;
  while (i < toDelete.length) {
    var pid = toDelete[i++];
    var keys = Object.keys(INST);
    for (var j = 0; j < keys.length; j++) {
      if (INST[keys[j]].parentId === pid) toDelete.push(keys[j]);
    }
  }
  toDelete.forEach(function(iid){
    var w = document.getElementById('dw-' + iid);
    if (w) w.remove();
    delete INST[iid];
  });
  if (parentId && parentField !== null && parentIndex !== null){
    findListChildren(parentId, parentField).forEach(function(c, idx){ c.parentIndex = idx; });
  }
}

// Display name in Editor / Canvas (generic, no id/Root)
function instDisplayName(id) {
  var inst = INST[id];
  if(!inst) return '?';
  var model = resolveSchema(inst.modelName);
  var base = inst.modelName === '__root__' ? (model.title || MODEL_NAME) : (model.title || inst.modelName);
  if (inst.dictKey !== undefined) return base + (inst.dictKey ? ' ["' + inst.dictKey + '"]' : ' [?]');
  return inst.parentIndex !== null ? base + ' #' + (inst.parentIndex + 1) : base;
}

// Agent tab label (Root / #N / id)
function agentTabName(id){
  var inst = INST[id];
  if(!inst || !isAgentModel(inst.modelName)) return instDisplayName(id);
  var aid = inst.data && inst.data.id;
  if (aid && String(aid).trim() !== '') return String(aid);
  if (id === ROOT_ID) return 'Root';
  return '#' + (inst.agentTabIndex || '?');
}

function instanceOrder() {
  var result = [];
  var visited = {};
  var queue = [ROOT_ID];
  while (queue.length) {
    var id = queue.shift();
    if (!id || visited[id]) continue;
    visited[id] = true;
    result.push(id);
    var kids = Object.values(INST).filter(function(i){ return i.parentId === id; });
    kids.sort(function(a, b){
      if (a.parentField < b.parentField) return -1;
      if (a.parentField > b.parentField) return 1;
      return (a.parentIndex || 0) - (b.parentIndex || 0);
    });
    kids.forEach(function(k){ queue.push(k.id); });
  }
  return result;
}

// ── Assembly ───────────────────────────────────────────────────────────────

function assemble(id) {
  var inst = INST[id];
  if (!inst) return null;
  var model = resolveSchema(inst.modelName);
  var props = model.properties || {};
  var result = {};

  for (var fn in props) {
    var fs = props[fn];
    var dr = getDirectRef(fs);
    var oref = getOptionalRef(fs);
    var amr = getArrayModelRef(fs);
    var ur = getUnionRefs(fs);

    if (ur) {
      var chosen = inst.unionChoice[fn];
      var uChildren = findUnionChildren(id, fn);
      var activeChild = uChildren.find(function(c){ return c.modelName === chosen; });
      if (!chosen && ur.nullable) {
        result[fn] = null;
      } else {
        result[fn] = activeChild ? assemble(activeChild.id) : null;
      }
    } else if (dr) {
      var child = findChild(id, fn);
      result[fn] = child ? assemble(child.id) : null;
    } else if (oref) {
      var child2 = findChild(id, fn);
      result[fn] = (child2 && child2.enabled) ? assemble(child2.id) : null;
    } else if (amr) {
      var isOptArr = isOptArrayRef(fs);
      if (isOptArr && inst.listState[fn] === false) {
        result[fn] = null;
      } else {
        result[fn] = findListChildren(id, fn).map(function(c){ return assemble(c.id); });
      }
    } else if (isPrimArray(fs)) {
      var raw = inst.data[fn];
      result[fn] = Array.isArray(raw)
        ? raw.filter(function(x){ return x !== ''; }).map(function(x){ return coerce(x, fs.items); })
        : (fs['default'] || []);
    } else if (isOptPrimArray(fs)) {
      if (inst.primListState[fn] === false) {
        result[fn] = null;
      } else {
        var rawO = inst.data[fn];
        var arrSchema = fs.anyOf.find(function(o){ return o.type === 'array'; }) || {};
        result[fn] = Array.isArray(rawO)
          ? rawO.filter(function(x){ return x !== ''; }).map(function(x){ return coerce(x, arrSchema.items); })
          : [];
      }
    } else if (isDictField(fs)) {
      var dmrA = getDictValueModelRef(fs);
      if (dmrA) {
        var objA = {};
        findListChildren(id, fn).forEach(function(c){
          var key = c.dictKey || '';
          if (key !== '') objA[key] = assemble(c.id);
        });
        result[fn] = objA;
      } else {
        var kvs = inst.data[fn];
        if (Array.isArray(kvs)) {
          var obj = {};
          kvs.forEach(function(kv){ if (kv.k !== '') obj[kv.k] = kv.v; });
          result[fn] = obj;
        } else {
          result[fn] = {};
        }
      }
    } else {
      var isOP = isOptPrim(fs);
      if (isOP && inst.data['__disabled__' + fn]) {
        result[fn] = null;
      } else {
        var v = inst.data[fn];
        if (v !== undefined && v !== null && v !== '') result[fn] = coerce(v, fs);
        else if (fs['default'] !== undefined) result[fn] = fs['default'];
        else result[fn] = null;
      }
    }
  }
  return result;
}

// ── Multiline editor ───────────────────────────────────────────────────────
var ML_CTX = null;
function firstNonEmptyLine(s){
  if(!s) return '';
  var lines = String(s).split(/\r?\n/);
  for(var i=0;i<lines.length;i++){ if(lines[i].trim()!=='') return lines[i]; }
  return '';
}
function openMultilineEditor(getVal, setVal, title){
  ML_CTX = {getVal: getVal, setVal: setVal};
  document.getElementById('ml-title').textContent = (title||'value') + ' – multiline';
  document.getElementById('ml-text').value = getVal() || '';
  document.getElementById('ml-overlay').classList.add('on');
  setTimeout(function(){ document.getElementById('ml-text').focus(); }, 30);
}
function closeMultilineEditor(){
  document.getElementById('ml-overlay').classList.remove('on');
  ML_CTX = null;
}
function saveMultilineEditor(){
  if(!ML_CTX) return;
  var v = document.getElementById('ml-text').value;
  ML_CTX.setVal(v);
  closeMultilineEditor();
  renderEditorPanel();
  Object.keys(INST).forEach(function(id){ if(INST[id].location==='desktop') renderDesktopWindow(id); });
  liveUpdate();
}
function makeMlInput(getVal, setVal, placeholder, title, onChangeExtra) {
  var wrap = document.createElement('div');
  wrap.style.cssText = 'display:flex;gap:4px;align-items:center;width:100%;';
  var inp = document.createElement('input');
  inp.type = 'text'; inp.className = 'fi'; inp.style.flex = '1'; inp.style.minWidth = '0';
  var fullVal = getVal();
  fullVal = (fullVal !== undefined && fullVal !== null) ? String(fullVal) : '';
  inp.value = firstNonEmptyLine(fullVal) || fullVal;
  if (placeholder) inp.placeholder = placeholder;
  var openEditor = function(){ openMultilineEditor(getVal, setVal, title); };
  inp.addEventListener('keydown', function(e){
    if (e.key === 'Enter'){ e.preventDefault(); openEditor(); }
  });
  inp.oninput = function(){
    var v = inp.value;
    setVal(v);
    if (v.indexOf('\n') !== -1){ openEditor(); return; }
    liveUpdate();
    if (onChangeExtra) onChangeExtra(v);
  };
  inp.addEventListener('paste', function(){
    setTimeout(function(){
      var v = getVal();
      if (v && String(v).indexOf('\n') !== -1) openEditor();
    }, 10);
  });
  inp.addEventListener('dblclick', openEditor);
  wrap.appendChild(inp);
  var editBtn = document.createElement('button');
  editBtn.className = 'ml-edit-btn';
  editBtn.title = 'Edit multiline';
  editBtn.textContent = '✎';
  editBtn.onclick = function(e){ e.stopPropagation(); openEditor(); };
  wrap.appendChild(editBtn);
  return wrap;
}

// ── Field row builder ──────────────────────────────────────────────────────

function buildFieldRow(instId, fn, fs, required) {
  var isReq = required.indexOf(fn) !== -1;
  var dr = getDirectRef(fs);
  var oref = getOptionalRef(fs);
  var amr = getArrayModelRef(fs);
  var ur = getUnionRefs(fs);
  var er = getEnumRef(fs);
  var isOptE = isOptEnum(fs);
  var isPA = isPrimArray(fs);
  var isOPA = isOptPrimArray(fs);
  var isOP = isOptPrim(fs);
  var isDict = isDictField(fs);
  var inst = INST[instId];
  var inlineEnum = getInlineEnum(fs);

  var row = document.createElement('div');
  row.className = 'f-row';
  row.id = 'fr-' + instId + '-' + fn;

  var meta = document.createElement('div');
  meta.className = 'f-meta';
  var nm = document.createElement('div');
  nm.className = 'f-nm';
  nm.textContent = fn;
  if (isReq) { var star = document.createElement('span'); star.className = 'f-req'; star.textContent = '*'; nm.appendChild(star); }
  var tp = document.createElement('div');
  tp.className = 'f-tp';
  tp.textContent = fieldTypeName(fs);
  meta.appendChild(nm);
  meta.appendChild(tp);
  if (fs.description && DESC_MODE === 'expanded') {
    var ds = document.createElement('div');
    ds.className = 'f-ds'; ds.title = fs.description; ds.textContent = fs.description;
    meta.appendChild(ds);
  }
  row.appendChild(meta);

  var ctl = document.createElement('div');
  ctl.className = 'f-ctl';

  if (ur) {
    row.appendChild(ctl);
    row.style.flexDirection = 'column';
    row.style.alignItems = 'stretch';
    var uWrap = document.createElement('div');
    uWrap.className = 'list-wrap';
    uWrap.id = 'uw-' + instId + '-' + fn;
    renderUnionControl(uWrap, instId, fn, ur);
    row.appendChild(uWrap);
    return row;
  }
  else if (dr) {
    var child = findChild(instId, fn);
    var isAgent = isAgentModel(dr);
    ctl.appendChild(makeRefChip(child ? child.id : null, instDisplayName(child ? child.id : null) || dr, instId, fn, null, isAgent));
  }
  else if (oref) {
    var child2 = findChild(instId, fn);
    ctl.appendChild(makeOptModelControl(instId, fn, child2, oref));
  }
  else if (amr) {
    row.appendChild(ctl);
    row.style.flexDirection = 'column';
    row.style.alignItems = 'stretch';
    var listWrap = document.createElement('div');
    listWrap.className = 'list-wrap';
    listWrap.id = 'mlw-' + instId + '-' + fn;
    if (isAgentModel(amr)) renderAgentList(listWrap, instId, fn, amr, fs);
    else renderModelList(listWrap, instId, fn, amr, fs);
    row.appendChild(listWrap);
    return row;
  }
  else if (isPA) {
    row.style.flexDirection = 'column';
    row.style.alignItems = 'stretch';
    var pw = makePrimListCtl(instId, fn, fs);
    row.appendChild(ctl);
    row.appendChild(pw);
    return row;
  }
  else if (isOPA) {
    row.style.flexDirection = 'column';
    row.style.alignItems = 'stretch';
    row.appendChild(ctl);
    row.appendChild(makeOptPrimListCtl(instId, fn, fs));
    return row;
  }
  else if (isDict) {
    row.style.flexDirection = 'column';
    row.style.alignItems = 'stretch';
    row.appendChild(ctl);
    row.appendChild(makeDictCtl(instId, fn, fs));
    return row;
  }
  else if (er || inlineEnum) {
    var isInlineOpt = inlineEnum && typeof inlineEnum === 'object' && inlineEnum.nullable;
    if (isOptE || isInlineOpt) {
      ctl.appendChild(makeOptEnumControl(instId, fn, fs, er, inlineEnum));
    } else {
      ctl.appendChild(makeEnumSelect(instId, fn, fs, er, inlineEnum));
    }
  }
  else if (fs.type === 'boolean') {
    ctl.appendChild(makeBoolToggle(instId, fn, fs));
  }
  else if (isOP) {
    ctl.appendChild(makeOptPrimControl(instId, fn, fs));
  }
  else if (fs.type === 'integer' || fs.type === 'number') {
    ctl.appendChild(makeNumInput(instId, fn, fs, isReq));
  }
  else {
    ctl.appendChild(makeTextInput(instId, fn, fs, isReq));
  }

  row.appendChild(ctl);
  return row;
}

function makeRefChip(targetId, label, sourceId, fieldName, listIndex, isAgent){
  var chip = document.createElement('span');
  chip.className = 'ref-chip' + (isAgent ? ' agent-chip' : '');
  chip.textContent = '↗ ' + label;
  if (targetId) {
    chip.dataset.target = targetId;
    chip.onclick = function(){ navigateFrom(sourceId, targetId, fieldName, listIndex); };
  } else {
    chip.classList.add('disabled');
  }
  return chip;
}

function makeOptModelControl(instId, fn, child, refLabel) {
  var wrap = document.createElement('div');
  wrap.style.cssText = 'display:flex;align-items:center;gap:6px;';
  var isAgent = isAgentModel(refLabel);

  var tog = document.createElement('div');
  tog.className = 'ftog' + (child && child.enabled ? ' on' : '');
  var chipLabel = child ? instDisplayName(child.id) : refLabel;
  var chip = document.createElement('span');
  chip.className = 'ref-chip' + (isAgent ? ' agent-chip' : '') + (child && child.enabled ? '' : ' disabled');
  chip.textContent = child && child.enabled ? '↗ ' + chipLabel : '× null';
  if (child && child.enabled) chip.onclick = function(){ navigateFrom(instId, child.id, fn, null); };

  tog.onclick = function() {
    if (!child) return;
    child.enabled = !child.enabled;
    if (child.enabled) ensureChildrenSpawned(child.id);
    tog.classList.toggle('on', child.enabled);
    chip.classList.toggle('disabled', !child.enabled);
    chip.textContent = child.enabled ? '↗ ' + chipLabel : '× null';
    chip.onclick = child.enabled ? function(){ navigateFrom(instId, child.id, fn, null); } : null;
    liveUpdate(); renderWires();
    renderEditorPanel();
  };
  wrap.appendChild(tog);
  wrap.appendChild(chip);
  return wrap;
}

function renderUnionControl(wrap, instId, fn, ur) {
  wrap.innerHTML = '';
  var inst = INST[instId];
  var topRow = document.createElement('div');
  topRow.style.cssText = 'display:flex;align-items:center;gap:6px;margin-bottom:5px;flex-wrap:wrap;';

  var tog = null;
  if (ur.nullable) {
    tog = document.createElement('div');
    var hasChoice = !!inst.unionChoice[fn];
    tog.className = 'ftog' + (hasChoice ? ' on' : '');
    tog.onclick = function(){
      if (inst.unionChoice[fn]) {
        var prevChoice = inst.unionChoice[fn];
        inst.data['__prevUnion_' + fn] = prevChoice;
        inst.unionChoice[fn] = null;
      } else {
        inst.unionChoice[fn] = inst.data['__prevUnion_' + fn] || ur.refs[0];
      }
      liveUpdate();
      renderUnionControl(wrap, instId, fn, ur);
    };
    topRow.appendChild(tog);
  }

  var sel = document.createElement('select');
  sel.className = 'fsel';
  sel.disabled = ur.nullable && !inst.unionChoice[fn];
  ur.refs.forEach(function(rn){
    var o = document.createElement('option');
    o.value = rn; o.textContent = rn;
    sel.appendChild(o);
  });
  sel.value = inst.unionChoice[fn] || ur.refs[0];
  sel.onchange = function(){
    inst.unionChoice[fn] = sel.value;
    findUnionChildren(instId, fn).forEach(function(c){ c.enabled = (c.modelName === sel.value); });
    liveUpdate();
    renderUnionControl(wrap, instId, fn, ur);
  };
  topRow.appendChild(sel);
  wrap.appendChild(topRow);

  if (!ur.nullable || inst.unionChoice[fn]) {
    var activeChild = findUnionChildren(instId, fn).find(function(c){ return c.modelName === (inst.unionChoice[fn] || ur.refs[0]); });
    if (activeChild) {
      var chipRow = document.createElement('div');
      var isAgent = isAgentModel(activeChild.modelName);
      var chip = makeRefChip(activeChild.id, instDisplayName(activeChild.id), instId, fn, null, isAgent);
      chipRow.appendChild(chip);
      wrap.appendChild(chipRow);
    }
  }
}

function makeEnumSelect(instId, fn, fs, er, inlineEnum) {
  var sel = document.createElement('select');
  sel.className = 'fsel';
  var options = [];
  if (er){
    var ed = (SCHEMA['$defs'] || {})[er] || {};
    options = ed['enum'] || [];
  } else if (inlineEnum){
    options = Array.isArray(inlineEnum) ? inlineEnum : (inlineEnum.values || []);
  }
  options.forEach(function(opt){
    var o = document.createElement('option');
    o.value = opt; o.textContent = opt; sel.appendChild(o);
  });
  var cur = INST[instId].data[fn];
  if (cur !== undefined) sel.value = cur;
  sel.onchange = function(){ INST[instId].data[fn] = sel.value; liveUpdate(); if(isAgentModel(INST[instId].modelName) && fn==='id'){ renderAgentTabs(); renderEditorPanel(); } };
  return sel;
}

function makeOptEnumControl(instId, fn, fs, er, inlineEnum) {
  var wrap = document.createElement('div');
  wrap.className = 'null-tog-row';
  var inst = INST[instId];
  var hasVal = inst.data[fn] !== undefined && inst.data[fn] !== null;

  var tog = document.createElement('div');
  tog.className = 'ftog' + (hasVal ? ' on' : '');
  var sel = document.createElement('select');
  sel.className = 'fsel';
  sel.disabled = !hasVal;
  sel.style.opacity = hasVal ? '1' : '0.3';
  var options = [];
  if (er){
    var ed = (SCHEMA['$defs'] || {})[er] || {};
    options = ed['enum'] || [];
  } else if (inlineEnum){
    options = Array.isArray(inlineEnum) ? inlineEnum : (inlineEnum.values || []);
  }
  options.forEach(function(opt){
    var o = document.createElement('option');
    o.value = opt; o.textContent = opt; sel.appendChild(o);
  });
  if (hasVal) sel.value = inst.data[fn];
  sel.onchange = function(){ inst.data[fn] = sel.value; liveUpdate(); };
  tog.onclick = function(){
    tog.classList.toggle('on');
    var on = tog.classList.contains('on');
    sel.disabled = !on; sel.style.opacity = on ? '1' : '0.3';
    if (!on) { inst.data[fn] = null; }
    else if (options.length) { inst.data[fn] = sel.value || options[0]; sel.value = inst.data[fn]; }
    liveUpdate();
  };
  wrap.appendChild(tog); wrap.appendChild(sel);
  return wrap;
}

function makeBoolToggle(instId, fn, fs) {
  var tog = document.createElement('div');
  var v = INST[instId].data[fn];
  tog.className = 'ftog' + (v ? ' on' : '');
  tog.onclick = function(){
    tog.classList.toggle('on');
    INST[instId].data[fn] = tog.classList.contains('on');
    liveUpdate();
  };
  return tog;
}

function makeOptPrimControl(instId, fn, fs) {
  var wrap = document.createElement('div');
  wrap.className = 'null-tog-row';
  var inst = INST[instId];
  var t = optPrimType(fs);
  var isText = !(t === 'integer' || t === 'number');
  var disabledFlag = '__disabled__' + fn;
  var hasVal = inst.data[fn] !== undefined && inst.data[fn] !== null && inst.data[fn] !== '';
  var isOn = hasVal && !inst.data[disabledFlag];

  var tog = document.createElement('div');
  tog.className = 'ftog' + (isOn ? ' on' : '');

  var ctl, inp, editBtn;
  if (isText) {
    ctl = makeMlInput(
      function(){ return inst.data[fn]; },
      function(v){ inst.data[fn] = v; },
      '—', fn
    );
    ctl.style.flex = '1'; ctl.style.minWidth = '0';
    inp = ctl.querySelector('input');
    editBtn = ctl.querySelector('.ml-edit-btn');
  } else {
    inp = document.createElement('input');
    inp.type = 'number'; inp.className = 'fi';
    if (inst.data[fn] !== undefined && inst.data[fn] !== null) inp.value = inst.data[fn];
    inp.placeholder = '—';
    inp.oninput = function(){ inst.data[fn] = inp.value; liveUpdate(); };
    ctl = inp;
  }
  inp.disabled = !isOn;
  inp.style.opacity = isOn ? '1' : '0.45';
  if (editBtn) editBtn.style.display = isOn ? '' : 'none';

  tog.onclick = function(){
    var turningOn = !tog.classList.contains('on');
    tog.classList.toggle('on', turningOn);
    inp.disabled = !turningOn; inp.style.opacity = turningOn ? '1' : '0.45';
    if (editBtn) editBtn.style.display = turningOn ? '' : 'none';
    inst.data[disabledFlag] = !turningOn;
    if (turningOn && (inst.data[fn] === undefined || inst.data[fn] === null)) {
      inst.data[fn] = '';
      if (isText) inp.value = '';
    }
    liveUpdate();
  };
  wrap.appendChild(tog); wrap.appendChild(ctl);
  return wrap;
}

function makeNumInput(instId, fn, fs, isReq) {
  var inp = document.createElement('input');
  inp.type = 'number'; inp.className = 'fi';
  if (fs.minimum !== undefined) inp.min = fs.minimum;
  if (fs.maximum !== undefined) inp.max = fs.maximum;
  var v = INST[instId].data[fn];
  if (v !== undefined) inp.value = v;
  inp.placeholder = isReq ? 'required' : '—';
  inp.oninput = function(){ INST[instId].data[fn] = inp.value; liveUpdate(); };
  return inp;
}

function makeTextInput(instId, fn, fs, isReq) {
  var inst = INST[instId];
  return makeMlInput(
    function(){ return inst.data[fn]; },
    function(v){ inst.data[fn] = v; },
    isReq ? 'required' : '—',
    fn,
    function(){
      if (isAgentModel(inst.modelName) && fn === 'id'){ renderAgentTabs(); }
    }
  );
}

function renderModelList(wrap, instId, fn, modelRef, fs) {
  wrap.innerHTML = '';
  var inst = INST[instId];
  var isOptArr = isOptArrayRef(fs);
  if (inst.listState[fn] === undefined) inst.listState[fn] = true;

  if (isOptArr) {
    var topRow = document.createElement('div');
    topRow.style.cssText = 'display:flex;align-items:center;gap:7px;margin-bottom:5px;';
    var tog = document.createElement('div');
    tog.className = 'ftog' + (inst.listState[fn] ? ' on' : '');
    var lbl = document.createElement('span');
    lbl.style.cssText = 'font-size:10px;color:var(--tx3);font-family:var(--mono);';
    lbl.textContent = 'List[' + modelRef + '] | None';
    tog.onclick = function(){
      inst.listState[fn] = !inst.listState[fn];
      tog.classList.toggle('on', inst.listState[fn]);
      liveUpdate();
      renderModelList(wrap, instId, fn, modelRef, fs);
    };
    topRow.appendChild(tog); topRow.appendChild(lbl);
    wrap.appendChild(topRow);
  }

  var offState = isOptArr && !inst.listState[fn];
  var children = findListChildren(instId, fn);
  var listDiv = document.createElement('div');
  listDiv.className = 'list-items';
  if (offState) listDiv.style.opacity = '0.45';
  children.forEach(function(child, i) {
    var li = document.createElement('div');
    li.className = 'list-item';
    li.id = 'fitem-' + instId + '-' + fn + '-' + i;
    var chip = document.createElement('span');
    chip.className = 'ref-chip' + (offState ? ' disabled' : '');
    chip.style.flex = '1';
    chip.textContent = '↗ ' + instDisplayName(child.id);
    if (!offState) chip.onclick = function(){ navigateFrom(instId, child.id, fn, i); };
    var del = document.createElement('button');
    del.className = 'li-del'; del.textContent = '×';
    del.disabled = offState;
    del.onclick = function(){
      if (offState) return;
      deleteInstTree(child.id);
      liveUpdate();
      renderModelList(wrap, instId, fn, modelRef, fs);
      if (inst.location === 'editor') renderEditorPanel();
      renderWires();
    };
    li.appendChild(chip); li.appendChild(del);
    listDiv.appendChild(li);
  });

  var addBtn = document.createElement('button');
  addBtn.className = 'li-add';
  addBtn.disabled = offState;
  if (offState) addBtn.style.opacity = '0.45';
  addBtn.innerHTML = '+ Add ' + modelRef;
  addBtn.onclick = function(){
    if (offState) return;
    var existing = findListChildren(instId, fn);
    var newIdx = existing.length;
    var nid = newInst(modelRef, instId, fn, newIdx);
    spawnChildren(nid);
    if (inst.location === 'desktop') {
      INST[nid].location = 'desktop';
      INST[nid].pos = { x: inst.pos.x + inst.size.w + 40, y: inst.pos.y + newIdx * 60 };
      INST[nid].size = { w: inst.size.w, h: inst.size.h };
      renderDesktopWindow(nid);
    }
    liveUpdate();
    renderModelList(wrap, instId, fn, modelRef, fs);
    if (inst.location === 'editor') renderEditorPanel();
    renderWires();
  };
  wrap.appendChild(listDiv);
  wrap.appendChild(addBtn);
}

function renderModelDictCtl(wrap, instId, fn, modelRef) {
  wrap.innerHTML = '';
  var inst = INST[instId];
  var children = findListChildren(instId, fn);

  var listDiv = document.createElement('div');
  listDiv.className = 'list-items';
  children.forEach(function(child, i) {
    var li = document.createElement('div');
    li.className = 'kv-item';
    li.id = 'fitem-' + instId + '-' + fn + '-' + i;

    var kiCtl = makeMlInput(
      function(){ return child.dictKey; },
      function(v){ child.dictKey = v; },
      'key', fn + ' key'
    );
    kiCtl.style.flex = '1'; kiCtl.style.minWidth = '0';

    var colon = document.createElement('span');
    colon.style.cssText = 'color:var(--tx3);font-family:var(--mono);font-size:11px;flex-shrink:0;';
    colon.textContent = ':';

    var chip = document.createElement('span');
    chip.className = 'ref-chip';
    chip.style.flex = '1';
    chip.textContent = '↗ ' + modelRef;
    chip.onclick = function(){ navigateFrom(instId, child.id, fn, i); };

    var del = document.createElement('button');
    del.className = 'li-del'; del.textContent = '×';
    del.onclick = function(){
      deleteInstTree(child.id);
      liveUpdate();
      renderModelDictCtl(wrap, instId, fn, modelRef);
      if (inst.location === 'editor') renderEditorPanel();
      renderWires();
    };

    li.appendChild(kiCtl); li.appendChild(colon); li.appendChild(chip); li.appendChild(del);
    listDiv.appendChild(li);
  });

  var addBtn = document.createElement('button');
  addBtn.className = 'li-add';
  addBtn.innerHTML = '+ Add ' + modelRef;
  addBtn.onclick = function(){
    var existing = findListChildren(instId, fn);
    var newIdx = existing.length;
    var nid = newInst(modelRef, instId, fn, newIdx);
    INST[nid].dictKey = '';
    spawnChildren(nid);
    if (inst.location === 'desktop') {
      INST[nid].location = 'desktop';
      INST[nid].pos = { x: inst.pos.x + inst.size.w + 40, y: inst.pos.y + newIdx * 60 };
      INST[nid].size = { w: inst.size.w, h: inst.size.h };
      renderDesktopWindow(nid);
    }
    liveUpdate();
    renderModelDictCtl(wrap, instId, fn, modelRef);
    if (inst.location === 'editor') renderEditorPanel();
    renderWires();
  };

  wrap.appendChild(listDiv);
  wrap.appendChild(addBtn);
}

function renderAgentList(wrap, instId, fn, modelRef, fs){
  wrap.innerHTML = '';
  var inst = INST[instId];
  var children = findListChildren(instId, fn);
  var listDiv = document.createElement('div');
  listDiv.className = 'list-items';
  children.forEach(function(child, i){
    var li = document.createElement('div');
    li.className = 'list-item';
    li.id = 'fitem-' + instId + '-' + fn + '-' + i;
    var chip = document.createElement('span');
    chip.className = 'ref-chip agent-chip';
    chip.style.flex = '1';
    chip.textContent = '↗ ' + agentTabName(child.id);
    chip.onclick = function(){ setActiveAgentTab(child.id); navigateTo(child.id, null, null); };
    var del = document.createElement('button');
    del.className = 'li-del'; del.textContent = '×';
    del.onclick = function(){
      deleteInstTree(child.id);
      liveUpdate(); renderAgentTabs(); renderEditorPanel(); renderWires();
    };
    li.appendChild(chip); li.appendChild(del);
    listDiv.appendChild(li);
  });
  var addBtn = document.createElement('button');
  addBtn.className = 'li-add';
  addBtn.innerHTML = '+ Add ' + modelRef;
  addBtn.onclick = function(){
    var existing = findListChildren(instId, fn);
    var newIdx = existing.length;
    var nid = newInst(modelRef, instId, fn, newIdx);
    spawnChildren(nid);
    liveUpdate(); renderAgentTabs(); setActiveAgentTab(nid); renderEditorPanel();
  };
  wrap.appendChild(listDiv);
  wrap.appendChild(addBtn);
}

function makePrimListCtl(instId, fn, fs) {
  var wrap = document.createElement('div');
  wrap.className = 'list-wrap';
  var inst = INST[instId];
  if (!Array.isArray(inst.data[fn])) inst.data[fn] = fs['default'] ? fs['default'].slice() : [];
  var items = inst.data[fn];

  function render() {
    wrap.innerHTML = '';
    var listDiv = document.createElement('div');
    listDiv.className = 'list-items';
    items.forEach(function(val, i) {
      (function(idx){
        var li = document.createElement('div'); li.className = 'list-item';
        var itype = fs.items ? (fs.items.type || 'string') : 'string';
        var isText = !(itype === 'integer' || itype === 'number');
        var ctl;
        if (isText) {
          ctl = makeMlInput(
            function(){ return items[idx]; },
            function(v){ items[idx] = v; inst.data[fn] = items; },
            itype, fn + '[' + idx + ']'
          );
          ctl.style.flex = '1'; ctl.style.minWidth = '0';
        } else {
          var inp = document.createElement('input');
          inp.type = 'number'; inp.className = 'fi'; inp.value = val; inp.placeholder = itype;
          inp.oninput = function(){ items[idx] = inp.value; inst.data[fn] = items; liveUpdate(); };
          ctl = inp;
        }
        var del = document.createElement('button');
        del.className = 'li-del'; del.textContent = '×';
        del.onclick = function(){ items.splice(idx, 1); inst.data[fn] = items; liveUpdate(); render(); };
        li.appendChild(ctl); li.appendChild(del);
        listDiv.appendChild(li);
      })(i);
    });
    var add = document.createElement('button');
    add.className = 'li-add';
    add.innerHTML = '+ Add item';
    add.onclick = function(){ items.push(''); inst.data[fn] = items; liveUpdate(); render(); };
    wrap.appendChild(listDiv); wrap.appendChild(add);
  }
  render();
  return wrap;
}

function makeOptPrimListCtl(instId, fn, fs) {
  var wrap = document.createElement('div');
  wrap.className = 'list-wrap';
  var inst = INST[instId];
  var arrSchema = fs.anyOf.find(function(o){ return o.type === 'array'; }) || {};
  if (inst.primListState[fn] === undefined) {
    inst.primListState[fn] = Array.isArray(inst.data[fn]);
  }
  if (!Array.isArray(inst.data[fn])) {
    inst.data[fn] = (inst.primListCache[fn] || []).slice();
  }

  function render() {
    wrap.innerHTML = '';
    var topRow = document.createElement('div');
    topRow.style.cssText = 'display:flex;align-items:center;gap:7px;margin-bottom:5px;';
    var tog = document.createElement('div');
    var isOn = inst.primListState[fn];
    tog.className = 'ftog' + (isOn ? ' on' : '');
    var lbl = document.createElement('span');
    lbl.style.cssText = 'font-size:10px;color:var(--tx3);font-family:var(--mono);';
    lbl.textContent = (arrSchema.items && arrSchema.items.type || 'any') + '[] | None';
    tog.onclick = function(){
      inst.primListState[fn] = !inst.primListState[fn];
      liveUpdate();
      render();
    };
    topRow.appendChild(tog); topRow.appendChild(lbl);
    wrap.appendChild(topRow);

    var offState = !inst.primListState[fn];
    var items = inst.data[fn];
    var listDiv = document.createElement('div');
    listDiv.className = 'list-items';
    if (offState) listDiv.style.opacity = '0.45';
    items.forEach(function(val, i) {
      (function(idx){
        var li = document.createElement('div'); li.className = 'list-item';
        var itype = arrSchema.items ? (arrSchema.items.type || 'string') : 'string';
        var isText = !(itype === 'integer' || itype === 'number');
        var ctl, inp;
        if (isText) {
          ctl = makeMlInput(
            function(){ return items[idx]; },
            function(v){ items[idx] = v; inst.data[fn] = items; },
            itype, fn + '[' + idx + ']'
          );
          ctl.style.flex = '1'; ctl.style.minWidth = '0';
          inp = ctl.querySelector('input');
          var editBtn = ctl.querySelector('.ml-edit-btn');
          if (editBtn) editBtn.disabled = offState, editBtn.style.opacity = offState ? '0.45' : '1';
        } else {
          inp = document.createElement('input');
          inp.type = 'number'; inp.className = 'fi'; inp.value = val; inp.placeholder = itype;
          inp.oninput = function(){ items[idx] = inp.value; inst.data[fn] = items; liveUpdate(); };
          ctl = inp;
        }
        inp.disabled = offState;
        var del = document.createElement('button');
        del.className = 'li-del'; del.textContent = '×'; del.disabled = offState;
        del.onclick = function(){ if (offState) return; items.splice(idx, 1); inst.data[fn] = items; liveUpdate(); render(); };
        li.appendChild(ctl); li.appendChild(del);
        listDiv.appendChild(li);
      })(i);
    });
    var add = document.createElement('button');
    add.className = 'li-add'; add.disabled = offState;
    if (offState) add.style.opacity = '0.45';
    add.innerHTML = '+ Add item';
    add.onclick = function(){ if (offState) return; items.push(''); inst.data[fn] = items; liveUpdate(); render(); };
    wrap.appendChild(listDiv); wrap.appendChild(add);
  }
  render();
  return wrap;
}

function makeDictCtl(instId, fn, fs) {
  var wrap = document.createElement('div');
  wrap.className = 'list-wrap';
  var inst = INST[instId];
  var dmr = getDictValueModelRef(fs);

  if (dmr) {
    renderModelDictCtl(wrap, instId, fn, dmr);
    return wrap;
  }

  var valType = (fs.additionalProperties && typeof fs.additionalProperties === 'object' && fs.additionalProperties.type) || 'string';
  if (!Array.isArray(inst.data[fn])) inst.data[fn] = [];
  var kvs = inst.data[fn];

  function render() {
    wrap.innerHTML = '';
    kvs.forEach(function(kv, i) {
      (function(idx){
        var li = document.createElement('div'); li.className = 'kv-item';
        var kiCtl = makeMlInput(
          function(){ return kvs[idx].k; },
          function(v){ kvs[idx].k = v; inst.data[fn] = kvs; },
          'key', fn + ' key'
        );
        kiCtl.style.flex = '1'; kiCtl.style.minWidth = '0';
        var colon = document.createElement('span');
        colon.style.cssText = 'color:var(--tx3);font-family:var(--mono);font-size:11px;flex-shrink:0;';
        colon.textContent = ':';
        var isTextVal = !(valType === 'integer' || valType === 'number');
        var vi;
        if (isTextVal) {
          vi = makeMlInput(
            function(){ return kvs[idx].v; },
            function(v){ kvs[idx].v = v; inst.data[fn] = kvs; },
            'value', fn + '.' + (kvs[idx].k || 'key')
          );
          vi.style.flex = '1'; vi.style.minWidth = '0';
        } else {
          vi = document.createElement('input');
          vi.type = 'number'; vi.className = 'fi'; vi.placeholder = 'value';
          vi.value = kv.v !== undefined ? kv.v : '';
          vi.oninput = function(){ kvs[idx].v = vi.value; inst.data[fn] = kvs; liveUpdate(); };
        }
        var del = document.createElement('button');
        del.className = 'li-del'; del.textContent = '×';
        del.onclick = function(){ kvs.splice(idx, 1); inst.data[fn] = kvs; liveUpdate(); render(); };
        li.appendChild(ki); li.appendChild(colon); li.appendChild(vi); li.appendChild(del);
        wrap.appendChild(li);
      })(i);
    });
    var add = document.createElement('button');
    add.className = 'li-add';
    add.innerHTML = '+ Add key-value';
    add.onclick = function(){ kvs.push({ k: '', v: '' }); inst.data[fn] = kvs; liveUpdate(); render(); };
    wrap.appendChild(add);
  }
  render();
  return wrap;
}

// ── Editor panel ───────────────────────────────────────────────────────────

function getAgentInstances(){
  return Object.values(INST).filter(function(i){ return isAgentModel(i.modelName); });
}
function getAgentParentChain(agentId){
  var chain = [];
  var cur = INST[agentId];
  while(cur && cur.parentId){
    var p = INST[cur.parentId];
    if(!p) break;
    if(isAgentModel(p.modelName)) chain.unshift(p);
    cur = p;
  }
  return chain;
}
function renderAgentTabs(){
  var wrap = document.getElementById('agent-tabs-wrap');
  var tabsEl = document.getElementById('agent-tabs');
  if(!AGENT_MANIFEST_MODE){ wrap.classList.remove('on'); return; }
  wrap.classList.add('on');
  tabsEl.innerHTML = '';
  var agents = getAgentInstances();
  agents.sort(function(a,b){ return (a.agentTabIndex||0) - (b.agentTabIndex||0); });
  agents.forEach(function(agent){
    var tab = document.createElement('div');
    tab.className = 'agent-tab' + (agent.id === ACTIVE_AGENT_TAB ? ' active' : '');
    var chain = getAgentParentChain(agent.id);
    if(chain.length > 0){
      var crumb = document.createElement('span');
      crumb.className = 'tab-crumb';
      crumb.textContent = chain.map(function(c){ return agentTabName(c.id); }).join(' → ');
      tab.appendChild(crumb);
    }
    var nameSpan = document.createElement('span');
    nameSpan.textContent = agentTabName(agent.id);
    tab.appendChild(nameSpan);
    tab.onclick = function(){ setActiveAgentTab(agent.id); };
    tabsEl.appendChild(tab);
  });
}
function setActiveAgentTab(agentId){
  ACTIVE_AGENT_TAB = agentId;
  renderAgentTabs();
  renderEditorPanel();
  if(INST[agentId] && INST[agentId].location === 'desktop'){
    focusDwin(agentId);
  }
}

function getAgentSubtree(agentId){
  var result = [];
  var queue = [agentId];
  var visited = {};
  while(queue.length){
    var id = queue.shift();
    if(visited[id]) continue;
    visited[id] = true;
    result.push(id);
    var kids = Object.values(INST).filter(function(i){ return i.parentId === id && !isAgentModel(i.modelName); });
    kids.forEach(function(k){ queue.push(k.id); });
  }
  return result;
}

function renderEditorPanel() {
  var scroll = document.getElementById('editor-scroll');
  var openState = {};
  scroll.querySelectorAll('.m-sec').forEach(function(s){
    openState[s.id] = s.classList.contains('open');
  });
  scroll.innerHTML = '';

  if (AGENT_MANIFEST_MODE){
    renderAgentTabs();
    var active = ACTIVE_AGENT_TAB || ROOT_ID;
    if (!INST[active]) active = ROOT_ID;
    ACTIVE_AGENT_TAB = active;
    var order = getAgentSubtree(active);
    for (var i = 0; i < order.length; i++) {
      var id = order[i];
      if (!INST[id] || INST[id].location !== 'editor') continue;
      var sec = buildInstSection(id);
      if (openState.hasOwnProperty(sec.id)) {
        sec.classList.toggle('open', openState[sec.id]);
      }
      scroll.appendChild(sec);
    }
    return;
  }

  var order = instanceOrder();
  for (var i = 0; i < order.length; i++) {
    var id = order[i];
    if (!INST[id] || INST[id].location !== 'editor') continue;
    var sec = buildInstSection(id);
    if (openState.hasOwnProperty(sec.id)) {
      sec.classList.toggle('open', openState[sec.id]);
    }
    scroll.appendChild(sec);
  }
}

function setAllSections(open) {
  document.querySelectorAll('#editor-scroll .m-sec').forEach(function(s){
    s.classList.toggle('open', open);
  });
}

function buildInstSection(id) {
  var inst = INST[id];
  var model = resolveSchema(inst.modelName);
  var props = model.properties || {};
  var required = model.required || [];
  var displayName = instDisplayName(id);
  var count = Object.keys(props).length;

  var sec = document.createElement('div');
  sec.className = 'm-sec open';
  sec.id = 'sec-' + id;

  var hdr = document.createElement('div');
  hdr.className = 'm-sec-hdr';
  hdr.draggable = !isPanelsOverlayBlocking();

  var arr = document.createElement('div');
  arr.className = 'm-arr';
  arr.innerHTML = '<svg width="6" height="10" viewBox="0 0 6 10" fill="none"><path d="M1 1l4 4-4 4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  var icon = document.createElement('div');
  icon.className = 'm-icon';
  icon.innerHTML = '<svg width="9" height="9" viewBox="0 0 10 10" fill="none"><polygon points="5,1 9,3.5 9,6.5 5,9 1,6.5 1,3.5" stroke="currentColor" stroke-width="1.3"/></svg>';

  var nameEl = document.createElement('span');
  nameEl.className = 'm-name';
  nameEl.textContent = displayName;

  var cntEl = document.createElement('span');
  cntEl.className = 'm-cnt';
  cntEl.textContent = count + 'f';

  hdr.appendChild(arr);
  hdr.appendChild(icon);
  hdr.appendChild(nameEl);
  hdr.appendChild(cntEl);

  if (inst.parentId){
    var parentBtn = document.createElement('button');
    parentBtn.className = 'nav-parent-btn';
    parentBtn.title = 'Go to parent';
    parentBtn.textContent = '↑';
    parentBtn.onclick = function(e){ e.stopPropagation(); navigateToParent(id); };
    hdr.appendChild(parentBtn);
  }
  var findBtn = document.createElement('button');
  findBtn.className = 'nav-find-btn';
  findBtn.title = 'Find child / descendants';
  findBtn.textContent = '↓';
  findBtn.onclick = function(e){ e.stopPropagation(); findChildInteractive(id); };
  hdr.appendChild(findBtn);

  var popBtn = document.createElement('button');
  popBtn.className = 'pop-btn';
  popBtn.title = 'Pop out to desktop';
  popBtn.textContent = '↗';
  popBtn.onclick = function(e){ e.stopPropagation(); popOut(id); if(isMobile()) collapseEditor(); };
  hdr.appendChild(popBtn);

  sec.appendChild(hdr);

  if (model.description && DESC_MODE === 'collapsed') {
    var dcol = document.createElement('div');
    dcol.className = 'm-desc-collapsed';
    dcol.textContent = model.description;
    sec.appendChild(dcol);
  }

  var body = document.createElement('div');
  body.className = 'm-sec-body';

  if (model.description && DESC_MODE === 'expanded') {
    var dexp = document.createElement('div');
    dexp.className = 'm-desc-expanded';
    dexp.textContent = model.description;
    body.appendChild(dexp);
  }

  for (var fn in props) {
    body.appendChild(buildFieldRow(id, fn, props[fn], required));
  }
  sec.appendChild(body);

  hdr.addEventListener('click', function(e){
    if (e.target.closest && e.target.closest('button')) return;
    if (hdr.dataset.justDragged === '1') { hdr.dataset.justDragged = '0'; return; }
    sec.classList.toggle('open');
  });

  hdr.addEventListener('dragstart', function(e){
    if (isPanelsOverlayBlocking()){ e.preventDefault(); return; }
    hdr.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', id);
  });
  hdr.addEventListener('dragend', function(){
    hdr.classList.remove('dragging');
    hideDropTarget();
  });

  return sec;
}

// ── Desktop windows ────────────────────────────────────────────────────────

function popOut(id, atDesktopPos) {
  var inst = INST[id];
  inst.location = 'desktop';
  if (atDesktopPos) {
    inst.pos = atDesktopPos;
  } else if (inst.pos.x === 0 && inst.pos.y === 0) {
    var vp = document.getElementById('desktop-viewport');
    var dw = vp.clientWidth, dh = vp.clientHeight;
    var center = viewportToCanvas(dw / 2, dh / 2);
    inst.pos = {
      x: Math.max(20, center.x - inst.size.w / 2 + (Math.random() * 80 - 40)),
      y: Math.max(20, center.y - inst.size.h / 2 + (Math.random() * 60 - 30))
    };
  }
  renderEditorPanel();
  renderDesktopWindow(id);
  updateDesktopHint();
  renderWires();
}

function returnToEditor(id) {
  var inst = INST[id];
  inst.location = 'editor';
  var el = document.getElementById('dw-' + id);
  if (el) el.remove();
  renderEditorPanel();
  updateDesktopHint();
  renderWires();
  setTimeout(function(){ flashSec(id); }, 80);
}

function renderDesktopWindow(id) {
  var inst = INST[id];
  var model = resolveSchema(inst.modelName);
  var props = model.properties || {};
  var required = model.required || [];

  var old = document.getElementById('dw-' + id);
  if (old) old.remove();

  var isRoot = (id === ROOT_ID);

  var win = document.createElement('div');
  win.className = 'dwin' + (inst.minimized ? ' dwin-min' : '') + (isRoot ? ' dwin-root' : '');
  win.id = 'dw-' + id;
  win.style.cssText = 'left:' + inst.pos.x + 'px;top:' + inst.pos.y + 'px;width:' + inst.size.w + 'px;height:' + (inst.minimized ? 'auto' : inst.size.h + 'px') + ';z-index:' + (++zTop) + ';';

  var hdr = document.createElement('div');
  hdr.className = 'dwin-hdr';

  var icon = document.createElement('div');
  icon.className = 'm-icon';
  icon.style.flexShrink = '0';
  icon.innerHTML = '<svg width="9" height="9" viewBox="0 0 10 10" fill="none"><polygon points="5,1 9,3.5 9,6.5 5,9 1,6.5 1,3.5" stroke="currentColor" stroke-width="1.3"/></svg>';

  var nameEl = document.createElement('span');
  nameEl.className = 'dwin-name';
  nameEl.textContent = instDisplayName(id) + (isRoot ? ' (root)' : '');

  var ctrls = document.createElement('div');
  ctrls.className = 'dwin-ctrls';

  if (inst.parentId){
    var pbtn = document.createElement('div');
    pbtn.className = 'dwc'; pbtn.style.cssText='width:15px;height:15px;background:rgba(255,107,53,.1);color:#ff8c5a;border:1px solid rgba(255,107,53,.25);border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:10px;cursor:pointer;';
    pbtn.textContent='↑';
    pbtn.title='Go to parent';
    (function(instId){ pbtn.onclick = function(){ navigateToParent(instId); }; })(id);
    ctrls.appendChild(pbtn);
  }

  var minBtn = document.createElement('div');
  minBtn.className = 'dwc dwc-m';
  minBtn.title = 'Minimize';
  minBtn.textContent = '−';
  (function(instId){
    minBtn.onclick = function(){
      var i = INST[instId];
      i.minimized = !i.minimized;
      win.classList.toggle('dwin-min', i.minimized);
      if (!i.minimized){
        win.style.height = i.size.h + 'px';
      } else {
        win.style.height = 'auto';
      }
      renderWires();
    };
  })(id);

  ctrls.appendChild(minBtn);

  var closeBtn = document.createElement('div');
  closeBtn.className = 'dwc dwc-x';
  closeBtn.title = 'Return to Editor';
  closeBtn.textContent = '×';
  (function(instId){ closeBtn.onclick = function(){ returnToEditor(instId); }; })(id);
  ctrls.appendChild(closeBtn);

  hdr.appendChild(icon); hdr.appendChild(nameEl); hdr.appendChild(ctrls);
  win.appendChild(hdr);

  var body = document.createElement('div');
  body.className = 'dwin-body';
  body.id = 'dwb-' + id;
  body.addEventListener('scroll', function(){ renderWires(); });

  if (model.description && (DESC_MODE === 'collapsed' || DESC_MODE === 'expanded')) {
    var dd = document.createElement('div');
    dd.className = 'm-desc-expanded';
    dd.textContent = model.description;
    body.appendChild(dd);
  }
  for (var fn in props) {
    body.appendChild(buildFieldRow(id, fn, props[fn], required));
  }
  win.appendChild(body);

  var rsz = document.createElement('div');
  rsz.className = 'win-rsz';
  win.appendChild(rsz);

  document.getElementById('desktop').appendChild(win);
  makeDwinDrag(win, hdr, id);
  makeDwinResize(win, rsz, id);
  win.addEventListener('mousedown', function(){ focusDwin(id); });
  win.addEventListener('touchstart', function(){ focusDwin(id); }, {passive:true});
  focusDwin(id);
}

function focusDwin(id) {
  document.querySelectorAll('.dwin').forEach(function(w){ w.classList.remove('dwin-focused'); });
  var win = document.getElementById('dw-' + id);
  if (win) { win.classList.add('dwin-focused'); win.style.zIndex = ++zTop; }
}

function makeDwinDrag(win, handle, id) {
  var ox, oy, sx, sy, on = false;
  function start(e, cx, cy){
    if (e.target.closest('.dwin-ctrls')) return;
    on = true; ox = cx; oy = cy;
    sx = parseInt(win.style.left); sy = parseInt(win.style.top);
    focusDwin(id);
    e.preventDefault();
  }
  handle.addEventListener('mousedown', function(e){ start(e, e.clientX, e.clientY); });
  handle.addEventListener('touchstart', function(e){
    if(e.touches.length !== 1) return;
    start(e, e.touches[0].clientX, e.touches[0].clientY);
  }, {passive:false});
  function move(cx, cy){
    if (!on) return;
    var nx = sx + (cx - ox) / canvasState.scale;
    var ny = sy + (cy - oy) / canvasState.scale;
    win.style.left = nx + 'px'; win.style.top = ny + 'px';
    INST[id].pos = { x: nx, y: ny };
    renderWires();
  }
  document.addEventListener('mousemove', function(e){ move(e.clientX, e.clientY); });
  document.addEventListener('touchmove', function(e){
    if(!on) return;
    if(e.touches.length===1) move(e.touches[0].clientX, e.touches[0].clientY);
  }, {passive:false});
  function end(){ on = false; }
  document.addEventListener('mouseup', end);
  document.addEventListener('touchend', end);
  document.addEventListener('touchcancel', end);
}

function makeDwinResize(win, handle, id) {
  var ox, oy, sw, sh, on = false;
  function start(e,cx,cy){
    on = true; ox = cx; oy = cy;
    sw = win.offsetWidth; sh = win.offsetHeight;
    e.preventDefault(); e.stopPropagation();
  }
  handle.addEventListener('mousedown', function(e){ start(e, e.clientX, e.clientY); });
  handle.addEventListener('touchstart', function(e){
    if(e.touches.length!==1) return;
    start(e, e.touches[0].clientX, e.touches[0].clientY);
  }, {passive:false});
  function move(cx,cy){
    if (!on) return;
    var nw = Math.max(240, sw + (cx - ox) / canvasState.scale);
    var nh = Math.max(100, sh + (cy - oy) / canvasState.scale);
    win.style.width = nw + 'px'; win.style.height = nh + 'px';
    INST[id].size = { w: nw, h: nh };
    renderWires();
  }
  document.addEventListener('mousemove', function(e){ move(e.clientX, e.clientY); });
  document.addEventListener('touchmove', function(e){
    if(!on) return;
    if(e.touches.length===1) move(e.touches[0].clientX, e.touches[0].clientY);
  }, {passive:false});
  function end(){ on = false; }
  document.addEventListener('mouseup', end);
  document.addEventListener('touchend', end);
}

function updateDesktopHint() {
  var any = Object.values(INST).some(function(i){ return i.location === 'desktop'; });
  var hint = document.getElementById('desktop-hint');
  if (hint) hint.classList.toggle('hidden', any);
}

// ── Wires ──────────────────────────────────────────────────────────────────

function getFieldAnchorEl(parentId, fieldName, listIndex){
  if (listIndex !== null && listIndex !== undefined){
    return document.getElementById('fitem-' + parentId + '-' + fieldName + '-' + listIndex);
  }
  return document.getElementById('fr-' + parentId + '-' + fieldName);
}

function renderWires() {
  var svg = document.getElementById('wires-svg');
  svg.innerHTML = '';
  var desktop = document.getElementById('desktop');
  if (!desktop) return;
  var dRect = desktop.getBoundingClientRect();
  var scale = canvasState.scale;

  Object.values(INST).forEach(function(child) {
    if (child.location !== 'desktop' || child.minimized || !child.parentId) return;
    var parent = INST[child.parentId];
    if (!parent || parent.location !== 'desktop' || parent.minimized) return;

    var pwEl = document.getElementById('dw-' + parent.id);
    var cwEl = document.getElementById('dw-' + child.id);
    if (!pwEl || !cwEl) return;

    var fieldIdx = child.parentIndex;
    var frEl = getFieldAnchorEl(parent.id, child.parentField, fieldIdx);
    if (!frEl){
      frEl = document.getElementById('fr-' + parent.id + '-' + child.parentField);
    }
    if (!frEl) return;

    var frRect = frEl.getBoundingClientRect();
    var pwRect = pwEl.getBoundingClientRect();
    var cwRect = cwEl.getBoundingClientRect();

    var bodyEl = pwEl.querySelector('.dwin-body');
    var bodyRect = bodyEl ? bodyEl.getBoundingClientRect() : pwRect;
    var fieldCenterY = frRect.top + frRect.height/2;
    // clamp to body visible area, never into header
    var sy_view = Math.max(bodyRect.top + 6, Math.min(bodyRect.bottom - 6, fieldCenterY));
    var sx_view = pwRect.right;
    var sx = (sx_view - dRect.left) / scale;
    var sy = (sy_view - dRect.top) / scale;

    var tx = (cwRect.left - dRect.left) / scale;
    var ty = (cwRect.top + 17 - dRect.top) / scale;

    var dx = Math.abs(tx - sx);
    var cpx = dx * 0.45;

    var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M' + sx + ',' + sy + ' C' + (sx + cpx) + ',' + sy + ' ' + (tx - cpx) + ',' + ty + ' ' + tx + ',' + ty);
    path.setAttribute('stroke', 'rgba(230,168,0,0.42)');
    path.setAttribute('stroke-width', 1.5 / scale);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke-dasharray', (5/scale) + ' ' + (4/scale));
    svg.appendChild(path);

    function dot(x, y, color) {
      var c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', 3/scale);
      c.setAttribute('fill', color);
      svg.appendChild(c);
    }
    dot(sx, sy, 'rgba(230,168,0,0.6)');
    dot(tx, ty, 'rgba(230,168,0,0.6)');
  });
}

// ── Canvas pan/zoom ─────────────────────────────────────────────────────────

var canvasState = { x: 0, y: 0, scale: 1 };
var ZOOM_MIN = 0.05, ZOOM_MAX = 8.0, ZOOM_STEP = 0.12;

function applyCanvasTransform() {
  var desktop = document.getElementById('desktop');
  desktop.style.transform = 'translate(' + canvasState.x + 'px,' + canvasState.y + 'px) scale(' + canvasState.scale + ')';
  document.getElementById('zoom-pct').textContent = Math.round(canvasState.scale * 100) + '%';
  var vp = document.getElementById('desktop-viewport');
  var dotSpacing = 24 * canvasState.scale;
  vp.style.backgroundSize = dotSpacing + 'px ' + dotSpacing + 'px';
  vp.style.backgroundPosition = canvasState.x + 'px ' + canvasState.y + 'px';
  renderWires();
}

function viewportToCanvas(vx, vy) {
  return {
    x: (vx - canvasState.x) / canvasState.scale,
    y: (vy - canvasState.y) / canvasState.scale
  };
}

function zoomAt(viewportX, viewportY, newScale) {
  newScale = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, newScale));
  var before = viewportToCanvas(viewportX, viewportY);
  canvasState.scale = newScale;
  canvasState.x = viewportX - before.x * newScale;
  canvasState.y = viewportY - before.y * newScale;
  applyCanvasTransform();
}

function zoomStep(dir) {
  var vp = document.getElementById('desktop-viewport');
  var rect = vp.getBoundingClientRect();
  zoomAt(rect.width / 2, rect.height / 2, canvasState.scale * (dir > 0 ? 1 + ZOOM_STEP : 1 - ZOOM_STEP));
}

function resetCanvasView() {
  canvasState = { x: 0, y: 0, scale: 1 };
  applyCanvasTransform();
  showToast('View reset');
}

function fitCanvasToIds(ids, duration){
  duration = duration || 300;
  var vp = document.getElementById('desktop-viewport');
  var vw = vp.clientWidth, vh = vp.clientHeight;
  var minX=Infinity, minY=Infinity, maxX=-Infinity, maxY=-Infinity;
  var found=false;
  ids.forEach(function(id){
    var inst = INST[id];
    if(!inst || inst.location !== 'desktop') return;
    found=true;
    minX = Math.min(minX, inst.pos.x);
    minY = Math.min(minY, inst.pos.y);
    maxX = Math.max(maxX, inst.pos.x + inst.size.w);
    maxY = Math.max(maxY, inst.pos.y + inst.size.h);
  });
  if(!found) return;
  var pad = 60;
  var bw = maxX - minX, bh = maxY - minY;
  var scaleX = (vw - pad*2) / (bw || 1);
  var scaleY = (vh - pad*2) / (bh || 1);
  var targetScale = Math.min(scaleX, scaleY, ZOOM_MAX);
  targetScale = Math.max(ZOOM_MIN, targetScale);
  var cx = minX + bw/2, cy = minY + bh/2;
  var targetX = vw/2 - cx * targetScale;
  var targetY = vh/2 - cy * targetScale;
  var start = {x: canvasState.x, y: canvasState.y, s: canvasState.scale};
  var t0 = performance.now();
  function easeOutCubic(t){ return 1 - Math.pow(1-t,3); }
  function step(now){
    var p = Math.min(1, (now - t0)/duration);
    var e = easeOutCubic(p);
    canvasState.x = start.x + (targetX - start.x)*e;
    canvasState.y = start.y + (targetY - start.y)*e;
    canvasState.scale = start.s + (targetScale - start.s)*e;
    applyCanvasTransform();
    if(p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function initCanvasControls() {
  var vp = document.getElementById('desktop-viewport');
  var desktop = document.getElementById('desktop');

  vp.addEventListener('wheel', function(e){
    var body = e.target.closest('.dwin-body');
    if (body){
      var scrollable = body.scrollHeight > body.clientHeight + 2;
      if (scrollable){
        e.stopPropagation();
        return; // allow scrolling inside a scrollable node window
      }
    }
    e.preventDefault();
    var rect = vp.getBoundingClientRect();
    var vx = e.clientX - rect.left, vy = e.clientY - rect.top;

    if (e.ctrlKey || e.altKey) {
      // Zoom
      var delta = -e.deltaY * 0.0015;
      zoomAt(vx, vy, canvasState.scale * (1 + delta));
    } else {
      // Canvas scroll (standard behavior)
      var scrollSpeed = (e.deltaMode === 1 ? 16 : 1) * 0.5;
      if (e.shiftKey) {
        // Horizontal scroll
        canvasState.x -= e.deltaY * scrollSpeed;
      } else {
        // Vertical scroll
        canvasState.y -= e.deltaY * scrollSpeed;
      }
      applyCanvasTransform();
    }
  }, { passive: false });

  var panning = false, panStartX, panStartY, camStartX, camStartY;
  function startPan(e, cx, cy) {
    if (e.target.closest('.dwin')) return;
    if (e.button !== undefined && e.button !== 0 && e.button !== 1) return;
    panning = true;
    panStartX = cx; panStartY = cy;
    camStartX = canvasState.x; camStartY = canvasState.y;
    desktop.classList.add('panning');
    e.preventDefault();
  }
  vp.addEventListener('mousedown', function(e){ startPan(e, e.clientX, e.clientY); });
  document.addEventListener('mousemove', function(e){
    if (!panning) return;
    canvasState.x = camStartX + (e.clientX - panStartX);
    canvasState.y = camStartY + (e.clientY - panStartY);
    applyCanvasTransform();
  });
  document.addEventListener('mouseup', function(){
    panning = false;
    desktop.classList.remove('panning');
  });
  vp.addEventListener('auxclick', function(e){ if (e.button === 1) e.preventDefault(); });
  vp.addEventListener('mousedown', function(e){ if (e.button === 1) e.preventDefault(); });

  var touchPan = false, touchStartX, touchStartY, touchCamX, touchCamY;
  var pinchStartDist = 0, pinchStartScale = 1, pinchMid = {x:0,y:0};
  function getTouchDist(touches){
    var dx = touches[0].clientX - touches[1].clientX;
    var dy = touches[0].clientY - touches[1].clientY;
    return Math.hypot(dx, dy);
  }
  vp.addEventListener('touchstart', function(e){
    var inWin = e.target.closest('.dwin');
    if (inWin) { return; }
    e.preventDefault();
    if (e.touches.length === 1){
      touchPan = true;
      touchStartX = e.touches[0].clientX;
      touchStartY = e.touches[0].clientY;
      touchCamX = canvasState.x; touchCamY = canvasState.y;
    } else if (e.touches.length === 2){
      touchPan = false;
      pinchStartDist = getTouchDist(e.touches);
      pinchStartScale = canvasState.scale;
      var rect = vp.getBoundingClientRect();
      pinchMid.x = (e.touches[0].clientX + e.touches[1].clientX)/2 - rect.left;
      pinchMid.y = (e.touches[0].clientY + e.touches[1].clientY)/2 - rect.top;
    }
  }, {passive:false});
  vp.addEventListener('touchmove', function(e){
    var inWin = e.target.closest('.dwin');
    if (inWin) return;
    e.preventDefault();
    if (e.touches.length === 1 && touchPan){
      canvasState.x = touchCamX + (e.touches[0].clientX - touchStartX);
      canvasState.y = touchCamY + (e.touches[0].clientY - touchStartY);
      applyCanvasTransform();
    } else if (e.touches.length === 2){
      var dist = getTouchDist(e.touches);
      var newScale = pinchStartScale * (dist / pinchStartDist);
      zoomAt(pinchMid.x, pinchMid.y, newScale);
    }
  }, {passive:false});
  vp.addEventListener('touchend', function(e){
    if (e.touches.length === 0){ touchPan = false; pinchStartDist = 0; }
  });

  document.getElementById('zoom-in-btn').onclick = function(){ zoomStep(1); };
  document.getElementById('zoom-out-btn').onclick = function(){ zoomStep(-1); };
  document.getElementById('zoom-reset-btn').onclick = resetCanvasView;

  document.addEventListener('keydown', function(e){
    var tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if ((e.ctrlKey || e.metaKey) && e.code === 'Space'){ resetCanvasView(); e.preventDefault(); return; }
    if (e.key === '+' || e.key === '=') { zoomStep(1); e.preventDefault(); }
    else if (e.key === '-' || e.key === '_') { zoomStep(-1); e.preventDefault(); }
    else if (e.key === '0') { resetCanvasView(); e.preventDefault(); }
  });

  applyCanvasTransform();
}

// ── Drag-and-drop ───────────────────────────────────────────────────

function isMobile(){ return window.innerWidth <= 780; }
function isPanelsOverlayBlocking(){
  var dimEl = document.getElementById('canvas-dim');
  return dimEl && dimEl.classList.contains('on');
}

function hideDropTarget() {
  document.getElementById('dz-target').classList.remove('show');
}

function initEditorDnD() {
  var vp = document.getElementById('desktop-viewport');

  vp.addEventListener('dragover', function(e){
    if (isPanelsOverlayBlocking()){ e.preventDefault(); return; }
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    var rect = vp.getBoundingClientRect();
    var dz = document.getElementById('dz-target');
    dz.classList.add('show');
    dz.style.left = (e.clientX - rect.left - 60) + 'px';
    dz.style.top = (e.clientY - rect.top - 30) + 'px';
    dz.style.width = '120px';
    dz.style.height = '60px';
  });
  vp.addEventListener('dragleave', function(e){
    if (e.target === vp) hideDropTarget();
  });
  vp.addEventListener('drop', function(e){
    e.preventDefault();
    hideDropTarget();
    if (isPanelsOverlayBlocking()) return;
    var id = e.dataTransfer.getData('text/plain');
    if (!id || !INST[id]) return;
    var rect = vp.getBoundingClientRect();
    var vx = e.clientX - rect.left, vy = e.clientY - rect.top;
    var canvasPos = viewportToCanvas(vx, vy);
    var inst = INST[id];
    popOut(id, { x: Math.max(0, canvasPos.x - inst.size.w / 2), y: Math.max(0, canvasPos.y - 17) });
  });
}

// ── Navigation ─────────────────────────────────────────────────────────────

var LAST_NAV_SOURCE = null;

function navigateFrom(sourceId, targetId, fieldName, listIndex){
  LAST_NAV_SOURCE = sourceId || null;
  navigateTo(targetId, fieldName, listIndex);
}

function getAgentOwner(instId){
  var cur = INST[instId];
  while(cur){
    if(isAgentModel(cur.modelName)) return cur.id;
    if(!cur.parentId) break;
    cur = INST[cur.parentId];
  }
  return ROOT_ID;
}

function navigateTo(id, fieldName, listIndex) {
  if (!INST[id]) return;
  var inst = INST[id];

  // switch to owning agent tab
  if (AGENT_MANIFEST_MODE){
    var ownerAgent = getAgentOwner(id);
    if (ownerAgent && ownerAgent !== ACTIVE_AGENT_TAB){
      setActiveAgentTab(ownerAgent);
    }
  } else if (isAgentModel(inst.modelName)){
    setActiveAgentTab(id);
  }

  if (inst.location === 'desktop') {
    focusDwin(id);
    var fitIds = [id];
    if (LAST_NAV_SOURCE && INST[LAST_NAV_SOURCE] && INST[LAST_NAV_SOURCE].location === 'desktop'){
      fitIds.push(LAST_NAV_SOURCE);
    }
    fitCanvasToIds(fitIds, 300);
    flashDwin(id);
    if (fieldName){
      setTimeout(function(){
        var el = getFieldAnchorEl(id, fieldName, listIndex);
        if(!el) el = document.getElementById('fr-' + id + '-' + fieldName);
        if(el){ flashField(el); el.scrollIntoView({behavior:'smooth', block:'center'}); }
      }, 120);
    }
  } else {
    var sec = document.getElementById('sec-' + id);
    if (sec) {
      sec.classList.add('open');
      sec.scrollIntoView({ behavior: 'smooth', block: 'center' });
      flashSec(id);
      if (fieldName){
        setTimeout(function(){
          var fel = document.getElementById('fr-' + id + '-' + fieldName);
          if (fel){
            fel.scrollIntoView({behavior:'smooth', block:'center'});
            flashField(fel);
            if (listIndex !== null && listIndex !== undefined){
              var liEl = document.getElementById('fitem-' + id + '-' + fieldName + '-' + listIndex);
              if(liEl) flashListItem(liEl);
            }
          }
        }, 220);
      }
    }
    openEditor();
  }
  LAST_NAV_SOURCE = null;
}

function flashSec(id) {
  var sec = document.getElementById('sec-' + id);
  if (!sec) return;
  sec.classList.add('flash-sec');
  setTimeout(function(){ sec.classList.remove('flash-sec'); }, 900);
}
function flashDwin(id) {
  var win = document.getElementById('dw-' + id);
  if (!win) return;
  var count = 0;
  var iv = setInterval(function(){
    win.style.boxShadow = count % 2 === 0 ? '0 0 0 2px var(--acc),0 0 20px rgba(230,168,0,.3)' : '';
    if (++count >= 4) { clearInterval(iv); win.style.boxShadow = ''; }
  }, 150);
}
function flashField(el){
  if(!el) return;
  el.classList.add('flash-field');
  setTimeout(function(){ el.classList.remove('flash-field'); }, 900);
}
function flashListItem(el){
  if(!el) return;
  el.classList.add('flash-list-item');
  setTimeout(function(){ el.classList.remove('flash-list-item'); }, 900);
}

function findParentInfo(id){
  var inst = INST[id];
  if(!inst || !inst.parentId) return null;
  return {parentId: inst.parentId, field: inst.parentField, index: inst.parentIndex};
}
function navigateToParent(id){
  var info = findParentInfo(id);
  if(!info){ showToast('No parent'); return; }
  LAST_NAV_SOURCE = id;
  navigateTo(info.parentId, info.field, info.index);
}

// Child picker modal
var CHILD_PICKER_CTX = null;
function closeChildPicker(){
  document.getElementById('child-picker-overlay').classList.remove('on');
  CHILD_PICKER_CTX = null;
}
function openChildPicker(parentId){
  var inst = INST[parentId];
  if(!inst) return;
  var descendants = [];
  Object.values(INST).forEach(function(i){
    if(i.id === parentId) return;
    var cur = i;
    while(cur && cur.parentId){
      if(cur.parentId === parentId){ descendants.push(i); break; }
      cur = INST[cur.parentId];
    }
  });
  if(!descendants.length){ showToast('No children'); return; }
  CHILD_PICKER_CTX = {parentId: parentId, list: descendants};
  var listEl = document.getElementById('child-picker-list');
  listEl.innerHTML = '';
  document.getElementById('child-picker-title').textContent = 'Children of ' + (AGENT_MANIFEST_MODE ? agentTabName(parentId) : instDisplayName(parentId));
  descendants.forEach(function(d){
    var item = document.createElement('div');
    item.className = 'child-picker-item';
    item.textContent = (isAgentModel(d.modelName) ? agentTabName(d.id) : instDisplayName(d.id));
    var small = document.createElement('small');
    small.textContent = d.modelName + (d.parentField ? ' · ' + d.parentField + (d.parentIndex !== null ? '['+d.parentIndex+']' : '') : '');
    item.appendChild(small);
    item.onclick = function(){
      closeChildPicker();
      navigateFrom(parentId, d.id, d.parentField, d.parentIndex);
    };
    listEl.appendChild(item);
  });
  document.getElementById('child-picker-overlay').classList.add('on');
}
function findChildInteractive(parentId){
  openChildPicker(parentId);
}

// ── Panel collapse ─────────────────────────────────────────────────────────

function openEditor(){
  var p = document.getElementById('editor-panel');
  var wasCollapsed = p.classList.contains('collapsed');
  p.classList.remove('collapsed');
  if (isMobile()){
    document.getElementById('output-panel').classList.add('collapsed');
  }
  updatePanelsOverlay();
}
function collapseEditor(){
  document.getElementById('editor-panel').classList.add('collapsed');
  updatePanelsOverlay();
}
function toggleEditor(){
  var p = document.getElementById('editor-panel');
  if(p.classList.contains('collapsed')) openEditor(); else collapseEditor();
}
function openOutput(){
  var p = document.getElementById('output-panel');
  p.classList.remove('collapsed');
  if (isMobile()){
    document.getElementById('editor-panel').classList.add('collapsed');
  }
  updatePanelsOverlay();
}
function collapseOutput(){
  document.getElementById('output-panel').classList.add('collapsed');
  updatePanelsOverlay();
}
function toggleOutput(){
  var p = document.getElementById('output-panel');
  if(p.classList.contains('collapsed')) openOutput(); else collapseOutput();
}
function updatePanelsOverlay(){
  var editorOpen = !document.getElementById('editor-panel').classList.contains('collapsed');
  var outputOpen = !document.getElementById('output-panel').classList.contains('collapsed');
  var mobile = isMobile();
  var dim = document.getElementById('canvas-dim');
  var openEditorBtn = document.getElementById('open-editor-btn');
  var openOutputBtn = document.getElementById('open-output-btn');
  var anyOpen = editorOpen || outputOpen;
  var dimOn = mobile && anyOpen;
  dim.classList.toggle('on', dimOn);
  openEditorBtn.classList.toggle('show', !editorOpen);
  openOutputBtn.classList.toggle('show', !outputOpen);
}
window.addEventListener('resize', function(){
  updatePanelsOverlay();
  renderWires();
});

// ── Panel resize ───────────────────────────────────────────────────────────

function initPanelResize() {
  var ePanel = document.getElementById('editor-panel');
  var eRes = document.getElementById('e-resize');
  var on = false, sx, sw;
  eRes.addEventListener('mousedown', function(e){
    on = true; sx = e.clientX; sw = ePanel.offsetWidth;
    eRes.classList.add('active');
    ePanel.classList.add('resizing');
    e.preventDefault();
  });
  document.addEventListener('mousemove', function(e){
    if (!on) return;
    ePanel.style.width = Math.max(180, Math.min(640, sw + e.clientX - sx)) + 'px';
    renderWires();
  });
  document.addEventListener('mouseup', function(){
    on = false; eRes.classList.remove('active');
    ePanel.classList.remove('resizing');
  });

  var oPanel = document.getElementById('output-panel');
  var oRes = document.getElementById('o-resize');
  var on2 = false, sx2, sw2;
  oRes.addEventListener('mousedown', function(e){
    on2 = true; sx2 = e.clientX; sw2 = oPanel.offsetWidth;
    oRes.classList.add('active');
    oPanel.classList.add('resizing');
    e.preventDefault();
  });
  document.addEventListener('mousemove', function(e){
    if (!on2) return;
    oPanel.style.width = Math.max(180, Math.min(640, sw2 - (e.clientX - sx2))) + 'px';
    renderWires();
  });
  document.addEventListener('mouseup', function(){
    on2 = false; oRes.classList.remove('active');
    oPanel.classList.remove('resizing');
  });
}

// ── Output ─────────────────────────────────────────────────────────────────

function liveUpdate() {
  var jout = document.getElementById('jout');
  if (!jout) return;
  try {
    jout.textContent = JSON.stringify(assemble(ROOT_ID), null, 2);
    jout.style.color = 'var(--acc3)';
  } catch(e) {
    jout.textContent = 'Error: ' + e.message;
    jout.style.color = 'var(--err)';
  }
}

function validateAndShow() {
  liveUpdate();
  var errors = [];
  Object.values(INST).forEach(function(inst) {
    if (!inst.enabled) return;
    var model = resolveSchema(inst.modelName);
    var req = model.required || [];
    var props = model.properties || {};
    req.forEach(function(fn) {
      var fs = props[fn] || {};
      if (getDirectRef(fs) || getOptionalRef(fs) || getArrayModelRef(fs) || getUnionRefs(fs)) return;
      var v = inst.data[fn];
      if (inst.data['__disabled__' + fn]) { errors.push(inst.modelName + '.' + fn); return; }
      if (v === undefined || v === null || v === '') errors.push(inst.modelName + '.' + fn);
    });
  });
  var msg = document.getElementById('vmsg');
  if (!msg) return;
  if (errors.length === 0) {
    msg.className = 'ok'; msg.style.display = 'block';
    msg.textContent = '✓ All required fields present';
    setTimeout(function(){ msg.style.display = 'none'; }, 3000);
  } else {
    msg.className = 'err'; msg.style.display = 'block';
    msg.textContent = '✗ Missing:\n  ' + errors.join('\n  ');
  }
}

function copyJson() {
  var t = document.getElementById('jout').textContent;
  if (t) navigator.clipboard.writeText(t).then(function(){ showToast('Copied!'); });
}

function downloadJson() {
  validateAndShow();
  var t = document.getElementById('jout').textContent;
  var a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([t], { type: 'application/json' }));
  a.download = MODEL_LOWER + '.json';
  a.click();
}

function resetAll() {
  document.querySelectorAll('.dwin').forEach(function(w){ w.remove(); });
  var keys = Object.keys(INST);
  keys.forEach(function(k){ delete INST[k]; });
  IC = 0; AGENT_COUNTER = 0;
  ROOT_ID = newInst('__root__', null, null, null);
  ACTIVE_AGENT_TAB = ROOT_ID;
  spawnChildren(ROOT_ID);
  resetCanvasView();
  renderEditorPanel(); liveUpdate(); renderWires(); updateDesktopHint();
  showToast('Reset');
}

function handleFileLoad(e) {
  var file = e.target.files[0];
  if (!file) return;
  var r = new FileReader();
  r.onload = function(ev) {
    try {
      var data = JSON.parse(ev.target.result);
      var inst = INST[ROOT_ID];
      if (inst) {
        Object.keys(data).forEach(function(k) {
          var v = data[k];
          if (typeof v !== 'object' || v === null) inst.data[k] = v;
        });
      }
      renderEditorPanel(); liveUpdate(); showToast('JSON loaded');
    } catch(err) { alert('JSON parse error: ' + err.message); }
  };
  r.readAsText(file); e.target.value = '';
}

// ── Description mode ──────────────────────────────────────────────────────

function cycleDescMode() {
  var modes = ['none', 'collapsed', 'expanded'];
  DESC_MODE = modes[(modes.indexOf(DESC_MODE) + 1) % 3];
  var labels = { none: 'Desc: Off', collapsed: 'Desc: ↓', expanded: 'Desc: ✦' };
  var btn = document.getElementById('desc-btn');
  if (btn) btn.textContent = labels[DESC_MODE];
  renderEditorPanel();
  Object.keys(INST).forEach(function(id) {
    if (INST[id].location === 'desktop') renderDesktopWindow(id);
  });
  renderWires();
}

// ── Toast ──────────────────────────────────────────────────────────────────

function showToast(msg) {
  var t = document.getElementById('toast');
  document.getElementById('toast-msg').textContent = msg;
  t.classList.add('show');
  setTimeout(function(){ t.classList.remove('show'); }, 2400);
}

// ── Modals init ────────────────────────────────────────────────────────────
(function initModals(){
  var mlOverlay = document.getElementById('ml-overlay');
  document.getElementById('ml-close').onclick = closeMultilineEditor;
  document.getElementById('ml-cancel').onclick = closeMultilineEditor;
  document.getElementById('ml-save').onclick = saveMultilineEditor;
  mlOverlay.addEventListener('click', function(e){ if(e.target === mlOverlay) closeMultilineEditor(); });

  var cpOverlay = document.getElementById('child-picker-overlay');
  document.getElementById('child-picker-close').onclick = closeChildPicker;
  document.getElementById('child-picker-cancel').onclick = closeChildPicker;
  cpOverlay.addEventListener('click', function(e){ if(e.target === cpOverlay) closeChildPicker(); });

  document.addEventListener('keydown', function(e){
    if(e.key==='Escape'){
      if (mlOverlay.classList.contains('on')){ closeMultilineEditor(); e.stopPropagation(); }
      else if (cpOverlay.classList.contains('on')){ closeChildPicker(); e.stopPropagation(); }
    }
  });
})();

// ── Agent tabs horizontal scroll ───────────────────────────────────────────
(function(){
  var tabs = document.getElementById('agent-tabs');
  if(!tabs) return;
  tabs.addEventListener('wheel', function(e){
    if(Math.abs(e.deltaY) > Math.abs(e.deltaX)){
      tabs.scrollLeft += e.deltaY;
      e.preventDefault();
    }
  }, {passive:false});
})();

// ── Init ───────────────────────────────────────────────────────────────────

(function init() {
  ROOT_ID = newInst('__root__', null, null, null);
  ACTIVE_AGENT_TAB = ROOT_ID;
  spawnChildren(ROOT_ID);
  initPanelResize();
  initCanvasControls();
  initEditorDnD();
  renderEditorPanel();
  liveUpdate();

  document.getElementById('expand-all-btn').onclick = function(){ setAllSections(true); };
  document.getElementById('collapse-all-btn').onclick = function(){ setAllSections(false); };

  document.getElementById('editor-collapse-btn').onclick = collapseEditor;
  document.getElementById('output-collapse-btn').onclick = collapseOutput;
  document.getElementById('open-editor-btn').onclick = function(e){ e.stopPropagation(); openEditor(); };
  document.getElementById('open-output-btn').onclick = function(e){ e.stopPropagation(); openOutput(); };
  document.getElementById('canvas-dim').onclick = function(){ collapseEditor(); collapseOutput(); };

  document.getElementById('find-child-btn').onclick = function(){
    var current = ACTIVE_AGENT_TAB || ROOT_ID;
    findChildInteractive(current);
  };
  document.getElementById('find-parent-btn').onclick = function(){
    var current = ACTIVE_AGENT_TAB || ROOT_ID;
    navigateToParent(current);
  };

  // initial panel state: desktop = both open, mobile = editor open only
  if (isMobile()){
    collapseOutput();
    openEditor();
  } else {
    document.getElementById('editor-panel').classList.remove('collapsed');
    document.getElementById('output-panel').classList.remove('collapsed');
  }
  updatePanelsOverlay();
})();
</script>
</body>
</html>"""


def _build_html(model_name: str, schema: dict, agent_manifest_mode: bool, accent_colors: dict | None = None) -> str:
    schema_json = json.dumps(schema, ensure_ascii=False)
    # default gold theme – visible on dark and light backgrounds
    acc = (accent_colors or {}).get('acc', '#e6a800')
    acc2 = (accent_colors or {}).get('acc2', '#ff6b35')
    acc3 = (accent_colors or {}).get('acc3', '#00c896')
    return (_TEMPLATE
            .replace('__SCHEMA__', schema_json)
            .replace('__MODEL_NAME__', json.dumps(model_name, ensure_ascii=False))
            .replace('__MODEL_LOWER__', json.dumps(model_name.lower(), ensure_ascii=False))
            .replace('__AGENT_MANIFEST_MODE__', 'true' if agent_manifest_mode else 'false')
            .replace('__ACC__', acc)
            .replace('__ACC2__', acc2)
            .replace('__ACC3__', acc3))


def up_server(
    model: Type[BaseModel],
    port: int = 8000,
    host: str = "0.0.0.0",
    open_browser: bool = True,
    agent_manifest_mode: bool = False,
    accent_colors: dict | None = None,
) -> None:
    """
    Launch a web editor for a Pydantic model.
    Args:
        model:        Pydantic model class (not instance).
        port:         Server port (default 8000).
        host:         Server host (default 0.0.0.0).
        open_browser: Auto-open browser tab (default True).
        agent_manifest_mode: If True, Editor switches to AgentManifest tab mode.
        accent_colors: dict with keys 'acc', 'acc2', 'acc3' to override theme colors.
            Example: accent_colors={'acc':'#e6a800','acc2':'#ff6b35','acc3':'#00c896'}
    """
    schema: dict[str, Any] = model.model_json_schema()
    model_name: str = model.__name__
    html = _build_html(model_name, schema, agent_manifest_mode, accent_colors)

    app = FastAPI(title=f"Pydantic Editor — {model_name}")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(html)

    @app.get("/schema", response_class=JSONResponse)
    async def get_schema() -> dict:
        return schema

    @app.post("/validate", response_class=JSONResponse)
    async def validate(request: Request) -> dict:
        try:
            data = await request.json()
            instance = model.model_validate(data)
            return {"ok": True, "data": instance.model_dump(mode="json")}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    url = f"http://localhost:{port}"
    print(f"\n  Pydantic Editor  |  {model_name}  |  agent_manifest_mode={agent_manifest_mode}  |  {url}\n")

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")


# ─────────────────────────────────────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from enum import Enum
    from typing import List, Optional, Dict
    from pydantic import Field

    class Environment(str, Enum):
        development = "development"
        staging = "staging"
        production = "production"

    class DatabaseConfig(BaseModel):
        """PostgreSQL connection settings."""
        host: str = Field("localhost", description="Database host")
        port: int = Field(5432, ge=1, le=65535, description="Database port")
        name: str = Field(..., description="Database name")
        user: str = Field(..., description="DB user")
        password: str = Field(..., min_length=8, description="Password (min 8 chars)")
        pool_size: int = Field(10, ge=1, le=100, description="Connection pool size")
        ssl: bool = Field(False, description="Use SSL")
        tags: Dict[str, str] = Field(default_factory=dict, description="Extra tags")

    class CacheConfig(BaseModel):
        """Redis cache settings."""
        enabled: bool = Field(True, description="Enable cache")
        host: str = Field("localhost", description="Redis host")
        port: int = Field(6379, description="Redis port")
        ttl_seconds: int = Field(3600, ge=0, description="Entry TTL")

    class ServiceConfig(BaseModel):
        """Microservice configuration."""
        name: str = Field(..., description="Service name")
        replicas: int = Field(1, ge=1, le=50, description="Replica count")
        memory_mb: int = Field(512, ge=64, description="Memory limit (MB)")
        cpu_cores: float = Field(0.5, ge=0.1, le=32.0, description="CPU cores")
        enabled: bool = Field(True)
        env: Optional[Environment] = Field(None, description="Override environment")

    class AppConfig(BaseModel):
        """Main application configuration."""
        app_name: str = Field(..., description="Application name", min_length=2)
        version: str = Field("1.0.0", description="Version (semver)")
        environment: Environment = Field(Environment.development)
        debug: bool = Field(False, description="Debug mode")
        allowed_hosts: List[str] = Field(default_factory=list, description="Allowed hosts")
        database: DatabaseConfig
        cache: Optional[CacheConfig] = Field(None, description="Cache (optional)")
        services: List[ServiceConfig] = Field(default_factory=list, description="Microservices")
        max_request_size_mb: int = Field(10, ge=1, le=1024)
        admin_email: Optional[str] = Field(None, description="Admin email")

    up_server(AppConfig, port=8000, agent_manifest_mode=False,
              accent_colors={'acc':'#e6a800','acc2':'#ff6b35','acc3':'#00c896'})
