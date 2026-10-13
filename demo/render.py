"""Render an explanatory animation from freshly executed C++ fixture results.
Requires Python 3, Pillow, g++/make and ffmpeg. No weights or remote services.
Run from the repository root: python3 demo/render.py
"""
import json, subprocess, hashlib, pathlib, tempfile, math
from PIL import Image, ImageDraw, ImageFont
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE='51a7a9f0aa6d3e792ec7b41c2e88d1c565877ea9'
EXPECTED_KERNEL='38f4b4692297314e6eb01d0e830cf8e496dcbaddb1d2ab922fb6acd42273fba0'
if hashlib.sha256((ROOT/'step3_attention.cpp').read_bytes()).hexdigest()!=EXPECTED_KERNEL:
    raise SystemExit('Kernel changed: reverify source provenance before regenerating media.')
def run(cmd):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,check=True)
    return p.stdout+p.stderr
log=run(['make','all','test','portable-demo'])
log+=run(['g++','-O2','-std=c++17','demo/fixture.cpp','-o','build/demo_fixture'])
result=json.loads(run(['./build/demo_fixture']))
assert result['earlier_channels_unchanged']==128 and result['finite']
(ROOT/'demo/core-run.txt').write_text(log)
receipt={'source_commit':SOURCE,'kernel_sha256':hashlib.sha256((ROOT/'step3_attention.cpp').read_bytes()).hexdigest(),'fixture':{'before':[1,2,3],'after':[1,2,1000],'channels':64,'heads':8,'projections':'identity'},'result':result,'presentation':'Rendered explanation of actual fixture outputs, not a screen recording.','scope':'Educational attention kernel only; no trained-model inference or performance claim.'}
(ROOT/'demo/results.json').write_text(json.dumps(receipt,indent=2)+'\n')
W,H,FPS,SECONDS=960,540,12,16
font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
mono='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
def f(n,m=False): return ImageFont.truetype(mono if m else font,n)
with tempfile.TemporaryDirectory() as td:
    frames=pathlib.Path(td)
    for i in range(FPS*SECONDS):
        t=i/FPS; stage=min(int(t//4),3)
        im=Image.new('RGB',(W,H),'#0b1020'); d=ImageDraw.Draw(im)
        d.rounded_rectangle((30,24,930,512),radius=22,fill='#141d32',outline='#283953',width=1)
        d.text((58,44),'MiniGPTEngine / C++',font=f(18),fill='#8eacc8')
        titles=['Causal attention, made visible','Only the past is visible','Change the future. Preserve the past.','Verified by the real C++ kernel']
        d.text((58,83),titles[stage],font=f(30),fill='#f2f6ff')
        d.text((58,130),'8 heads  /  64 channels  /  identity Q, K, V  /  fixture data',font=f(16),fill='#8eacc8')
        if stage in (0,1):
            d.text((65,184),'QUERY  →  VISIBLE KEYS',font=f(16,True),fill='#79dcca')
            for q in range(3):
                d.text((65,232+q*63),f'Token {q+1}',font=f(20),fill='#ecf2ff')
                for k in range(3):
                    x=240+k*170; y=219+q*63; visible=k<=q
                    d.rounded_rectangle((x,y,x+140,y+46),radius=8,fill='#185a5d' if visible else '#222e45')
                    d.text((x+18,y+10),f'{k+1}  '+('visible' if visible else 'masked'),font=f(16),fill='#a2f5df' if visible else '#718198')
            d.text((65,437),'Future keys never participate in an earlier query.',font=f(20),fill='#b9c9dc')
        elif stage==2:
            d.text((65,187),'INPUT FIXTURE',font=f(16,True),fill='#79dcca')
            for n,v in enumerate([1,2,1000]):
                x=65+n*280; color='#ffb878' if n==2 else '#a2f5df'
                d.rounded_rectangle((x,226,x+250,346),radius=12,fill='#202d46')
                d.text((x+20,241),f'Token {n+1}',font=f(18),fill='#8eacc8')
                d.text((x+20,278),'3 → 1000' if n==2 else str(v),font=f(28,True),fill=color)
            d.text((65,392),'128 earlier output channels: exactly unchanged',font=f(23),fill='#a2f5df')
            d.text((65,434),'The updated final token stays finite.',font=f(20),fill='#b9c9dc')
        else:
            d.text((65,184),'ACTUAL OUTPUT · FIRST CHANNEL PER TOKEN',font=f(15,True),fill='#79dcca')
            d.text((65,225),'              before         after',font=f(20,True),fill='#8eacc8')
            for n,(a,b) in enumerate(zip(result['before'],result['after'])):
                d.text((65,270+n*43),f'Token {n+1}     {a:10.6f}    {b:10.6f}',font=f(20,True),fill='#a2f5df' if n<2 else '#ffb878')
            d.text((65,421),'make all test portable-demo  ·  PASS',font=f(20,True),fill='#a2f5df')
        d.text((58,477),'Educational kernel · no trained-model generation · source '+SOURCE[:12],font=f(13),fill='#8eacc8')
        d.rounded_rectangle((30,529,30+int(900*(i+1)/(FPS*SECONDS)),533),radius=2,fill='#79dcca')
        im.save(frames/f'{i:04}.png')
    subprocess.run(['ffmpeg','-v','error','-y','-framerate',str(FPS),'-i',str(frames/'%04d.png'),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(ROOT/'demo/causal-attention.mp4')],check=True)
    subprocess.run(['ffmpeg','-v','error','-y','-framerate',str(FPS),'-i',str(frames/'%04d.png'),'-filter_complex','[0:v]split[a][b];[a]palettegen[p];[b][p]paletteuse','-loop','0',str(ROOT/'demo/causal-attention.gif')],check=True)
    Image.open(frames/'0170.png').save(ROOT/'demo/preview.png')
    sheet=Image.new('RGB',(960,540))
    for k,n in enumerate([24,72,120,170]):
        shot=Image.open(frames/f'{n:04}.png').resize((480,270))
        sheet.paste(shot,((k%2)*480,(k//2)*270))
    sheet.save(ROOT/'demo/contact.png')
print(json.dumps(receipt,indent=2))
