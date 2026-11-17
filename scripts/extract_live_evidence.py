import os
from fractions import Fraction
import av
import numpy as np
from PIL import Image, ImageDraw

VIDEO_SRC = r"FormatFactory Screen Record20261117_045656.mp4"
OUT_DIR = "docs/assets"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_MP4 = os.path.join(OUT_DIR, "roomforge_live_restaging_evidence.mp4")
OUT_GIF = os.path.join(OUT_DIR, "roomforge_live_restaging_evidence.gif")
OUT_COMPARISON = os.path.join(OUT_DIR, "before_after_restaging.jpg")

print(f"Opening {VIDEO_SRC}...")
container = av.open(VIDEO_SRC)
fps = 24

selected_pil_frames = []
before_frame_img = None
after_frame_img = None

frame_idx = 0
try:
    for frame in container.decode(video=0):
        sec = frame_idx / fps
        
        # Save before comparison frame around 22s
        if before_frame_img is None and 20.0 <= sec <= 24.0:
            before_frame_img = frame.to_image()
            
        # Save after comparison frame around 142s
        if after_frame_img is None and 140.0 <= sec <= 145.0:
            after_frame_img = frame.to_image()

        if 18.0 <= sec <= 48.0:
            # Normal speed: active prompt input & selection
            if frame_idx % 2 == 0: # 12 fps is ultra smooth and fast to encode
                selected_pil_frames.append(frame.to_image().resize((1280, 800), Image.Resampling.BILINEAR))
        elif 50.0 <= sec <= 64.0:
            # Fast-forward transition
            if frame_idx % 6 == 0:
                selected_pil_frames.append(frame.to_image().resize((1280, 800), Image.Resampling.BILINEAR))
        elif 126.0 <= sec <= 151.0:
            # Normal speed: viewing transformed room in 3D
            if frame_idx % 2 == 0:
                selected_pil_frames.append(frame.to_image().resize((1280, 800), Image.Resampling.BILINEAR))

        frame_idx += 1
except Exception as e:
    print(f"Decoded up to frame {frame_idx}: {e}")

print(f"Collected {len(selected_pil_frames)} processed highlight frames.")

# 1. Export Side-by-Side Comparison Image
if before_frame_img and after_frame_img:
    w, h = 960, 600
    b_thumb = before_frame_img.resize((w, h), Image.Resampling.LANCZOS)
    a_thumb = after_frame_img.resize((w, h), Image.Resampling.LANCZOS)
    
    comp = Image.new("RGB", (w * 2 + 30, h + 80), (15, 23, 42))
    comp.paste(b_thumb, (10, 70))
    comp.paste(a_thumb, (w + 20, 70))
    
    draw = ImageDraw.Draw(comp)
    draw.text((20, 20), "ROOMFORGE: LIVE SPATIAL RESTAGING EVIDENCE (KAGGLE DUAL TESLA T4 GPU)", fill=(245, 158, 11))
    draw.text((20, 48), "Original Interactive 3D Room (Base Royal Navy)", fill=(148, 163, 184))
    draw.text((w + 20, 48), "Live Restaged 3D Room (Generative Red Velvet via Kaggle GPU)", fill=(56, 189, 248))
    
    comp.save(OUT_COMPARISON, quality=92)
    print(f"Saved comparison image: {OUT_COMPARISON}")

# 2. Export Highlight MP4
output_container = av.open(OUT_MP4, mode="w")
out_stream = output_container.add_stream("libx264", rate=Fraction(12, 1))
out_stream.width = 1280
out_stream.height = 800
out_stream.pix_fmt = "yuv420p"

for pil_img in selected_pil_frames:
    vframe = av.VideoFrame.from_image(pil_img)
    for packet in out_stream.encode(vframe):
        output_container.mux(packet)

for packet in out_stream.encode():
    output_container.mux(packet)
output_container.close()
print(f"Saved highlight MP4: {OUT_MP4}")

# 3. Export Lightweight Animated GIF Showcase
print("Generating animated GIF showcase...")
gif_frames = []
# Sample every 2nd frame from selected_pil_frames (6 fps) and downscale to 640x400
for idx, pil_img in enumerate(selected_pil_frames):
    if idx % 2 == 0:
        im = pil_img.resize((640, 400), Image.Resampling.BILINEAR)
        im = im.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        gif_frames.append(im)

if gif_frames:
    gif_frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=gif_frames[1:],
        duration=160,
        loop=0,
        optimize=True,
    )
    print(f"Saved animated GIF: {OUT_GIF} ({len(gif_frames)} frames)")

print("All live evidence extracted successfully!")
