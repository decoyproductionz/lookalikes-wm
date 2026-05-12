import os
import json
import datetime
import numpy as np
import torch
import gradio as gr
import open_clip
from pathlib import Path
from PIL import Image
from insightface.app import FaceAnalysis

# --- Config ---
HERE        = Path(__file__).parent
ARC_FILE    = HERE / "player_embeddings.npz"
CLIP_FILE   = HERE / "clip_player_embeddings.npz"
REPS_DIR    = HERE / "representatives"
TOP_K       = 3
ARC_WEIGHT  = 0.6
CLIP_WEIGHT = 0.4
GA_ID       = os.environ.get("GA_MEASUREMENT_ID", "G-XXXXXXXXXX")
LOG_FILE    = Path("/data/usage.jsonl") if Path("/data").exists() else HERE / "usage.jsonl"

# --- Load models once on cold start ---
face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(320, 320))

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
clip_model.eval()

# --- Load player matrices ---
arc_data    = np.load(ARC_FILE)
clip_data   = np.load(CLIP_FILE)
COMMON_KEYS = sorted(set(arc_data.files) & set(clip_data.files))
ARC_MATRIX  = np.stack([arc_data[k]  for k in COMMON_KEYS])
CLIP_MATRIX = np.stack([clip_data[k] for k in COMMON_KEYS])
print(f"Loaded {len(COMMON_KEYS)} players.")

# --- Server-side event logging ---
def log_event(event_type, details=None):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps({
                "ts": datetime.datetime.utcnow().isoformat(),
                "event": event_type,
                "details": details or {}
            }) + "\n")
    except Exception:
        pass  # don't let logging crash the request

# --- Matching logic ---
@torch.no_grad()
def clip_embed_path(img_path):
    with Image.open(img_path) as pil:
        pil = pil.convert("RGB")
        tensor = clip_preprocess(pil).unsqueeze(0)
    feat = clip_model.encode_image(tensor)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy()

def find_lookalike(selfie_path):
    img = np.array(Image.open(selfie_path).convert("RGB"))
    faces = face_app.get(img)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]),
               reverse=True)
    q_arc  = faces[0].normed_embedding
    q_clip = clip_embed_path(selfie_path)
    arc_scores  = ARC_MATRIX  @ q_arc
    clip_scores = CLIP_MATRIX @ q_clip
    arc_n  = (arc_scores  - arc_scores.min())  / (arc_scores.ptp()  + 1e-9)
    clip_n = (clip_scores - clip_scores.min()) / (clip_scores.ptp() + 1e-9)
    combined = ARC_WEIGHT * arc_n + CLIP_WEIGHT * clip_n
    top = np.argsort(-combined)[:TOP_K]
    return [(COMMON_KEYS[i], float(combined[i]),
             float(arc_scores[i]), float(clip_scores[i])) for i in top]

# --- Gradio handler ---
def gradio_lookalike(selfie_path):
    blank = [None, "", None, "", None, ""]
    log_event("page_submit")

    if selfie_path is None:
        return [*blank, "Please upload a selfie first."]

    results = find_lookalike(selfie_path)
    if results is None:
        log_event("no_face_detected")
        return [*blank, "No face detected — try a clearer, forward-facing photo."]

    log_event("match_success",
              {"top_player": results[0][0], "top_combined": results[0][1]})

    images, captions = [None]*3, [""]*3
    for i, (player_key, combined, _, _) in enumerate(results):
        team, name = player_key.split("__", 1)
        rep_path = REPS_DIR / f"{player_key}.jpg"
        if rep_path.exists():
            pct = int(round(combined * 100))
            images[i] = str(rep_path)
            captions[i] = (
                f"### {name.replace('_', ' ')}\n"
                f"**{team}**  \n"
                f"{pct}% match"
            )
    return [images[0], captions[0],
            images[1], captions[1],
            images[2], captions[2],
            ""]

# --- Google Analytics snippet ---
GA_SNIPPET = f"""
<script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_ID}');
</script>
"""

# --- UI ---
arc_pct  = int(round(ARC_WEIGHT  * 100))
clip_pct = int(round(CLIP_WEIGHT * 100))

with gr.Blocks(
    title=f"Footballer Lookalike — ArcFace {arc_pct}% / CLIP {clip_pct}%",
    theme=gr.themes.Soft(),
    head=GA_SNIPPET
) as demo:
    gr.Markdown(
        f"# Which footballer do you look like? ⚽️👤  "
        f"<span style='font-size:0.55em; opacity:0.65; font-weight:normal;'>"
        f"ArcFace {arc_pct}% &nbsp;·&nbsp; CLIP {clip_pct}%</span>"
    )
    gr.Markdown(
        "Upload a clear, forward-facing selfie. We combine **facial structure** "
        "(ArcFace) with **appearance similarity** (CLIP) to find your closest "
        "match. Your picture is processed in memory and never stored."
    )

    with gr.Row():
        with gr.Column(scale=1):
            selfie_in  = gr.Image(type="filepath", label="Your selfie",
                                  sources=["upload"])
            submit_btn = gr.Button("Find my lookalikes", variant="primary")
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column():
                    img1 = gr.Image(show_label=False, height=240, interactive=False)
                    cap1 = gr.Markdown("")
                with gr.Column():
                    img2 = gr.Image(show_label=False, height=240, interactive=False)
                    cap2 = gr.Markdown("")
                with gr.Column():
                    img3 = gr.Image(show_label=False, height=240, interactive=False)
                    cap3 = gr.Markdown("")
            status_out = gr.Markdown("")

    gr.Markdown(
        f"<div style='text-align:center; font-size:0.85em; opacity:0.6; "
        f"margin-top:1em;'>Model weights: ArcFace {arc_pct}% &nbsp;·&nbsp; "
        f"CLIP {clip_pct}%</div>"
    )

    submit_btn.click(
        gradio_lookalike, inputs=[selfie_in],
        outputs=[img1, cap1, img2, cap2, img3, cap3, status_out]
    )

if __name__ == "__main__":
    demo.launch()