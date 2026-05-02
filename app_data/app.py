import gradio as gr
import numpy as np
from pathlib import Path
from PIL import Image
from insightface.app import FaceAnalysis

# --- Config ---
HERE              = Path(__file__).parent
EMBEDDINGS_FILE   = HERE / "player_embeddings.npz"
REPS_DIR          = HERE / "representatives"
TOP_K             = 3
MAX_IMAGE_SIDE    = 1024

# --- One-time startup (loads model + embeddings once) ---
print("Loading face model...")
face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(320, 320))

print("Loading player fingerprints...")
data           = np.load(EMBEDDINGS_FILE)
PLAYER_KEYS    = list(data.files)
PLAYER_MATRIX  = np.stack([data[k] for k in PLAYER_KEYS])    # (P, 512)
print(f"Ready. {len(PLAYER_KEYS)} players loaded.")

# --- Core matching function ---
def find_lookalikes(selfie_pil):
    if selfie_pil is None:
        return [], "Please upload a selfie first."
    pil = selfie_pil.convert("RGB")
    if max(pil.size) > MAX_IMAGE_SIDE:
        pil.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    img = np.array(pil)

    faces = face_app.get(img)
    if not faces:
        return [], "No face detected — try a clearer, forward-facing photo."
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    q = faces[0].normed_embedding

    scores = PLAYER_MATRIX @ q
    top_idx = np.argsort(-scores)[:TOP_K]

    gallery, lines = [], []
    for rank, i in enumerate(top_idx, 1):
        key = PLAYER_KEYS[i]
        score = float(scores[i])
        team, name = key.split("__", 1)
        path = REPS_DIR / f"{key}.jpg"
        if path.exists():
            caption = f"#{rank}  {name.replace('_', ' ')} ({team}) — {score:+.3f}"
            gallery.append((str(path), caption))
            lines.append(caption)
    return gallery, "\n".join(lines)


# --- UI ---
with gr.Blocks(title="Footballer Lookalike", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Which footballer do you look like?")
    gr.Markdown(
        "Upload a clear, forward-facing selfie and we'll match you "
        f"against players from the 2026 FIFA World Cup. "
        "Your photo is processed in memory and never stored."
    )

    with gr.Row():
        with gr.Column(scale=1):
            selfie_in   = gr.Image(type="pil", label="Your selfie", sources=["upload", "webcam"])
            submit_btn  = gr.Button("Find my lookalikes", variant="primary")
        with gr.Column(scale=2):
            gallery_out = gr.Gallery(label=f"Top {TOP_K} matches",
                                     columns=TOP_K, height=360, object_fit="contain")
            text_out    = gr.Textbox(label="Scores", lines=3, interactive=False)

    submit_btn.click(find_lookalikes, inputs=[selfie_in],
                     outputs=[gallery_out, text_out])

if __name__ == "__main__":
    demo.launch()