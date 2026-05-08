import streamlit as st
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
import torch
import open_clip
from insightface.app import FaceAnalysis

# --- Config ---
HERE             = Path(__file__).parent
ARC_FILE         = HERE / "player_embeddings.npz"
CLIP_FILE        = HERE / "clip_embeddings.npz"
REPS_DIR         = HERE / "representatives"
TOP_K            = 3
MAX_IMAGE_SIDE   = 1024


# --- Cached one-time loads ---
@st.cache_resource
def load_arc_model():
    a = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    a.prepare(ctx_id=0, det_size=(320, 320))
    return a

@st.cache_resource
def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    model.eval()
    return model, preprocess

@st.cache_resource
def load_player_matrices():
    arc  = np.load(ARC_FILE)
    clip = np.load(CLIP_FILE)
    arc_keys   = list(arc.files)
    clip_keys  = list(clip.files)
    arc_matrix  = np.stack([arc[k]  for k in arc_keys])
    clip_matrix = np.stack([clip[k] for k in clip_keys])
    return arc_keys, arc_matrix, clip_keys, clip_matrix


face_app                                            = load_arc_model()
clip_model, clip_preprocess                         = load_clip_model()
ARC_KEYS, ARC_MATRIX, CLIP_KEYS, CLIP_MATRIX        = load_player_matrices()


# --- Embedding + matching ---
def match_arcface(image_path):
    img = np.array(Image.open(image_path).convert("RGB"))
    faces = face_app.get(img)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    q = faces[0].normed_embedding
    scores = ARC_MATRIX @ q
    top = np.argsort(-scores)[:TOP_K]
    return [(ARC_KEYS[i], float(scores[i])) for i in top]


@torch.no_grad()
def match_clip(image_path):
    image = clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    feat = clip_model.encode_image(image)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    q = feat.squeeze(0).cpu().numpy()
    scores = CLIP_MATRIX @ q
    top = np.argsort(-scores)[:TOP_K]
    return [(CLIP_KEYS[i], float(scores[i])) for i in top]


def render_results(results, title, emoji):
    st.markdown(f"### {emoji} {title}")
    if results is None:
        st.error("No face detected.")
        return
    for rank, (player_key, score) in enumerate(results, start=1):
        team, name = player_key.split("__", 1)
        display    = name.replace("_", " ")
        rep_path   = REPS_DIR / f"{player_key}.jpg"
        st.markdown(f"**#{rank} — {display} ({team})**  &nbsp; `{score:+.3f}`")
        if rep_path.exists():
            st.image(str(rep_path), width=240)
        else:
            st.warning("Representative image missing.")


# --- Page ---
st.set_page_config(page_title="ArcFace vs CLIP", layout="wide")
st.title("Footballer Lookalike — ArcFace vs CLIP")
st.write(
    "Upload one or more selfies. Each is matched against the same 1,200-player pool "
    "using two different embedding models, side by side. **Your photos are processed "
    "in memory and never stored.**"
)

uploads = st.file_uploader(
    "Upload selfies",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploads:
    for i, uploaded in enumerate(uploads, start=1):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.divider()
        st.subheader(f"Selfie {i}: `{uploaded.name}`")

        with st.spinner(f"Matching selfie {i}..."):
            arc_results  = match_arcface(tmp_path)
            clip_results = match_clip(tmp_path)

        # Layout: selfie on far left, ArcFace column, CLIP column
        col_selfie, col_arc, col_clip = st.columns([1, 2, 2])
        with col_selfie:
            st.image(tmp_path, caption="Your selfie", width=220)
        with col_arc:
            render_results(arc_results,  "ArcFace (face-specific)", "🎯")
        with col_clip:
            render_results(clip_results, "CLIP (general image)",    "🌐")