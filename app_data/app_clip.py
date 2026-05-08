import streamlit as st
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image
import torch
import open_clip

# --- Config ---
HERE             = Path(__file__).parent
EMBEDDINGS_FILE  = HERE / "clip_player_embeddings.npz"
REPS_DIR         = HERE / "representatives"
TOP_K            = 3


# --- Cached loads (run once per Streamlit session) ---
@st.cache_resource
def load_clip_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    return model, preprocess

@st.cache_resource
def load_player_data():
    data   = np.load(EMBEDDINGS_FILE)
    keys   = list(data.files)
    matrix = np.stack([data[k] for k in keys])
    return keys, matrix


clip_model, clip_preprocess = load_clip_model()
PLAYER_KEYS, PLAYER_MATRIX  = load_player_data()


# --- Embedding + matching ---
@torch.no_grad()
def find_lookalikes(image_path):
    image = clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    feat = clip_model.encode_image(image)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    q = feat.squeeze(0).cpu().numpy()
    scores = PLAYER_MATRIX @ q
    top_idx = np.argsort(-scores)[:TOP_K]
    return [(PLAYER_KEYS[i], float(scores[i])) for i in top_idx]


# --- Page ---
st.set_page_config(page_title="Footballer Lookalike (CLIP)", layout="centered")
st.title("Which footballer do you look like? — CLIP version")
st.write(
    "Upload a clear selfie. We'll match you against players from the 2026 FIFA "
    "World Cup using CLIP, a general-purpose image transformer rather than a "
    "face-specific model. Your photo is processed in memory and never stored."
)

uploaded = st.file_uploader("Your selfie", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.image(tmp_path, caption="Your selfie", width=200)

    if st.button("Find my lookalikes", type="primary"):
        with st.spinner("Matching..."):
            results = find_lookalikes(tmp_path)

        st.subheader(f"Top {TOP_K} matches")

        # ---- Vertical column: rank 1 at the top, rank 3 at the bottom ----
        for rank, (player_key, score) in enumerate(results, start=1):
            team, name      = player_key.split("__", 1)
            display_name    = name.replace("_", " ")
            rep_path        = REPS_DIR / f"{player_key}.jpg"

            st.markdown(f"### #{rank} — {display_name} ({team})")
            if rep_path.exists():
                st.image(str(rep_path), width=300)
            else:
                st.warning("No representative image found for this player.")
            st.markdown(f"**Cosine similarity:** {score:+.3f}")
            st.divider()