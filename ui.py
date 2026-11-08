"""Streamlit app to verify / correct the transcripts in review.jsonl.

Run:
    pip install streamlit
    streamlit run ui.py

Reads  <prepared>/review.jsonl   (made by evaluation_preparing.py)
Saves  <prepared>/annotations.json  progress, one entry per clip (resumable)
Writes <prepared>/verified.jsonl    NeMo manifest of human-checked clips, ready for WER

Only clips you approved or edited go into verified.jsonl; rejected and pending
clips are left out. The stored text goes through the same normalize_text() that
should be applied to model output when scoring.
"""

import base64
import html
import json
import os
import time
from difflib import SequenceMatcher
from pathlib import Path

import soundfile as sf
import streamlit as st

from evaluation_preparing import normalize_text

st.set_page_config(page_title="Transcript review", page_icon="🎧", layout="wide")

STATUS = {  # key: (label, colour)
    "pending": ("Pending", "#94a3b8"),
    "verified": ("Verified", "#16a34a"),
    "edited": ("Edited", "#2563eb"),
    "rejected": ("Rejected", "#dc2626"),
}



@st.cache_resource
def font_face_css() -> str:
    """Vazirmatn embedded as base64 from ./fonts, so it works offline / behind a firewall."""
    rules = []
    for weight, name in ((400, "Regular"), (600, "SemiBold")):
        f = Path(__file__).parent / "fonts" / f"Vazirmatn-{name}.woff2"
        if f.exists():
            data = base64.b64encode(f.read_bytes()).decode()
            rules.append(f"@font-face {{font-family: 'Vazirmatn'; font-weight: {weight}; "
                         f"src: url(data:font/woff2;base64,{data}) format('woff2');}}")
    return "\n".join(rules)


FONT = "'Vazirmatn', 'Vazir', Tahoma, 'Noto Naskh Arabic', sans-serif"

st.markdown(
    "<style>" + font_face_css() + "</style>"
    + """
<style>
html, body, .stApp, .stMarkdown, button, input, textarea {font-family: %s !important;}
.block-container {padding-top: 2rem; max-width: 1200px;}
.fa {direction: rtl; text-align: right; font-size: 1.25rem; line-height: 2.2;}
.stTextArea textarea {direction: rtl; text-align: right; font-size: 1.25rem; line-height: 2.1;}
.card {background: var(--secondary-background-color); border-radius: 14px; padding: 1rem 1.3rem;
       margin-bottom: 1rem; border: 1px solid rgba(128,128,128,.18);}
.card h4 {margin: 0 0 .6rem 0; font-size: .8rem; letter-spacing: .08em; text-transform: uppercase;
          opacity: .65; font-weight: 600;}
.badge {display: inline-block; padding: .15rem .7rem; border-radius: 999px; color: #fff;
        font-size: .8rem; font-weight: 600; vertical-align: middle;}
.meta {opacity: .65; font-size: .9rem;}
ins {background: rgba(22,163,74,.25); text-decoration: none; border-radius: 4px; padding: 0 3px;}
del {background: rgba(220,38,38,.25); border-radius: 4px; padding: 0 3px;}
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
""" % FONT,
    unsafe_allow_html=True,
)


# ---------- helpers ----------

def word_errors(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def diff_html(old: str, new: str) -> str:
    a, b = old.split(), new.split()
    out = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(html.escape(" ".join(b[j1:j2])))
        else:
            if i2 > i1:
                out.append(f"<del>{html.escape(' '.join(a[i1:i2]))}</del>")
            if j2 > j1:
                out.append(f"<ins>{html.escape(' '.join(b[j1:j2]))}</ins>")
    return " ".join(out)


def write_json_atomic(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@st.cache_data(show_spinner=False)
def load_items(folder: str) -> list[dict]:
    root = Path(folder)
    review = read_jsonl(root / "review.jsonl")
    meta = {}
    if (root / "manifest.jsonl").exists():
        meta = {r["audio_filepath"]: r for r in read_jsonl(root / "manifest.jsonl")}
    items = []
    for r in review:
        path = r["audio_filepath"]
        m = meta.get(path, {})
        duration = m.get("duration")
        if duration is None and Path(path).exists():
            duration = round(sf.info(path).duration, 3)
        items.append({
            "id": path,
            "gpt": r.get("text_gpt", ""),
            "prefill": r.get("text_corrected", ""),
            "source": m.get("source", Path(path).stem.rsplit("_", 1)[0]),
            "duration": duration or 0.0,
        })
    return items


def load_ann(folder: str) -> dict:
    p = Path(folder) / "annotations.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_ann(folder: str, items: list[dict], ann: dict):
    """Persist progress and rebuild the verified manifest used for WER."""
    root = Path(folder)
    write_json_atomic(root / "annotations.json", ann)
    tmp = root / "verified.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            a = ann.get(it["id"])
            if a and a["status"] in ("verified", "edited") and a["text"]:
                f.write(json.dumps({"audio_filepath": it["id"], "duration": it["duration"],
                                    "text": a["text"], "source": it["source"]},
                                   ensure_ascii=False) + "\n")
    os.replace(tmp, root / "verified.jsonl")


# ---------- sidebar ----------

with st.sidebar:
    st.title("🎧 Transcript review")
    folder = st.text_input("Prepared folder", "evaluation_dataset/prepared")

if not (Path(folder) / "review.jsonl").exists():
    st.error(f"`{folder}/review.jsonl` not found. Run evaluation_preparing.py first, "
             "or change the folder in the sidebar.")
    st.stop()

items = load_items(folder)
if not items:
    st.warning("review.jsonl is empty.")
    st.stop()

ann = load_ann(folder)
ids = [it["id"] for it in items]
by_id = {it["id"]: it for it in items}


def status_of(i: str) -> str:
    return ann.get(i, {}).get("status", "pending")


with st.sidebar:
    flt = st.radio("Show", ["all", *STATUS], format_func=lambda k: "All" if k == "all" else STATUS[k][0])

    counts = {k: sum(status_of(i) == k for i in ids) for k in STATUS}
    done = len(ids) - counts["pending"]
    st.progress(done / len(ids), text=f"{done} / {len(ids)} reviewed")
    c1, c2 = st.columns(2)
    c1.metric("Verified", counts["verified"])
    c2.metric("Edited", counts["edited"])
    c1.metric("Rejected", counts["rejected"])
    c2.metric("Pending", counts["pending"])

    # How far the raw model transcripts were from the human version (on checked clips).
    errs = words = 0
    for i in ids:
        a = ann.get(i)
        if a and a["status"] in ("verified", "edited"):
            ref = a["text"].split()
            errs += word_errors(ref, normalize_text(by_id[i]["gpt"]).split())
            words += len(ref)
    if words:
        st.metric("GPT vs human WER", f"{100 * errs / words:.2f} %",
                  help="Error rate of the OpenAI transcripts against your corrections.")

    hours = sum(by_id[i]["duration"] for i in ids if status_of(i) in ("verified", "edited")) / 3600
    st.caption(f"Verified set: {counts['verified'] + counts['edited']} clips, {hours:.2f} h  \n"
               f"→ `{folder}/verified.jsonl`")

# ---------- navigation ----------

if st.session_state.get("cur") not in by_id:
    first = next((i for i in ids if status_of(i) == "pending"), ids[0])
    st.session_state.cur = first
cur = st.session_state.cur
pos = ids.index(cur)


def matches(i: str) -> bool:
    return flt == "all" or status_of(i) == flt


def go(step: int):
    """Move to the next/previous clip that fits the filter (stay if there is none)."""
    k = pos + step
    while 0 <= k < len(ids):
        if matches(ids[k]):
            st.session_state.cur = ids[k]
            return
        k += step


def jump():
    st.session_state.cur = ids[st.session_state[f"jump_{pos}"] - 1]


with st.sidebar:  # key includes pos so the box always shows the current clip number
    st.number_input("Go to clip #", 1, len(ids), pos + 1, key=f"jump_{pos}", on_change=jump)

# ---------- main ----------

item = by_id[cur]
a = ann.get(cur, {})
label, colour = STATUS[status_of(cur)]

st.markdown(
    f"### Clip {pos + 1} / {len(ids)} &nbsp;<span class='badge' style='background:{colour}'>{label}</span>",
    unsafe_allow_html=True,
)
st.markdown(f"<span class='meta'>{html.escape(item['source'])} · {item['duration']:.1f} s · "
            f"{html.escape(Path(cur).name)}</span>", unsafe_allow_html=True)

if Path(cur).exists():
    st.audio(cur, format="audio/wav")
else:
    st.error(f"Audio file missing: {cur}")

key = f"txt_{cur}"
if key not in st.session_state:
    st.session_state[key] = a.get("raw") or item["prefill"] or item["gpt"]


def reset_text():
    st.session_state[key] = item["gpt"]


left, right = st.columns(2, gap="large")
with left:
    st.markdown("<div class='card'><h4>Your transcript (editable)</h4></div>", unsafe_allow_html=True)
    text = st.text_area("transcript", key=key, height=220, label_visibility="collapsed")
    st.button("↺ Reset to GPT text", on_click=reset_text)

final = normalize_text(text)
with right:
    st.markdown(f"<div class='card'><h4>Original (GPT)</h4><div class='fa'>{html.escape(item['gpt'])}</div></div>",
                unsafe_allow_html=True)
    changed = final != normalize_text(item["gpt"])
    body = diff_html(normalize_text(item["gpt"]), final) if changed else html.escape(final)
    title = "Stored text (normalized) — changes highlighted" if changed else "Stored text (normalized)"
    st.markdown(f"<div class='card'><h4>{title}</h4><div class='fa'>{body or '—'}</div></div>",
                unsafe_allow_html=True)


def commit(status: str):
    ann[cur] = {"status": status, "raw": text, "text": final, "ts": int(time.time())}
    save_ann(folder, items, ann)


def advance():
    """After a decision, move on; with the 'pending' filter the item just left the list."""
    go(1)
    if st.session_state.cur == cur:
        go(-1)


b1, b2, b3, b4, b5 = st.columns([1, 1, 1.4, 1.2, 1])
if b1.button("◀ Prev", use_container_width=True, disabled=pos == 0):
    go(-1)
    st.rerun()
if b2.button("Next ▶", use_container_width=True, disabled=pos == len(ids) - 1):
    go(1)
    st.rerun()
if b3.button("✅ Save & verify", type="primary", use_container_width=True):
    if not final:
        st.warning("The transcript is empty. Reject the clip instead.")
    else:
        commit("edited" if changed else "verified")
        advance()
        st.rerun()
if b4.button("🗑 Reject clip", use_container_width=True, help="Bad audio / unusable: excluded from the WER set"):
    commit("rejected")
    advance()
    st.rerun()
if b5.button("⏮ Reset", use_container_width=True, disabled=status_of(cur) == "pending",
             help="Mark as pending again"):
    ann.pop(cur, None)
    save_ann(folder, items, ann)
    st.rerun()

st.caption("Tip: listen to the whole clip. Fix wrong, missing or extra words; punctuation and "
           "Arabic/Persian letter variants are normalized automatically when saving.")
