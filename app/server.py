"""OPTIONAL demo app — a thin UI over the same adapters (stub).

Minimal upload UI: POST an audio clip → normalized ONCE → IndicConformer (RNNT+CTC) +
the three APIs → transcripts side by side, with a toggle to apply Method A correction.
No new transcription logic here — it reuses the same adapters and normalize.py.

Run:  uvicorn app.server:app --reload    # from the asr_bench/ root, venv active
"""
from __future__ import annotations

import io

from envload import load_env
from adapters.registry import ALL_SYSTEMS, build_adapter

load_env()
from benchmark.normalize import load_audio_16k_mono
from biasing.glossary import glossary_for, load_glossaries
from biasing.llm_postcorrect import correct

try:
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import HTMLResponse
except Exception as e:  # pragma: no cover - app is optional
    raise SystemExit(
        "FastAPI not installed. `pip install fastapi uvicorn python-multipart` to run the demo."
    ) from e

app = FastAPI(title="asr_bench demo")

_FORM = """
<!doctype html><title>asr_bench</title>
<h2>ASR contextual-biasing demo</h2>
<form action="/transcribe" method="post" enctype="multipart/form-data">
  <input type="file" name="file" accept="audio/*" required><br><br>
  lang: <input name="lang" value="hi" size="4">
  systems: <input name="systems" value="all" size="30">
  <label><input type="checkbox" name="llm" value="1"> apply LLM correction</label><br><br>
  <button type="submit">Transcribe</button>
</form>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _FORM


@app.post("/transcribe", response_class=HTMLResponse)
async def transcribe(
    file: UploadFile = File(...),
    lang: str = Form("hi"),
    systems: str = Form("all"),
    llm: str = Form(""),
) -> str:
    raw = await file.read()
    wav = load_audio_16k_mono(io.BytesIO(raw))  # normalize ONCE
    names = list(ALL_SYSTEMS) if systems.strip() == "all" else [
        s.strip() for s in systems.split(",") if s.strip()
    ]
    gloss = glossary_for(load_glossaries(), lang)

    rows = ["<table border=1 cellpadding=6><tr><th>system</th><th>transcript</th></tr>"]
    for name in names:
        try:
            out = await build_adapter(name).transcribe(wav, lang)
            text = out.text
            if out.alt_text and name == "indicconformer":
                rows.append(f"<tr><td>{name} (CTC head)</td><td>{out.alt_text}</td></tr>")
            if llm:
                text = (await correct(out.text, gloss)).text
            rows.append(f"<tr><td>{name}</td><td>{text}</td></tr>")
        except Exception as e:
            rows.append(f"<tr><td>{name}</td><td><i>error: {e}</i></td></tr>")
    rows.append("</table>")
    return _FORM + "".join(rows)
