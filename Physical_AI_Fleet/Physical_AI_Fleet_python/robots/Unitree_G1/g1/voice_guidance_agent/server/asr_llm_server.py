import glob, os, re, tempfile, base64
import nemo.collections.asr as nemo_asr
from fastapi import FastAPI, UploadFile, File
from openai import OpenAI

app = FastAPI()
asr = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8001/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "none")
LLM_MODEL    = os.getenv("LLM_MODEL", "nvidia/NVIDIA-Nemotron-Nano-9B-v2")
llm = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

def load_sops():
    return "\n\n".join(open(f).read() for f in sorted(glob.glob("sops/*.md")))

SYSTEM = """/no_think
You are a voice assistant on an oil rig helping a field worker repair equipment.
Use ONLY the SOPs below.
RULES:
- Give exactly ONE numbered step per reply, never two. Say the step number, e.g. "Step 1: ...".
- End with: "Say next when done."
- When the worker says next, give the following step. When they say repeat, repeat the same step.
- When the last step is done, say the repair is complete and to report to the supervisor.
- Tool questions: name only the tools from that SOP.
- If the procedure is not covered, say exactly: "I don't have that procedure, please contact the supervisor."
- If the worker's message is unclear, off-topic or does not make sense, say exactly: Sorry, I did not catch that. Please say it again. Never give a step in that case.\n- Plain spoken English, under 25 words, no markdown.
SOPs:
{sops}"""

history = []

def clean(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    t = t.replace("</think>", "")
    return re.sub(r"[*#`]", "", t).strip()

@app.post("/ask")
async def ask(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(await audio.read()); f.flush()
        out = asr.transcribe([f.name])
        text = out[0].text if hasattr(out[0], "text") else out[0]
    KEYS = ("next","repeat","again","tool","step","valve","leak","pump","motor","gas","alarm","pressure",
            "gauge","fault","overheat","hot","previous","back","done","what","how","fix","repair","which","need","fail")
    if not any(k in text.lower() for k in KEYS):
        return {"heard": text, "reply": "Sorry, I did not catch that. Please say it again."}
    history.append({"role": "user", "content": text})
    r = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM.format(sops=load_sops())}] + history[-10:],
        max_tokens=150, temperature=0.2)
    reply = clean(r.choices[0].message.content)
    history.append({"role": "assistant", "content": reply})
    return {"heard": text, "reply": reply}

@app.post("/reset")
def reset():
    history.clear(); return {"ok": True}


VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8002/v1")
VLM_MODEL    = os.getenv("VLM_MODEL", "nvidia/Cosmos-Reason1-7B")
vlm = OpenAI(base_url=VLM_BASE_URL, api_key=os.getenv("VLM_API_KEY", "none"))

@app.post("/ask_image")
async def ask_image(audio: UploadFile = File(...), image: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(await audio.read()); f.flush()
        out = asr.transcribe([f.name])
        text = out[0].text if hasattr(out[0], "text") else out[0]
    if len(text.strip()) < 3:
        return {"heard": text, "reply": "Sorry, I did not catch that. Please say it again."}
    b64 = base64.b64encode(await image.read()).decode()
    prompt = ("You are a voice assistant on an oil rig. Look at the image and answer the worker's question "
              "in at most 2 short spoken sentences, plain English, no markdown. If unsure, say so. "
              "Worker said: " + text)
    r = vlm.chat.completions.create(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            {"type": "text", "text": prompt}]}],
        max_tokens=150, temperature=0.2)
    reply = clean(r.choices[0].message.content)
    history.append({"role": "user", "content": text + " [showed an image]"})
    history.append({"role": "assistant", "content": reply})
    return {"heard": text, "reply": reply}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(await audio.read()); f.flush()
        out = asr.transcribe([f.name])
        text = out[0].text if hasattr(out[0], "text") else out[0]
    return {"heard": text}
