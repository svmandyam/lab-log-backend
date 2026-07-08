"""
Voice-to-Notion lab log pipeline (OpenAI-only version).

Flow:
  Tasker (phone) --audio file--> this server
    -> OpenAI Whisper: audio -> transcript
    -> OpenAI GPT (JSON mode): transcript -> structured tags
       (rules loaded from tagging_config.json, not hardcoded)
    -> Notion API: create tagged page in the Log database

Required environment variables:
  OPENAI_API_KEY      - from platform.openai.com
  NOTION_TOKEN         - your internal integration secret (ntn_...)
  NOTION_DATABASE_ID   - e.g. e13261f51edf4020a9893d27c8fa3524 (your Log database)

tagging_config.json lives alongside this file in the repo. Claude can review
your actual Notion Log database periodically and propose edits to that file
(e.g. new Equipment values worth adding) -- always review the diff before
pushing, since Render redeploys automatically on push to main.

Run locally for testing:
  pip install flask requests --break-system-packages
  export OPENAI_API_KEY=...
  export NOTION_TOKEN=...
  export NOTION_DATABASE_ID=...
  python app.py
"""

import os
import json
import requests
from datetime import datetime, timezone
from flask import Flask, request, jsonify

app = Flask(__name__)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DATABASE_ID = os.environ["NOTION_DATABASE_ID"]

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "tagging_config.json")


def load_tagging_config():
    """Reload the config on every request -- cheap, and means a git pull/redeploy
    picks up changes immediately with no server restart logic needed."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def transcribe_audio(audio_bytes, filename):
    """Send audio to OpenAI's gpt-4o-transcribe endpoint (better WER on technical
    jargon than legacy whisper-1, same price, same endpoint/response shape)."""
    resp = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        files={"file": (filename, audio_bytes)},
        data={"model": "gpt-4o-transcribe"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["text"]


def tag_transcript(transcript, config):
    """Send transcript to GPT (JSON mode) for structured tagging."""
    system_prompt = (
        config["tagging_instructions"]
        + "\n\ntype_options: " + json.dumps(config["type_options"])
        + "\nequipment_options: " + json.dumps(config["equipment_options"])
        + "\nproject_examples: " + json.dumps(config["project_examples"])
        + "\nstatus_options: " + json.dumps(config["status_options"])
    )

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o-mini",
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return json.loads(raw)


def upload_image_to_notion(image_bytes, filename):
    """
    Two-step Notion File Upload API:
      1. Create a file_upload object
      2. Send the actual bytes to it
    No public hosting (e.g. GitHub bridge) needed for live capture.
    """
    notion_headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
    }

    create_resp = requests.post(
        "https://api.notion.com/v1/file_uploads",
        headers={**notion_headers, "Content-Type": "application/json"},
        json={"filename": filename},
        timeout=30,
    )
    create_resp.raise_for_status()
    file_upload_id = create_resp.json()["id"]

    send_resp = requests.post(
        f"https://api.notion.com/v1/file_uploads/{file_upload_id}/send",
        headers=notion_headers,
        files={"file": (filename, image_bytes)},
        timeout=60,
    )
    send_resp.raise_for_status()
    return file_upload_id


def chunk_text(text, size=2000):
    """Notion enforces a 2000-character limit per rich_text object.
    Long transcripts must be split across multiple paragraph blocks or the
    pages API rejects the whole request with a 400. Split on the last
    newline/space before the limit when possible so blocks break at natural
    boundaries instead of mid-word."""
    chunks = []
    while len(text) > size:
        cut = text.rfind("\n", 0, size)
        if cut == -1:
            cut = text.rfind(" ", 0, size)
        if cut == -1:
            cut = size
        chunks.append(text[:cut])
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def clean_option(name):
    """Notion disallows commas inside select/multi_select option names --
    a single comma in a model-generated tag 400s the whole page create."""
    return name.replace(",", " -").strip()


def write_to_notion(tags, image_files=None):
    """
    Create a tagged page in the Notion Log database.
    image_files: optional list of (filename, bytes) tuples for photos taken
    alongside the voice note.
    """
    children = [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": chunk}}]},
        }
        for chunk in chunk_text(tags["body"])
    ]

    if image_files:
        for filename, image_bytes in image_files:
            file_upload_id = upload_image_to_notion(image_bytes, filename)
            children.append({
                "object": "block",
                "type": "image",
                "image": {
                    "type": "file_upload",
                    "file_upload": {"id": file_upload_id},
                },
            })

    # Timestamp of capture, not of whenever the pipeline happens to finish --
    # taken at the top of the request in log_entry() and passed in, since
    # transcription/tagging can add a few seconds of lag before we get here.
    payload = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Title": {"title": [{"text": {"content": tags["title"][:2000]}}]},
            "Type": {"multi_select": [{"name": clean_option(t)} for t in tags["type"]]},
            "Equipment": {"multi_select": [{"name": clean_option(e)} for e in tags["equipment"]]},
            "Status": {"select": {"name": clean_option(tags["status"])}},
            "Project": {"multi_select": [{"name": clean_option(p)} for p in tags["project"]]},
            "Date": {"date": {"start": tags["captured_at"]}},
        },
        "children": children,
    }

    resp = requests.post(
        "https://api.notion.com/v1/pages",
        headers={
            "Authorization": f"Bearer {NOTION_TOKEN}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    if not resp.ok:
        # Surface Notion's actual complaint -- a bare raise_for_status() hides
        # the response body, leaving only an opaque "400 Bad Request" upstream.
        raise Exception(f"Notion API error {resp.status_code}: {resp.text[:500]}")
    return resp.json()


@app.route("/log", methods=["POST"])
def log_entry():
    """
    Accepts audio two different ways, since Tasker's built-in HTTP Request
    action does NOT send proper multipart/form-data when you use its
    "File To Send" field -- it just PUTs the raw file bytes as the plain
    request body. So:

      - Preferred: multipart form-data with field name "file" (what curl -F
        and most other HTTP clients send)
      - Fallback: raw bytes in the request body with no file field at all
        (what Tasker's "File To Send" actually does)

    Optional photos ("images" field, multipart only) are still supported
    when sent that way.
    """
    if "file" in request.files:
        audio_file = request.files["file"]
        audio_bytes = audio_file.read()
        filename = audio_file.filename or "voice_log.mp4"
    elif request.data:
        audio_bytes = request.data
        filename = "voice_log.mp4"
    else:
        return jsonify({"error": "no audio file uploaded"}), 400

    # Capture time-of-recording now, before transcription/tagging add lag
    captured_at = datetime.now(timezone.utc).isoformat()

    image_files = [
        (f.filename or f"photo_{i}.jpg", f.read())
        for i, f in enumerate(request.files.getlist("images"))
    ]

    try:
        config = load_tagging_config()
        transcript = transcribe_audio(audio_bytes, filename)
        tags = tag_transcript(transcript, config)
        tags["captured_at"] = captured_at
        notion_result = write_to_notion(tags, image_files=image_files or None)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "status": "ok",
        "transcript": transcript,
        "title": tags["title"],
        "type": tags["type"],
        "equipment": tags["equipment"],
        "images_attached": len(image_files),
        "captured_at": captured_at,
        "notion_url": notion_result.get("url"),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
