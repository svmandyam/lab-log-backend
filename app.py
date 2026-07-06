"""
Voice-to-Notion lab log pipeline.

Flow:
  Tasker (phone) --audio file--> this server
    -> OpenAI Whisper: audio -> transcript
    -> Claude API: transcript -> structured tags (Type/Equipment/Project/Status/Title)
    -> Notion API: create tagged page in the Log database

Required environment variables (set these, never hardcode keys in this file):
  OPENAI_API_KEY      - from platform.openai.com
  ANTHROPIC_API_KEY   - from console.anthropic.com
  NOTION_TOKEN        - your internal integration secret (ntn_...)
  NOTION_DATABASE_ID  - e.g. e13261f51edf4020a9893d27c8fa3524 (your Log database)

Run locally for testing:
  pip install flask requests anthropic --break-system-packages
  export OPENAI_API_KEY=...
  export ANTHROPIC_API_KEY=...
  export NOTION_TOKEN=...
  export NOTION_DATABASE_ID=...
  python app.py

Then point Tasker's HTTP Request at:  http://<your-server-ip>:5000/log
"""

import os
import json
import requests
from flask import Flask, request, jsonify
from anthropic import Anthropic

app = Flask(__name__)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DATABASE_ID = os.environ["NOTION_DATABASE_ID"]

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Keep this in sync with your actual Notion schema.
TYPE_OPTIONS = ["Event", "Plan", "Decision", "Solution", "Data-taking",
                 "Analysis", "Idea", "Reading/Link", "Tutorial"]
EQUIPMENT_OPTIONS = ["Cryostat", "DAC", "MW Source", "Confocal Setup",
                      "Optics", "Electronics"]

TAGGING_SYSTEM_PROMPT = f"""You are tagging a lab notebook entry for a high-pressure NV-diamond
sensing research group. Given a raw voice-transcribed note, output ONLY a JSON object
(no prose, no markdown fences) with these fields:

- "title": a short (under 10 words) descriptive title you generate
- "type": array, choose one or more from exactly: {TYPE_OPTIONS}
- "equipment": array, choose zero or more from exactly: {EQUIPMENT_OPTIONS}
  (leave empty if nothing clearly matches; do not invent new equipment names)
- "project": a short project name if identifiable from context (e.g. "Hydride superconductor run",
  "327 Nickelate", "GSLAC", "AC Calorimetry"), otherwise "General equipment"
- "status": choose exactly one of: "Open", "Resolved", "Reference" -- use your best
  judgment from context (e.g. a described fix that worked = "Resolved", an unanswered
  question or to-do = "Open", a procedure/tutorial/reference note = "Reference")
- "body": the transcript, lightly cleaned up (fix obvious transcription errors/fragments)
  but preserve the original meaning and technical detail exactly. Do not summarize or shorten.

Respond with raw JSON only."""


def transcribe_audio(audio_bytes, filename):
    """Send audio to OpenAI's Whisper transcription endpoint."""
    resp = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        files={"file": (filename, audio_bytes)},
        data={"model": "whisper-1"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["text"]


def tag_transcript(transcript):
    """Send transcript to Claude for structured tagging."""
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=TAGGING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": transcript}],
    )
    raw = message.content[0].text.strip()
    # Defensive: strip accidental code fences if the model adds them
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
    return json.loads(raw)


def upload_image_to_notion(image_bytes, filename):
    """
    Two-step Notion File Upload API:
      1. Create a file_upload object
      2. Send the actual bytes to it
    Returns the file_upload id, which can then be referenced in a page's
    children blocks without needing any public hosting (no GitHub bridge needed
    for live capture -- that was only ever a workaround for the PDF migration).
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


def write_to_notion(tags, image_files=None):
    """
    Create a tagged page in the Notion Log database.
    image_files: optional list of (filename, bytes) tuples for photos taken
    alongside the voice note. Each becomes an inline image block after the
    caption text.
    """
    children = [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"text": {"content": tags["body"]}}]},
        }
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

    payload = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Title": {"title": [{"text": {"content": tags["title"]}}]},
            "Type": {"multi_select": [{"name": t} for t in tags["type"]]},
            "Equipment": {"multi_select": [{"name": e} for e in tags["equipment"]]},
            "Status": {"select": {"name": tags["status"]}},
            "Project": {"rich_text": [{"text": {"content": tags["project"]}}]},
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
    resp.raise_for_status()
    return resp.json()


@app.route("/log", methods=["POST"])
def log_entry():
    """
    Tasker should POST here as multipart form-data:
      - "file": the recorded audio (required)
      - "images": zero or more photo files (optional, same field name repeated
        for multiple files -- this is what the future photo-capture routine
        will add; the text-only routine simply omits this field)
    """
    if "file" not in request.files:
        return jsonify({"error": "no audio file uploaded"}), 400

    audio_file = request.files["file"]
    audio_bytes = audio_file.read()

    image_files = [
        (f.filename or f"photo_{i}.jpg", f.read())
        for i, f in enumerate(request.files.getlist("images"))
    ]

    try:
        transcript = transcribe_audio(audio_bytes, audio_file.filename or "audio.m4a")
        tags = tag_transcript(transcript)
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
        "notion_url": notion_result.get("url"),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
