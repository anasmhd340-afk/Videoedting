import asyncio
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


OUTPUT_DIR = Path("output_scripts")
TEMP_DIR = Path("temp_video")
DEFAULT_MALAYALAM_VOICE = "ml-IN-MidhunNeural"


def ensure_edge_tts():
    try:
        import edge_tts  # noqa: F401
        return edge_tts
    except ImportError:
        print("Installing edge-tts for Malayalam voiceover audio generation...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "edge-tts"])
        import edge_tts  # noqa: F401
        return edge_tts


def get_drive_service():
    client_id = os.environ.get("GDRIVE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GDRIVE_CLIENT_SECRET", "").strip()
    refresh_token = os.environ.get("GDRIVE_REFRESH_TOKEN", "").strip()

    if not (client_id and client_secret and refresh_token):
        raise RuntimeError(
            "Missing Google Drive OAuth secrets (GDRIVE_CLIENT_ID, GDRIVE_CLIENT_SECRET, GDRIVE_REFRESH_TOKEN)."
        )

    # Omit explicit scopes so Google refreshes using the exact scope(s) granted to the refresh token
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
    )
    creds.refresh(Request())
    return build("drive", "v3", credentials=creds)


def extract_drive_file_id(raw_input: str) -> str | None:
    if not raw_input:
        return None
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"^([a-zA-Z0-9_-]{25,})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_input.strip())
        if match:
            return match.group(1)
    return None


def resolve_target_video(service, video_input: str, folder_id: str) -> dict:
    file_id = extract_drive_file_id(video_input)
    if file_id:
        return (
            service.files()
            .get(fileId=file_id, fields="id, name, parents, mimeType, size")
            .execute()
        )

    if video_input:
        safe_name = video_input.replace("'", "\\'")
        query = f"name = '{safe_name}' and trashed = false"
        if folder_id:
            query += f" and '{folder_id}' in parents"
        res = (
            service.files()
            .list(
                q=query,
                orderBy="createdTime desc",
                pageSize=1,
                fields="files(id, name, parents, mimeType, size)",
            )
            .execute()
        )
        files = res.get("files", [])
        if files:
            return files[0]

    # Default: pick latest Edited_Vlog_*.mp4
    query = "mimeType contains 'video/' and name contains 'Edited_Vlog_' and trashed = false"
    if folder_id:
        query += f" and '{folder_id}' in parents"

    res = (
        service.files()
        .list(
            q=query,
            orderBy="createdTime desc",
            pageSize=1,
            fields="files(id, name, parents, mimeType, size)",
        )
        .execute()
    )
    files = res.get("files", [])
    if not files:
        fallback_q = "mimeType = 'video/mp4' and trashed = false"
        res = (
            service.files()
            .list(
                q=fallback_q,
                orderBy="createdTime desc",
                pageSize=1,
                fields="files(id, name, parents, mimeType, size)",
            )
            .execute()
        )
        files = res.get("files", [])

    if not files:
        raise FileNotFoundError("No matching video found in Google Drive.")
    return files[0]


def download_drive_file(service, file_id: str, dest_path: Path):
    request = service.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=10 * 1024 * 1024)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Download progress: {int(status.progress() * 100)}%", flush=True)


def upload_to_drive(service, local_path: Path, parent_id: str | None, mime_type: str) -> str:
    metadata = {"name": local_path.name}
    if parent_id:
        metadata["parents"] = [parent_id]
    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
    created = (
        service.files()
        .create(body=metadata, media_body=media, fields="id, webViewLink")
        .execute()
    )
    return created.get("webViewLink", "")


def generate_with_retry(client, uploaded_file, prompt: str, max_attempts: int = 4) -> str:
    delays = [15, 30, 60]
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(
                f"Generating Malayalam voiceover script with gemini-2.5-flash (attempt {attempt}/{max_attempts})...",
                flush=True,
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.7,
                ),
            )
            return response.text or ""
        except Exception as err:
            last_error = err
            err_str = str(err)
            is_transient = any(
                code in err_str
                for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "502", "504")
            )
            if is_transient and attempt < max_attempts:
                wait_sec = delays[min(attempt - 1, len(delays) - 1)]
                print(
                    f"Gemini server busy/rate-limited ({err_str[:120]}...). Retrying in {wait_sec}s...",
                    flush=True,
                )
                time.sleep(wait_sec)
            else:
                raise
    raise last_error


def generate_malayalam_script(video_path: Path, style: str, extra_notes: str) -> tuple[str, str]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY secret is not set in GitHub repository secrets.")

    client = genai.Client(api_key=api_key)
    print(f"Uploading {video_path.name} to Gemini File API...", flush=True)
    uploaded_file = client.files.upload(file=str(video_path))

    try:
        while uploaded_file.state.name == "PROCESSING":
            print("Waiting for Gemini video processing...", flush=True)
            time.sleep(8)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name != "ACTIVE":
            raise RuntimeError(f"Gemini file processing failed with state: {uploaded_file.state.name}")

        extra_section = f"\nAdditional creator notes/context: {extra_notes}" if extra_notes else ""

        prompt = f"""Watch this entire video and write a concise, engaging Malayalam voiceover script that highlights all key moments, locations, and transitions without narrating every single second or minor camera cut.

Style requested: {style}{extra_section}

Strict Output Requirements:
1. Provide TWO clearly separated sections using the exact markers `===SCRIPT_TXT===` and `===SUBTITLES_SRT===`.
2. Under `===SCRIPT_TXT===`:
   - Group related shots into broader, natural story segments (e.g., 15–45 second chapters/beats) instead of second-by-second commentary.
   - Start with a crisp, warm, conversational Malayalam hook suited to the opening mood.
   - Use `[MM:SS - MM:SS]` timestamps for each major segment, preceded by a short one-line Malayalam scene tag `(ദൃശ്യം: ...)`.
   - Keep each narration block short, punchy, and natural (1–3 crisp sentences per segment)—capture every important highlight, shift in scenery, or key moment without over-describing what is already obvious on screen.
   - Use natural, contemporary, spoken-friendly Malayalam—avoid stiff or bookish Sanskritized phrasing, and leave generous breathing pauses so the visuals shine.
3. Under `===SUBTITLES_SRT===`:
   - Output valid `.srt` subtitle blocks (`00:MM:SS,000 --> 00:MM:SS,000`) matching those concise narration blocks with clean Malayalam lines only.
"""

        raw_text = generate_with_retry(client, uploaded_file, prompt)
    finally:
        try:
            client.files.delete(name=uploaded_file.name)
            print("Cleaned up temporary file from Gemini File API.", flush=True)
        except Exception as cleanup_err:
            print(f"Warning: could not delete remote Gemini file: {cleanup_err}", flush=True)

    script_txt = raw_text
    srt_txt = ""
    if "===SCRIPT_TXT===" in raw_text and "===SUBTITLES_SRT===" in raw_text:
        after_script = raw_text.split("===SCRIPT_TXT===", 1)[1]
        script_part, srt_part = after_script.split("===SUBTITLES_SRT===", 1)
        script_txt = script_part.strip()
        srt_txt = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", srt_part.strip()).strip()

    return script_txt, srt_txt


def extract_spoken_narration(script_txt: str, srt_txt: str) -> str:
    """Extract clean spoken Malayalam sentences without timestamps, SRT indices, or visual cues."""
    if srt_txt:
        lines = []
        for raw_line in srt_txt.splitlines():
            line = raw_line.strip()
            if not line or line.isdigit() or "-->" in line:
                continue
            lines.append(line)
        cleaned = " ".join(lines).strip()
        if cleaned:
            return cleaned

    lines = []
    for raw_line in script_txt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^\[?\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\]?$", line):
            continue
        if line.startswith("(ദൃശ്യം:") or line.startswith("**"):
            continue
        line = re.sub(r"^\[?\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\]?\s*", "", line)
        line = re.sub(r"\(ദൃശ്യം:.*?\)", "", line).strip()
        if line:
            lines.append(line)
    return " ".join(lines).strip()


async def _synthesize_tts(text: str, voice: str, output_mp3: Path):
    edge_tts = ensure_edge_tts()
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(output_mp3))


def generate_voiceover_audio(script_txt: str, srt_txt: str, output_mp3: Path, voice: str) -> bool:
    spoken_text = extract_spoken_narration(script_txt, srt_txt)
    if not spoken_text:
        print("Warning: No spoken narration text extracted for TTS audio.", flush=True)
        return False

    print(f"Generating Malayalam MP3 audio with Edge-TTS ({voice})...", flush=True)
    asyncio.run(_synthesize_tts(spoken_text, voice, output_mp3))
    return output_mp3.exists() and output_mp3.stat().st_size > 0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    video_input = os.environ.get("VIDEO_INPUT", "").strip()
    style = os.environ.get(
        "VOICEOVER_STYLE",
        "Engaging, warm, conversational Malayalam travel vlog storytelling",
    ).strip()
    extra_notes = os.environ.get("EXTRA_NOTES", "").strip()
    folder_id = os.environ.get("GDRIVE_OUTPUT_FOLDER_ID", "").strip()
    tts_voice = os.environ.get("VOICEOVER_VOICE", DEFAULT_MALAYALAM_VOICE).strip() or DEFAULT_MALAYALAM_VOICE

    drive = get_drive_service()
    video_meta = resolve_target_video(drive, video_input, folder_id)
    video_id = video_meta["id"]
    video_name = video_meta["name"]
    parents = video_meta.get("parents", [])
    target_parent = parents[0] if parents else (folder_id or None)

    print(f"Target video found: {video_name}", flush=True)
    local_video = TEMP_DIR / video_name
    download_drive_file(drive, video_id, local_video)

    try:
        script_txt, srt_txt = generate_malayalam_script(local_video, style, extra_notes)
    finally:
        if local_video.exists():
            local_video.unlink()
            print("Cleaned up local temporary video.", flush=True)

    stem = Path(video_name).stem
    txt_path = OUTPUT_DIR / f"{stem}_voiceover.txt"
    txt_path.write_text(script_txt + "\n", encoding="utf-8")
    txt_link = upload_to_drive(drive, txt_path, target_parent, "text/plain")
    print(f"Uploaded TXT script to Google Drive: {txt_link}", flush=True)

    if srt_txt:
        srt_path = OUTPUT_DIR / f"{stem}_voiceover.srt"
        srt_path.write_text(srt_txt + "\n", encoding="utf-8")
        srt_link = upload_to_drive(drive, srt_path, target_parent, "application/x-subrip")
        print(f"Uploaded SRT subtitles to Google Drive: {srt_link}", flush=True)

    mp3_path = OUTPUT_DIR / f"{stem}_voiceover.mp3"
    if generate_voiceover_audio(script_txt, srt_txt, mp3_path, tts_voice):
        mp3_link = upload_to_drive(drive, mp3_path, target_parent, "audio/mpeg")
        print(f"Uploaded MP3 voiceover audio to Google Drive: {mp3_link}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stdout.flush()
        print(f"\nError: {exc}", flush=True)
        sys.exit(1)
