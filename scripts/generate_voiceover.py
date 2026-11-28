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


def get_media_duration(file_path: Path) -> float:
    """Return exact duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


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


def format_mm_ss(seconds: float) -> str:
    total = int(round(seconds))
    mins, secs = divmod(total, 60)
    return f"{mins:02d}:{secs:02d}"


def generate_malayalam_script(
    video_path: Path, video_duration: float, style: str, extra_notes: str
) -> tuple[str, str]:
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
        dur_str = format_mm_ss(video_duration)

        prompt = f"""Watch this entire video (total duration: {dur_str}, i.e. {video_duration:.1f} seconds) and write a concise, engaging Malayalam voiceover script that highlights all key moments, locations, and transitions without narrating every single second or minor camera cut.

Style requested: {style}{extra_section}

Strict Output Requirements:
1. Provide TWO clearly separated sections using the exact markers `===SCRIPT_TXT===` and `===SUBTITLES_SRT===`.
2. Under `===SCRIPT_TXT===`:
   - Group related shots into natural story segments with intentional silent pauses between segments wherever the visuals or ambient sounds should breathe without voiceover (for example, leaving a 4–10 second gap with no entry when nothing needs to be spoken).
   - Start with a crisp, warm, conversational Malayalam hook suited to the opening mood.
   - Use `[MM:SS - MM:SS]` timestamps for each spoken segment (never exceeding `{dur_str}`), preceded by a short one-line Malayalam scene tag `(ദൃശ്യം: ...)`.
   - Keep each narration block concise enough (1–2 crisp sentences, roughly 2–2.5 words per second of that segment's window) so it can be spoken comfortably within its `[MM:SS - MM:SS]` timeframe without rushing.
   - Use natural, contemporary, spoken-friendly Malayalam—avoid stiff or bookish Sanskritized phrasing.
3. Under `===SUBTITLES_SRT===`:
   - Output valid `.srt` subtitle blocks (`00:MM:SS,000 --> 00:MM:SS,000`) matching the exact start and end timestamps of those spoken segments (leaving gaps where there is no voiceover) with clean Malayalam lines only.
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


def _srt_time_to_seconds(ts: str) -> float:
    match = re.match(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", ts.strip())
    if not match:
        return 0.0
    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000.0


def _mmss_to_seconds(ts: str) -> float:
    parts = [int(p) for p in ts.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(parts[0])


def parse_timed_segments(script_txt: str, srt_txt: str, max_duration: float) -> list[dict]:
    """Parse timed narration segments (start, end, text) from SRT or TXT."""
    segments: list[dict] = []

    if srt_txt:
        blocks = re.split(r"\n\s*\n", srt_txt.strip())
        for block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            time_line_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
            if time_line_idx is None:
                continue
            start_str, end_str = lines[time_line_idx].split("-->", 1)
            start_sec = max(0.0, _srt_time_to_seconds(start_str))
            end_sec = min(max_duration, _srt_time_to_seconds(end_str))
            text = " ".join(lines[time_line_idx + 1 :]).strip()
            if text and start_sec < max_duration:
                segments.append({"start": start_sec, "end": max(end_sec, start_sec + 1.0), "text": text})

    if not segments:
        current_start = None
        current_end = None
        current_lines: list[str] = []
        ts_pattern = re.compile(r"^\[?(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)\]?")

        for raw_line in script_txt.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("(ദൃശ്യം:") or line.startswith("**"):
                continue
            m = ts_pattern.match(line)
            if m:
                if current_start is not None and current_lines:
                    segments.append(
                        {
                            "start": current_start,
                            "end": min(max_duration, current_end),
                            "text": " ".join(current_lines).strip(),
                        }
                    )
                current_start = max(0.0, _mmss_to_seconds(m.group(1)))
                current_end = _mmss_to_seconds(m.group(2))
                remainder = ts_pattern.sub("", line).strip()
                current_lines = [remainder] if remainder else []
            elif current_start is not None:
                clean_line = re.sub(r"\(ദൃശ്യം:.*?\)", "", line).strip()
                if clean_line:
                    current_lines.append(clean_line)

        if current_start is not None and current_lines:
            segments.append(
                {
                    "start": current_start,
                    "end": min(max_duration, current_end),
                    "text": " ".join(current_lines).strip(),
                }
            )

    segments.sort(key=lambda s: s["start"])
    return segments


async def _synthesize_all_segments(segments: list[dict], voice: str, work_dir: Path) -> list[Path]:
    edge_tts = ensure_edge_tts()
    clip_paths: list[Path] = []
    for idx, seg in enumerate(segments):
        raw_clip = work_dir / f"seg_{idx:03d}_raw.mp3"
        communicate = edge_tts.Communicate(text=seg["text"], voice=voice)
        await communicate.save(str(raw_clip))
        clip_paths.append(raw_clip)
    return clip_paths


def build_time_synced_mp3(
    script_txt: str,
    srt_txt: str,
    video_duration: float,
    output_mp3: Path,
    voice: str,
) -> bool:
    segments = parse_timed_segments(script_txt, srt_txt, video_duration)
    if not segments:
        print("Warning: No timed narration segments found for synced TTS audio.", flush=True)
        return False

    print(
        f"Synthesizing {len(segments)} timed Malayalam voiceover segments ({voice}) to match {video_duration:.2f}s video...",
        flush=True,
    )
    seg_dir = TEMP_DIR / "tts_segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    raw_clips = asyncio.run(_synthesize_all_segments(segments, voice, seg_dir))

    placed_clips: list[tuple[int, Path]] = []
    try:
        for idx, (seg, raw_clip) in enumerate(zip(segments, raw_clips)):
            if not raw_clip.exists() or raw_clip.stat().st_size == 0:
                continue

            next_start = segments[idx + 1]["start"] if idx + 1 < len(segments) else video_duration
            max_allowed = max(0.8, min(next_start - seg["start"] - 0.15, video_duration - seg["start"]))
            if max_allowed <= 0.3:
                continue

            actual_dur = get_media_duration(raw_clip)
            final_clip = raw_clip

            # Gently speed up only if spoken audio would spill over into the next segment's timestamp
            if actual_dur > max_allowed:
                speed = min(1.35, actual_dur / max_allowed)
                fitted_clip = seg_dir / f"seg_{idx:03d}_fit.mp3"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(raw_clip),
                        "-filter:a",
                        f"atempo={speed:.4f}",
                        str(fitted_clip),
                    ],
                    check=True,
                )
                final_clip = fitted_clip

            delay_ms = int(round(seg["start"] * 1000))
            placed_clips.append((delay_ms, final_clip))

        if not placed_clips:
            return False

        # Build silent base track of exact video duration + place each segment at its exact timestamp
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=24000:cl=mono:d={video_duration:.3f}",
        ]
        filter_parts = []
        mix_labels = ["[0:a]"]

        for i, (delay_ms, clip_path) in enumerate(placed_clips, start=1):
            ffmpeg_cmd.extend(["-i", str(clip_path)])
            filter_parts.append(f"[{i}:a]aresample=24000,adelay={delay_ms}|{delay_ms}[a{i}]")
            mix_labels.append(f"[a{i}]")

        filter_complex = (
            ";".join(filter_parts)
            + f";{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=first:normalize=0,atrim=0:{video_duration:.3f}[out]"
        )

        ffmpeg_cmd.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[out]",
                "-c:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(output_mp3),
            ]
        )
        subprocess.run(ffmpeg_cmd, check=True)
    finally:
        for f in seg_dir.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass
        try:
            seg_dir.rmdir()
        except Exception:
            pass

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
        video_duration = get_media_duration(local_video)
        print(f"Detected exact video duration: {video_duration:.2f}s ({format_mm_ss(video_duration)})", flush=True)
        script_txt, srt_txt = generate_malayalam_script(local_video, video_duration, style, extra_notes)
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
    if build_time_synced_mp3(script_txt, srt_txt, video_duration, mp3_path, tts_voice):
        final_audio_dur = get_media_duration(mp3_path)
        print(f"Generated time-synced MP3 duration: {final_audio_dur:.2f}s", flush=True)
        mp3_link = upload_to_drive(drive, mp3_path, target_parent, "audio/mpeg")
        print(f"Uploaded time-synced MP3 voiceover audio to Google Drive: {mp3_link}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stdout.flush()
        print(f"\nError: {exc}", flush=True)
        sys.exit(1)
