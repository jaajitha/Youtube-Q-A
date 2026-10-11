import re
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats or raw ID."""
    if not url:
        raise ValueError("URL cannot be empty")
    url = url.strip()

    # Raw 11-character video ID
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url):
        return url

    parsed = urlparse(url)
    netloc = parsed.netloc.lower()

    if "youtu.be" in netloc:
        path = parsed.path.lstrip("/")
        return path.split("?")[0].split("&")[0]

    if "youtube.com" in netloc or "m.youtube.com" in netloc:
        if "/shorts/" in parsed.path:
            return parsed.path.split("/shorts/")[1].split("/")[0].split("?")[0]
        elif "/embed/" in parsed.path:
            return parsed.path.split("/embed/")[1].split("/")[0].split("?")[0]
        elif "/v/" in parsed.path:
            return parsed.path.split("/v/")[1].split("/")[0].split("?")[0]
        else:
            qs = parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                return qs["v"][0]

    # Regex fallback for embedded or uncommon patterns
    match = re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})(?:\?|&|$)", url)
    if match:
        return match.group(1)

    raise ValueError(f"Could not extract a valid YouTube video ID from: '{url}'. Please provide a valid YouTube URL.")


def get_video_transcript_data(video_url: str):
    """
    Fetches transcript segments with timestamp information.
    Supports both youtube-transcript-api 1.2+ and older versions.
    """
    video_id = extract_video_id(video_url)

    # Newer youtube-transcript-api 1.2+ uses instance methods fetch() and list()
    if hasattr(YouTubeTranscriptApi, "fetch"):
        yt = YouTubeTranscriptApi()
        try:
            # Try default fetch (English)
            fetched = yt.fetch(video_id)
            items = []
            for s in fetched:
                text = getattr(s, "text", s["text"] if isinstance(s, dict) else str(s))
                start = getattr(s, "start", s["start"] if isinstance(s, dict) else 0.0)
                duration = getattr(s, "duration", s["duration"] if isinstance(s, dict) else 0.0)
                items.append({"text": text, "start": start, "duration": duration})
            return video_id, items
        except Exception:
            # Try listing available transcripts and fallback to any available
            try:
                transcript_list = yt.list(video_id)
                # Try finding English or auto-generated English
                try:
                    transcript_obj = transcript_list.find_transcript(["en", "en-US", "en-GB"])
                except Exception:
                    # Pick the first transcript available in any language
                    transcript_obj = next(iter(transcript_list))
                
                fetched = transcript_obj.fetch()
                items = []
                for s in fetched:
                    text = getattr(s, "text", s["text"] if isinstance(s, dict) else str(s))
                    start = getattr(s, "start", s["start"] if isinstance(s, dict) else 0.0)
                    duration = getattr(s, "duration", s["duration"] if isinstance(s, dict) else 0.0)
                    items.append({"text": text, "start": start, "duration": duration})
                return video_id, items
            except Exception as e:
                raise RuntimeError(
                    f"Unable to extract captions for video ID '{video_id}'. "
                    f"Transcripts might be disabled for this video, or it is unavailable. Details: {e}"
                )

    # Legacy youtube-transcript-api class method get_transcript()
    elif hasattr(YouTubeTranscriptApi, "get_transcript"):
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            items = []
            for s in transcript_list:
                text = s.get("text", "") if isinstance(s, dict) else getattr(s, "text", str(s))
                start = s.get("start", 0.0) if isinstance(s, dict) else getattr(s, "start", 0.0)
                duration = s.get("duration", 0.0) if isinstance(s, dict) else getattr(s, "duration", 0.0)
                items.append({"text": text, "start": start, "duration": duration})
            return video_id, items
        except Exception as e:
            raise RuntimeError(
                f"Unable to extract captions for video ID '{video_id}'. Details: {e}"
            )
    else:
        raise RuntimeError("No supported transcript extraction method found on YouTubeTranscriptApi.")


def get_video_transcript(video_url: str) -> str:
    """Returns the combined text transcript for a YouTube video."""
    _, items = get_video_transcript_data(video_url)
    clean_texts = [item["text"].strip() for item in items if item.get("text")]
    return " ".join(clean_texts)
