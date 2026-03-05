"""Edit raw auto-generated transcripts into professional educational textbook content.

Reads raw transcripts from course_content/transcripts/raw/,
sends each to Gemini for editing, and saves
the polished versions to course_content/transcripts/.

Requires GOOGLE_API_KEY in .env at the project root.
"""

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

import os

from google import genai
from google.genai import types

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RAW_DIR = SCRIPT_DIR / "raw"
OUTPUT_DIR = SCRIPT_DIR
MODEL = "gemini-3.1-pro-preview"
DELAY_SECONDS = 5

SYSTEM_PROMPT = """\
You are an expert academic editor specializing in computer science and \
artificial intelligence textbooks. Your task is to transform a raw, \
auto-generated video lecture transcript into a polished, professional \
educational textbook chapter.

## Instructions

1. **Fix all speech-to-text errors**: Correct misspelled technical terms, \
library names (e.g., "Secret Line" → "scikit-learn", "pandas 9" → "pandas", \
"DataEco" → "Dataiku"), proper nouns, and garbled words. Use your knowledge \
of the subject matter to infer the correct term.

2. **Remove verbal artifacts**: Eliminate filler words (uh, um, so, okay, \
right, you know, let's see), false starts, repetitions, and self-corrections. \
Remove greetings, sign-offs, and off-topic digressions.

3. **Structure into sections**: Organize the content into logical sections \
with clear headings (using Markdown ## and ### format). Each section should \
cover a coherent subtopic.

4. **Write proper paragraphs**: Merge the fragmented caption lines into \
well-formed, complete sentences grouped into coherent paragraphs. Each \
paragraph should convey a single idea.

5. **Preserve technical accuracy**: Keep ALL technical content, code \
references, examples, definitions, and explanations. Do not add information \
that was not in the original transcript. Do not remove substantive content.

6. **Maintain the instructor's voice**: Keep the educational tone and \
teaching style. The text should read as a written lecture — authoritative \
yet approachable — not as a dry reference manual.

7. **Format code references**: Use inline code formatting (`backticks`) \
for code elements like function names, variable names, library names, \
file names, and commands.

8. **Use Markdown formatting**: Use bullet points, numbered lists, and \
bold text where they improve readability and structure.

## Output format

Return ONLY the edited textbook chapter in Markdown format. Do not include \
any preamble, commentary, or explanation about your edits. Start directly \
with the chapter title as a top-level heading (# Title).
"""


def get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(
            "ERROR: GOOGLE_API_KEY not found in environment.\n"
            f"Ensure it is set in {PROJECT_ROOT / '.env'}",
            file=sys.stderr,
        )
        sys.exit(1)
    return genai.Client(api_key=api_key)


def derive_chapter_title(filename: str) -> str:
    """Extract a human-readable title hint from the filename."""
    name = filename.removesuffix(".txt")
    # Remove leading index number (e.g., "1. Introduction" → "Introduction")
    parts = name.split(". ", 1)
    if len(parts) > 1 and parts[0].isdigit():
        name = parts[1]
    return name


def edit_transcript(client: genai.Client, raw_text: str, title_hint: str) -> str:
    """Send a raw transcript to Gemini for editing."""
    user_prompt = (
        f"The following is a raw auto-generated transcript from a lecture video "
        f"titled \"{title_hint}\" in the PUM (Designing Services Using AI Methods / "
        f"Machine Learning Fundamentals) course at PJATK. "
        f"Edit it into a professional textbook chapter.\n\n"
        f"---\n\n{raw_text}"
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=30000,
        ),
    )

    if not response.text:
        raise RuntimeError(f"Empty response from model. Finish reason: {response.candidates[0].finish_reason}")

    return response.text


def main():
    if not RAW_DIR.exists():
        print(f"ERROR: Raw transcripts directory not found: {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        print(f"ERROR: No .txt files found in {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    client = get_client()

    print(f"Model: {MODEL}")
    print(f"Input directory: {RAW_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files to process: {len(raw_files)}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, raw_file in enumerate(raw_files, start=1):
        output_file = OUTPUT_DIR / raw_file.name
        title_hint = derive_chapter_title(raw_file.name)

        # Skip if output already exists and is non-empty
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"[{i}/{len(raw_files)}] {raw_file.name} — SKIPPED (already exists)")
            skip_count += 1
            results.append({
                "filename": raw_file.name,
                "status": "skipped",
                "error": None,
            })
            continue

        print(f"[{i}/{len(raw_files)}] {raw_file.name}...", end=" ", flush=True)

        raw_text = raw_file.read_text(encoding="utf-8")

        try:
            edited_text = edit_transcript(client, raw_text, title_hint)
            output_file.write_text(edited_text, encoding="utf-8")
            print("OK")
            success_count += 1
            results.append({
                "filename": raw_file.name,
                "status": "success",
                "error": None,
            })
        except Exception as e:
            print(f"FAILED: {e}")
            fail_count += 1
            results.append({
                "filename": raw_file.name,
                "status": "failed",
                "error": str(e),
            })

        # Rate-limit between requests
        if i < len(raw_files):
            time.sleep(DELAY_SECONDS)

    # Write processing log
    log_path = OUTPUT_DIR / "edit_log.json"
    log_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDone. {success_count} edited, {fail_count} failed, {skip_count} skipped.")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Log: {log_path}")

    if fail_count > 0:
        print("\nRe-run the script to retry failed files (existing outputs are skipped).")


if __name__ == "__main__":
    main()
