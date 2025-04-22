from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import uuid
import subprocess
import time
import asyncio
import httpx
import shutil
import json
from datetime import datetime
import aiofiles
import logging
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(title="Multilingual Meeting Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(os.path.dirname(BASE_DIR), "uploads")
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Create necessary directories
for directory in [UPLOADS_DIR, CHUNKS_DIR, OUTPUTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Alibaba Cloud ASR API configuration from environment variables
ALIBABA_CONFIG = {
    "appKey": os.getenv("ALIBABA_APP_KEY", "PVI9DP7cToUpEhvN"),
    "token": os.getenv("ALIBABA_TOKEN", "1ec68967dee1422daa102e1e037bf145"),
    "apiEndpoint": os.getenv("ALIBABA_ENDPOINT", "https://nls-gateway-cn-shanghai.aliyuncs.com/stream/v1/file"),
    "sampleRate": os.getenv("ALIBABA_SAMPLE_RATE", "16000"),
}

# Audio processing configuration from environment variables
AUDIO_CONFIG = {
    "chunkDuration": int(os.getenv("CHUNK_DURATION", "60")),  # in seconds
    # MB to bytes
    "maxFileSizeForDirectProcessing": int(os.getenv("MAX_FILE_SIZE_DIRECT_MB", "5")) * 1024 * 1024,
}

# Initialize Groq client with API key from environment
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Response models - updated for compatibility with older Pydantic
class TranscriptionResponse(BaseModel):
    transcript: str
    translated: str
    usedFallback: bool = False

    class Config:
        # Needed for Pydantic v1
        schema_extra = {
            "example": {
                "transcript": "This is the transcribed text.",
                "translated": "This is the translated text.",
                "usedFallback": False
            }
        }

# Helper function to get file extension


def get_file_extension(filename: str) -> str:
    """Extract extension from filename."""
    return os.path.splitext(filename)[1].lower()

# Validate allowed audio file extensions


def is_valid_audio_file(filename: str) -> bool:
    """Check if the file has an allowed audio extension."""
    valid_extensions = ['.mp3', '.wav', '.m4a']
    return get_file_extension(filename) in valid_extensions

# Audio processing functions


async def process_audio_with_ffmpeg(input_file: str, output_prefix: str) -> List[str]:
    """
    Process audio file with ffmpeg, splitting into chunks if needed.
    Returns a list of file paths to the chunks.
    """
    try:
        # Check file size
        file_size = os.path.getsize(input_file)

        # For small files, no need to chunk
        if file_size <= AUDIO_CONFIG["maxFileSizeForDirectProcessing"]:
            logger.info("File is small enough for direct processing")
            return [input_file]

        logger.info(
            f"Processing large audio file ({file_size / (1024 * 1024):.2f} MB)")

        # Get audio duration using ffprobe
        cmd = [
            "ffprobe",
            "-i", input_file,
            "-show_entries", "format=duration",
            "-v", "quiet",
            "-of", "csv=p=0"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Error getting audio duration: {result.stderr}")
            raise Exception("Could not determine audio duration")

        # Parse duration
        total_seconds = float(result.stdout.strip())
        logger.info(f"Audio duration: {total_seconds:.2f} seconds")

        # Calculate number of chunks
        num_chunks = int(
            (total_seconds + AUDIO_CONFIG["chunkDuration"] - 1) // AUDIO_CONFIG["chunkDuration"])
        output_files = []

        # Create chunks
        for i in range(num_chunks):
            start_time = i * AUDIO_CONFIG["chunkDuration"]
            output_file = f"{output_prefix}_chunk{i}{get_file_extension(input_file)}"

            logger.info(
                f"Creating chunk {i+1}/{num_chunks}: {start_time}s to {start_time + AUDIO_CONFIG['chunkDuration']}s")

            cmd = [
                "ffmpeg",
                "-ss", str(start_time),
                "-t", str(AUDIO_CONFIG["chunkDuration"]),
                "-i", input_file,
                "-acodec", "copy",
                output_file,
                "-y"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Error creating chunk {i+1}: {result.stderr}")
                continue

            output_files.append(output_file)

        return output_files

    except Exception as e:
        logger.error(f"Error processing audio with ffmpeg: {str(e)}")
        raise e

# Transcription functions


async def transcribe_with_alibaba_asr(file_path: str) -> Optional[str]:
    """Transcribe audio using Alibaba Cloud ASR API with retry mechanism."""
    # Get file extension without the dot
    ext = get_file_extension(file_path)[1:]

    # Construct the URL with query parameters
    url = f"{ALIBABA_CONFIG['apiEndpoint']}?appkey={ALIBABA_CONFIG['appKey']}&token={ALIBABA_CONFIG['token']}&format={ext}&sample_rate={ALIBABA_CONFIG['sampleRate']}"

    # Define retry strategy
    max_retries = 2
    retries = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        while retries <= max_retries:
            try:
                logger.info(
                    f"Attempt {retries + 1}/{max_retries + 1} to call Alibaba ASR API...")

                # Read file as binary data
                async with aiofiles.open(file_path, 'rb') as f:
                    file_data = await f.read()

                # Make the API request
                response = await client.post(
                    url,
                    content=file_data,
                    headers={"Content-Type": "application/octet-stream"}
                )
                response.raise_for_status()

                data = response.json()

                # Check for API errors
                if data.get("status") != 20000000:
                    raise Exception(
                        f"Alibaba ASR API error: {data.get('message')}")

                logger.info("Transcription successful")
                return data.get("result")

            except Exception as e:
                retries += 1
                logger.error(
                    f"Alibaba ASR transcription error (attempt {retries}/{max_retries + 1}): {str(e)}")

                if retries <= max_retries:
                    # Wait before retrying (exponential backoff)
                    delay = 1.0 * (2 ** retries)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    # All retries failed
                    logger.error(
                        "All retry attempts failed. Using fallback mechanism.")
                    return None

    return None


async def transcribe_with_groq(file_path: str) -> Optional[str]:
    """Transcribe audio using Groq's Whisper API as a fallback."""
    try:
        logger.info("Attempting transcription with Groq's service...")

        with open(file_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )

        logger.info("Groq transcription successful")
        return transcription.text

    except Exception as e:
        logger.error(f"Groq transcription error: {str(e)}")
        return None


def get_mock_transcription(file_path: str) -> str:
    """Generate a mock transcription as a last resort fallback."""
    filename = os.path.basename(file_path)
    logger.info(f"Using mock transcription for file: {filename}")

    return "[This is a mock transcription because both Alibaba Cloud ASR and Groq transcription services are currently unavailable. The actual transcription would appear here.]"


async def transcribe_audio_chunks(chunks: List[str]) -> str:
    """Transcribe all audio chunks and combine them."""
    transcriptions = []

    for i, chunk in enumerate(chunks):
        logger.info(f"Transcribing chunk {i+1}/{len(chunks)}")

        # Try primary method (Alibaba ASR)
        transcription = await transcribe_with_alibaba_asr(chunk)

        # If primary method fails, try Groq
        if transcription is None:
            transcription = await transcribe_with_groq(chunk)

        # If both fail, use mock
        if transcription is None:
            transcription = get_mock_transcription(chunk)

        transcriptions.append(transcription)

    # Combine transcriptions
    return " ".join(transcriptions)


async def enhance_transcription_with_groq(raw_transcription: str) -> str:
    """Improve transcription quality using Groq."""
    try:
        logger.info("Enhancing transcription with Groq...")

        # If the transcription is too short, no need for enhancement
        if len(raw_transcription) < 1000:
            return raw_transcription

        # Create a simple function to split text into manageable chunks
        def split_text_into_chunks(text, chunk_size=4000, overlap=200):
            chunks = []
            start_index = 0

            while start_index < len(text):
                # Calculate end index for this chunk
                end_index = min(start_index + chunk_size, len(text))

                # If we're not at the end of the text, find a good break point
                if end_index < len(text) and end_index != len(text):
                    # Look back from end_index to find a sentence break
                    for i in range(end_index, max(start_index, end_index - overlap), -1):
                        if i < len(text) and text[i-1] in ".!?" and (i == len(text) or text[i].isspace()):
                            end_index = i
                            break

                # Add this chunk to our list
                chunks.append(text[start_index:end_index])

                # Move to the next chunk, with overlap
                start_index = max(start_index + 1, end_index - overlap)

            return chunks

        # Split the text into manageable chunks
        text_chunks = split_text_into_chunks(raw_transcription)
        logger.info(
            f"Split transcription into {len(text_chunks)} chunks for enhancement")

        # Process each chunk to improve/correct transcription
        enhanced_chunks = []
        for i, chunk in enumerate(text_chunks):
            try:
                logger.info(f"Enhancing chunk {i+1}/{len(text_chunks)}")

                prompt = """You are a transcription correction assistant. Your job is to fix any errors in this automatic speech recognition output. 
                Make minimal changes, only correcting obvious errors while preserving the original speaker's words and intent.
                
                Original transcription:
                {}
                
                Enhanced transcription:""".format(chunk)

                # Use Groq for enhancement
                completion = groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    model="qwen-qwq-32b",
                    temperature=0.2,  # Lower temperature for more accurate corrections
                    max_tokens=4096,
                    top_p=0.95,
                )

                enhanced_text = completion.choices[0].message.content if completion.choices else chunk
                enhanced_chunks.append(enhanced_text)

            except Exception as e:
                logger.error(f"Error enhancing chunk {i+1}: {str(e)}")
                # Fall back to original chunk on error
                enhanced_chunks.append(chunk)

        # Join the enhanced chunks
        return " ".join(enhanced_chunks)

    except Exception as e:
        logger.error(f"Error enhancing transcription: {str(e)}")
        return raw_transcription  # Return original transcription if enhancement fails


async def process_audio_file(file_path: str) -> tuple:
    """
    Process audio file including chunking, transcription, enhancement, and translation.
    Returns a tuple of (transcription, translation, used_fallback).
    """
    try:
        # Generate a unique ID for the chunk output prefix
        filename = os.path.basename(file_path)
        file_id = os.path.splitext(filename)[0]
        chunk_output_prefix = os.path.join(CHUNKS_DIR, file_id)

        # Process audio file with ffmpeg to get chunks
        chunks = await process_audio_with_ffmpeg(file_path, chunk_output_prefix)

        # Flag to track if fallback was used
        used_fallback = False

        # Transcribe all chunks
        transcription = await transcribe_audio_chunks(chunks)

        # Check if mock transcription was used (simplified check)
        if "[This is a mock transcription" in transcription:
            used_fallback = True

        # Enhance transcription with Groq
        enhanced_transcription = await enhance_transcription_with_groq(transcription)

        # Translate the text with Groq
        prompt = f"Translate to English:\n{enhanced_transcription}"

        try:
            completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                model="qwen-qwq-32b",
                temperature=0.6,
                max_tokens=4096,
                top_p=0.95,
            )

            translated = completion.choices[0].message.content if completion.choices else "Translation failed"
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            translated = "Translation failed: " + str(e)

        # Clean up chunks except the original file
        for chunk in chunks:
            if chunk != file_path and os.path.exists(chunk):
                os.remove(chunk)

        return enhanced_transcription, translated, used_fallback

    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        raise e

# FastAPI endpoints


@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and transcribe an audio file.
    Returns the transcription and translation.
    """
    try:
        # Validate file extension
        if not is_valid_audio_file(audio.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Only .mp3, .wav, and .m4a are supported."
            )

        # Generate a unique filename
        file_id = str(uuid.uuid4()).replace("-", "")
        file_extension = get_file_extension(audio.filename)
        file_path = os.path.join(UPLOADS_DIR, f"{file_id}{file_extension}")

        # Save the uploaded file
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await audio.read()
            await out_file.write(content)

        logger.info(f"Processing file: {file_path}")

        # Process the audio file
        transcription, translated, used_fallback = await process_audio_file(file_path)

        # Clean up the original file in the background
        if background_tasks:
            background_tasks.add_task(lambda: os.remove(
                file_path) if os.path.exists(file_path) else None)

        return TranscriptionResponse(
            transcript=transcription,
            translated=translated,
            usedFallback=used_fallback
        )

    except Exception as e:
        logger.error(f"Error in /api/transcribe: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def read_root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Multilingual Meeting Assistant API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
