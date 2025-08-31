# main.py

import os
import json
import shutil
import traceback
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from document_parser import extract_text_from_document
from processor import get_structured_data

# Load environment variables
load_dotenv()

# Create FastAPI app instance
app = FastAPI()

# --- START: RECOMMENDED CORS CONFIGURATION ---

# Define the list of domains that are allowed to make requests to your API
allowed_origins = [
    "https://sof-event-extractor-hackathon.netlify.app",
    "http://localhost:3000", # Optional: for local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins, # Use the specific list of origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)
# --- END: RECOMMENDED CORS CONFIGURATION ---


@app.get("/")
async def root():
    """Returns a simple message indicating the server is running."""
    return {"message": "Server is running"}


def clear_output_file(file_path: str = "output.json"):
    """
    Wipes the given file for privacy by replacing its contents with an empty JSON array.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("[]")
        print(f"--- ✅ Cleared {file_path} ---")
    except Exception as e:
        print(f"--- ❌ Could not clear {file_path}. Reason: {e} ---")


@app.post("/process")
async def process_uploaded_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Handles uploaded files:
    - saves them temporarily
    - extracts text
    - processes with AI
    - returns structured JSON
    - schedules cleanup in the background
    """
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    all_detailed_data = []

    for file in files:
        file_path = os.path.join(uploads_dir, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            raw_text = extract_text_from_document(file_path)
            detailed_ai_data = get_structured_data(raw_text)

            detailed_ai_data['fileName'] = file.filename
            all_detailed_data.append(detailed_ai_data)

        except Exception as e:
            print(f"--- ❌ Failed while processing {file.filename} ---")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Error while processing {file.filename}: {e}"
            )
        finally:
            if not file.file.closed:
                file.file.close()
            if os.path.exists(file_path):
                os.remove(file_path)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(all_detailed_data, f, indent=2, ensure_ascii=False)

    background_tasks.add_task(clear_output_file)

    return all_detailed_data
