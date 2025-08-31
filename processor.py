# processor.py

import os
import json
from typing import Optional, List
from google import genai
from pydantic import BaseModel, ValidationError


# ---------------- Data Models ---------------- #
class PartyDetails(BaseModel):
    shipowner_name: Optional[str]
    charterer_name: Optional[str]
    port_agent_name: Optional[str]
    confidence: Optional[float]


class CargoDetails(BaseModel):
    operation_type: Optional[str]
    cargo_type: Optional[str]
    quantity: Optional[float]
    unit: Optional[str]
    confidence: Optional[float]


class Signatory(BaseModel):
    role: Optional[str]
    name: Optional[str]
    date_signed: Optional[str]
    confidence: Optional[float]


class DocumentDetails(BaseModel):
    document_source: Optional[str]
    date_of_document: Optional[str]
    port_name: Optional[str]
    vessel_name: Optional[str]
    voyage_number: Optional[str]
    parties: Optional[PartyDetails]
    cargo: Optional[CargoDetails]
    confidence: Optional[float]


class Event(BaseModel):
    event_id: Optional[int]
    event_type: Optional[str]
    start_date: Optional[str]
    start_time: Optional[str]
    end_date: Optional[str]
    end_time: Optional[str]
    duration_hours: Optional[float]
    weather_conditions: Optional[str]
    remarks: Optional[str]
    confidence: Optional[float]


class LaytimeNotes(BaseModel):
    free_time_periods_identified: Optional[str]
    suspension_periods_identified: Optional[str]
    remarks_on_interruptions_or_delays: Optional[str]
    confidence: Optional[float]


class SoFSchema(BaseModel):
    """Represents the structured format for a Statement of Facts document."""
    document_details: DocumentDetails
    events: List[Event]
    laytime_notes: LaytimeNotes
    approvals: Optional[List[Signatory]]


# ---------------- Helpers ---------------- #
def _run_extraction(sof_text: str, temperature: float) -> dict:
    """Run a single extraction with given temperature."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY environment variable.")
    client = genai.Client(api_key=api_key)

    config = {
        "response_mime_type": "application/json",
        "response_schema": SoFSchema,
        "temperature": temperature,
    }

    prompt = f"""
    You are given a Statement of Facts (SOF) document.
    Extract its details into the provided schema (SoFSchema).

    Guidelines:
     If information is clearly present in the document, do not leave fields blank.

     For every recorded event, ensure both start_time and end_time are populated. If an event represents a single point in time, use that same timestamp for both the start and end.

     If distinct start and end times are given, calculate the duration_hours.

     Include all relevant notes on weather, delays, tug usage, approvals, and laytime.

     Only leave a field as null if the data is truly missing from the document.

     Add a confidence score (from 0.0 to 1.0) for each extracted value.
    """

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=sof_text + "\n\n" + prompt,
        config=config,
    )

    try:
        parsed: SoFSchema = response.parsed
        return parsed.model_dump()
    except ValidationError:
        return json.loads(response.text)


def _refine_extraction(sof_text: str, extracted_jsons: list[dict]) -> dict:
    """Merge two extracted JSONs into a final one using the original text."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY environment variable.")
    client = genai.Client(api_key=api_key)

    config = {
        "response_mime_type": "application/json",
        "response_schema": SoFSchema,
        "temperature": 0.0,
    }

    prompt = f"""
**Role & Goal:**
You are an expert data adjudicator specializing in maritime Statement of Facts (SOF) documents. Your task is to act as the final authority, producing a single, definitive, and highly accurate JSON record by critically analyzing the provided information.

**Inputs:**
1.  **The Document Text:** The original, unaltered source of truth.
2.  **Extraction 1:** A first attempt to structure the data.
3.  **Extraction 2:** A second, independent attempt.

**Core Logic & Rules:**
1.  **The Source is Final:** The Document Text is the absolute ground truth. Every field in your final output must be directly verifiable from this text.
2.  **Resolve Conflicts:** When the extractions disagree, you must re-examine the Document Text to determine the correct value. Do not guess or choose randomly.
3.  **Merge for Completeness:** Create the most complete record possible. If one extraction captured a detail (e.g., a remark or event) that the other missed, include it in your final output, but only after confirming it exists in the source text.
4.  **Determine Final Confidence:** Your confidence score for each field should reflect the evidence.
    - If both extractions agree and are confirmed by the text, confidence should be high (e.g., 0.95-1.0).
    - If you resolved a conflict or found a value only one extraction caught, your confidence should be slightly lower but still high if the text is clear (e.g., 0.85-0.9).
    - If the text itself is ambiguous, reflect that in the score (e.g., 0.7-0.8).

**Output Requirements:**
- **Strict Schema:** The output MUST be a single, valid JSON object that strictly follows the SoFSchema.
- **No Fabrication:** Do not invent, hallucinate, or infer data that is not explicitly present in the Document Text. If information is absent, the value MUST be `null`.

Begin your analysis. Compare the two extractions against the source text and generate the final, consolidated JSON object.

--- DOCUMENT TEXT ---
{sof_text}
--- END DOCUMENT ---

--- EXTRACTION 1 (Initial Pass) ---
{json.dumps(extracted_jsons[0], indent=2)}
--- END EXTRACTION 1 ---

--- EXTRACTION 2 (Second Pass) ---
{json.dumps(extracted_jsons[1], indent=2)}
--- END EXTRACTION 2 ---

**Final Consolidated JSON:**
"""

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config=config,
    )

    try:
        parsed: SoFSchema = response.parsed
        return parsed.model_dump()
    except ValidationError:
        return json.loads(response.text)


# ---------------- Main Public Function ---------------- #
def get_structured_data(sof_text: str) -> dict:
    """
    Runs dual extraction + refinement.
    Returns a single final JSON (SoFSchema).
    """
    print(f"\n[DEBUG] Text length: {len(sof_text)} characters")
    print("[DEBUG] Text preview >>>")
    print(sof_text[:500], "...\n")

    # first pass
    extraction1 = _run_extraction(sof_text, 0.0)
    extraction2 = _run_extraction(sof_text, 0.3)

    # refinement
    final = _refine_extraction(sof_text, [extraction1, extraction2])
    return final