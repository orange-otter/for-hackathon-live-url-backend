# document_parser.py

"""A module for extracting text from documents using LlamaParse."""

import os
from llama_parse import LlamaParse


def extract_text_from_document(
    file_path: str,
    result_type: str = "text",
    split_documents: bool = False
) -> str:
    """
    Extracts structured text from a document using the LlamaParse API.

    This function requires the 'LLAMA_CLOUD_API_KEY' environment variable to be set.

    Args:
        file_path (str): The local path to the document file (e.g., PDF, DOCX).
        result_type (str, optional): The desired output format.
                                     Can be "text" or "markdown". Defaults to "text".
        split_documents (bool, optional): If True, keeps each parsed section separate.
                                          If False, merges them into a single string.
                                          Defaults to False.

    Returns:
        str: The extracted content as a single string (or structured markdown).

    Raises:
        ValueError: If the API key environment variable is not set.
        FileNotFoundError: If the specified file_path does not exist.
        Exception: For any errors encountered during the API call or parsing.
    """

    # 1. Check for API Key
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError(
            "LLAMA_CLOUD_API_KEY environment variable is not set. "
            "You can get a key from https://cloud.llamaindex.ai/regions."
        )

    # 2. Check for File Existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at the specified path: {file_path}")

    try:
        # 3. Initialize Parser
        parser = LlamaParse(result_type=result_type, api_key=api_key)

        # 4. Parse File
        documents = parser.load_data(file_path)

        if not documents:
            return "Warning: Document was parsed, but no content was extracted."

        # 5. Return Cleaned Text
        if split_documents:
            return "\n\n--- DOCUMENT SPLIT ---\n\n".join(doc.text.strip() for doc in documents if doc.text.strip())
        else:
            return "\n\n".join(doc.text.strip() for doc in documents if doc.text.strip())

    except Exception as e:
        raise Exception(f"❌ Failed to parse document '{file_path}'. Error: {e}") from e
