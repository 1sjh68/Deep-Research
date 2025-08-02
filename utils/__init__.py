# utils/__init__.py
from .file_handler import (
    load_external_data,
    save_checkpoint,
    load_checkpoint,
    delete_checkpoint
)
from .text_processor import (
    truncate_text_for_context,
    calculate_checksum,
    preprocess_json_string,
    extract_knowledge_gaps,
    chunk_document_for_rag
)