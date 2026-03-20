from sorakai.common.logging_utils import get_logger

logger = get_logger("sorakai.ingest_logic")


def process_file(file_content: str, chunk_size: int = 500) -> list[str]:
    logger.info("Chunking with size %s", chunk_size)
    return [file_content[i : i + chunk_size] for i in range(0, len(file_content), chunk_size)]
