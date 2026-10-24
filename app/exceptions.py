"""Domain exceptions for PDF ingest and indexing."""


class AppError(Exception):
    """Base application error with an HTTP-friendly message."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class InvalidPDFError(AppError):
    def __init__(self, message: str = "Invalid PDF file") -> None:
        super().__init__(message, status_code=400)


class OversizedPDFError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=413)


class CorruptPDFError(AppError):
    def __init__(self, message: str = "Corrupt or unreadable PDF") -> None:
        super().__init__(message, status_code=400)


class EncryptedPDFError(AppError):
    def __init__(self, message: str = "Encrypted PDFs are not supported") -> None:
        super().__init__(message, status_code=400)


class NoExtractableTextError(AppError):
    def __init__(
        self,
        message: str = (
            "No extractable text found in this PDF. "
            "OCR is not yet supported — please upload a text-based PDF."
        ),
    ) -> None:
        super().__init__(message, status_code=422)


class PageExtractionError(AppError):
    def __init__(self, page_number: int, detail: str) -> None:
        super().__init__(
            f"Failed to extract text from page {page_number}: {detail}",
            status_code=400,
        )
        self.page_number = page_number
