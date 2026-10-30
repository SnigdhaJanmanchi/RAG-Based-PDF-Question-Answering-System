"""Streamlit startup / AppTest smoke (mocked backend)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from frontend.api_client import DocumentInfo

APP_PATH = Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py"


@pytest.fixture()
def mock_client() -> MagicMock:
    client = MagicMock()
    client.base_url = "http://localhost:8000"
    client.list_documents.return_value = []
    client.health.return_value = {"status": "ok"}
    return client


def test_streamlit_module_imports() -> None:
    import frontend.streamlit_app as app

    assert callable(app.main)
    assert APP_PATH.is_file()


def test_streamlit_app_smoke_empty_library(mock_client: MagicMock) -> None:
    from streamlit.testing.v1 import AppTest

    with patch("frontend.api_client.APIClient", return_value=mock_client):
        at = AppTest.from_file(str(APP_PATH), default_timeout=15)
        at.run()
        assert not at.exception
        assert mock_client.list_documents.called


def test_streamlit_app_smoke_with_documents(mock_client: MagicMock) -> None:
    from streamlit.testing.v1 import AppTest

    mock_client.list_documents.return_value = [
        DocumentInfo("d1", "sample.pdf", 2, 3),
    ]
    with patch("frontend.api_client.APIClient", return_value=mock_client):
        at = AppTest.from_file(str(APP_PATH), default_timeout=15)
        at.run()
        assert not at.exception
        assert at.radio
        labels = list(at.radio[0].options)
        assert any("sample.pdf" in str(opt) for opt in labels)
