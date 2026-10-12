import pytest
from pathlib import Path
from pyconcepticon import Concepticon


@pytest.fixture
def mockcepticon():
    mockcepticon_path = Path(__file__).parent / "test_data" / "mockcepticon"
    return Concepticon(mockcepticon_path)


@pytest.fixture
def mock_concept(mockcepticon):
    return list(mockcepticon.conceptsets.values())[0].gloss


@pytest.fixture
def mock_definition(mockcepticon):
    return list(mockcepticon.conceptsets.values())[0].definition
