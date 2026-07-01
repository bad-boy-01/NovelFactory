import pytest
import codecs
from pathlib import Path
from core.pipeline.dataset_provider import LocalDatasetProvider, RegexChapterSplitter
from core.domain.dataset import ValidationResult

@pytest.fixture
def empty_dataset(tmp_path):
    d = tmp_path / "dataset_empty"
    d.mkdir()
    (d / "novel.txt").touch() # Empty file
    return d

@pytest.fixture
def valid_dataset(tmp_path):
    d = tmp_path / "dataset_valid"
    d.mkdir()
    (d / "novel.txt").write_text("Hello World\n\nAnother paragraph", encoding="utf-8")
    (d / "metadata.json").write_text('{"title": "My Novel"}', encoding="utf-8")
    return d

@pytest.fixture
def missing_novel_dataset(tmp_path):
    d = tmp_path / "dataset_missing"
    d.mkdir()
    (d / "metadata.json").write_text('{"title": "My Novel"}', encoding="utf-8")
    return d

def test_validation_report_correctness(valid_dataset, missing_novel_dataset, empty_dataset):
    provider_valid = LocalDatasetProvider(valid_dataset)
    res = provider_valid.validate()
    assert res.is_valid is True
    assert len(res.errors) == 0
    assert len(res.warnings) == 0

    provider_missing = LocalDatasetProvider(missing_novel_dataset)
    res = provider_missing.validate()
    assert res.is_valid is False
    assert "Missing novel.txt file." in res.errors

    provider_empty = LocalDatasetProvider(empty_dataset)
    res = provider_empty.validate()
    assert res.is_valid is False
    assert "novel.txt is empty." in res.errors

def test_missing_metadata(tmp_path):
    d = tmp_path / "dataset_no_meta"
    d.mkdir()
    (d / "novel.txt").write_text("Test", encoding="utf-8")
    
    provider = LocalDatasetProvider(d)
    res = provider.validate()
    assert res.is_valid is True
    assert "Missing metadata.json file. Defaults will be used." in res.warnings

def test_encoding_fallback(tmp_path):
    d = tmp_path / "dataset_encoding"
    d.mkdir()
    # Write some invalid utf-8 bytes (Windows-1252 smart quotes or accents)
    (d / "novel.txt").write_bytes(b"L'H\xf4tel")
    
    provider = LocalDatasetProvider(d)
    text = provider.load_novel()
    assert text == "L'Hôtel"
    assert provider.load_manifest().encoding == "windows-1252"

def test_utf16_encoding(tmp_path):
    d = tmp_path / "dataset_utf16"
    d.mkdir()
    (d / "novel.txt").write_bytes(codecs.BOM_UTF16_LE + "Hello in UTF16".encode("utf-16-le"))
    
    provider = LocalDatasetProvider(d)
    text = provider.load_novel()
    assert text == "Hello in UTF16"
    assert provider.load_manifest().encoding == "utf-16"

def test_load_manifest_defaults(tmp_path):
    d = tmp_path / "dataset_no_meta"
    d.mkdir()
    (d / "novel.txt").write_text("Test", encoding="utf-8")
    
    provider = LocalDatasetProvider(d)
    manifest = provider.load_manifest()
    assert manifest.title == "Unknown"
    assert manifest.language == "en"

def test_iterator_behavior(valid_dataset):
    provider = LocalDatasetProvider(valid_dataset)
    chapters = list(provider.iter_chapters())
    assert len(chapters) == 1
    assert chapters[0].scenes[0].beats[0].text == "Hello World"
    assert chapters[0].scenes[0].beats[1].text == "Another paragraph"

def test_multi_chapter(tmp_path):
    d = tmp_path / "dataset_multi"
    d.mkdir()
    # For MVP splitter, we just return 1 chapter with multiple scenes/beats for now.
    # To truly test multi-chapter, we'd need a better RegexChapterSplitter.
    # We will test that we can iterate.
    (d / "novel.txt").write_text("Scene 1\n\nScene 2\n\nScene 3", encoding="utf-8")
    provider = LocalDatasetProvider(d)
    chapters = list(provider.iter_chapters())
    assert len(chapters) == 1
    assert len(chapters[0].scenes[0].beats) == 3

def test_load_chapter(valid_dataset):
    provider = LocalDatasetProvider(valid_dataset)
    chap = provider.load_chapter(0)
    assert chap.title == "Chapter 1"

def test_load_chapter_out_of_bounds(valid_dataset):
    provider = LocalDatasetProvider(valid_dataset)
    with pytest.raises(IndexError):
        provider.load_chapter(1)
