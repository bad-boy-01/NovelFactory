import pytest
from pathlib import Path
from core.pipeline.project_builder import ProjectBuilder

def test_project_builder_missing_dataset(tmp_path):
    projects_dir = tmp_path / "projects"
    datasets_dir = tmp_path / "datasets"
    
    builder = ProjectBuilder(projects_dir, datasets_dir)
    
    with pytest.raises(ValueError, match="Dataset not found"):
        builder.build_project("MyVideo", "MyNovel")

def test_project_builder_success(tmp_path):
    projects_dir = tmp_path / "projects"
    datasets_dir = tmp_path / "datasets"
    projects_dir.mkdir()
    datasets_dir.mkdir()
    
    dataset_path = datasets_dir / "MyNovel"
    dataset_path.mkdir()
    (dataset_path / "novel.txt").touch()
    
    builder = ProjectBuilder(projects_dir, datasets_dir)
    manifest = builder.build_project("MyVideo", "MyNovel")
    
    assert manifest.project_name == "MyVideo"
    assert manifest.dataset_id == "MyNovel"
    
    project_dir = projects_dir / "MyVideo"
    assert (project_dir / "project.json").exists()
    assert (project_dir / "cache" / "v1" / "scenes").exists()
    assert (project_dir / "assets").exists()
