def test_package_imports_and_has_version():
    import emotive_gait

    assert isinstance(emotive_gait.__version__, str)
    assert emotive_gait.__version__


def test_package_version_matches_pyproject():
    import tomllib
    from pathlib import Path

    import emotive_gait

    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert emotive_gait.__version__ == data["project"]["version"] == "0.3.0"
