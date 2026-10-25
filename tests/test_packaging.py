def test_package_imports_and_has_version():
    import emotive_gait

    assert isinstance(emotive_gait.__version__, str)
    assert emotive_gait.__version__
