def test_package_exposes_public_api():
    import saf_ms

    assert saf_ms.__version__ == "0.1.0"
    assert saf_ms.SAFMSModel is not None
