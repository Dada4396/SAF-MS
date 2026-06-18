def test_package_exposes_public_api():
    import sparse_ms_flow

    assert sparse_ms_flow.__version__ == "0.1.0"
    assert sparse_ms_flow.SparseMSFlowModel is not None
