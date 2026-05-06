from clifft_cuda.backends import check_cuda


def test_cuda_diagnostics_shape() -> None:
    diag = check_cuda()
    assert isinstance(diag.nvidia_smi, bool)
    assert hasattr(diag, "missing_summary")
