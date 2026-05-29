from clifft_cuda.backends import check_cuda, check_rocm


def test_cuda_diagnostics_shape() -> None:
    diag = check_cuda()
    assert isinstance(diag.nvidia_smi, bool)
    assert hasattr(diag, "missing_summary")


def test_rocm_diagnostics_shape() -> None:
    diag = check_rocm()
    assert isinstance(diag.rocminfo, bool)
    assert isinstance(diag.rocm_smi, bool)
    assert hasattr(diag, "missing_summary")
