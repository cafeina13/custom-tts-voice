"""Wrapper around ``piper.train.export_onnx`` that forces the legacy
TorchScript-based ONNX exporter instead of the new dynamo path.

PyTorch >= 2.6 flipped ``torch.onnx.export``'s default to the
``torch.export``-based ("dynamo") backend. That backend trips on a few
VITS-side runtime operations (``assert (discriminant >= 0).all()`` inside
``rational_quadratic_spline``, dynamic-shape guards, etc.) and fails with
``GuardOnDataDependentSymNode``. Piper's ``export_onnx.py`` calls
``torch.onnx.export`` without explicitly setting ``dynamo=False``, so we
patch the default here before importing it.

After the patch we just call ``piper.train.export_onnx.main()`` — same CLI,
same outputs, same behaviour as the legacy PyTorch / Piper combo.
"""
import torch

_original_export = torch.onnx.export


def _patched_export(*args, **kwargs):
    kwargs.setdefault("dynamo", False)
    return _original_export(*args, **kwargs)


torch.onnx.export = _patched_export  # type: ignore[assignment]


from piper.train.export_onnx import main  # noqa: E402

if __name__ == "__main__":
    main()
