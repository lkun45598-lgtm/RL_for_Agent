"""
@file validate_loss.py
@description Thin CLI/testability wrapper around loss_transfer.validation.validate_loss.
@author kongzhiquan
@contributors Leizheng
@date 2026-03-28
@version 1.0.0

@changelog
  - 2026-03-28 kongzhiquan: v1.0.0 expose patchable wrapper functions for validate_loss tests
"""

from __future__ import annotations

from typing import Any

from loss_transfer.validation import validate_loss as _impl
from loss_transfer.validation.validate_loss import *  # noqa: F401,F403


# Explicitly mirror patchable helpers so tests can patch `validate_loss.<name>`
# and the delegated implementation will observe the overridden binding.
load_formula_spec = _impl.load_formula_spec
_copy_loss_to_sandbox = _impl._copy_loss_to_sandbox
_resolve_sandbox_override_dir = _impl._resolve_sandbox_override_dir
_prepare_config_path = _impl._prepare_config_path
_run_subprocess_with_combined_log = _impl._run_subprocess_with_combined_log


def _sync_impl_bindings() -> None:
    _impl.load_formula_spec = load_formula_spec
    _impl._copy_loss_to_sandbox = _copy_loss_to_sandbox
    _impl._resolve_sandbox_override_dir = _resolve_sandbox_override_dir
    _impl._prepare_config_path = _prepare_config_path
    _impl._run_subprocess_with_combined_log = _run_subprocess_with_combined_log


def validate_static(*args: Any, **kwargs: Any):
    _sync_impl_bindings()
    return _impl.validate_static(*args, **kwargs)


def validate_smoke(*args: Any, **kwargs: Any):
    _sync_impl_bindings()
    return _impl.validate_smoke(*args, **kwargs)


def validate_single_model(*args: Any, **kwargs: Any):
    _sync_impl_bindings()
    return _impl.validate_single_model(*args, **kwargs)


def validate_full_run(*args: Any, **kwargs: Any):
    _sync_impl_bindings()
    return _impl.validate_full_run(*args, **kwargs)


def main() -> None:
    _sync_impl_bindings()
    _impl.main()


if __name__ == '__main__':
    main()
