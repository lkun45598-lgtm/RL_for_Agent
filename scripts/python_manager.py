"""
@file python_manager.py

@description 管理和查找系统中可能的 Python 可执行文件路径（同步实现）
@author kongzhiquan
@date 2026-03-23
@version 1.1.0

@changelog
  - 2026-03-23 kongzhiquan: v1.0.0
    - 从 src/utils/python-manager.ts 移植为 Python 实现
    - 保留相同的扫描逻辑：环境变量、pyenv、conda、系统路径
    - 新增模块检测缓存（find_python_with_module）
  - 2026-03-23 kongzhiquan: v1.1.0
    - 改为同步实现，移除 asyncio 依赖
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# 模块检测结果缓存
_module_cache: dict[str, Optional[str]] = {}

# 合法的 Python 模块名格式（防止命令注入）
_MODULE_NAME_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_.]*$')


def find_possible_python_paths() -> list[str]:
    """扫描系统中可能的 Python 可执行文件路径"""
    home = Path.home()
    pyenv_root = Path(os.environ.get('PYENV_ROOT', home / '.pyenv'))

    candidates = (
        _collect_from_env()
        + _collect_pyenv_versions(pyenv_root)
        + _collect_common_locations(home, pyenv_root)
    )
    return _dedupe_and_filter_existing(candidates)


def find_first_python_path() -> Optional[str]:
    """返回第一个可用的 Python 路径，找不到则返回 None"""
    paths = find_possible_python_paths()
    return paths[0] if paths else None


def find_python_with_module(module_name: str) -> Optional[str]:
    """
    查找包含指定模块的 Python 路径（如 find_python_with_module('torch')）
    结果会缓存以避免重复检测
    """
    if not _MODULE_NAME_PATTERN.match(module_name):
        print(f'[python-manager] Invalid module name: "{module_name}"', file=sys.stderr)
        return None

    if module_name in _module_cache:
        return _module_cache[module_name]

    for py_path in find_possible_python_paths():
        try:
            result = subprocess.run(
                [py_path, '-c', f'import {module_name}'],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                _module_cache[module_name] = py_path
                return py_path
        except (subprocess.TimeoutExpired, OSError):
            pass

    _module_cache[module_name] = None
    return None


def _collect_from_env() -> list[Optional[str]]:
    """从环境变量收集 Python 路径候选"""
    is_win = sys.platform == 'win32'
    results: list[Optional[str]] = [
        os.environ.get('PYTHON3'),
        os.environ.get('PYTHON'),
    ]

    python_home = os.environ.get('PYTHON_HOME')
    if python_home:
        exe = 'python.exe' if is_win else 'python3'
        sub = '' if is_win else 'bin'
        results.append(str(Path(python_home, sub, exe) if sub else Path(python_home, exe)))

    virtual_env = os.environ.get('VIRTUAL_ENV')
    if virtual_env:
        if is_win:
            results.append(str(Path(virtual_env) / 'Scripts' / 'python.exe'))
        else:
            results.append(str(Path(virtual_env) / 'bin' / 'python'))

    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        if is_win:
            results.append(str(Path(conda_prefix) / 'python.exe'))
        else:
            results.append(str(Path(conda_prefix) / 'bin' / 'python'))

    return results


def _collect_pyenv_versions(pyenv_root: Path) -> list[str]:
    """扫描 pyenv 版本目录"""
    versions_dir = pyenv_root / 'versions'
    try:
        return [
            str(entry / 'bin' / 'python')
            for entry in versions_dir.iterdir()
            if entry.is_dir()
        ]
    except (OSError, PermissionError):
        return []


def _collect_common_locations(home: Path, pyenv_root: Path) -> list[Optional[str]]:
    """收集系统常见位置的 Python 路径"""
    is_win = sys.platform == 'win32'

    if is_win:
        program_files = os.environ.get('ProgramFiles')
        program_files_x86 = os.environ.get('ProgramFiles(x86)')
        paths: list[Optional[str]] = [str(home / 'anaconda3' / 'python.exe')]
        if program_files:
            paths.append(str(Path(program_files) / 'Anaconda3' / 'python.exe'))
        if program_files_x86:
            paths.append(str(Path(program_files_x86) / 'Anaconda3' / 'python.exe'))
        paths.append(r'C:\ProgramData\Anaconda3\python.exe')
        return paths

    return [
        '/usr/bin/python3',
        '/usr/local/bin/python3',
        '/opt/homebrew/bin/python3',
        '/opt/local/bin/python3',
        '/usr/bin/python',
        '/usr/local/bin/python',
        str(pyenv_root / 'shims' / 'python'),
        *_collect_conda_envs(home),
    ]


def _collect_conda_envs(home: Path) -> list[str]:
    """扫描当前用户的 conda 环境 (miniconda3/envs, anaconda3/envs)"""
    results: list[str] = []
    for conda_dir in ('miniconda3', 'anaconda3'):
        envs_dir = home / conda_dir / 'envs'
        try:
            for env in envs_dir.iterdir():
                if env.is_dir():
                    results.append(str(env / 'bin' / 'python'))
        except (OSError, PermissionError):
            pass
    return results


def _dedupe_and_filter_existing(paths: list[Optional[str]]) -> list[str]:
    """去重并过滤出实际存在且可执行的路径"""
    is_win = sys.platform == 'win32'
    seen: set[str] = set()
    results: list[str] = []

    for raw in paths:
        if not raw:
            continue
        candidate = raw.strip()
        if not candidate:
            continue
        key = candidate.lower() if is_win else candidate
        if key in seen:
            continue
        p = Path(candidate)
        if p.is_file() and (is_win or os.access(candidate, os.X_OK)):
            seen.add(key)
            results.append(candidate)

    return results


# CLI 入口：直接运行时打印所有可用 Python 路径
if __name__ == '__main__':
    import argparse

    paths = find_possible_python_paths()
    if paths:
        print('Found Python executables:')
        for p in paths:
            print(f'  {p}')
    else:
        print('No Python executables found.')

    parser = argparse.ArgumentParser(description='Python path manager')
    parser.add_argument('--module', help='Find Python with specific module')
    args, _ = parser.parse_known_args()
    if args.module:
        result = find_python_with_module(args.module)
        if result:
            print(f'\nPython with module "{args.module}": {result}')
        else:
            print(f'\nNo Python found with module "{args.module}"')
