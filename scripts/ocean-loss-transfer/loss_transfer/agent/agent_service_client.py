"""
@file agent_service_client.py
@description Minimal client for the local KODE agent HTTP service (SSE stream).
@author OpenAI Codex
@date 2026-03-25
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from loss_transfer.common.paths import PROJECT_ENV_FILE

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise RuntimeError('requests is required for agent_service_client.py') from exc

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ENV_FILE)
except ImportError:
    pass


def resolve_service_url(service_url: Optional[str] = None) -> str:
    return resolve_service_descriptor(service_url)['resolved_url']


def resolve_service_descriptor(service_url: Optional[str] = None) -> Dict[str, str]:
    if service_url and service_url.strip():
        return {
            'resolved_url': service_url.rstrip('/'),
            'source': 'explicit',
        }

    env_url = os.getenv('KODE_API_URL')
    if env_url and env_url.strip():
        return {
            'resolved_url': env_url.rstrip('/'),
            'source': 'env',
        }

    port = os.getenv('KODE_API_PORT', '8787').strip() or '8787'
    return {
        'resolved_url': f'http://localhost:{port}',
        'source': 'default_localhost',
    }


def resolve_api_key(api_key: Optional[str] = None) -> str:
    key = api_key or os.getenv('KODE_API_SECRET')
    if not key:
        raise ValueError('Missing service API key. Set KODE_API_SECRET or pass api_key explicitly.')
    return key


def fetch_service_health(
    *,
    service_url: Optional[str] = None,
    timeout_sec: int = 2,
) -> Dict[str, Any]:
    descriptor = resolve_service_descriptor(service_url)
    try:
        response = requests.get(
            descriptor['resolved_url'] + '/health',
            timeout=timeout_sec,
        )
    except requests.RequestException as exc:
        return {
            'status': 'unavailable',
            'resolved_url': descriptor['resolved_url'],
            'error': str(exc),
        }

    if response.status_code != 200:
        return {
            'status': 'error',
            'resolved_url': descriptor['resolved_url'],
            'http_status': response.status_code,
            'body': response.text[:500],
        }

    try:
        payload = response.json()
    except ValueError as exc:
        return {
            'status': 'error',
            'resolved_url': descriptor['resolved_url'],
            'error': f'Invalid health payload: {exc}',
        }

    if isinstance(payload, dict):
        return {
            'status': 'ok',
            'resolved_url': descriptor['resolved_url'],
            **payload,
        }
    return {
        'status': 'error',
        'resolved_url': descriptor['resolved_url'],
        'error': 'Health payload was not a JSON object',
    }


def run_agent_chat(
    *,
    message: str,
    working_dir: str,
    outputs_path: str,
    notebook_path: str,
    files: Optional[List[str]] = None,
    mode: str = 'edit',
    user_id: str = 'loss-transfer-agent',
    service_url: Optional[str] = None,
    api_key: Optional[str] = None,
    agent_id: Optional[str] = None,
    timeout_sec: int = 900,
) -> Dict[str, Any]:
    service_descriptor = resolve_service_descriptor(service_url)
    url = service_descriptor['resolved_url'] + '/api/chat/stream'
    headers = {
        'Content-Type': 'application/json',
        'X-API-Key': resolve_api_key(api_key),
    }
    body: Dict[str, Any] = {
        'message': message,
        'mode': mode,
        'outputsPath': outputs_path,
        'context': {
            'userId': user_id,
            'workingDir': working_dir,
            'notebookPath': notebook_path,
            'files': files or [],
        },
    }
    if agent_id:
        body['agentId'] = agent_id

    text_chunks: List[str] = []
    events: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    current_agent_id: Optional[str] = None
    session_scope = 'reused_agent_session' if agent_id else 'new_request_session'

    try:
        response = requests.post(
            url,
            headers=headers,
            json=body,
            stream=True,
            timeout=(15, timeout_sec),
        )
    except requests.RequestException as exc:
        return {
            'status': 'error',
            'error': f'Failed to connect to agent service: {exc}',
            'events': [],
            'text': '',
            'agent_id': None,
            'requested_agent_id': agent_id,
            'session_scope': session_scope,
            'service_url': service_descriptor['resolved_url'],
            'service_url_source': service_descriptor['source'],
            'tool_calls': [],
            'tool_results': [],
        }

    with response:
        if response.status_code != 200:
            try:
                error_text = response.text
            except Exception:
                error_text = ''
            return {
                'status': 'error',
                'error': f'Agent service returned {response.status_code}: {error_text[:500]}',
                'events': [],
                'text': '',
                'agent_id': None,
                'requested_agent_id': agent_id,
                'session_scope': session_scope,
                'service_url': service_descriptor['resolved_url'],
                'service_url_source': service_descriptor['source'],
                'tool_calls': [],
                'tool_results': [],
            }

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith('data: '):
                continue
            payload = raw_line[6:]
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if not isinstance(event, dict):
                continue

            events.append(event)
            event_type = event.get('type')

            if event_type == 'start':
                current_agent_id = event.get('agentId')
            elif event_type == 'text':
                content = event.get('content')
                if isinstance(content, str):
                    text_chunks.append(content)
            elif event_type == 'tool_use':
                tool_calls.append(event)
            elif event_type == 'tool_result':
                tool_results.append(event)
            elif event_type in {'tool_error', 'agent_error', 'error'}:
                errors.append(event)

    return {
        'status': 'ok' if not errors else 'error',
        'error': errors[0].get('error') if errors else None,
        'events': events,
        'text': ''.join(text_chunks),
        'agent_id': current_agent_id,
        'requested_agent_id': agent_id,
        'session_scope': session_scope,
        'service_url': service_descriptor['resolved_url'],
        'service_url_source': service_descriptor['source'],
        'tool_calls': tool_calls,
        'tool_results': tool_results,
        'errors': errors,
    }
