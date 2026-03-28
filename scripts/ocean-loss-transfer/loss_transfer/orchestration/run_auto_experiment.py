"""
@file run_auto_experiment.py
@description Build task context and execute the new agentic loss-transfer loop.
@author Leizheng
@date 2026-03-25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from loss_transfer.agent.agent_artifact_generator import generate_analysis_plan
from loss_transfer.common.contract_validation import write_contract_validation
from loss_transfer.common.run_manifest import write_run_manifest
from loss_transfer.context.context_builder import build_task_context


def run_auto_experiment(
    paper_slug: str,
    *,
    paper_pdf_path: Optional[str] = None,
    code_repo_path: Optional[str] = None,
    loss_ir_yaml: Optional[str] = None,
    dataset_root: Optional[str] = None,
    analysis_plan_json: Optional[str] = None,
    output_dir: Optional[str] = None,
    mode: str = 'agent_loop',
    bootstrap_formula: bool = True,
    max_attempts: int = 4,
    auto_generate_plan: bool = False,
    service_url: Optional[str] = None,
    service_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    task_context = build_task_context(
        paper_slug=paper_slug,
        paper_pdf_path=paper_pdf_path,
        code_repo_path=code_repo_path,
        loss_ir_yaml=loss_ir_yaml,
        dataset_root=dataset_root,
        output_dir=output_dir,
    )
    paths = task_context.get('paths', {}) if isinstance(task_context.get('paths'), dict) else {}
    experiment_dir = Path(str(paths.get('experiment_dir') or output_dir or '.')).expanduser().resolve()
    run_manifest_result = write_run_manifest(
        experiment_dir=experiment_dir,
        paper_slug=paper_slug,
        task_context=task_context,
        mode=mode,
        bootstrap_formula=bootstrap_formula,
        max_attempts=max_attempts,
        auto_generate_plan=auto_generate_plan,
        service_url=service_url,
        analysis_plan_path=analysis_plan_json,
    )
    context_validation = write_contract_validation(
        experiment_dir=experiment_dir,
        paper_slug=paper_slug,
        task_context=task_context,
    )
    run_manifest_result = write_run_manifest(
        experiment_dir=experiment_dir,
        paper_slug=paper_slug,
        task_context=task_context,
        mode=mode,
        bootstrap_formula=bootstrap_formula,
        max_attempts=max_attempts,
        auto_generate_plan=auto_generate_plan,
        service_url=service_url,
        analysis_plan_path=analysis_plan_json,
    )
    if context_validation.get('status') == 'error':
        return {
            'status': 'context_error',
            'paper_slug': paper_slug,
            'task_context_path': task_context.get('paths', {}).get('task_context_path'),
            'routing_audit_path': context_validation.get('paths', {}).get('routing_audit_path'),
            'contract_validation_path': context_validation.get('contract_validation_path'),
            'run_manifest_path': run_manifest_result.get('run_manifest_path'),
            'contract_validation_errors': context_validation.get('errors', []),
            'contract_validation_warnings': context_validation.get('warnings', []),
        }

    if mode == 'context_only':
        return {
            'status': 'context_ready',
            'paper_slug': paper_slug,
            'task_context_path': task_context.get('paths', {}).get('task_context_path'),
            'loss_formula_path': task_context.get('paths', {}).get('loss_formula_path'),
            'loss_ir_path': task_context.get('paths', {}).get('loss_ir_path'),
            'routing_audit_path': context_validation.get('paths', {}).get('routing_audit_path'),
            'contract_validation_path': context_validation.get('contract_validation_path'),
            'run_manifest_path': run_manifest_result.get('run_manifest_path'),
            'decision_trace_path': task_context.get('paths', {}).get('decision_trace_path'),
            'rl_dataset_path': task_context.get('paths', {}).get('rl_dataset_path'),
        }

    plan_generation: Optional[Dict[str, Any]] = None
    if not analysis_plan_json and auto_generate_plan:
        plan_generation = generate_analysis_plan(
            task_context.get('paths', {}).get('task_context_path'),
            max_attempts=max_attempts,
            service_url=service_url,
            api_key=service_api_key,
        )
        if plan_generation.get('status') != 'success':
            return {
                'status': 'plan_generation_failed',
                'paper_slug': paper_slug,
                'task_context_path': task_context.get('paths', {}).get('task_context_path'),
                'loss_formula_path': task_context.get('paths', {}).get('loss_formula_path'),
                'routing_audit_path': context_validation.get('paths', {}).get('routing_audit_path'),
                'contract_validation_path': context_validation.get('contract_validation_path'),
                'run_manifest_path': run_manifest_result.get('run_manifest_path'),
                'plan_generation': plan_generation,
            }
        analysis_plan_json = plan_generation.get('analysis_plan_path')

    from loss_transfer.agent.agent_repair_loop import run_agent_repair_loop

    result = run_agent_repair_loop(
        task_context,
        analysis_plan_path=analysis_plan_json,
        max_attempts=max_attempts,
        bootstrap_formula=bootstrap_formula,
        dataset_root=dataset_root,
        output_dir=output_dir,
        agent_service_url=service_url,
        agent_api_key=service_api_key,
    )
    if plan_generation is not None:
        result['plan_generation'] = plan_generation
    result.setdefault('run_manifest_path', run_manifest_result.get('run_manifest_path'))
    result.setdefault('contract_validation_path', context_validation.get('contract_validation_path'))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the agentic loss-transfer loop')
    parser.add_argument('--paper_slug', required=True)
    parser.add_argument('--paper_pdf', default=None)
    parser.add_argument('--code_repo', default=None)
    parser.add_argument('--loss_ir_yaml', default=None)
    parser.add_argument('--dataset_root', default=None)
    parser.add_argument('--analysis_plan_json', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--mode', choices=['context_only', 'agent_loop'], default='agent_loop')
    parser.add_argument('--max_attempts', type=int, default=4)
    parser.add_argument('--bootstrap_formula', action='store_true', default=True)
    parser.add_argument('--no_bootstrap_formula', action='store_false', dest='bootstrap_formula')
    parser.add_argument('--auto_generate_plan', action='store_true', default=False)
    parser.add_argument('--service_url', default=None)
    parser.add_argument('--service_api_key', default=None)
    args = parser.parse_args()

    result = run_auto_experiment(
        paper_slug=args.paper_slug,
        paper_pdf_path=args.paper_pdf,
        code_repo_path=args.code_repo,
        loss_ir_yaml=args.loss_ir_yaml,
        dataset_root=args.dataset_root,
        analysis_plan_json=args.analysis_plan_json,
        output_dir=args.output_dir,
        mode=args.mode,
        bootstrap_formula=args.bootstrap_formula,
        max_attempts=args.max_attempts,
        auto_generate_plan=args.auto_generate_plan,
        service_url=args.service_url,
        service_api_key=args.service_api_key,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
