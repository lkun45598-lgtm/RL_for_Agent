# Repository Guidelines

## Project Structure

- `src/`: TypeScript Agent HTTP service (Express + KODE SDK), SSE streaming, tool registration.
- `src/tools/`: Tool wrappers that call into Python scripts via `ctx.sandbox.exec`.
- `scripts/`: Python workflows.
  - `scripts/ocean-loss-transfer/`: loss migration pipeline (context prep, IR, validators, trials).
  - `scripts/ocean-SR-training-masked/`: training code used by `sandbox/` runners.
- `sandbox/`: experiment harness. Training dynamically loads `sandbox/sandbox_loss.py` via `sandbox/sandbox_trainer.py`.
- `workflow/loss_transfer/`: specs and safety rules (e.g. target interface and blocked patterns).

## Build, Test, And Dev Commands

```bash
npm install
pip install -r requirements.txt

npm run dev        # start server with watch reload
npm run start      # start server
npm run typecheck  # TypeScript typecheck
npm run test:client  # run the TS client test script
```

Loss transfer entry points:

```bash
python scripts/ocean-loss-transfer/prepare_context.py --code_repo <repo> --paper_slug <slug> [--paper_pdf <pdf>]
python scripts/ocean-loss-transfer/run_auto_experiment.py --paper_slug <slug> --code_repo <repo>
```

## Coding Style & Naming

- TypeScript: keep tool params small and explicit; always use `ctx.sandbox.exec` (no direct shelling outside sandbox).
- Python: prefer pure functions in `scripts/ocean-loss-transfer/`; keep outputs JSON/YAML and machine-parseable.
- Avoid hardcoding Python paths in scripts; use `scripts/python_manager.py` (or TS `src/utils/python-manager.ts`).

## Testing Guidelines

- For loss changes, always run `scripts/ocean-loss-transfer/validate_loss.py` (static + smoke at minimum).
- If a paper has a formula spec, keep `loss_formula.json` in `sandbox/loss_transfer_experiments/<slug>/` and ensure `symbol_map` is a 1:1 mapping.

## Commit & Pull Request Guidelines

- Commit messages follow Conventional Commits (`feat: ...`, `fix: ...`, `docs: ...`), often with Chinese detail after the prefix.
- PRs should include: what changed, why, how you validated (commands + key logs/metrics), and any new artifacts under `sandbox/loss_transfer_experiments/<slug>/`.

### Get-Content Usage in PowerShell

When utilizing the `Get-Content` cmdlet on Windows PowerShell, explicitly specify the encoding as **UTF8** to guarantee correct text decoding and avoid garbled characters. Always follow standard PowerShell syntax with formal parameter naming and valid Windows file paths.

Standard compliant example:
```powershell
Get-Content -Path "C:\target\file.txt" -Encoding UTF8
```
For higher robustness with special characters or literal file paths, use:
```powershell
Get-Content -LiteralPath "C:\target\file.txt" -Encoding UTF8 -Raw
```

## Author Documentation before Modifications or Generations
When generating or modifying code files, always add/update a standardized header comment at the top following this format:
```typescript
/**
 * @file filename.ext
 *
 * @description [Brief description]
 * @author kongzhiquan
 * @date YYYY-MM-DD
 * @version x.x.x
 *
 * @changelog
 *   - YYYY-MM-DD kongzhiquan: version description
 */
```
If the file already has a header, update the "Changelog" section and append `kongzhiquan` to the "@contributors" list if not already present. If '@contributors' is not present, add it below the author line.