/**
 * @file run-python-unittest.mjs
 *
 * @description Read python3 or PYTHON3 from .env and run Python unittest with that interpreter.
 * @author kongzhiquan
 * @date 2026-03-31
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-03-31 kongzhiquan: v1.0.0 add npm entrypoint for Python unittest using the interpreter configured in .env
 */

import { spawn } from 'child_process'
import path from 'path'
import process from 'process'
import { fileURLToPath } from 'url'

import dotenv from 'dotenv'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const projectRoot = path.resolve(__dirname, '..')
const envPath = path.join(projectRoot, '.env')

dotenv.config({ path: envPath, override: false })

function pickPythonPath() {
  const candidates = [
    process.env.python3,
    process.env.PYTHON3,
    'python3',
  ]

  for (const candidate of candidates) {
    if (typeof candidate !== 'string') {
      continue
    }
    const trimmed = candidate.trim()
    if (trimmed) {
      return trimmed
    }
  }
  return 'python3'
}

const pythonPath = pickPythonPath()
const forwardedArgs = process.argv.slice(2)
const unittestArgs = forwardedArgs.length > 0
  ? ['-m', 'unittest', ...forwardedArgs]
  : ['-m', 'unittest', 'discover', 'scripts/ocean-loss-transfer/tests']

const child = spawn(pythonPath, unittestArgs, {
  cwd: projectRoot,
  stdio: 'inherit',
  env: {
    ...process.env,
    python3: pythonPath,
    PYTHON3: pythonPath,
  },
})

child.on('error', (error) => {
  console.error(`[test:py] failed to start python interpreter "${pythonPath}": ${error.message}`)
  process.exit(1)
})

child.on('exit', (code, signal) => {
  if (signal) {
    process.exit(1)
  }
  process.exit(code ?? 1)
})
