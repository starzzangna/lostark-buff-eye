param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) {
        $taskCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($taskCommand) { $taskPython = $taskCommand.Source }
    }
}
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Install Python 3.12 and requirements.txt first.' }
$taskOldPythonPath = $env:PYTHONPATH
$taskOldPythonUtf8 = $env:PYTHONUTF8
try {
    $env:PYTHONUTF8 = '1'
    $env:PYTHONPATH = (Join-Path $PSScriptRoot '.deps') + [IO.Path]::PathSeparator + $PSScriptRoot
    & $taskPython @Arguments
    $taskExitCode = $LASTEXITCODE
} finally { $env:PYTHONPATH = $taskOldPythonPath; $env:PYTHONUTF8 = $taskOldPythonUtf8 }
exit $taskExitCode
