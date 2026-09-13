<#
.SYNOPSIS
  Copy Bracket Gambit to the robot over SSH and (optionally) install Stockfish + espeak-ng there.

.EXAMPLE
  .\deploy_to_robot.ps1 -RobotHost bracketbot-092.local
  .\deploy_to_robot.ps1 -RobotHost 10.88.111.42 -InstallDeps
  .\deploy_to_robot.ps1 -RobotHost bracketbot-092.local -DryRun      # just print what would run

  The robot's login is user 'bracketbot' (the bbos checkout lives in /home/bracketbot/bbos).
  Its hostname is printed on the robot / by the view_*.py demos as http://<hostname>.local:...
#>
param(
    [Parameter(Mandatory = $true)] [string] $RobotHost,
    [string] $User = "bracketbot",
    [string] $Dest = "~/bbapps/gambit",
    [switch] $InstallDeps,
    [switch] $DryRun
)
$ErrorActionPreference = "Stop"
$target = "$User@$RobotHost"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Run($cmd) {
    Write-Host ">> $cmd" -ForegroundColor Cyan
    if (-not $DryRun) { Invoke-Expression $cmd; if ($LASTEXITCODE -ne 0) { throw "command failed: $cmd" } }
}

# 1. is the robot there?
Write-Host "1. checking SSH to $target ..." -ForegroundColor Green
Run "ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new $target 'echo connected to `$(hostname) as `$(whoami); uname -m; test -d /home/bracketbot/bbos && echo bbos: ok || echo bbos: MISSING; command -v uv || echo uv: MISSING'"

# 2. copy the code (entry script + package; tests/sim are not needed on the robot)
Write-Host "2. copying gambit.py + bracket_gambit/ to $target`:$Dest ..." -ForegroundColor Green
Run "ssh $target 'mkdir -p $Dest'"
Run "scp -r `"$here\gambit.py`" `"$here\bracket_gambit`" `"$target`:$Dest/`""
# a Linux Stockfish binary dropped in tools/stockfish/ is found automatically on the robot
$linux_sf = Get-ChildItem "$here\tools\stockfish" -Filter "stockfish-linux*" -ErrorAction SilentlyContinue
if ($linux_sf) {
    Run "ssh $target 'mkdir -p $Dest/tools/stockfish'"
    foreach ($f in $linux_sf) { Run "scp `"$($f.FullName)`" `"$target`:$Dest/tools/stockfish/`""; Run "ssh $target 'chmod +x $Dest/tools/stockfish/$($f.Name)'" }
}
Run "ssh $target 'rm -rf $Dest/bracket_gambit/__pycache__'"

# 3. optional: Stockfish + espeak-ng
if ($InstallDeps) {
    Write-Host "3. installing stockfish + espeak-ng on the robot (needs sudo) ..." -ForegroundColor Green
    Run "ssh -t $target 'sudo apt-get update && sudo apt-get install -y stockfish espeak-ng && command -v stockfish && stockfish --help | head -1'"
} else {
    Write-Host "3. (skipped dependency install; add -InstallDeps, or install stockfish/espeak-ng yourself)" -ForegroundColor Yellow
}

# 4. resolve dependencies once, so the first real run is not slowed by uv
Write-Host "4. resolving python deps on the robot (uv, first run only) ..." -ForegroundColor Green
Run "ssh $target 'cd $Dest && uv run gambit.py --help > /dev/null && echo deps: ok'"

Write-Host @"

Done. Next, on the robot (ssh $target):
    cd $Dest
    uv run gambit.py check --robot                      # dry run: IK reaches every square + tray?
    uv run gambit.py calibrate --robot --execute        # empty-board reference, teach a1/h1/a8, gripper
    uv run gambit.py play --robot                       # dry run: prints waypoints, nothing moves
    uv run gambit.py play --robot --execute --moves 6   # the demo (web UI: http://$RobotHost`:8010/)
Emergency stop while playing: 'e' + Enter in that terminal, the red web button, or Ctrl-C.
"@ -ForegroundColor Green
