[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet('Check', 'Start', 'Status')][string]$Action,
    [string]$SshAlias = 'autodl-monthly',
    [string]$SshConfig = (Join-Path $env:USERPROFILE '.ssh\config'),
    [string]$RemoteProjectRoot = '/root/autodl-tmp/codex-original',
    [string]$PythonPath = '/root/miniconda3/bin/python',
    [string]$ControlRoot = '/root/autodl-tmp/autodl-job-control',
    [string]$TaskName,
    [string]$RunDir,
    [string]$CommandFile,
    [string]$RunId,
    [ValidateRange(1, 200)][int]$TailLines = 20
)

$ErrorActionPreference = 'Stop'

function Assert-Profile {
    if ($SshAlias -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$') { throw 'Invalid SSH alias.' }
    if (-not (Test-Path -LiteralPath $SshConfig -PathType Leaf)) { throw "SSH config not found: $SshConfig" }
    if ($RemoteProjectRoot -notmatch '^/root/autodl-tmp/[A-Za-z0-9._/-]+$' -or $RemoteProjectRoot.Contains('..')) { throw 'RemoteProjectRoot must be below /root/autodl-tmp.' }
    if ($ControlRoot -notmatch '^/root/autodl-tmp/[A-Za-z0-9._/-]+$' -or $ControlRoot.Contains('..')) { throw 'ControlRoot must be below /root/autodl-tmp.' }
    $projectPath = $RemoteProjectRoot.TrimEnd('/')
    $controlPath = $ControlRoot.TrimEnd('/')
    if ($controlPath -eq $projectPath -or $controlPath.StartsWith($projectPath + '/') -or $projectPath.StartsWith($controlPath + '/')) { throw 'ControlRoot and RemoteProjectRoot must be separate directories.' }
    if ($PythonPath -notmatch '^/[A-Za-z0-9._/+:-]+$') { throw 'PythonPath must be an explicit absolute Linux path.' }
}

function Invoke-Ssh {
    param([Parameter(Mandatory = $true)][string]$RemoteCommand)
    $ssh = (Get-Command ssh.exe -ErrorAction Stop).Source
    $output = @(& $ssh -F $SshConfig -o BatchMode=yes -o ConnectTimeout=15 $SshAlias $RemoteCommand 2>&1)
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "SSH exited ${code}: $($output -join [Environment]::NewLine)" }
    return ($output -join [Environment]::NewLine)
}

function To-Base64 {
    param([Parameter(Mandatory = $true)][string]$Text)
    $lf = $Text -replace "`r`n?", "`n"
    [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($lf))
}

function Invoke-RemoteBash {
    param([Parameter(Mandatory = $true)][string]$Text)
    $encoded = To-Base64 $Text
    Invoke-Ssh "printf '%s' '$encoded' | base64 -d | bash"
}

function Assert-RunDir {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ($Path -notmatch '^outputs/[0-9]{8}_[A-Za-z0-9][A-Za-z0-9._-]*/runs/[A-Za-z0-9][A-Za-z0-9._-]*$') { throw 'RunDir must be an explicit versioned outputs/.../runs/... path.' }
    if ($Path -match '(?i)(^|/)(final_pt|_wsl_restore_metadata_v4|\.tmp|legacy)(/|$)') { throw 'RunDir points inside a protected output namespace.' }
}

function Assert-TaskName {
    if ($TaskName -notmatch '^[a-z0-9][a-z0-9_-]{0,39}$') { throw 'TaskName must be 1-40 lowercase letters, digits, underscores, or hyphens.' }
}

function Invoke-Check {
    $settings = @{ project = $RemoteProjectRoot; python = $PythonPath; watcher = "$RemoteProjectRoot/scripts/maintenance/autodl_watch.py" } | ConvertTo-Json -Compress
    $settings64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($settings))
    $probe = @'
import base64,json,pathlib,shutil,subprocess,sys
c=json.loads(base64.b64decode(sys.argv[1]).decode())
p=c['python']; root=pathlib.Path(c['project'])
r={'project':str(root),'python':p,'python_exists':pathlib.Path(p).is_file(),'project_exists':root.is_dir(),'watcher_script_exists':pathlib.Path(c['watcher']).is_file()}
if r['python_exists']:
 q=subprocess.run([p,'-c',"import json,sys,torch,ldm_project; print(json.dumps({'python':sys.executable,'python_version':sys.version.split()[0],'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'project_import':ldm_project.__file__}))"],cwd=str(root),text=True,capture_output=True)
 r['python_exit_code']=q.returncode
 if q.returncode==0:r['runtime']=json.loads(q.stdout.strip().splitlines()[-1])
 else:r['python_error']=q.stderr[-3000:]
for key,path in [('system_disk','/'),('data_disk','/root/autodl-tmp')]:
 try:
  d=shutil.disk_usage(path);r[key]={'total_bytes':d.total,'free_bytes':d.free,'used_bytes':d.used}
 except OSError as e:r[key]={'error':str(e)}
try:r['screen_version']=subprocess.run(['screen','--version'],text=True,capture_output=True,check=True).stdout.strip()
except Exception as e:r['screen_error']=str(e)
try:
 q=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True,check=True);r['nvidia_smi']=q.stdout.strip().splitlines()
except Exception as e:r['nvidia_smi']='unavailable: '+str(e)
print(json.dumps(r,ensure_ascii=False));sys.exit(0 if r['python_exists'] and r['project_exists'] and r['watcher_script_exists'] and r.get('python_exit_code')==0 and 'screen_version' in r else 2)
'@
    $probe64 = To-Base64 $probe
    Invoke-Ssh "printf '%s' '$probe64' | base64 -d | '$PythonPath' - '$settings64'"
}

function Assert-SafeCommandFile {
    if (-not $CommandFile -or -not (Test-Path -LiteralPath $CommandFile -PathType Leaf)) { throw 'CommandFile must name an existing local file.' }
    $item = Get-Item -LiteralPath $CommandFile
    if ($item.Length -gt 1MB) { throw 'CommandFile must be no larger than 1 MiB.' }
    $text = [IO.File]::ReadAllText($item.FullName, [Text.Encoding]::UTF8).TrimStart([char]0xFEFF)
    if ([string]::IsNullOrWhiteSpace($text) -or $text.Contains([char]0)) { throw 'CommandFile must contain non-empty UTF-8 shell commands without NUL bytes.' }
    return ($text -replace "`r`n?", "`n") + "`n"
}

function Invoke-Start {
    Assert-TaskName
    Assert-RunDir $RunDir
    $commandText = Assert-SafeCommandFile
    $RunId = [Guid]::NewGuid().ToString('N')
    $jobDir = "$ControlRoot/$TaskName-$RunId"
    $stageTemplate = "$ControlRoot/.stage-$TaskName.XXXXXX"
    $stageScript = @'
set -eu
mkdir -p '__CONTROL__'
stage=$(mktemp -d '__STAGE__')
printf '%s\n' "$stage"
'@
    $stageScript = $stageScript.Replace('__CONTROL__', $ControlRoot).Replace('__STAGE__', $stageTemplate)
    $stageDir = (Invoke-RemoteBash $stageScript).Trim()
    if ($stageDir -notmatch '^/root/autodl-tmp/[A-Za-z0-9._/-]+$' -or -not $stageDir.StartsWith("$ControlRoot/.stage-$TaskName.")) { throw 'Remote staging directory failed validation.' }

    $tempCommand = [IO.Path]::GetTempFileName()
    $tempRunner = [IO.Path]::GetTempFileName()
    try {
        [IO.File]::WriteAllText($tempCommand, $commandText, [Text.UTF8Encoding]::new($false))
        $runner = @'
#!/usr/bin/env bash
set -u
job_dir='__JOB__'
project_root='__PROJECT__'
task_name='__TASK__'
write_status() {
  local state="$1" code="$2" stamp tmp
  stamp=$(date -Is)
  tmp="$job_dir/status.json.tmp"
  printf '{"task":"%s","run_id":"%s","state":"%s","exit_code":%s,"updated_at":"%s"}\n' "$task_name" '__RUNID__' "$state" "$code" "$stamp" > "$tmp"
  mv "$tmp" "$job_dir/status.json"
}
cd "$project_root" || { write_status START_FAILED 125; exit 125; }
write_status RUNNING null
bash "$job_dir/command.sh" > "$job_dir/run.log" 2>&1 &
command_pid=$!
'__PYTHON__' "$project_root/scripts/maintenance/autodl_watch.py" \
  --run-id '__RUNID__' --control-dir "$job_dir" --run-dir "$project_root/__RUNDIR__" \
  --process-pid "$command_pid" --interval-seconds 60 > "$job_dir/telemetry.log" 2>&1 &
watcher_pid=$!
wait "$command_pid"
code=$?
touch "$job_dir/telemetry.stop"
wait "$watcher_pid" || true
run_output="$project_root/__RUNDIR__"
telemetry_output="$run_output/resource_telemetry.csv"
if [ -f "$job_dir/telemetry.csv" ] && [ ! -e "$telemetry_output" ]; then
  mkdir -p "$run_output"
  cp -- "$job_dir/telemetry.csv" "$telemetry_output"
  printf '{"run_id":"%s","status":"COPIED","path":"resource_telemetry.csv"}\n' '__RUNID__' > "$job_dir/telemetry_copy_status.json"
elif [ -e "$telemetry_output" ]; then
  printf '{"run_id":"%s","status":"DESTINATION_EXISTS","path":"resource_telemetry.csv"}\n' '__RUNID__' > "$job_dir/telemetry_copy_status.json"
else
  printf '{"run_id":"%s","status":"MISSING","path":"resource_telemetry.csv"}\n' '__RUNID__' > "$job_dir/telemetry_copy_status.json"
fi
if [ "$code" -eq 0 ]; then state=COMMAND_EXITED_ZERO; else state=COMMAND_EXITED_NONZERO; fi
write_status "$state" "$code"
exit "$code"
'@
        $runner = $runner.Replace('__JOB__', $jobDir).Replace('__PROJECT__', $RemoteProjectRoot).Replace('__TASK__', $TaskName).Replace('__RUNID__', $RunId).Replace('__PYTHON__', $PythonPath).Replace('__RUNDIR__', $RunDir)
        [IO.File]::WriteAllText($tempRunner, $runner, [Text.UTF8Encoding]::new($false))
        $scp = (Get-Command scp.exe -ErrorAction Stop).Source
        & $scp -F $SshConfig -o BatchMode=yes $tempCommand "${SshAlias}:$stageDir/command.sh" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "SCP command upload failed with exit code $LASTEXITCODE." }
        & $scp -F $SshConfig -o BatchMode=yes $tempRunner "${SshAlias}:$stageDir/runner.sh" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "SCP runner upload failed with exit code $LASTEXITCODE." }
    }
    finally {
        Remove-Item -LiteralPath $tempCommand, $tempRunner -Force -ErrorAction SilentlyContinue
    }

    $launch = @'
set -eu
stage='__STAGE_DIR__'
job='__JOB_DIR__'
if screen -ls 2>/dev/null | grep -Fq '.ad-__TASK__'; then
  rm -f "$stage/command.sh" "$stage/runner.sh"
  rmdir "$stage" 2>/dev/null || true
  printf 'An active session already uses task name __TASK__.\n' >&2
  exit 23
fi
test -d "$stage"
test ! -e "$job"
mv "$stage" "$job"
chmod 700 "$job"
chmod 600 "$job/command.sh"
chmod 700 "$job/runner.sh"
screen -dmS 'ad-__TASK__' bash "$job/runner.sh"
printf '{"state":"DETACHED","task":"__TASK__","run_id":"__RUNID__","control_dir":"%s","run_dir":"__RUNDIR__"}\n' "$job"
'@
    $launch = $launch.Replace('__STAGE_DIR__', $stageDir).Replace('__JOB_DIR__', $jobDir).Replace('__TASK__', $TaskName).Replace('__RUNID__', $RunId).Replace('__RUNDIR__', $RunDir)
    $result = Invoke-RemoteBash $launch
    [pscustomobject]@{ TaskName = $TaskName; RunId = $RunId; RemoteControlDir = $jobDir; RunDir = $RunDir; Launch = $result }
}

function Invoke-Status {
    Assert-TaskName
    if ($RunId -notmatch '^[a-f0-9]{32}$') { throw 'RunId must be the 32-character identifier returned by Start.' }
    $jobDir = "$ControlRoot/$TaskName-$RunId"
    $statusScript = @'
set -u
job='__JOB__'
if [ ! -d "$job" ]; then printf '{"state":"UNKNOWN","reason":"control directory missing"}\n'; exit 0; fi
has_screen=0
if screen -ls 2>/dev/null | grep -Fq '.ad-__TASK__'; then has_screen=1; fi
if [ -f "$job/status.json" ]; then
  if [ "$has_screen" -eq 0 ] && grep -q '"state":"RUNNING"' "$job/status.json"; then
    printf '{"state":"UNKNOWN","reason":"screen missing without an exit record"}\n'
  else
    cat "$job/status.json"
  fi
else
  printf '{"state":"UNKNOWN","reason":"no exit record"}\n'
fi
if [ -f "$job/telemetry_status.json" ]; then
  printf '\nTELEMETRY\n'
  cat "$job/telemetry_status.json"
  if [ -f "$job/telemetry.csv" ]; then tail -n 4 "$job/telemetry.csv"; fi
fi
printf '\nSCREEN\n'
screen -ls 2>&1 | grep -F '.ad-__TASK__' || true
printf 'LOG\n'
if [ -f "$job/run.log" ]; then tail -n __TAIL__ "$job/run.log"; else printf '(log not created)\n'; fi
'@
    $statusScript = $statusScript.Replace('__JOB__', $jobDir).Replace('__TASK__', $TaskName).Replace('__TAIL__', [string]$TailLines)
    Invoke-RemoteBash $statusScript
}

Assert-Profile
switch ($Action) {
    'Check' { Invoke-Check }
    'Start' { Invoke-Start }
    'Status' { Invoke-Status }
}
