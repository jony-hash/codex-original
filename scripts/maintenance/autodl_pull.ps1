[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RemoteRunDir,
    [Parameter(Mandatory = $true, ParameterSetName = 'ExplicitFiles')][string[]]$Files,
    [Parameter(Mandatory = $true, ParameterSetName = 'Manifest')][string]$ManifestFile,
    [Parameter(Mandatory = $true)][string]$Destination,
    [string]$SshAlias = 'autodl-monthly',
    [string]$SshConfig = (Join-Path $env:USERPROFILE '.ssh\config'),
    [string]$RemoteProjectRoot = '/root/autodl-tmp/codex-original',
    [string]$PythonPath = '/root/miniconda3/bin/python'
)

$ErrorActionPreference = 'Stop'

function Assert-Profile {
    if ($SshAlias -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$') { throw 'Invalid SSH alias.' }
    if (-not (Test-Path -LiteralPath $SshConfig -PathType Leaf)) { throw "SSH config not found: $SshConfig" }
    if ($RemoteProjectRoot -notmatch '^/root/autodl-tmp/[A-Za-z0-9._/-]+$' -or $RemoteProjectRoot.Contains('..')) { throw 'RemoteProjectRoot must be below /root/autodl-tmp.' }
    if ($PythonPath -notmatch '^/[A-Za-z0-9._/+:-]+$') { throw 'PythonPath must be an explicit absolute Linux path.' }
}

function Assert-RemoteRunDir {
    if ($RemoteRunDir -notmatch '^outputs/[0-9]{8}_[A-Za-z0-9][A-Za-z0-9._-]*/runs/[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*$') { throw 'RemoteRunDir must be an explicit versioned outputs/.../runs/... path.' }
    if ($RemoteRunDir -match '(?i)(^|/)(final_pt|_wsl_restore_metadata_v4|\.tmp|legacy)(/|$)') { throw 'RemoteRunDir points inside a protected output namespace.' }
}

function Assert-FileList {
    param([Parameter(Mandatory = $true)][string[]]$Paths)
    if (-not $Paths -or $Paths.Count -eq 0) { throw 'The file manifest must list at least one exact relative file path.' }
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($file in $Paths) {
        if ($file -notmatch '^[A-Za-z0-9][A-Za-z0-9._/-]*$' -or $file.Contains('\') -or $file.Contains('*') -or $file.Contains('?')) { throw "Invalid relative file path: $file" }
        $segments = $file.Split('/')
        if ($segments | Where-Object { $_ -in @('', '.', '..') }) { throw "Path traversal is not allowed: $file" }
        if (-not $seen.Add($file)) { throw "Duplicate file path: $file" }
    }
}

function Resolve-BackupPaths {
    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path.TrimEnd('\')
    $destinationPath = [IO.Path]::GetFullPath($Destination)
    $repoPrefix = $repoRoot + '\'
    if (-not $destinationPath.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Destination must be inside this repository.' }
    $relative = $destinationPath.Substring($repoPrefix.Length).Replace('\', '/')
    if ($relative -notmatch '^outputs/[0-9]{8}_[A-Za-z0-9][A-Za-z0-9._-]*/runs/autodl_backup_[A-Za-z0-9][A-Za-z0-9._-]*$') { throw 'Destination must be a new outputs/<version>/runs/autodl_backup_<name> directory.' }
    if ($relative -match '(?i)(^|/)(final_pt|_wsl_restore_metadata_v4|\.tmp|legacy)(/|$)') { throw 'Destination points inside a protected output namespace.' }
    if (-not (Test-Path -LiteralPath $repoRoot -PathType Container)) { throw 'Repository root could not be resolved.' }
    $outputsRoot = Join-Path $repoRoot 'outputs'
    if (-not (Test-Path -LiteralPath $outputsRoot -PathType Container)) { throw 'Repository outputs directory is missing.' }
    $resolvedOutputs = (Resolve-Path -LiteralPath $outputsRoot).Path.TrimEnd('\')
    if (-not $resolvedOutputs.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Repository outputs directory resolves outside the repository.' }
    $segments = $relative.Split('/')
    $versionPath = Join-Path $resolvedOutputs $segments[1]
    if (-not (Test-Path -LiteralPath $versionPath)) { New-Item -ItemType Directory -Path $versionPath | Out-Null }
    if (-not (Test-Path -LiteralPath $versionPath -PathType Container)) { throw 'Version output path is not a directory.' }
    $resolvedVersion = (Resolve-Path -LiteralPath $versionPath).Path.TrimEnd('\')
    if (-not $resolvedVersion.StartsWith($resolvedOutputs + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Version output path resolves outside the outputs directory.' }
    $runsPath = Join-Path $resolvedVersion 'runs'
    if (-not (Test-Path -LiteralPath $runsPath)) { New-Item -ItemType Directory -Path $runsPath | Out-Null }
    if (-not (Test-Path -LiteralPath $runsPath -PathType Container)) { throw 'Backup parent path is not a directory.' }
    $parent = (Resolve-Path -LiteralPath $runsPath).Path.TrimEnd('\')
    if (-not $parent.StartsWith($resolvedVersion + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Backup parent resolves outside its version output directory.' }
    $destinationPath = Join-Path $parent $segments[3]
    if (Test-Path -LiteralPath $destinationPath) { throw "Destination already exists; refusing to overwrite: $destinationPath" }
    return @{ RepoRoot = $repoRoot; Destination = $destinationPath; Parent = $parent }
}

function Invoke-Ssh {
    param([Parameter(Mandatory = $true)][string]$RemoteCommand)
    $ssh = (Get-Command ssh.exe -ErrorAction Stop).Source
    $output = @(& $ssh -F $SshConfig -o BatchMode=yes -o ConnectTimeout=15 $SshAlias $RemoteCommand 2>&1)
    $code = $LASTEXITCODE
    if ($code -ne 0) { throw "SSH exited ${code}: $($output -join [Environment]::NewLine)" }
    return ($output -join [Environment]::NewLine)
}

function Invoke-SshWithUtf8Input {
    param([Parameter(Mandatory = $true)][string]$RemoteCommand, [Parameter(Mandatory = $true)][string]$InputText)
    $ssh = (Get-Command ssh.exe -ErrorAction Stop).Source
    $process = [Diagnostics.Process]::new()
    $process.StartInfo.FileName = $ssh
    $arguments = @('-F', $SshConfig, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', $SshAlias, $RemoteCommand)
    $process.StartInfo.Arguments = ($arguments | ForEach-Object {
        $value = [string]$_
        '"' + ($value -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
    }) -join ' '
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.CreateNoWindow = $true
    $process.StartInfo.RedirectStandardInput = $true
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.RedirectStandardError = $true
    try { $process.StartInfo.StandardInputEncoding = [Text.UTF8Encoding]::new($false) } catch { }
    $process.Start() | Out-Null
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $inputError = $null
    try { $process.StandardInput.Write($InputText) } catch { $inputError = $_.Exception.Message }
    finally { $process.StandardInput.Close() }
    $process.WaitForExit()
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    if ($process.ExitCode -ne 0) { throw "SSH exited $($process.ExitCode): $($stdout + $stderr) $inputError" }
    if ($inputError) { throw "SSH did not receive the complete UTF-8 manifest: $inputError" }
    return $stdout.TrimEnd([char[]]@([char]13, [char]10))
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

Assert-Profile
Assert-RemoteRunDir
$requiredLocalDependencies = @()
if ($PSCmdlet.ParameterSetName -eq 'Manifest') {
    if (-not (Test-Path -LiteralPath $ManifestFile -PathType Leaf)) { throw "ManifestFile not found: $ManifestFile" }
    $savedManifest = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $ManifestFile).Path, [Text.Encoding]::UTF8) | ConvertFrom-Json
    if ($savedManifest.schema -ne 'autodl_review_manifest_v1' -or $savedManifest.files -isnot [System.Array]) { throw 'ManifestFile must use autodl_review_manifest_v1 with an exact files array.' }
    $Files = @($savedManifest.files | ForEach-Object { [string]$_ })
    $requiredLocalDependencies = @($savedManifest.required_local_dependencies | ForEach-Object { [string]$_ })
}
Assert-FileList -Paths $Files
if ($requiredLocalDependencies.Count -gt 0) { Assert-FileList -Paths $requiredLocalDependencies }
$repoRootForDependencies = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
foreach ($dependency in $requiredLocalDependencies) {
    if (-not $dependency.StartsWith('outputs/', [StringComparison]::Ordinal) -or -not (Test-Path -LiteralPath (Join-Path $repoRootForDependencies $dependency) -PathType Leaf)) {
        throw "Required local start dependency is missing: $dependency"
    }
}
$paths = Resolve-BackupPaths
$stage = Join-Path $paths.Parent ('.autodl-pull-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $paths.Parent | Out-Null
New-Item -ItemType Directory -Path $stage -ErrorAction Stop | Out-Null

$manifestInput = @{ root = $RemoteProjectRoot; run = $RemoteRunDir; files = @($Files) } | ConvertTo-Json -Compress -Depth 5
$manifestScript = @'
import json,pathlib,sys
c=json.load(sys.stdin)
root=pathlib.Path(c['root']).resolve(strict=True)
run=(root/c['run']).resolve(strict=False)
if not run.is_relative_to(root):
 print(json.dumps({'complete':False,'error':'run path resolves outside project'})); sys.exit(0)
files=[]; missing=[]
for rel in c['files']:
 p=(run/rel).resolve(strict=False)
 if not p.is_relative_to(run) or not p.is_file(): missing.append(rel)
 else: files.append({'path':rel,'size':p.stat().st_size})
print(json.dumps({'complete':not missing,'missing':missing,'files':files},ensure_ascii=False))
'@
$manifestScript64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($manifestScript))
$remoteProbe = "'$PythonPath' -c 'import base64,sys; exec(base64.b64decode(`"$manifestScript64`"))'"
$manifestProbe = $null
try {
    $manifestProbe = Invoke-SshWithUtf8Input -RemoteCommand $remoteProbe -InputText $manifestInput
    $manifest = $manifestProbe | ConvertFrom-Json
}
catch {
    $receipt = [pscustomobject]@{
        status = 'PREFLIGHT_FAILED'
        remote_alias = $SshAlias
        remote_run_dir = "$RemoteProjectRoot/$RemoteRunDir"
        destination = $paths.Destination
        created_at = (Get-Date).ToString('o')
        error = $_.Exception.Message
        files = @()
    }
    $receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $stage 'transfer_receipt.json') -Encoding utf8
    throw "Remote preflight failed; no backup was completed. Receipt: $stage\transfer_receipt.json"
}
if (-not $manifest.complete) {
    $missingText = @($manifest.missing) -join ', '
    if ($manifest.error) { $missingText = $manifest.error }
    $receipt = [pscustomobject]@{
        status = 'INCOMPLETE'
        remote_alias = $SshAlias
        remote_run_dir = "$RemoteProjectRoot/$RemoteRunDir"
        destination = $paths.Destination
        created_at = (Get-Date).ToString('o')
        missing = @($manifest.missing)
        error = $manifest.error
        files = @($manifest.files)
    }
    $receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $stage 'transfer_receipt.json') -Encoding utf8
    throw "Remote preflight incomplete; no backup was completed. Missing or unsafe entries: $missingText. Receipt: $stage\transfer_receipt.json"
}

$scp = (Get-Command scp.exe -ErrorAction Stop).Source
$transfer = @()
$failed = $false
foreach ($entry in $manifest.files) {
    $relative = [string]$entry.path
    $localFile = Join-Path $stage ($relative.Replace('/', '\'))
    $localParent = Split-Path -Parent $localFile
    New-Item -ItemType Directory -Force -Path $localParent | Out-Null
    $remoteFile = "${SshAlias}:$RemoteProjectRoot/$RemoteRunDir/$relative"
    $output = @(& $scp -F $SshConfig -o BatchMode=yes $remoteFile $localFile 2>&1)
    $code = $LASTEXITCODE
    $exists = Test-Path -LiteralPath $localFile -PathType Leaf
    $localSize = if ($exists) { (Get-Item -LiteralPath $localFile).Length } else { $null }
    $sizeMatches = $exists -and ($localSize -eq [long]$entry.size)
    $ok = ($code -eq 0) -and $sizeMatches
    if (-not $ok) { $failed = $true }
    $transfer += [pscustomobject]@{
        path = $relative
        scp_exit_code = $code
        remote_size_bytes = [long]$entry.size
        local_size_bytes = $localSize
        status = if ($ok) { 'COPIED' } else { 'FAILED_OR_SIZE_MISMATCH' }
        detail = ($output -join [Environment]::NewLine)
    }
}

$receipt = [pscustomobject]@{
    status = if ($failed) { 'PARTIAL' } else { 'COMPLETE' }
    remote_alias = $SshAlias
    remote_run_dir = "$RemoteProjectRoot/$RemoteRunDir"
    destination = $paths.Destination
    created_at = (Get-Date).ToString('o')
    files = $transfer
}
$receiptPath = Join-Path $stage 'transfer_receipt.json'
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $receiptPath -Encoding utf8
if ($failed) {
    throw "Transfer incomplete; remote originals retained. Review partial staging and receipt: $stage"
}
[IO.Directory]::Move($stage, $paths.Destination)
$receipt.destination = $paths.Destination
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $paths.Destination 'transfer_receipt.json') -Encoding utf8
$receipt
