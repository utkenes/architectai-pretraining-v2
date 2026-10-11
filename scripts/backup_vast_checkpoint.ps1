[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$VastHost,
    [Parameter(Mandatory)]
    [int]$Port,
    [Parameter(Mandatory)]
    [string]$Checkpoint,
    [string]$RemoteRepo = '/workspace/architectai-pretraining-v2',
    [string]$RemoteBackupDir = '/workspace/architectai-checkpoint-backups',
    [string]$LocalBackupRoot = (Join-Path $PSScriptRoot '..\backups\vast')
)

$ErrorActionPreference = 'Stop'
$checkpointName = Split-Path -Leaf $Checkpoint
if ($checkpointName -notmatch '^checkpoint-(\d+)$') {
    throw "Checkpoint must end in checkpoint-N: $Checkpoint"
}
$expectedStep = [int]$Matches[1]
$target = "root@$VastHost"
$remoteScript = "$RemoteRepo/scripts/prepare_checkpoint_backup.sh"

& ssh -p $Port $target "bash '$remoteScript' '$Checkpoint' '$RemoteBackupDir'"
if ($LASTEXITCODE -ne 0) { throw 'Remote checkpoint archive preparation failed.' }

$destination = Join-Path $LocalBackupRoot $checkpointName
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$archive = Join-Path $destination "$checkpointName.tar.gz"
$archiveManifest = Join-Path $destination "$checkpointName.archive.sha256"
$fileManifest = Join-Path $destination "$checkpointName.files.sha256"

foreach ($name in @("$checkpointName.tar.gz", "$checkpointName.archive.sha256", "$checkpointName.files.sha256")) {
    & scp -P $Port "${target}:$RemoteBackupDir/$name" $destination
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $name" }
}

$expectedArchiveHash = ((Get-Content -Raw -LiteralPath $archiveManifest).Trim() -split '\s+')[0].ToLowerInvariant()
$actualArchiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
if ($actualArchiveHash -ne $expectedArchiveHash) {
    throw "Archive SHA-256 mismatch: expected $expectedArchiveHash, got $actualArchiveHash"
}

tar -xzf $archive -C $destination
if ($LASTEXITCODE -ne 0) { throw 'Could not extract the verified checkpoint archive.' }
$extracted = Join-Path $destination $checkpointName

foreach ($line in Get-Content -LiteralPath $fileManifest) {
    if ($line -notmatch '^(?<hash>[a-fA-F0-9]{64})  \./(?<path>.+)$') {
        throw "Invalid file-manifest entry: $line"
    }
    $path = Join-Path $extracted $Matches.path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing extracted file: $($Matches.path)" }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
    if ($actual -ne $Matches.hash.ToLowerInvariant()) { throw "File SHA-256 mismatch: $($Matches.path)" }
}

$required = @('checkpoint_metadata.json', 'trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'model/adapter_config.json')
foreach ($relative in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $extracted $relative) -PathType Leaf)) { throw "Required checkpoint file missing: $relative" }
}
if (-not ((Test-Path -LiteralPath (Join-Path $extracted 'model/adapter_model.safetensors')) -or (Test-Path -LiteralPath (Join-Path $extracted 'model/adapter_model.bin')))) {
    throw 'Required adapter weights are missing.'
}

$metadata = Get-Content -Raw -LiteralPath (Join-Path $extracted 'checkpoint_metadata.json') | ConvertFrom-Json
$trainerState = Get-Content -Raw -LiteralPath (Join-Path $extracted 'trainer_state.json') | ConvertFrom-Json
if ([int]$metadata.global_step -ne $expectedStep -or [int]$trainerState.global_step -ne $expectedStep) {
    throw "Checkpoint step mismatch after backup: expected $expectedStep, metadata=$($metadata.global_step), trainer=$($trainerState.global_step)"
}

[pscustomobject]@{
    checkpoint = $checkpointName
    global_step = $expectedStep
    archive_sha256 = $actualArchiveHash
    verified_file_count = (Get-Content -LiteralPath $fileManifest).Count
    local_path = $extracted
} | ConvertTo-Json
