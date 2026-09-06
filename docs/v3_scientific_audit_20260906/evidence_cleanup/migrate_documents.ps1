[CmdletBinding()]
param(
    [string]$PlanPath = (Join-Path $PSScriptRoot '../DOCUMENT_CLEANUP_PLAN.json'),
    [ValidateSet('historical_docs', 'component_reference', 'reuse_immutable', 'all')]
    [string]$Batch = 'historical_docs',
    [string[]]$SourcePath = @(),
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'
$planFile = (Resolve-Path -LiteralPath $PlanPath).Path
$plan = Get-Content -LiteralPath $planFile -Raw | ConvertFrom-Json
$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..')).TrimEnd('\', '/')
if (-not $workspaceRoot.Equals([System.IO.Path]::GetFullPath($plan.workspace).TrimEnd('\', '/'), [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The plan belongs to a different worktree.'
}
$docsRoot = Join-Path $workspaceRoot 'docs'
$auditRoot = Join-Path $docsRoot 'v3_scientific_audit_20260906'
$immutableRoot = Join-Path $workspaceRoot 'archives/successful_contributions_20260828/c3fd_v2_5'

function Test-Within([string]$Candidate, [string]$Boundary) {
    $prefix = [System.IO.Path]::GetFullPath($Boundary).TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    return [System.IO.Path]::GetFullPath($Candidate).StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
}

function Assert-NoReparseParent([string]$Candidate) {
    $cursorPath = [System.IO.Path]::GetFullPath($Candidate)
    while (Test-Within $cursorPath $workspaceRoot) {
        if (Test-Path -LiteralPath $cursorPath) {
            $item = Get-Item -LiteralPath $cursorPath -Force
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Reparse point is outside the approved ordinary-file migration: $cursorPath"
            }
        }
        $cursorPath = [System.IO.Path]::GetDirectoryName($cursorPath)
    }
}

function Get-Sha([string]$FilePath) {
    return (Get-FileHash -LiteralPath $FilePath -Algorithm SHA256).Hash.ToLowerInvariant()
}

$eligible = @($plan.entries | Where-Object { $_.action -in @('archive', 'move_reference', 'reuse_immutable_archive') })
if ($SourcePath.Count -gt 0) {
    $wanted = @($SourcePath | ForEach-Object { $_.Replace('\', '/') })
    $selected = @($eligible | Where-Object { $_.source_path -in $wanted })
    $missing = @($wanted | Where-Object { $_ -notin @($selected.source_path) })
    if ($missing.Count -gt 0) { throw "Source is missing or protected in the plan: $($missing -join ', ')" }
} else {
    $selected = @($eligible | Where-Object { $Batch -eq 'all' -or $_.batch -eq $Batch })
}

$prepared = @()
foreach ($entry in $selected) {
    $sourceFull = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot $entry.source_path))
    $destinationFull = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot $entry.destination))
    if (-not (Test-Within $sourceFull $docsRoot) -or (Test-Within $sourceFull $auditRoot) -or
        $entry.source_path -in @($plan.protected_exact_paths)) { throw "Protected or outside source: $sourceFull" }
    if ($entry.action -eq 'reuse_immutable_archive') {
        if (-not (Test-Within $destinationFull $immutableRoot)) { throw "Archive reuse destination outside approved immutable archive: $destinationFull" }
    } elseif (-not ((Test-Within $destinationFull (Join-Path $auditRoot 'historical/docs')) -or
                     (Test-Within $destinationFull (Join-Path $auditRoot 'reference/component_specs')))) {
        throw "Destination outside approved documentation archive/reference roots: $destinationFull"
    }
    Assert-NoReparseParent $sourceFull
    Assert-NoReparseParent $destinationFull
    if (-not (Test-Path -LiteralPath $sourceFull -PathType Leaf)) { throw "Missing source file: $sourceFull" }
    if ((Get-Sha $sourceFull) -ne $entry.sha256) { throw "Source changed since plan; rebuild and coordinate again: $sourceFull" }
    if ($entry.action -eq 'reuse_immutable_archive') {
        if (-not (Test-Path -LiteralPath $destinationFull -PathType Leaf) -or (Get-Sha $destinationFull) -ne $entry.sha256) {
            throw "Immutable archive is missing or no longer byte-identical: $destinationFull"
        }
    } elseif (Test-Path -LiteralPath $destinationFull) {
        throw "Destination already exists; nothing will be overwritten: $destinationFull"
    }
    $prepared += [pscustomobject]@{ Source = $sourceFull; Destination = $destinationFull; Action = $entry.action; Sha256 = $entry.sha256 }
}

if (-not $Apply) {
    [pscustomobject]@{
        Status = 'read_only_preflight_passed'; SelectedFiles = $prepared.Count; Batch = $Batch
        Actions = @($prepared | Group-Object Action | ForEach-Object { @{ Action = $_.Name; Files = $_.Count } })
        Message = 'No files moved, removed, or rewritten. Apply only after the parent coordinates these sources and reference owners.'
    } | ConvertTo-Json -Depth 5
    return
}

$journalPath = Join-Path $PSScriptRoot ('MIGRATION_LOG_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '_' + [guid]::NewGuid().ToString('N') + '.jsonl')
foreach ($entry in $prepared) {
    # Revalidate immediately before each mutation, after the all-file preflight.
    if ((Get-Sha $entry.Source) -ne $entry.Sha256) { throw "Source changed during batch: $($entry.Source)" }
    Assert-NoReparseParent $entry.Source
    Assert-NoReparseParent $entry.Destination
    $event = @{ utc = [DateTime]::UtcNow.ToString('o'); source = $entry.Source; destination = $entry.Destination; action = $entry.Action; original_sha256 = $entry.Sha256 }
    try {
        if ($entry.Action -eq 'reuse_immutable_archive') {
            if ((Get-Sha $entry.Destination) -ne $entry.Sha256) { throw 'Archive changed during batch.' }
            Remove-Item -LiteralPath $entry.Source
        } else {
            if (Test-Path -LiteralPath $entry.Destination) { throw 'Destination appeared during batch.' }
            [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($entry.Destination)) | Out-Null
            Move-Item -LiteralPath $entry.Source -Destination $entry.Destination
            if ((Get-Sha $entry.Destination) -ne $entry.Sha256) { throw 'Destination hash differs after move.' }
        }
        $event.status = 'completed'
        $event.destination_sha256 = Get-Sha $entry.Destination
        $event | ConvertTo-Json -Compress | Add-Content -LiteralPath $journalPath -Encoding utf8
    } catch {
        $event.status = 'error'
        $event.error = $_.Exception.Message
        $event | ConvertTo-Json -Compress | Add-Content -LiteralPath $journalPath -Encoding utf8
        throw
    }
}
[pscustomobject]@{ Status = 'completed'; Files = $prepared.Count; Journal = $journalPath; ReferenceEdits = 'Not performed by this tool; apply only the separately coordinated link_updates.' } | ConvertTo-Json
