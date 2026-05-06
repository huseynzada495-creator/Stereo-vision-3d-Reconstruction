param(
    [string]$RunName = ""
)

$timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"

if ($RunName -eq "") {
    $runFolder = "runs\$timestamp"
} else {
    $runFolder = "runs\${timestamp}_$RunName"
}

$subfolders = @(
    "$runFolder\mono_calibration",
    "$runFolder\stereo_calibration",
    "$runFolder\rectification",
    "$runFolder\disparity",
    "$runFolder\depth",
    "$runFolder\pointcloud",
    "$runFolder\mesh",
    "$runFolder\logs"
)

foreach ($folder in $subfolders) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
}

New-Item -ItemType File -Force -Path "$runFolder\summary.json" | Out-Null
New-Item -ItemType File -Force -Path "$runFolder\config_snapshot.yaml" | Out-Null

Write-Host "Created run folder: $runFolder"