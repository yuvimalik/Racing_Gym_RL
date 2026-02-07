# Simple PowerShell script to fix racecar_gym dataclass bug
# Direct file patching approach

Write-Host "Fixing racecar_gym dataclass bug..." -ForegroundColor Cyan

# Find the specs.py file
$specsPaths = @(
    "$env:LOCALAPPDATA\Packages\PythonSoftwareFoundation.Python.3.11_*\LocalCache\local-packages\Python311\site-packages\racecar_gym\core\specs.py",
    "$env:APPDATA\Python\Python311\site-packages\racecar_gym\core\specs.py"
)

$specsFile = $null
foreach ($path in $specsPaths) {
    $found = Get-ChildItem -Path ($path -replace '\*', '*') -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) {
        $specsFile = $found.FullName
        break
    }
}

# Try pip show method
if (-not $specsFile) {
    $pipOutput = pip show racecar-gym 2>$null
    if ($pipOutput) {
        $locationLine = $pipOutput | Select-String "Location:"
        if ($locationLine) {
            $location = ($locationLine -split "Location:")[1].Trim()
            $specsFile = Join-Path $location "racecar_gym\core\specs.py"
        }
    }
}

if (-not $specsFile -or -not (Test-Path $specsFile)) {
    Write-Host "ERROR: Could not find racecar_gym/core/specs.py" -ForegroundColor Red
    Write-Host "Trying alternative method..." -ForegroundColor Yellow
    
    # Use Python to find it
    $pythonFind = python -c "import site; import os; sp = site.getsitepackages()[0]; print(os.path.join(sp, 'racecar_gym', 'core', 'specs.py'))" 2>$null
    if ($pythonFind -and (Test-Path $pythonFind)) {
        $specsFile = $pythonFind
    }
}

if (-not $specsFile -or -not (Test-Path $specsFile)) {
    Write-Host "ERROR: Could not locate specs.py file" -ForegroundColor Red
    Write-Host "Please ensure racecar_gym is installed" -ForegroundColor Yellow
    exit 1
}

Write-Host "Found: $specsFile" -ForegroundColor Green

# Create backup
$backupFile = "$specsFile.backup"
Copy-Item $specsFile $backupFile -Force
Write-Host "Backup created: $backupFile" -ForegroundColor Green

# Read and fix the file
$content = Get-Content $specsFile -Raw

# Check if already fixed
if ($content -match "field\(default_factory=VehicleSpec\)") {
    Write-Host "File appears to already be fixed" -ForegroundColor Yellow
    $test = python -c "import racecar_gym.core.specs" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: Already working!" -ForegroundColor Green
        exit 0
    }
}

# Apply fixes
Write-Host "Applying fixes..." -ForegroundColor Yellow

# Fix 1: Ensure field is imported
if ($content -notmatch "from dataclasses import.*field") {
    $content = $content -replace "(from dataclasses import dataclass)", "from dataclasses import dataclass, field"
}

# Fix 2: Replace mutable defaults
$content = $content -replace "(\s+)(\w+):\s*VehicleSpec\s*=\s*VehicleSpec\(\)", '$1$2: VehicleSpec = field(default_factory=VehicleSpec)'

# Write back
Set-Content -Path $specsFile -Value $content -NoNewline

Write-Host "Fix applied. Testing..." -ForegroundColor Yellow

# Test
$test = python -c "import racecar_gym.core.specs; print('SUCCESS')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: racecar_gym now works!" -ForegroundColor Green
    Write-Host "You can now run: python train.py" -ForegroundColor Cyan
} else {
    Write-Host "ERROR: Fix failed. Restoring backup..." -ForegroundColor Red
    Copy-Item $backupFile $specsFile -Force
    Write-Host $test
    exit 1
}
