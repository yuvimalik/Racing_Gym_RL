# PowerShell script to fix racecar_gym dataclass bug
# This fixes the "mutable default" error in Python 3.11+

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fixing racecar_gym dataclass bug" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Find racecar_gym installation path
Write-Host "Finding racecar_gym installation..." -ForegroundColor Yellow

$pythonCmd = "python"
$findPathCmd = "import racecar_gym; import os; print(os.path.dirname(racecar_gym.__file__))"

try {
    # Try to get the path without importing (to avoid the error)
    $racecarPath = python -c "import sys; import site; paths = site.getsitepackages(); print([p for p in paths if 'racecar' in p.lower() or 'site-packages' in p][0] if any('racecar' in p.lower() or 'site-packages' in p for p in paths) else paths[0])" 2>$null
    
    if (-not $racecarPath) {
        # Alternative: search common locations
        $possiblePaths = @(
            "$env:LOCALAPPDATA\Packages\PythonSoftwareFoundation.Python.3.11_*\LocalCache\local-packages\Python311\site-packages\racecar_gym",
            "$env:APPDATA\Python\Python311\site-packages\racecar_gym",
            "$env:USERPROFILE\.local\lib\python3.11\site-packages\racecar_gym"
        )
        
        foreach ($path in $possiblePaths) {
            $expanded = $path -replace '\*', '*'
            $found = Get-ChildItem -Path $expanded -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($found) {
                $racecarPath = $found.FullName
                break
            }
        }
    }
    
    # If still not found, try pip show
    if (-not $racecarPath) {
        $pipShow = pip show racecar-gym 2>$null | Select-String "Location:"
        if ($pipShow) {
            $location = ($pipShow -split "Location:")[1].Trim()
            $racecarPath = Join-Path $location "racecar_gym"
        }
    }
    
    # Final fallback: use site-packages
    if (-not $racecarPath -or -not (Test-Path $racecarPath)) {
        $sitePackages = python -c "import site; print(site.getsitepackages()[0])" 2>$null
        if ($sitePackages) {
            $racecarPath = Join-Path $sitePackages "racecar_gym"
        }
    }
    
    Write-Host "Racecar_gym path: $racecarPath" -ForegroundColor Green
    
    if (-not (Test-Path $racecarPath)) {
        Write-Host "ERROR: Could not find racecar_gym installation" -ForegroundColor Red
        Write-Host "Please install racecar_gym first: pip install git+https://github.com/axelbr/racecar_gym.git --no-deps" -ForegroundColor Yellow
        exit 1
    }
    
    $specsFile = Join-Path $racecarPath "core\specs.py"
    
    if (-not (Test-Path $specsFile)) {
        Write-Host "ERROR: Could not find specs.py at $specsFile" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Found specs.py: $specsFile" -ForegroundColor Green
    Write-Host ""
    Write-Host "Reading file..." -ForegroundColor Yellow
    
    # Read the file
    $content = Get-Content $specsFile -Raw
    
    # Check if already fixed
    if ($content -match "default_factory") {
        Write-Host "File appears to already be fixed (contains 'default_factory')" -ForegroundColor Green
        Write-Host "Checking if fix is correct..." -ForegroundColor Yellow
        
        # Try to import to verify
        $testImport = python -c "import racecar_gym.core.specs" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "SUCCESS: racecar_gym imports correctly!" -ForegroundColor Green
            exit 0
        } else {
            Write-Host "File has default_factory but still has errors. Re-applying fix..." -ForegroundColor Yellow
        }
    }
    
    Write-Host "Applying fix..." -ForegroundColor Yellow
    
    # Create backup
    $backupFile = "$specsFile.backup"
    Copy-Item $specsFile $backupFile -Force
    Write-Host "Backup created: $backupFile" -ForegroundColor Green
    
    # Fix the dataclass issue
    # Pattern: Look for dataclass fields with VehicleSpec as default
    # Need to replace: vehicle: VehicleSpec = VehicleSpec()
    # With: vehicle: VehicleSpec = field(default_factory=VehicleSpec)
    
    # Read line by line to fix properly
    $lines = Get-Content $specsFile
    $fixedLines = @()
    $inDataclass = $false
    $needsImport = $true
    
    foreach ($line in $lines) {
        # Check if we need to add field import
        if ($line -match "^from dataclasses import" -and $needsImport) {
            if ($line -notmatch "field") {
                $fixedLines += ($line -replace "from dataclasses import", "from dataclasses import field,")
                $needsImport = $false
            } else {
                $fixedLines += $line
                $needsImport = $false
            }
            continue
        }
        
        # Check if we're in a dataclass
        if ($line -match "@dataclass") {
            $inDataclass = $true
            $fixedLines += $line
            continue
        }
        
        # Check for class definition (end of dataclass)
        if ($line -match "^class " -and $inDataclass) {
            $inDataclass = $false
            $fixedLines += $line
            continue
        }
        
        # Fix mutable defaults in dataclass fields
        if ($inDataclass -and $line -match "vehicle:\s*VehicleSpec\s*=\s*VehicleSpec\(\)") {
            Write-Host "Fixing line: $line" -ForegroundColor Cyan
            $fixedLines += "    vehicle: VehicleSpec = field(default_factory=VehicleSpec)"
            continue
        }
        
        # Fix any other mutable defaults with VehicleSpec
        if ($inDataclass -and $line -match ":\s*VehicleSpec\s*=\s*VehicleSpec\(\)") {
            Write-Host "Fixing line: $line" -ForegroundColor Cyan
            $fieldName = ($line -split ":")[0].Trim()
            $fixedLines += "    $fieldName : VehicleSpec = field(default_factory=VehicleSpec)"
            continue
        }
        
        $fixedLines += $line
    }
    
    # If we still need the import, add it
    if ($needsImport) {
        $newLines = @()
        foreach ($line in $fixedLines) {
            $newLines += $line
            if ($line -match "^from dataclasses import dataclass") {
                $newLines += ($line -replace "from dataclasses import dataclass", "from dataclasses import dataclass, field")
                $needsImport = $false
            }
        }
        if ($needsImport) {
            # Add import at the top
            $newLines = @("from dataclasses import dataclass, field") + $fixedLines
        }
        $fixedLines = $newLines
    }
    
    # Write fixed content
    $fixedContent = $fixedLines -join "`n"
    Set-Content -Path $specsFile -Value $fixedContent -NoNewline
    
    Write-Host ""
    Write-Host "Fix applied!" -ForegroundColor Green
    Write-Host ""
    
    # Test the fix
    Write-Host "Testing fix..." -ForegroundColor Yellow
    $testResult = python -c "import racecar_gym.core.specs; print('SUCCESS: racecar_gym imports correctly!')" 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: racecar_gym now imports correctly!" -ForegroundColor Green
        Write-Host ""
        Write-Host "You can now run: python train.py" -ForegroundColor Cyan
    } else {
        Write-Host "ERROR: Fix did not work. Error:" -ForegroundColor Red
        Write-Host $testResult -ForegroundColor Red
        Write-Host ""
        Write-Host "Restoring backup..." -ForegroundColor Yellow
        Copy-Item $backupFile $specsFile -Force
        Write-Host "Backup restored. Please check the error above." -ForegroundColor Yellow
        exit 1
    }
    
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fix complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
