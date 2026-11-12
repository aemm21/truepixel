<#
.SYNOPSIS
Evalua un checkpoint de TruePixel sobre cada carpeta ai_* de un directorio OOD.

.EXAMPLE
.\evaluar_ood.ps1 -DirReales C:\datasets\ood\real -DirOod C:\datasets\ood -Modelo truepixel_v2.pth
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DirReales,

    [Parameter(Mandatory = $true)]
    [string]$DirOod,

    [string]$Modelo = "truepixel_v2.pth",
    [string]$Salida = ".\resultados_ood",
    [string]$Python = "",
    [int]$BatchSize = 16,
    [int]$NumWorkers = 2,
    [double]$Umbral = 0.5,
    [int]$MaxImagePixels = 100000000,
    [switch]$SinRecursivo
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot

function Resolver-Ruta([string]$ruta) {
    if (Test-Path -LiteralPath $ruta) {
        return (Resolve-Path -LiteralPath $ruta).Path
    }
    $enRepo = Join-Path $repo $ruta
    if (Test-Path -LiteralPath $enRepo) {
        return (Resolve-Path -LiteralPath $enRepo).Path
    }
    throw "No se encontro la ruta: $ruta"
}

$DirReales = Resolver-Ruta $DirReales
$DirOod = Resolver-Ruta $DirOod
$Modelo = Resolver-Ruta $Modelo

if (-not $Python) {
    $venv = Join-Path $repo ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venv) {
        $Python = $venv
    } else {
        $Python = "python"
    }
} elseif (Test-Path -LiteralPath $Python) {
    $Python = (Resolve-Path -LiteralPath $Python).Path
} else {
    $comando = Get-Command $Python -ErrorAction SilentlyContinue
    if ($comando) {
        $Python = $comando.Source
    } else {
        throw "No se encontro Python: $Python"
    }
}

if (-not [System.IO.Path]::IsPathRooted($Salida)) {
    $Salida = Join-Path (Get-Location).Path $Salida
}
New-Item -ItemType Directory -Force -Path $Salida | Out-Null

$evaluate = Join-Path $repo "evaluate.py"
if (-not (Test-Path -LiteralPath $evaluate)) {
    throw "No se encontro evaluate.py en $repo"
}

$env:TRUEPIXEL_MAX_IMAGE_PIXELS = "$MaxImagePixels"
$umbralTexto = $Umbral.ToString([System.Globalization.CultureInfo]::InvariantCulture)

$generadores = @(Get-ChildItem -LiteralPath $DirOod -Directory -Filter "ai_*" | Sort-Object Name)
if ($generadores.Count -eq 0) {
    throw "No hay carpetas ai_* en $DirOod"
}

$resumenes = New-Object System.Collections.Generic.List[object]

foreach ($generador in $generadores) {
    $nombre = $generador.Name
    $csv = Join-Path $Salida "$nombre.csv"
    $json = Join-Path $Salida "$nombre.json"

    $argumentos = @(
        "-u", $evaluate,
        "--dir_reales", $DirReales,
        "--dir_sinteticas", $generador.FullName,
        "--modelo", $Modelo,
        "--batch_size", "$BatchSize",
        "--num_workers", "$NumWorkers",
        "--umbral", $umbralTexto,
        "--csv", $csv,
        "--json"
    )
    if (-not $SinRecursivo) {
        $argumentos += "--recursivo"
    }

    Write-Host ""
    Write-Host "=== Evaluando $nombre ===" -ForegroundColor Cyan
    & $Python @argumentos | Set-Content -LiteralPath $json -Encoding UTF8
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Fallo la evaluacion de $nombre (codigo $LASTEXITCODE). Se omite."
        continue
    }

    try {
        $resumen = Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
    } catch {
        Write-Warning "No se pudo leer el JSON de $nombre. Se omite."
        continue
    }

    $resumenes.Add([pscustomobject]@{
        generador        = $nombre
        total            = [int]$resumen.total
        reales           = [int]$resumen.reales
        sinteticas       = [int]$resumen.sinteticas
        accuracy_pct     = [math]::Round([double]$resumen.metricas.accuracy * 100, 2)
        auc              = [math]::Round([double]$resumen.metricas.auc, 4)
        f1               = [math]::Round([double]$resumen.metricas.f1, 4)
        precision        = [math]::Round([double]$resumen.metricas.precision, 4)
        recall           = [math]::Round([double]$resumen.metricas.recall, 4)
        especificidad    = [math]::Round([double]$resumen.especificidad, 4)
        mejor_f1_umbral  = [double]$resumen.mejor_f1.umbral
        mejor_f1         = [math]::Round([double]$resumen.mejor_f1.f1, 4)
        mejor_bal_umbral = [double]$resumen.mejor_balanced_accuracy.umbral
        mejor_bal        = [math]::Round([double]$resumen.mejor_balanced_accuracy.balanced_accuracy, 4)
    }) | Out-Null
}

if ($resumenes.Count -eq 0) {
    throw "No se pudo evaluar ningun generador."
}

$rutaResumen = Join-Path $Salida "resumen_ood.csv"
$resumenes | Export-Csv -LiteralPath $rutaResumen -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "=== Resumen OOD ===" -ForegroundColor Green
$resumenes | Format-Table -AutoSize
Write-Host "CSV resumen: $rutaResumen"
