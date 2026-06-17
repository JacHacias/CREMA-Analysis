# Two-phase map at FIXED 2.4 MHz:
#  A) CONTEXT: wide coarse (Vrf 400-1480, Vdc 0-240) to place the 36S window in the landscape.
#  B) DETAIL : fine high-stats around the apex (Vrf 1320-1440, Vdc 105-155) to resolve the transition.
$ErrorActionPreference = "Continue"
$Py     = "C:\Program Files\Python312\python.exe"
$Script = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\scan_vrf_vdc_purity.py"
$Plotter= "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\plot_purity_map.py"
$DataDir= "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data"
$Log    = "$DataDir\context_detail_driver.log"

$ctxOut = "$DataDir\purity_context.csv"
$ctxPlotDir = "$DataDir\plots_purity_context"
$ctxTarget = 49   # 7 Vrf x 7 Vdc
$ctxArgs = @("--rf-freq-mhz","2.4","--vrf-min","400","--vrf-max","1480","--vrf-step","180",
  "--vdc-min","0","--vdc-max","240","--vdc-step","40","--num-particles","15","--chunk-size","15",
  "--max-flight-time-us","100","--mean-ke-ev","0.7367136539184808","--fwhm-ke-ev","0.11451862433053576",
  "--output",$ctxOut,"--plot-dir",$ctxPlotDir)

$detOut = "$DataDir\purity_detail.csv"
$detPlotDir = "$DataDir\plots_purity_detail"
$detTarget = 42   # 7 Vrf x 6 Vdc
$detArgs = @("--rf-freq-mhz","2.4","--vrf-min","1320","--vrf-max","1440","--vrf-step","20",
  "--vdc-min","105","--vdc-max","155","--vdc-step","10","--num-particles","40","--chunk-size","40",
  "--max-flight-time-us","100","--mean-ke-ev","0.7367136539184808","--fwhm-ke-ev","0.11451862433053576",
  "--output",$detOut,"--plot-dir",$detPlotDir)

function Rows($f) { if (Test-Path -LiteralPath $f) { return @(Import-Csv -LiteralPath $f).Count } ; return 0 }
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -LiteralPath $Log -Append -Encoding utf8 }
function RunPhase($name, $out, $target, $cargs) {
  Log "$name start, rows=$(Rows $out)/$target"
  $prev=-1; $stuck=0
  while ((Rows $out) -lt $target) {
    $cur = Rows $out
    if ($cur -le $prev) { $stuck++ } else { $stuck = 0 }
    if ($stuck -ge 5) { Log "$name STUCK at $cur"; break }
    $prev = $cur
    & $Py $Script @cargs *>> $Log
    Get-ChildItem -LiteralPath $DataDir -Filter "_pur_m*.fly2" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
  }
  Log "$name done at $(Rows $out)/$target"
}

RunPhase "CONTEXT" $ctxOut $ctxTarget $ctxArgs
& $Py $Plotter --csv $ctxOut --out "$ctxPlotDir\purity_composite.png" --title "Context: (Vrf,Vdc) landscape @ 2.4 MHz - green=36S-only" *>> $Log
RunPhase "DETAIL" $detOut $detTarget $detArgs
& $Py $Plotter --csv $detOut --out "$detPlotDir\purity_composite.png" --title "Detail: 36S-only window @ 2.4 MHz (40 ions/pt)" *>> $Log
Log "ALL DONE ctx=$(Rows $ctxOut)/$ctxTarget det=$(Rows $detOut)/$detTarget"
