# High-stats map of the 36S-only transmission window at FIXED 2.4 MHz, near the apex.
# Maps (Vrf, Vdc) with all 3 isotopes; finds window extent + lowest usable Vrf.
$ErrorActionPreference = "Continue"
$Py     = "C:\Program Files\Python312\python.exe"
$Script = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\scan_vrf_vdc_purity.py"
$Out    = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data\apex24_36S_window.csv"
$PlotDir= "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data\plots_apex24_36S_window"
$DataDir= "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data"
$Log    = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data\apex24_36S_window_driver.log"
$Target = 20   # 5 Vrf (1300..1460/40) x 4 Vdc (110..155/15)

$common = @(
  "--rf-freq-mhz","2.4",
  "--vrf-min","1300","--vrf-max","1460","--vrf-step","40",
  "--vdc-min","110","--vdc-max","155","--vdc-step","15",
  "--num-particles","20","--chunk-size","20","--max-flight-time-us","100",
  "--mean-ke-ev","0.7367136539184808","--fwhm-ke-ev","0.11451862433053576",
  "--output",$Out,"--plot-dir",$PlotDir
)

function Get-Rows { if (Test-Path -LiteralPath $Out) { return @(Import-Csv -LiteralPath $Out).Count } ; return 0 }
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -LiteralPath $Log -Append -Encoding utf8 }

Log "driver start, rows=$(Get-Rows)/$Target"
$prev=-1; $stuck=0
while ((Get-Rows) -lt $Target) {
  $cur = Get-Rows
  if ($cur -le $prev) { $stuck++ } else { $stuck = 0 }
  if ($stuck -ge 5) { Log "STUCK at $cur rows"; break }
  $prev = $cur
  Log "chunk start at $cur/$Target rows"
  & $Py $Script @common *>> $Log
  Get-ChildItem -LiteralPath $DataDir -Filter "_pur_m*.fly2" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
}
Log "loop done at $(Get-Rows)/$Target rows, plotting"
& $Py $Script @common --plot-only *>> $Log
Log "DONE rows=$(Get-Rows)/$Target"
