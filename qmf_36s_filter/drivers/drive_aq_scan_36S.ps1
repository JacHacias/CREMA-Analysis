# Self-healing driver: (a,q)-plane ACCEPTANCE map for 36S (realistic emittance).
# Fine grid concentrated in the active region. Scheduled-task durable.
$ErrorActionPreference = "Continue"
$Py     = "C:\Program Files\Python312\python.exe"
$Script = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\scan_aq_plane.py"
$Out    = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data\aq_plane_36S_emit.csv"
$PlotDir= "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data\plots_aq_plane_36S_emit"
$DataDir= "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data"
$Log    = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data\aq_plane_36S_emit_driver.log"
$Target = 495   # 33 q (0.15..0.95/0.025) x 15 a (0.0..0.14/0.01)

$common = @(
  "--mass","36","--r0-mm","5.1942","--rf-freq-mhz","2.4",
  "--q-min","0.15","--q-max","0.95","--q-step","0.025",
  "--a-min","0.0","--a-max","0.14","--a-step","0.01",
  "--num-particles","24","--chunk-size","24","--trajectory-quality","1",
  "--mean-ke-ev","0.7367136539184808","--fwhm-ke-ev","0.11451862433053576",
  "--source-radius-mm","0.5","--half-angle-deg","1.0","--pressure-pa","0",
  "--max-flight-time-us","100","--stop-y-mm","170",
  "--output",$Out,"--plot-dir",$PlotDir
)

function Get-Rows { if (Test-Path -LiteralPath $Out) { return @(Import-Csv -LiteralPath $Out).Count } ; return 0 }
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -LiteralPath $Log -Append -Encoding utf8 }

Log "driver start, rows=$(Get-Rows)/$Target"
$prev=-1; $stuck=0
while ((Get-Rows) -lt $Target) {
  $cur = Get-Rows
  if ($cur -le $prev) { $stuck++ } else { $stuck = 0 }
  if ($stuck -ge 4) { Log "STUCK at $cur rows"; break }
  $prev = $cur
  Log "chunk start at $cur/$Target rows"
  & $Py $Script @common *>> $Log
  Get-ChildItem -LiteralPath $DataDir -Filter "_aq_m*.fly2" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
}
Log "loop done at $(Get-Rows)/$Target rows, plotting"
& $Py $Script @common --plot-only *>> $Log
Log "DONE rows=$(Get-Rows)/$Target"
