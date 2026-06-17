# Idealized 36S window scan: pressure=0, source radius=0, cone angle=0.
# Isolates intrinsic (RF-phase + finite-time) broadening from collisional/emittance broadening.
# Self-healing; launched as a Scheduled Task.

$ErrorActionPreference = "Continue"

$Py     = "C:\Program Files\Python312\python.exe"
$Script = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\sweep_vdc_transmission.py"
$Abund  = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\plot_abundance_weighted_trapping.py"
$Annot  = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\plot_trapping_annotated.py"
$Out    = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data\vdc_trapping_36s_ideal_27_30_100p.csv"
$PlotDir= "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data\plots_36s_ideal_27_30_100p"
$DataDir= "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data"
$Log    = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data\vdc_trapping_36s_ideal_27_30_100p_driver.log"
$Target = 78   # 2 scenarios x 13 vdcs x 3 masses

$common = @(
  "--masses","32,34,36","--num-particles","100","--chunk-size","25","--trajectory-quality","1",
  "--rf-freq-mhz","2.4","--qmf-rf","448","--brubaker-rf","448",
  "--mean-ke-ev","0.7367136539184808","--realistic-fwhm-ke-ev","0.11451862433053576",
  "--source-radius-mm","0.0","--half-angle-deg","0.0","--pressure-pa","0",
  "--vdc-min","27.0","--vdc-max","30.0","--vdc-step","0.25",
  "--max-flight-time-us","100","--stop-y-mm","170",
  "--min-x-mm","12","--max-x-mm","26","--min-z-mm","12","--max-z-mm","26","--min-y-mm","10",
  "--output",$Out,"--plot-dir",$PlotDir
)

function Get-Rows {
  if (Test-Path -LiteralPath $Out) { return @(Import-Csv -LiteralPath $Out).Count }
  return 0
}
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -LiteralPath $Log -Append -Encoding utf8 }

Log "driver start, rows=$(Get-Rows)/$Target"
$prev = -1; $stuck = 0
while ((Get-Rows) -lt $Target) {
  $cur = Get-Rows
  if ($cur -le $prev) { $stuck++ } else { $stuck = 0 }
  if ($stuck -ge 4) { Log "STUCK at $cur rows, aborting loop"; break }
  $prev = $cur
  Log "chunk start at $cur/$Target rows"
  & $Py $Script @common *>> $Log
  Get-ChildItem -LiteralPath $DataDir -Filter "_vdc_sweep_*.fly2" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
}
Log "loop done at $(Get-Rows)/$Target rows, plotting"
& $Py $Script @common --plot-only *>> $Log
& $Py $Abund --csv $Out --plot-dir $PlotDir *>> $Log
& $Py $Annot --csv $Out --out (Join-Path $PlotDir "vdc_trapping_annotated_compare.png") *>> $Log
Log "DONE rows=$(Get-Rows)/$Target (ideal: p=0, r=0, angle=0)"
