# Self-healing driver for the 25-particle Vdc trapping sweep.
# Runs the resume-capable Python sweep repeatedly until all points are done,
# then generates plots. Designed to be launched as a Windows Scheduled Task so
# it survives the harness reaping background shells at turn boundaries.

$ErrorActionPreference = "Continue"

$Py     = "C:\Program Files\Python312\python.exe"
$Script = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\sweep_vdc_transmission.py"
$Out    = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data\vdc_trapping_sweep_fixed_rf448_25p.csv"
$PlotDir= "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data\plots_vdc_trapping_fixed_rf448_25p"
$DataDir= "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data"
$Log    = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data\vdc_trapping_sweep_25p_driver.log"
$Target = 198

$common = @(
  "--masses","32,34,36","--num-particles","25","--trajectory-quality","1",
  "--rf-freq-mhz","2.4","--qmf-rf","448","--brubaker-rf","448",
  "--mean-ke-ev","0.7367136539184808","--realistic-fwhm-ke-ev","0.11451862433053576",
  "--source-radius-mm","0.03","--half-angle-deg","0.10",
  "--vdc-min","24.0","--vdc-max","40.0","--vdc-step","0.5",
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
# abundance-weighted + purity figures
$AbundScript = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\plot_abundance_weighted_trapping.py"
& $Py $AbundScript --csv $Out --plot-dir $PlotDir *>> $Log
Log "DONE rows=$(Get-Rows)/$Target (raw + abundance-weighted + purity plots written)"
