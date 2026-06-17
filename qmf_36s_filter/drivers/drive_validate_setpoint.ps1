# High-statistics validation of the 36S filter setpoint: Vrf=1400, Vdc=125, 2.4 MHz.
# 200 ions/isotope to tighten the 34S/32S suppression bound and nail the 36S transmission.
$ErrorActionPreference = "Continue"
$Py     = "C:\Program Files\Python312\python.exe"
$Script = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\scan_vrf_vdc_purity.py"
$Out    = "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data\setpoint_validate.csv"
$PlotDir= "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data\plots_setpoint_validate"
$DataDir= "C:\Users\EMALAB\Documents\Jackson\Beamline Simion\QMF\data"
$Log    = "$DataDir\setpoint_validate_driver.log"
$Target = 1

$common = @("--rf-freq-mhz","2.4",
  "--vrf-min","1400","--vrf-max","1400","--vrf-step","40",
  "--vdc-min","125","--vdc-max","125","--vdc-step","10",
  "--num-particles","200","--chunk-size","50","--max-flight-time-us","150",
  "--mean-ke-ev","0.7367136539184808","--fwhm-ke-ev","0.11451862433053576",
  "--output",$Out,"--plot-dir",$PlotDir)

function Rows { if (Test-Path -LiteralPath $Out) { return @(Import-Csv -LiteralPath $Out).Count } ; return 0 }
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -LiteralPath $Log -Append -Encoding utf8 }
Log "validate start"
$tries=0
while ((Rows) -lt $Target -and $tries -lt 6) {
  $tries++
  & $Py $Script @common *>> $Log
  Get-ChildItem -LiteralPath $DataDir -Filter "_pur_m*.fly2" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
}
Log "DONE rows=$(Rows)/$Target"
