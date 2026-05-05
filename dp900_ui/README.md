# Rigol DP900 Control UI

Small browser UI for the Rigol DP932A / DP900 series power supply.

## Run

From this folder:

```powershell
python .\app.py
```

Open:

```text
http://127.0.0.1:8765
```

The UI defaults to LAN socket mode for the DP932A at `192.168.1.181`, port `5555`.

## Main Controls

- Control `CH1`, `CH2`, and `CH3` from separate panels.
- Set each channel's CV voltage setpoint.
- Set each channel's current limit.
- Set each channel's OVP and OCP protection limits. The app enables both protections when applying the setup.
- Turn each output on/off independently, or use `All Off`.
- Read back each channel's measured voltage, current, power, setpoints, output state, CV/CC/UR/OFF mode, and OVP/OCP alarm state.

## Notes

The app uses SCPI commands from `DP900_ProgrammingGuide_en.pdf`, including:

- `:SOURce<n>:VOLTage`
- `:SOURce<n>:CURRent`
- `:OUTPut:OVP:VALue`
- `:OUTPut:OCP:VALue`
- `:OUTPut CHx,ON/OFF`
- `:MEASure:ALL? CHx`
- `:OUTPut:MODE? CHx`

There are no required Python packages for LAN socket mode. VISA mode is present, but it requires `pyvisa` and a working VISA installation.
