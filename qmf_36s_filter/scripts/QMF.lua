simion.workbench_program()

adjustable RF_freq_MHz = 0.6
adjustable entrance_Brubaker_RF_amp_Vp = 341.4  -- The RF zero-to-peak voltage on one electrode, = V/2
adjustable QMF_RF_amp_Vp = 341.4                -- The RF zero-to-peak voltage on one electrode, = V/2
adjustable QMF_DC_V = 57.04                     -- The DC voltage on one electrode, = U/2
adjustable exit_Brubaker_RF_amp_Vp = 341.4      -- The RF zero-to-peak voltage on one electrode, = V/2

adjustable pressure_pa = 4e-3
adjustable temperature_k = 300
adjustable reduced_ion_mobility = 2e-4
adjustable stop_y_mm = 180
adjustable max_flight_time_us = 250
adjustable min_x_mm = -5
adjustable max_x_mm = 43
adjustable min_z_mm = -5
adjustable max_z_mm = 43
adjustable min_y_mm = -10


-- Configuration
local Y_MIN   = 25
local Y_MAX   =  670
local Y_STEP  =   5
local OUTPUT_FILENAME = 'data/data_test.csv'
local EVENT_FILENAME = 'data/trap_events.csv'
local FALLBACK_OUTPUT_FILENAME = '../../qmf_data_test.csv'
local FALLBACK_EVENT_FILENAME = '../../qmf_trap_events.csv'

-- Build list of y-planes (in mm)
local y_planes = {}
do
  local i = 1
  for y = Y_MIN, Y_MAX, Y_STEP do
    y_planes[i] = y
    i = i + 1
  end
end

-- Per-ion previous state (to detect crossings and interpolate)
local prev_state = {}   -- prev_state[ion_number] = {x,y,z,vx,vy,vz,t}
local pass_count = {}   -- pass_count[ion_number] = integer count
local finished = {}     -- finished[ion_number] = true after final event recorded

-- Output file handle
local fout = nil
local fevent = nil

local function open_writable(primary, fallback)
  local f = io.open(primary, 'w')
  if f then return f end
  return assert(io.open(fallback, 'w'))
end

-- Helper: open output and start a fresh CSV for each run.
local function open_output()
  fout = open_writable(OUTPUT_FILENAME, FALLBACK_OUTPUT_FILENAME)
  -- Keep column names simple; values come from *_mm variables.
  fout:write('ion,pass_index,time_us,y_plane,x_mm,y_mm,z_mm,vx_mm_per_us,vy_mm_per_us,vz_mm_per_us\n')
  fout:flush()
  fevent = open_writable(EVENT_FILENAME, FALLBACK_EVENT_FILENAME)
  fevent:write('ion,event,time_us,x_mm,y_mm,z_mm,vx_mm_per_us,vy_mm_per_us,vz_mm_per_us,pass_count\n')
  fevent:flush()
end

-- Helper: write one CSV line
local function write_row(id, pass_idx, t, yp, x, y, z, vx, vy, vz)
  fout:write(string.format(
    '%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n',
    id, pass_idx, t, yp, x, y, z, vx, vy, vz
  ))
end

local function write_event(id, event_name)
  if finished[id] then return end
  finished[id] = true
  if not fevent then return end
  fevent:write(string.format(
    '%d,%s,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d\n',
    id,
    event_name,
    ion_time_of_flight,
    ion_px_mm,
    ion_py_mm,
    ion_pz_mm,
    ion_vx_mm,
    ion_vy_mm,
    ion_vz_mm,
    pass_count[id] or 0
  ))
  fevent:flush()
end

-- Called when potentials are initialized (typically at start of a run)
function segment.initialize_run()
  prev_state = {}
  pass_count = {}
  finished = {}
  if fout then
    fout:flush()
    fout:close()
    fout = nil
  end
  if fevent then
    fevent:flush()
    fevent:close()
    fevent = nil
  end
  open_output()
end

-- Called when each ion is created (born)
function segment.initialize()
  local id = ion_number
  pass_count[id] = 0
  prev_state[id] = {
    x  = ion_px_mm,
    y  = ion_py_mm,
    z  = ion_pz_mm,
    vx = ion_vx_mm,
    vy = ion_vy_mm,
    vz = ion_vz_mm,
    t  = ion_time_of_flight
  }
end

function segment.fast_adjust()
    -- Viewed from top (exit), upper left and lower right are phase 1, upper right and lower left are phase 2
    -- 01 is exit Brubaker lens with phase 1
    -- 02 is exit Brubaker lens with phase 2
    -- 03 is filter with phase 1
    -- 04 is filter with phase 2
    -- 05 is entrance Brubaker lens with phase 1
    -- 06 is entrance Brubaker lens with phase 2
    adj_elect05 = - entrance_Brubaker_RF_amp_Vp * cos(2 * math.pi * RF_freq_MHz * ion_time_of_flight)
    adj_elect06 = - (- entrance_Brubaker_RF_amp_Vp * cos(2 * math.pi * RF_freq_MHz * ion_time_of_flight))

    adj_elect03 = QMF_DC_V - QMF_RF_amp_Vp * cos(2 * math.pi * RF_freq_MHz * ion_time_of_flight)
    adj_elect04 = - (QMF_DC_V - QMF_RF_amp_Vp * cos(2 * math.pi * RF_freq_MHz * ion_time_of_flight))

    adj_elect01 = - exit_Brubaker_RF_amp_Vp * cos(2 * math.pi * RF_freq_MHz * ion_time_of_flight)
    adj_elect02 = - (- exit_Brubaker_RF_amp_Vp * cos(2 * math.pi * RF_freq_MHz * ion_time_of_flight))
end

function segment.accel_adjust()
    -- For modeling pressure effect as a viscous drag using the Stokes' law.
    pressure_mbar = pressure_pa / 100

    if ion_time_step == 0 then return end                   -- Skip if zero time step
    if pressure_mbar == 0 then return end                   -- Skip if pressure set to zero

    -- Compute correction factor.
    mobility = reduced_ion_mobility * 1013 / pressure_mbar * temperature_k / 273.15
    linear_damping = abs((ion_charge * 1.602176462e-19) / (ion_mass * 1.66053873e-27) / mobility) * 1e-6            -- force damping factor positive, ion_mass in amu, ion_charge in unit of elementary charge                                                                                            -- in usec^-1

    local tterm = ion_time_step * linear_damping            -- time constant
    local factor = (1 - exp(-tterm)) / tterm                -- correction factor

    -- Compute new x, y, and z accelerations.
    -- This following the differential equation
    --   da/dt = -v*linear_damping
    -- with the correction factor for dt being finite.
    -- Note: ion_v[xyz]_mm is particle velocity in mm/usec.
    --       ion_a[xyz]_mm is particle acceleration in mm/usec^2.
    ion_ax_mm = factor * (ion_ax_mm - ion_vx_mm * linear_damping)
    ion_ay_mm = factor * (ion_ay_mm - ion_vy_mm * linear_damping)
    ion_az_mm = factor * (ion_az_mm - ion_vz_mm * linear_damping)
end

-- Called on each ion motion step
function segment.other_actions()
  local id = ion_number

  if ion_time_of_flight > max_flight_time_us then
    write_event(id, 'timeout_trapped')
    ion_splat = 1
    return
  end

  if ion_px_mm < min_x_mm or ion_px_mm > max_x_mm or
     ion_pz_mm < min_z_mm or ion_pz_mm > max_z_mm or
     ion_py_mm < min_y_mm then
    write_event(id, 'out_of_bounds')
    ion_splat = 1
    return
  end

  local p = prev_state[id]
  if not p then
    prev_state[id] = {
      x  = ion_px_mm, y  = ion_py_mm, z  = ion_pz_mm,
      vx = ion_vx_mm, vy = ion_vy_mm, vz = ion_vz_mm,
      t  = ion_time_of_flight
    }
    return
  end

  local x2, y2, z2 = ion_px_mm, ion_py_mm, ion_pz_mm
  local vx2, vy2, vz2 = ion_vx_mm, ion_vy_mm, ion_vz_mm
  local t2 = ion_time_of_flight

  local y1 = p.y
  local ydelta = y2 - y1
  local crossed_stop_plane = false
  if ydelta ~= 0 then
    local going_up = (ydelta > 0)

    -- Iterate planes in time order:
    if going_up then
      for i = 1, #y_planes do
        local yp = y_planes[i]
        if (y1 < yp) and (y2 >= yp) then
          local a = (yp - y1) / ydelta
          local x = p.x  + (x2  - p.x ) * a
          local z = p.z  + (z2  - p.z ) * a
          local vx= p.vx + (vx2 - p.vx) * a
          local vy= p.vy + (vy2 - p.vy) * a
          local vz= p.vz + (vz2 - p.vz) * a
          local t = p.t  + (t2  - p.t ) * a
          pass_count[id] = (pass_count[id] or 0) + 1
          write_row(id, pass_count[id], t, yp, x, yp, z, vx, vy, vz)
          if yp >= stop_y_mm then crossed_stop_plane = true end
        end
      end
    else
      for i = #y_planes, 1, -1 do
        local yp = y_planes[i]
        if (y1 > yp) and (y2 <= yp) then
          local a = (yp - y1) / ydelta
          local x = p.x  + (x2  - p.x ) * a
          local z = p.z  + (z2  - p.z ) * a
          local vx= p.vx + (vx2 - p.vx) * a
          local vy= p.vy + (vy2 - p.vy) * a
          local vz= p.vz + (vz2 - p.vz) * a
          local t = p.t  + (t2  - p.t ) * a
          pass_count[id] = (pass_count[id] or 0) + 1
          write_row(id, pass_count[id], t, yp, x, yp, z, vx, vy, vz)
        end
      end
    end

    if fout then fout:flush() end
  end

  if crossed_stop_plane or y2 >= stop_y_mm then
    write_event(id, 'reached_stop_plane')
    ion_splat = 1
    return
  end

  -- Update previous state
  prev_state[id] = {
    x  = x2,  y  = y2,  z  = z2,
    vx = vx2, vy = vy2, vz = vz2,
    t  = t2
  }
end

-- Called when ion run finishes; flush output
function segment.terminate()
  if fout then
    fout:flush()
    fout:close()
    fout = nil
  end
  if fevent then
    fevent:flush()
    fevent:close()
    fevent = nil
  end
end

