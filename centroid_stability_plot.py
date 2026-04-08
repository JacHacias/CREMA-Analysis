from datetime import datetime
import re

import matplotlib.pyplot as plt
import numpy as np


GHZ_TO_MHZ = 1000.0


def _to_mhz(value_ghz):
    return float(value_ghz) * GHZ_TO_MHZ


def _parse_timestamp(value):
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _build_day_positions(timestamps):
    day_keys = [ts.date().isoformat() for ts in timestamps]
    unique_days = []
    for key in day_keys:
        if key not in unique_days:
            unique_days.append(key)

    x_positions = []
    tick_positions = []
    tick_labels = []

    for day_index, day_key in enumerate(unique_days):
        indices = [i for i, key in enumerate(day_keys) if key == day_key]
        count = len(indices)
        offsets = np.arange(count, dtype=float) - 0.5 * (count - 1)
        offsets *= 0.18

        for offset in offsets:
            x_positions.append(day_index + offset)

        tick_positions.append(day_index)
        tick_labels.append(datetime.fromisoformat(day_key).strftime("%m/%d/%y"))

    return np.array(x_positions, dtype=float), tick_positions, tick_labels


def _build_scan_labels(labels, timestamps):
    day_keys = [ts.date().isoformat() for ts in timestamps]
    counts_by_day = {}
    for key in day_keys:
        counts_by_day[key] = counts_by_day.get(key, 0) + 1

    seen_by_day = {}
    out = []
    for label, key in zip(labels, day_keys):
        seen_by_day[key] = seen_by_day.get(key, 0) + 1
        if counts_by_day[key] == 1:
            out.append("")
        else:
            out.append(f"s{seen_by_day[key]}")
    return out


def parse_centroid_output_blocks(text):
    """
    Parse pasted centroid-output blocks into the format expected by plot_centroid_stability.

    Expected block structure resembles:
        #3/27/26 Data 14:30
        32S center: -0.109326 +/- 0.051235 GHz
          fit contribution: 0.003180 GHz
          voltage contribution: 0.051136 GHz
        34S center: 0.124142 +/- 0.050042 GHz
          fit contribution: 0.006557 GHz
          voltage contribution: 0.049611 GHz
    """
    if not text.strip():
        raise ValueError("text must not be empty.")

    header_re = re.compile(r"^\s*#?\s*(?P<label>.+?)\s*$")
    center_re = re.compile(r"^\s*(?P<iso>32S|34S)\s+center:\s+(?P<center>[-+0-9.]+)\s+\+/-\s+(?P<total>[-+0-9.]+)\s+GHz\s*$")
    fit_re = re.compile(r"^\s*fit contribution:\s+(?P<fit>[-+0-9.]+)\s+GHz\s*$")
    voltage_re = re.compile(r"^\s*voltage contribution:\s+(?P<voltage>[-+0-9.]+)\s+GHz\s*$")
    wavemeter_re = re.compile(r"^\s*wavemeter contribution:\s+(?P<wavemeter>[-+0-9.]+)\s+GHz\s*$")
    timestamp_in_label_re = re.compile(
        r"(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{2,4})(?:\s+(?P<hour>\d{1,2}):(?P<minute>\d{2}))?"
    )

    entries = []
    current = None
    pending_iso = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current and "center_32_GHz" in current and "center_34_GHz" in current:
                entries.append(current)
                current = None
                pending_iso = None
            continue

        center_match = center_re.match(line)
        if center_match:
            if current is None:
                current = {}

            iso = center_match.group("iso")
            current[f"center_{iso[:2]}_GHz"] = float(center_match.group("center"))
            current[f"center_{iso[:2]}_total_unc_GHz"] = float(center_match.group("total"))
            pending_iso = iso[:2]
            continue

        fit_match = fit_re.match(line)
        if fit_match and current is not None and pending_iso is not None:
            current[f"center_{pending_iso}_fit_unc_GHz"] = float(fit_match.group("fit"))
            continue

        voltage_match = voltage_re.match(line)
        if voltage_match and current is not None and pending_iso is not None:
            current[f"center_{pending_iso}_voltage_unc_GHz"] = float(voltage_match.group("voltage"))
            continue

        wavemeter_match = wavemeter_re.match(line)
        if wavemeter_match and current is not None and pending_iso is not None:
            current[f"center_{pending_iso}_wavemeter_unc_GHz"] = float(wavemeter_match.group("wavemeter"))
            continue

        header_match = header_re.match(line)
        if header_match:
            if current and "center_32_GHz" in current and "center_34_GHz" in current:
                entries.append(current)
            current = {"label": header_match.group("label")}
            pending_iso = None

            ts_match = timestamp_in_label_re.search(header_match.group("label"))
            if ts_match:
                year = int(ts_match.group("year"))
                if year < 100:
                    year += 2000
                hour = int(ts_match.group("hour") or 12)
                minute = int(ts_match.group("minute") or 0)
                current["timestamp"] = datetime(
                    year,
                    int(ts_match.group("month")),
                    int(ts_match.group("day")),
                    hour,
                    minute,
                )
            continue

    if current and "center_32_GHz" in current and "center_34_GHz" in current:
        entries.append(current)

    if not entries:
        raise ValueError("No centroid result blocks were parsed from the provided text.")

    for item in entries:
        if "timestamp" not in item:
            raise ValueError(f"Could not infer a timestamp from label: {item.get('label', '<missing>')}")

    return entries


def plot_centroid_stability(results, title="Sulfur Centroid Stability"):
    """
    Plot 32S/34S centroid positions by day in MHz with uncertainty breakdowns.

    Parameters
    ----------
    results : list of dict
        Each dict should contain:
        - timestamp
        - label (optional, for annotations)
        - center_32_GHz
        - center_32_fit_unc_GHz
        - center_32_voltage_unc_GHz
        - center_32_wavemeter_unc_GHz (optional)
        - center_34_GHz
        - center_34_fit_unc_GHz
        - center_34_voltage_unc_GHz
        - center_34_wavemeter_unc_GHz (optional)
        Optional:
        - center_32_total_unc_GHz
        - center_34_total_unc_GHz

    Returns
    -------
    fig, axes
        Matplotlib figure and axes.
    """
    if not results:
        raise ValueError("results must contain at least one entry.")

    entries = sorted(results, key=lambda item: _parse_timestamp(item["timestamp"]))
    times = [_parse_timestamp(item["timestamp"]) for item in entries]
    x_positions, tick_positions, tick_labels = _build_day_positions(times)

    center_32 = np.array([_to_mhz(item["center_32_GHz"]) for item in entries], dtype=float)
    center_34 = np.array([_to_mhz(item["center_34_GHz"]) for item in entries], dtype=float)

    fit_32 = np.array([_to_mhz(item["center_32_fit_unc_GHz"]) for item in entries], dtype=float)
    fit_34 = np.array([_to_mhz(item["center_34_fit_unc_GHz"]) for item in entries], dtype=float)

    voltage_32 = np.array([_to_mhz(item["center_32_voltage_unc_GHz"]) for item in entries], dtype=float)
    voltage_34 = np.array([_to_mhz(item["center_34_voltage_unc_GHz"]) for item in entries], dtype=float)

    wavemeter_32 = np.array(
        [_to_mhz(item.get("center_32_wavemeter_unc_GHz", 0.0)) for item in entries],
        dtype=float,
    )
    wavemeter_34 = np.array(
        [_to_mhz(item.get("center_34_wavemeter_unc_GHz", 0.0)) for item in entries],
        dtype=float,
    )

    total_32 = np.array(
        [
            _to_mhz(
                item.get(
                    "center_32_total_unc_GHz",
                    np.sqrt(
                        item["center_32_fit_unc_GHz"] ** 2
                        + item["center_32_voltage_unc_GHz"] ** 2
                        + item.get("center_32_wavemeter_unc_GHz", 0.0) ** 2
                    ),
                )
            )
            for item in entries
        ],
        dtype=float,
    )
    total_34 = np.array(
        [
            _to_mhz(
                item.get(
                    "center_34_total_unc_GHz",
                    np.sqrt(
                        item["center_34_fit_unc_GHz"] ** 2
                        + item["center_34_voltage_unc_GHz"] ** 2
                        + item.get("center_34_wavemeter_unc_GHz", 0.0) ** 2
                    ),
                )
            )
            for item in entries
        ],
        dtype=float,
    )

    labels = [item.get("label", _parse_timestamp(item["timestamp"]).strftime("%Y-%m-%d %H:%M")) for item in entries]
    point_labels = _build_scan_labels(labels, times)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.2, 1.0]},
    )

    centroid_style = {
        "elinewidth": 1.2,
        "capsize": 4,
        "markersize": 6,
        "linewidth": 1.2,
    }
    component_style = {
        "elinewidth": 1.0,
        "capsize": 3,
        "markersize": 4,
        "linewidth": 0.9,
        "alpha": 0.75,
    }

    axes[0].errorbar(x_positions, center_32, yerr=total_32, fmt="o-", color="C0", label="32S centroid (total)", **centroid_style)
    axes[0].errorbar(
        x_positions - 0.035,
        center_32,
        yerr=fit_32,
        fmt=".",
        color="C0",
        linestyle="none",
        label="32S fit contribution",
        **component_style,
    )
    axes[0].errorbar(
        x_positions + 0.035,
        center_32,
        yerr=voltage_32,
        fmt=".",
        color="C0",
        linestyle="none",
        label="32S voltage contribution",
        **component_style,
    )
    if np.any(wavemeter_32 > 0):
        axes[0].errorbar(
            x_positions + 0.070,
            center_32,
            yerr=wavemeter_32,
            fmt=".",
            color="C4",
            linestyle="none",
            label="32S wavemeter contribution",
            **component_style,
        )
    axes[0].set_ylabel("32S centroid (MHz)", fontweight="bold")
    axes[0].set_title(title, fontweight="bold")
    axes[0].legend(loc="best")

    axes[1].errorbar(x_positions, center_34, yerr=total_34, fmt="o-", color="C1", label="34S centroid (total)", **centroid_style)
    axes[1].errorbar(
        x_positions - 0.035,
        center_34,
        yerr=fit_34,
        fmt=".",
        color="C1",
        linestyle="none",
        label="34S fit contribution",
        **component_style,
    )
    axes[1].errorbar(
        x_positions + 0.035,
        center_34,
        yerr=voltage_34,
        fmt=".",
        color="C1",
        linestyle="none",
        label="34S voltage contribution",
        **component_style,
    )
    if np.any(wavemeter_34 > 0):
        axes[1].errorbar(
            x_positions + 0.070,
            center_34,
            yerr=wavemeter_34,
            fmt=".",
            color="C5",
            linestyle="none",
            label="34S wavemeter contribution",
            **component_style,
        )
    axes[1].set_ylabel("34S centroid (MHz)", fontweight="bold")
    axes[1].legend(loc="best")

    axes[2].plot(x_positions, fit_32, "o:", color="C0", label="32S fit")
    axes[2].plot(x_positions, voltage_32, "o--", color="C0", label="32S voltage")
    if np.any(wavemeter_32 > 0):
        axes[2].plot(x_positions, wavemeter_32, "o-.", color="C4", label="32S wavemeter")
    axes[2].plot(x_positions, total_32, "o-", color="C0", alpha=0.8, label="32S total")
    axes[2].plot(x_positions, fit_34, "s:", color="C1", label="34S fit")
    axes[2].plot(x_positions, voltage_34, "s--", color="C1", label="34S voltage")
    if np.any(wavemeter_34 > 0):
        axes[2].plot(x_positions, wavemeter_34, "s-.", color="C5", label="34S wavemeter")
    axes[2].plot(x_positions, total_34, "s-", color="C1", alpha=0.8, label="34S total")
    axes[2].set_ylabel("Uncertainty (MHz)", fontweight="bold")
    axes[2].set_xlabel("Scan day", fontweight="bold")
    axes[2].legend(loc="best", ncol=2)

    for ax, centers in zip(axes[:2], [center_32, center_34]):
        for x, y, text in zip(x_positions, centers, point_labels):
            if text:
                ax.annotate(text, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, alpha=0.85)

    axes[2].set_xticks(tick_positions)
    axes[2].set_xticklabels(tick_labels)
    for ax in axes:
        ax.set_xlim(min(x_positions) - 0.35, max(x_positions) + 0.35)

    fig.tight_layout()
    return fig, axes


if __name__ == "__main__":
    example_text = """
    #3/24/26 backup Data
    32S center: -0.109326 +/- 0.051235 GHz
      fit contribution: 0.003180 GHz
      voltage contribution: 0.051136 GHz
    34S center: 0.124142 +/- 0.050042 GHz
      fit contribution: 0.006557 GHz
      voltage contribution: 0.049611 GHz

    #3/27/26 Data
    32S center: -0.109326 +/- 0.051235 GHz
      fit contribution: 0.003180 GHz
      voltage contribution: 0.051136 GHz
    34S center: 0.124142 +/- 0.050042 GHz
      fit contribution: 0.006557 GHz
      voltage contribution: 0.049611 GHz
    """

    plot_centroid_stability(parse_centroid_output_blocks(example_text))
    plt.show()
