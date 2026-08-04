"""Export engine for the Digital Twin Validation Tool.

Handles the two export buttons on the Report tab:

- **``btnExportData``** (Export Table…) — writes the aligned comparison
  table (twin + ECU + error columns for every matched sample) to CSV
  or Excel (pandas ``to_csv`` / ``to_excel``).
- **``btnExportReport``** (Export Report…) — writes a full report
  (summary cards + per-signal breakdown + worst mismatches) to PDF
  (reportlab) or HTML (pandas ``to_html``).

Both methods append a timestamped log line to ``textEditExportLog`` so
the user has a visible audit trail of what was exported and when.

Like the statistics engine, this module is **pure Python** — it touches
neither Qt nor the UI widgets directly.  It returns paths or strings,
and ``back.py`` is responsible for appending to the log text edit and
showing any message boxes.  This keeps the engine unit-testable from the
command line.

The ``QFileDialog`` calls live in ``back.py`` too — the engine receives
an output path and writes to it.  Two reasons:

1. ``QFileDialog`` must run on the GUI thread; mixing it into a
   backend module would blur the line between logic and presentation.
2. The engine can be re-used by the live-MQTT path, which has no file
   dialog at all (messages arrive, get logged, optionally flushed to
   disk by a non-interactive path).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from alignment_engine import AlignedData
from statistics_engine import StatisticsResult, WorstMismatch


# ---------------------------------------------------------------------------
# Format constants (mirror the UI combo-box indices)
# ---------------------------------------------------------------------------
FORMAT_DATA_CSV   = "csv"
FORMAT_DATA_EXCEL = "excel"
FORMAT_REPORT_PDF  = "pdf"
FORMAT_REPORT_HTML = "html"


class ExportError(Exception):
    """Fatal error during export (missing library, bad path, etc.)."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class ExportResult:
    """Outcome of an export operation — fed back to back.py for the log."""

    success: bool
    output_path: str
    fmt: str
    n_rows: int               # rows written (0 for a report's "sections")
    message: str              # human-readable line for textEditExportLog
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def export_data(
    aligned: AlignedData,
    fmt: str,
    output_path: str,
) -> ExportResult:
    """Export the aligned comparison table to CSV or Excel.

    Builds a DataFrame with one row per matched sample and one column
    per (signal × twin/ecu/error), plus a ``timestamp`` column.  Then
    writes it via pandas.

    Parameters
    ----------
    aligned
        Output of :func:`alignment_engine.align`.
    fmt
        ``FORMAT_DATA_CSV`` or ``FORMAT_DATA_EXCEL``.  Any other value
        raises :class:`ExportError`.
    output_path
        Destination filesystem path.  The extension is the caller's
        responsibility; we don't enforce ``.csv`` vs ``.xlsx``.

    Returns
    -------
    ExportResult
        ``n_rows`` is the number of rows written.
    """
    df = _aligned_to_dataframe(aligned)

    try:
        if fmt == FORMAT_DATA_CSV:
            df.to_csv(output_path, index=False)
        elif fmt == FORMAT_DATA_EXCEL:
            df.to_excel(output_path, index=False, engine="openpyxl")
        else:
            raise ExportError(
                f"Unknown data format '{fmt}'. "
                f"Expected '{FORMAT_DATA_CSV}' or '{FORMAT_DATA_EXCEL}'."
            )
    except ExportError:
        raise
    except Exception as exc:
        return ExportResult(
            success=False, output_path=output_path, fmt=fmt, n_rows=0,
            message="", error=str(exc),
        )

    return ExportResult(
        success=True, output_path=output_path, fmt=fmt, n_rows=len(df),
        message=_log_line(fmt, output_path, n_rows=len(df)),
    )


def export_report(
    aligned: AlignedData,
    stats: StatisticsResult,
    fmt: str,
    output_path: str,
    title: str = "Digital Twin Validation Report",
) -> ExportResult:
    """Export a full report (summary + per-signal + worst mismatches) to PDF or HTML.

    Parameters
    ----------
    aligned
        Output of :func:`alignment_engine.align`.
    stats
        Output of :func:`statistics_engine.compute`.
    fmt
        ``FORMAT_REPORT_PDF`` or ``FORMAT_REPORT_HTML``.  Any other
        value raises :class:`ExportError`.
    output_path
        Destination filesystem path.
    title
        Header shown at the top of the report.  Defaults to
        "Digital Twin Validation Report".

    Returns
    -------
    ExportResult
        ``n_rows`` is 0 (a report has sections, not rows).
    """
    try:
        if fmt == FORMAT_REPORT_HTML:
            html = _report_html(aligned, stats, title)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
        elif fmt == FORMAT_REPORT_PDF:
            _report_pdf(aligned, stats, title, output_path)
        else:
            raise ExportError(
                f"Unknown report format '{fmt}'. "
                f"Expected '{FORMAT_REPORT_PDF}' or '{FORMAT_REPORT_HTML}'."
            )
    except ExportError:
        raise
    except Exception as exc:
        return ExportResult(
            success=False, output_path=output_path, fmt=fmt, n_rows=0,
            message="", error=str(exc),
        )

    return ExportResult(
        success=True, output_path=output_path, fmt=fmt, n_rows=0,
        message=_log_line(fmt, output_path, n_rows=0, is_report=True),
    )


# ---------------------------------------------------------------------------
# Internals: aligned-table DataFrame
# ---------------------------------------------------------------------------
def _aligned_to_dataframe(aligned: AlignedData) -> pd.DataFrame:
    """Build the wide-format comparison table for the CSV/Excel export."""
    cols: dict = {"timestamp": aligned.timestamps}
    for name, sig in aligned.signals.items():
        cols[f"{name}_twin"] = sig.twin_values
        cols[f"{name}_ecu"]  = sig.ecu_values
        cols[f"{name}_error"] = sig.errors
    df = pd.DataFrame(cols)
    # Round floats for readability; timestamps keep full precision.
    signal_cols = [c for c in df.columns if c != "timestamp"]
    df[signal_cols] = df[signal_cols].round(6)
    return df


# ---------------------------------------------------------------------------
# Internals: HTML report
# ---------------------------------------------------------------------------
def _report_html(
    aligned: AlignedData,
    stats: StatisticsResult,
    title: str,
) -> str:
    """Render the full report as a self-contained HTML document."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Summary cards table ---
    summary_rows = [
        ("Total Samples",     stats.total_samples),
        ("Matched Samples",   stats.matched_samples),
        ("Mean Absolute Error", f"{stats.mean_abs_error:.6f}"),
        ("Max Error",         f"{stats.max_error:.6f}"),
        ("RMSE",              f"{stats.rmse:.6f}"),
        ("Match %",           f"{stats.match_pct:.2f}"),
    ]
    summary_html = _html_table(
        ["Metric", "Value"], summary_rows, classes="summary"
    )

    # --- Per-signal breakdown table ---
    signal_rows = []
    for name, s in stats.signal_stats.items():
        signal_rows.append((
            name,
            f"{s.mae:.6f}",
            f"{s.rmse:.6f}",
            f"{s.max_error:.6f}",
            f"{s.mean_error:+.6f}",
            f"{s.match_pct:.2f}",
            f"{s.tolerance_pct:.2f}",
        ))
    signal_html = _html_table(
        ["Signal", "MAE", "RMSE", "Max Error", "Bias", "Match %",
         "Tol %"],
        signal_rows, classes="signals",
    )

    # --- Worst-mismatches table ---
    worst_rows = [
        (
            f"{w.timestamp:.3f}",
            w.signal,
            f"{w.twin_value:.4f}",
            f"{w.ecu_value:.4f}",
            f"{w.error:+.4f}",
            "yes" if w.within_tolerance else "no",
        )
        for w in stats.worst_mismatches
    ]
    if not worst_rows:
        worst_rows.append(("", "(none)", "", "", "", ""))
    worst_html = _html_table(
        ["Timestamp", "Signal", "Twin", "ECU", "Error", "Within Tol?"],
        worst_rows, classes="worst",
    )

    # --- Tolerances used ---
    tols = ", ".join(f"{k}={v:.2f}%" for k, v in stats.tolerances.items())
    method = aligned.alignment_method

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 2em; }}
  h1, h2 {{ color: #333; }}
  table {{ border-collapse: collapse; margin-bottom: 1.5em; }}
  th, td {{ border: 1px solid #999; padding: 6px 14px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #eee; }}
  .summary td:last-child {{ font-weight: bold; }}
  .worst td:last-child {{ text-align: center; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 1em; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">
  Generated {generated} &middot;
  Alignment method: <code>{method}</code> &middot;
  Tolerances: {tols or "(defaults)"}
</div>

<h2>Summary</h2>
{summary_html}

<h2>Per-Signal Breakdown</h2>
{signal_html}

<h2>Worst Mismatches (top {len(stats.worst_mismatches)})</h2>
{worst_html}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Internals: PDF report (reportlab)
# ---------------------------------------------------------------------------
def _report_pdf(
    aligned: AlignedData,
    stats: StatisticsResult,
    title: str,
    output_path: str,
) -> None:
    """Render the report to a PDF using reportlab's platypus framework."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        )
    except ImportError as exc:
        raise ExportError(
            "PDF export requires the 'reportlab' package. "
            "Install it with: pip install reportlab"
        ) from exc

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            topMargin=50, bottomMargin=50,
                            leftMargin=50, rightMargin=50)
    styles = getSampleStyleSheet()
    story: list = []

    # --- Title + meta ---
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 8))
    meta = (
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; "
        f"Alignment method: <b>{aligned.alignment_method}</b> &nbsp;|&nbsp; "
        f"Tolerances: {_tol_summary(stats)}"
    )
    story.append(Paragraph(meta, styles["Normal"]))
    story.append(Spacer(1, 18))

    # --- Summary table ---
    story.append(Paragraph("Summary", styles["Heading2"]))
    summary_data = [
        ["Metric", "Value"],
        ["Total Samples",     str(stats.total_samples)],
        ["Matched Samples",   str(stats.matched_samples)],
        ["Mean Absolute Error", f"{stats.mean_abs_error:.6f}"],
        ["Max Error",         f"{stats.max_error:.6f}"],
        ["RMSE",              f"{stats.rmse:.6f}"],
        ["Match %",           f"{stats.match_pct:.2f}"],
    ]
    story.append(_pdf_table(summary_data, header_row=True))
    story.append(Spacer(1, 18))

    # --- Per-signal breakdown ---
    story.append(Paragraph("Per-Signal Breakdown", styles["Heading2"]))
    sig_rows = [["Signal", "MAE", "RMSE", "Max Err", "Bias",
                 "Match %", "Tol %"]]
    for name, s in stats.signal_stats.items():
        sig_rows.append([
            name,
            f"{s.mae:.6f}",
            f"{s.rmse:.6f}",
            f"{s.max_error:.6f}",
            f"{s.mean_error:+.6f}",
            f"{s.match_pct:.2f}",
            f"{s.tolerance_pct:.2f}",
        ])
    story.append(_pdf_table(sig_rows, header_row=True))
    story.append(Spacer(1, 18))

    # --- Worst mismatches ---
    story.append(Paragraph(
        f"Worst Mismatches (top {len(stats.worst_mismatches)})",
        styles["Heading2"],
    ))
    worst_rows = [["Timestamp", "Signal", "Twin", "ECU", "Error",
                   "Within Tol?"]]
    for w in stats.worst_mismatches:
        worst_rows.append([
            f"{w.timestamp:.3f}",
            w.signal,
            f"{w.twin_value:.4f}",
            f"{w.ecu_value:.4f}",
            f"{w.error:+.4f}",
            "yes" if w.within_tolerance else "no",
        ])
    if len(worst_rows) == 1:
        worst_rows.append(["", "(none)", "", "", "", ""])
    story.append(_pdf_table(worst_rows, header_row=True))

    doc.build(story)


def _pdf_table(rows: List[list], header_row: bool = False):
    """Build a styled reportlab Table from a list-of-lists."""
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    # Column widths: first column narrow (label), rest flexible.  reportlab
    # expects explicit widths for consistent rendering.
    n_cols = len(rows[0])
    col_widths = [120] + [None] * (n_cols - 1) if n_cols > 1 else None
    # Setting half the widths to None tells reportlab to share remaining
    # space equally among them.
    tbl = Table(rows, colWidths=col_widths, hAlign="LEFT")
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID",     (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN",    (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN",    (0, 0), (0, -1),  "LEFT"),
        ("VALIGN",   (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.whitesmoke, colors.white]),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    if header_row:
        style_cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _html_table(
    headers: List[str],
    rows: list,
    classes: str = "",
) -> str:
    """Render a minimal HTML table from headers + rows of strings."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    cls = f' class="{classes}"' if classes else ""
    return f'<table{cls}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _tol_summary(stats: StatisticsResult) -> str:
    if not stats.tolerances:
        return "(defaults)"
    return ", ".join(f"{k}={v:.2f}%"
                     for k, v in stats.tolerances.items())


def _log_line(
    fmt: str,
    output_path: str,
    n_rows: int,
    is_report: bool = False,
) -> str:
    """Format a timestamped log line for textEditExportLog."""
    ts = datetime.now().strftime("%H:%M:%S")
    kind = "report" if is_report else "table"
    detail = f"{n_rows} rows" if n_rows else "report"
    return (f"[{ts}] Exported {kind} ({fmt}) → {output_path}  ({detail})")