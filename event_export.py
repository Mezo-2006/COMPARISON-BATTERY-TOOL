"""Report export for the Event Detection module (Task 15).

Writes an :class:`event_detector.EventLog` to CSV, Excel, or PDF. Mirrors
``export_engine.py``'s split: this module is pure Python (pandas +
optionally reportlab) and only ever receives an output path — any
``QFileDialog`` belongs in the caller (``event_window.py`` / ``back.py``),
not here, so this stays unit-testable from the command line.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd

from event_detector import EventLog

# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------
FORMAT_CSV = "csv"
FORMAT_EXCEL = "excel"
FORMAT_PDF = "pdf"

# Column order + display names required by Task 15.
_COLUMNS = [
    ("signal", "Signal"),
    ("real_value", "Real Value"),
    ("digital_twin_value", "Digital Twin Value"),
    ("difference", "Difference"),
    ("threshold", "Threshold"),
    ("event", "Event Type"),
    ("severity", "Severity"),
    ("timestamp", "Timestamp"),
]


class EventExportError(Exception):
    """Fatal error during export (missing library, bad path, etc.)."""


@dataclass
class EventExportResult:
    success: bool
    output_path: str
    fmt: str
    n_rows: int
    message: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def export_event_log(
    log: EventLog,
    fmt: str,
    output_path: str,
    title: str = "Event Detection Report",
) -> EventExportResult:
    """Export the event log to CSV, Excel, or PDF.

    Parameters
    ----------
    log
        The accumulated :class:`event_detector.EventLog`.
    fmt
        One of :data:`FORMAT_CSV`, :data:`FORMAT_EXCEL`, :data:`FORMAT_PDF`.
    output_path
        Destination filesystem path.
    """
    df = _display_dataframe(log)

    try:
        if fmt == FORMAT_CSV:
            df.to_csv(output_path, index=False)
        elif fmt == FORMAT_EXCEL:
            df.to_excel(output_path, index=False, engine="openpyxl")
        elif fmt == FORMAT_PDF:
            _export_pdf(df, title, output_path)
        else:
            raise EventExportError(
                f"Unknown export format '{fmt}'. Expected "
                f"'{FORMAT_CSV}', '{FORMAT_EXCEL}', or '{FORMAT_PDF}'."
            )
    except EventExportError:
        raise
    except Exception as exc:
        return EventExportResult(
            success=False, output_path=output_path, fmt=fmt, n_rows=0,
            message="", error=str(exc),
        )

    return EventExportResult(
        success=True, output_path=output_path, fmt=fmt, n_rows=len(df),
        message=_log_line(fmt, output_path, len(df)),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _display_dataframe(log: EventLog) -> pd.DataFrame:
    """Reorder/rename the event log's columns to the Task-15 field list."""
    df = log.to_dataframe()
    src_cols = [c for c, _ in _COLUMNS]
    for col in src_cols:
        if col not in df.columns:
            df[col] = None
    df = df[src_cols].rename(columns=dict(_COLUMNS))
    return df


def _export_pdf(df: pd.DataFrame, title: str, output_path: str) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        )
    except ImportError as exc:
        raise EventExportError(
            "PDF export requires the 'reportlab' package. "
            "Install it with: pip install reportlab"
        ) from exc

    doc = SimpleDocTemplate(
        output_path, pagesize=landscape(A4),
        topMargin=40, bottomMargin=40, leftMargin=40, rightMargin=40,
    )
    styles = getSampleStyleSheet()
    story: list = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 8))
    meta = (
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; "
        f"{len(df)} event(s)"
    )
    story.append(Paragraph(meta, styles["Normal"]))
    story.append(Spacer(1, 14))

    rows = [list(df.columns)] + df.astype(object).where(df.notna(), "").values.tolist()
    rows = [[str(c) for c in row] for row in rows]

    tbl = Table(rows, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)
    doc.build(story)


def _log_line(fmt: str, output_path: str, n_rows: int) -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    return f"[{ts}] Exported event log ({fmt}) -> {output_path}  ({n_rows} rows)"
