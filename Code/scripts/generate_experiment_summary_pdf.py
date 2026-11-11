"""Render the SD_Guard experiment Markdown summary as a standalone PDF.

The generated PDF embeds the four chart previews, so the final document can be
shared as one file without preserving any local directory structure.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ACCENT = colors.HexColor("#0F766E")
ACCENT_DARK = colors.HexColor("#134E4A")
NAVY = colors.HexColor("#183B56")
TEXT = colors.HexColor("#243746")
MUTED = colors.HexColor("#64748B")
PALE = colors.HexColor("#ECFDF5")
PALE_BLUE = colors.HexColor("#EFF6FF")
TABLE_HEADER = colors.HexColor("#DCEFEA")
TABLE_ALT = colors.HexColor("#F7FAFC")
WARNING_BG = colors.HexColor("#FFF7ED")
WARNING_BORDER = colors.HexColor("#F97316")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def register_fonts() -> tuple[str, str]:
    candidates = [
        (
            Path(r"C:\Windows\Fonts\Deng.ttf"),
            Path(r"C:\Windows\Fonts\Dengb.ttf"),
        ),
        (
            Path(r"C:\Windows\Fonts\simhei.ttf"),
            Path(r"C:\Windows\Fonts\simhei.ttf"),
        ),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            try:
                pdfmetrics.registerFont(TTFont("DocSans", str(regular)))
                pdfmetrics.registerFont(TTFont("DocSansBold", str(bold)))
                pdfmetrics.registerFontFamily(
                    "DocSans",
                    normal="DocSans",
                    bold="DocSansBold",
                    italic="DocSans",
                    boldItalic="DocSansBold",
                )
                return "DocSans", "DocSansBold"
            except Exception:
                continue
    raise RuntimeError("No usable Chinese font was found in C:\\Windows\\Fonts")


def inline_markup(text: str, *, regular_font: str) -> str:
    """Convert the small Markdown inline subset used by the report."""

    placeholders: list[str] = []

    def save_code(match: re.Match[str]) -> str:
        value = html.escape(match.group(1))
        placeholders.append(
            f'<font name="{regular_font}" color="#0F766E">{value}</font>'
        )
        return f"@@CODE{len(placeholders) - 1}@@"

    text = re.sub(r"`([^`]+)`", save_code, text)
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+?)\*", r"<i>\1</i>", text)
    for index, value in enumerate(placeholders):
        text = text.replace(f"@@CODE{index}@@", value)
    return text


def build_styles(regular: str, bold: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName=bold,
            fontSize=28,
            leading=36,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=9 * mm,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=12,
            leading=20,
            textColor=MUTED,
            borderColor=ACCENT,
            borderWidth=0,
            borderPadding=(0, 0, 0, 5 * mm),
            leftIndent=5 * mm,
            spaceAfter=8 * mm,
        ),
        "h2": ParagraphStyle(
            "Heading2CN",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=18,
            leading=24,
            textColor=NAVY,
            spaceBefore=6 * mm,
            spaceAfter=4 * mm,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Heading3CN",
            parent=base["Heading3"],
            fontName=bold,
            fontSize=12.5,
            leading=17,
            textColor=ACCENT_DARK,
            spaceBefore=4 * mm,
            spaceAfter=2.5 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=9.2,
            leading=14.5,
            textColor=TEXT,
            wordWrap="CJK",
            spaceAfter=2.5 * mm,
        ),
        "bullet": ParagraphStyle(
            "BulletCN",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=9.2,
            leading=14.5,
            leftIndent=6 * mm,
            firstLineIndent=-3.5 * mm,
            bulletIndent=1.5 * mm,
            bulletFontName=regular,
            textColor=TEXT,
            wordWrap="CJK",
            spaceAfter=1.2 * mm,
        ),
        "number": ParagraphStyle(
            "NumberCN",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=9.2,
            leading=14.5,
            leftIndent=7 * mm,
            firstLineIndent=-5 * mm,
            bulletFontName=regular,
            textColor=TEXT,
            wordWrap="CJK",
            spaceAfter=1.2 * mm,
        ),
        "quote": ParagraphStyle(
            "QuoteCN",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=9.3,
            leading=15,
            leftIndent=6 * mm,
            rightIndent=6 * mm,
            borderColor=ACCENT,
            borderWidth=1.2,
            borderPadding=4 * mm,
            backColor=PALE,
            textColor=ACCENT_DARK,
            wordWrap="CJK",
            spaceBefore=2 * mm,
            spaceAfter=4 * mm,
        ),
        "warning": ParagraphStyle(
            "WarningCN",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=9.3,
            leading=15,
            leftIndent=5 * mm,
            rightIndent=5 * mm,
            borderColor=WARNING_BORDER,
            borderWidth=1.1,
            borderPadding=4 * mm,
            backColor=WARNING_BG,
            textColor=colors.HexColor("#9A3412"),
            wordWrap="CJK",
            spaceBefore=2 * mm,
            spaceAfter=4 * mm,
        ),
        "table_header": ParagraphStyle(
            "TableHeaderCN",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=7.2,
            leading=9.4,
            textColor=NAVY,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "table_cell": ParagraphStyle(
            "TableCellCN",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=7.0,
            leading=9.2,
            textColor=TEXT,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "CodeBlock",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.2,
            leading=10,
            textColor=colors.HexColor("#1E293B"),
            backColor=PALE_BLUE,
            borderColor=colors.HexColor("#BFDBFE"),
            borderWidth=0.7,
            borderPadding=4 * mm,
            spaceBefore=2 * mm,
            spaceAfter=4 * mm,
        ),
        "caption": ParagraphStyle(
            "CaptionCN",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=7.8,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=3 * mm,
        ),
    }


def table_col_widths(rows: list[list[str]], available_width: float) -> list[float]:
    columns = max(len(row) for row in rows)
    weights: list[float] = []
    for column in range(columns):
        values = [row[column] if column < len(row) else "" for row in rows]
        visual = max(
            sum(1.8 if ord(char) > 127 else 1.0 for char in value)
            for value in values
        )
        weights.append(min(max(visual, 7.0), 28.0))
    total = sum(weights)
    return [available_width * value / total for value in weights]


def make_table(
    raw_rows: list[list[str]],
    *,
    available_width: float,
    styles: dict[str, ParagraphStyle],
    regular_font: str,
) -> Table:
    columns = max(len(row) for row in raw_rows)
    normalized = [row + [""] * (columns - len(row)) for row in raw_rows]
    data: list[list[Paragraph]] = []
    for row_index, row in enumerate(normalized):
        style = styles["table_header"] if row_index == 0 else styles["table_cell"]
        data.append(
            [
                Paragraph(inline_markup(cell.strip(), regular_font=regular_font), style)
                for cell in row
            ]
        )
    table = Table(
        data,
        colWidths=table_col_widths(normalized, available_width),
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
    )
    commands: list[tuple] = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER),
        ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_index in range(2, len(data), 2):
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), TABLE_ALT))
    table.setStyle(TableStyle(commands))
    return table


def resolve_chart(markdown_path: Path, markdown_target: str) -> Path:
    target = (markdown_path.parent / markdown_target).resolve()
    if target.suffix.lower() == ".svg":
        preview = target.parent / "preview" / f"{target.stem}.png"
        if preview.exists():
            return preview
    return target


def parse_markdown(
    markdown_path: Path,
    *,
    styles: dict[str, ParagraphStyle],
    regular_font: str,
    available_width: float,
) -> list:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    story: list = []
    index = 0
    first_title = True

    while index < len(lines):
        stripped = lines[index].strip()

        if not stripped:
            index += 1
            continue

        if stripped.startswith("```"):
            language = stripped[3:].strip()
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            prefix = f"# {language}\n" if language else ""
            story.append(Preformatted(prefix + "\n".join(code_lines), styles["code"]))
            continue

        image_match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            alt, target = image_match.groups()
            chart = resolve_chart(markdown_path, target)
            if not chart.exists():
                raise FileNotFoundError(f"Chart not found: {chart}")
            max_width = min(available_width, 255 * mm)
            max_height = 132 * mm
            image = Image(str(chart))
            scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
            image.drawWidth = image.imageWidth * scale
            image.drawHeight = image.imageHeight * scale
            image.hAlign = "CENTER"
            story.append(image)
            if alt:
                story.append(
                    Paragraph(
                        inline_markup(f"图：{alt}", regular_font=regular_font),
                        styles["caption"],
                    )
                )
            index += 1
            continue

        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if first_title:
                story.append(Spacer(1, 22 * mm))
                story.append(
                    Paragraph(
                        inline_markup(title, regular_font=regular_font),
                        styles["cover_title"],
                    )
                )
                story.append(
                    Paragraph(
                        "vLLM 推测解码、严格安全评估、K 值消融与在线 Guard 性能诊断",
                        styles["cover_subtitle"],
                    )
                )
                first_title = False
            index += 1
            continue

        if stripped.startswith("## "):
            heading = stripped[3:].strip()
            if re.match(r"(?:4|5|6|7)\. 图", heading) or heading.startswith("8."):
                story.append(PageBreak())
            story.append(
                Paragraph(
                    inline_markup(heading, regular_font=regular_font), styles["h2"]
                )
            )
            index += 1
            continue

        if stripped.startswith("### "):
            heading = stripped[4:].strip()
            story.append(
                Paragraph(
                    inline_markup(heading, regular_font=regular_font), styles["h3"]
                )
            )
            index += 1
            continue

        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                value = lines[index].strip()[1:].strip()
                if value:
                    quote_lines.append(value)
                index += 1
            quote_text = "<br/>".join(
                inline_markup(value, regular_font=regular_font) for value in quote_lines
            )
            style = styles["warning"] if "口径提醒" in " ".join(quote_lines) else styles["quote"]
            story.append(Paragraph(quote_text, style))
            continue

        if stripped.startswith("|"):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            rows: list[list[str]] = []
            for table_index, line in enumerate(table_lines):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if table_index == 1 and all(
                    re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells
                ):
                    continue
                rows.append(cells)
            story.append(
                make_table(
                    rows,
                    available_width=available_width,
                    styles=styles,
                    regular_font=regular_font,
                )
            )
            story.append(Spacer(1, 3 * mm))
            continue

        if stripped.startswith("- "):
            while index < len(lines) and lines[index].strip().startswith("- "):
                value = lines[index].strip()[2:].strip()
                story.append(
                    Paragraph(
                        inline_markup(value, regular_font=regular_font),
                        styles["bullet"],
                        bulletText="•",
                    )
                )
                index += 1
            story.append(Spacer(1, 1.5 * mm))
            continue

        numbered = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if numbered:
            while index < len(lines):
                match = re.match(r"^(\d+)\.\s+(.+)$", lines[index].strip())
                if not match:
                    break
                story.append(
                    Paragraph(
                        inline_markup(match.group(2), regular_font=regular_font),
                        styles["number"],
                        bulletText=f"{match.group(1)}.",
                    )
                )
                index += 1
            story.append(Spacer(1, 1.5 * mm))
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if (
                not candidate
                or candidate.startswith(("#", ">", "|", "- ", "```", "!["))
                or re.match(r"^\d+\.\s+", candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        text = " ".join(paragraph_lines)
        story.append(
            Paragraph(inline_markup(text, regular_font=regular_font), styles["body"])
        )

    return story


def draw_page(canvas, document, *, regular_font: str, bold_font: str) -> None:
    width, height = landscape(A4)
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8E2EA"))
    canvas.setLineWidth(0.6)
    canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
    canvas.setFont(bold_font, 7.5)
    canvas.setFillColor(NAVY)
    canvas.drawString(18 * mm, height - 10.5 * mm, "SD_Guard 推测解码实验总结")
    canvas.setFont(regular_font, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(
        width - 18 * mm,
        9 * mm,
        f"SD_Guard · 第 {document.page} 页",
    )
    canvas.restoreState()


def build_pdf(markdown_path: Path, output_path: Path) -> None:
    regular, bold = register_fonts()
    styles = build_styles(regular, bold)
    page_size = landscape(A4)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=page_size,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=16 * mm,
        title="SD_Guard 推测解码实验总结",
        author="SD_Guard Project",
        subject="推测解码、严格安全评估、K 值消融与在线 Guard 性能诊断",
    )
    story = parse_markdown(
        markdown_path,
        styles=styles,
        regular_font=regular,
        available_width=document.width,
    )
    callback = lambda canvas, doc: draw_page(
        canvas, doc, regular_font=regular, bold_font=bold
    )
    document.build(story, onFirstPage=callback, onLaterPages=callback)


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--markdown",
        type=Path,
        default=root / "results" / "ppt_figures" / "SD_experiment_summary.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "output" / "pdf" / "SD_Guard推测解码实验总结.pdf",
    )
    args = parser.parse_args()
    build_pdf(args.markdown.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
