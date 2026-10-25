// Document helpers for the report: Times New Roman, A4, template formatting.
const fs = require("fs");
const {
  Paragraph, TextRun, AlignmentType, Table, TableRow, TableCell, WidthType, ImageRun,
  BorderStyle, ShadingType, HeadingLevel, ExternalHyperlink, VerticalAlign,
} = require("docx");

const FONT = "Times New Roman";
const BODY = 24; // half-points: 12 pt
const CONTENT_W = 9026; // A4 width 11906 - 2 x 1440 margins (DXA)

// Inline markup: **bold**, *italic*, ^{superscript}, _{subscript}, [[url|text]]
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^\{[^}]+\}|_\{[^}]+\}|\[\[[^\]]+\]\])/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: BODY, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, size: BODY, ...base }));
    else if (t.startsWith("^{")) out.push(new TextRun({ text: t.slice(2, -1), superScript: true, font: FONT, size: BODY, ...base }));
    else if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), subScript: true, font: FONT, size: BODY, ...base }));
    else if (t.startsWith("[[")) {
      const [url, label] = t.slice(2, -2).split("|");
      out.push(new ExternalHyperlink({ link: url, children: [new TextRun({ text: label || url, style: "Hyperlink", font: FONT, size: BODY, ...base })] }));
    } else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, size: BODY, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size: BODY, ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({
  alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 120, line: 276 },
  indent: opts.indent,
  keepNext: opts.keepNext,
  children: runs(text, opts.run || {}),
});

const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 280, after: 140 },
  keepNext: true,
  children: [new TextRun({ text, bold: true, font: FONT, size: 30 })],
});
const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 200, after: 100 },
  keepNext: true,
  children: [new TextRun({ text, bold: true, font: FONT, size: BODY })],
});
const H3 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_3,
  spacing: { before: 140, after: 80 },
  keepNext: true,
  children: [new TextRun({ text, bold: true, italics: true, font: FONT, size: BODY })],
});

const bullets = (items, ref = "bullets") => items.map((t) => new Paragraph({
  numbering: { reference: ref, level: 0 },
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 60, line: 276 },
  children: runs(t),
}));

const numbered = (items, ref) => items.map((t) => new Paragraph({
  numbering: { reference: ref, level: 0 },
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 60, line: 276 },
  children: runs(t),
}));

// Display equation: centred, italic-free (symbols carry the meaning)
const EQ = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 60, after: 120 },
  children: runs(text, { font: "Cambria Math" }),
});

const caption = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 60, after: 200 },
  children: runs(text, { size: 20 }),
});
const tableCaption = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 160, after: 80 },
  keepNext: true,
  children: runs(text, { size: 20 }),
});

function pngSize(path) {
  const b = fs.readFileSync(path);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}

// Figure scaled to `frac` of the text width, aspect ratio preserved.
function figure(path, cap, frac = 1.0) {
  const { w, h, data } = pngSize(path);
  const width = Math.round(600 * frac);
  const plain = cap.replace(/\*\*/g, "").replace(/[_^]\{([^}]*)\}/g, "$1");
  const height = Math.round(width * h / w);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 40 },
      keepNext: true,
      children: [new ImageRun({ type: "png", data, transformation: { width, height }, altText: { title: plain.slice(0, 60), description: plain, name: path.split("/").pop() } })],
    }),
    caption(cap),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "8A8A85" };
const borders = { top: border, bottom: border, left: border, right: border };

// Table: header row shaded, widths in DXA (sum = CONTENT_W unless given).
function table(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const size = opts.size || 18; // 9 pt in tables
  const cell = (text, isHead, w, align) => new TableCell({
    borders,
    width: { size: w, type: WidthType.DXA },
    shading: isHead ? { fill: "E8EEF6", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    verticalAlign: VerticalAlign.CENTER,
    children: String(text).split("\n").map((line) => new Paragraph({
      alignment: align,
      spacing: { after: 0, line: 240 },
      children: runs(line, { size, bold: isHead || undefined }),
    })),
  });
  const align = (i) => (opts.align ? opts.align[i] : AlignmentType.LEFT);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, true, widths[i], align(i))) }),
      ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, false, widths[i], align(i))) })),
    ],
  });
}

module.exports = { FONT, BODY, CONTENT_W, runs, P, H1, H2, H3, bullets, numbered, EQ, caption, tableCaption, figure, table };
