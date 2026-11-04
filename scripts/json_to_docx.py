import json, sys
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

FONT = "Tahoma"

def rtl_run(run, size, bold=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold = bold
    rpr = run._r.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), FONT)
    for tag in ("w:rtl",):
        rpr.append(OxmlElement(tag))
    cs = OxmlElement("w:szCs"); cs.set(qn("w:val"), str(size * 2)); rpr.append(cs)
    if bold:
        rpr.append(OxmlElement("w:bCs"))

def rtl_par(doc, text, size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=8):
    p = doc.add_paragraph()
    p.alignment = align
    p._p.get_or_add_pPr().append(OxmlElement("w:bidi"))
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.5
    rtl_run(p.add_run(text), size, bold)
    return p

src, dst = sys.argv[1], sys.argv[2]
data = json.load(open(src, encoding="utf-8"))
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.5)
sec.top_margin = sec.bottom_margin = Cm(2.5)
sec._sectPr.append(OxmlElement("w:bidi"))

rtl_par(doc, "متن پیاده‌سازی مکالمه", 20, True, WD_ALIGN_PARAGRAPH.CENTER, 16)
text = " ".join(c.strip() for c in data["chunks"])
# one paragraph per ~4 chunks for readability
chunks = [c.strip() for c in data["chunks"]]
for i in range(0, len(chunks), 4):
    rtl_par(doc, " ".join(chunks[i:i+4]))
doc.save(dst)
