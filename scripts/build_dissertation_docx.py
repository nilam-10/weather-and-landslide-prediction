"""Build final revised dissertation DOCX for B.Tech Honours Major Project.
Synthesizes the complete audited dissertation text into a polished DOCX file
using OpenXML package structure.
"""
import zipfile
import xml.etree.ElementTree as ET
import pathlib
import re

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
TEMPLATE_DOCX = pathlib.Path(r"D:\Downloads\RBU_Project_Synopsis_Multimodal_Hazard_Platform_Updated (1).docx")
OUT_DOCX = PROJECT_ROOT / "outputs" / "reports" / "thesis_weather_model_integration" / "FINAL_DISSERTATION_REVISED.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W_NS)

def make_element(tag, attrs=None):
    qname = f"{{{W_NS}}}{tag}"
    el = ET.Element(qname)
    if attrs:
        for k, v in attrs.items():
            if ":" in k:
                prefix, local = k.split(":")
                el.set(f"{{{W_NS}}}{local}", v)
            else:
                el.set(k, v)
    return el

def add_pPr(p, style=None, jc="left", spacing_after=120, line="276"):
    pPr = make_element("pPr")
    if style:
        pStyle = make_element("pStyle", {"w:val": style})
        pPr.append(pStyle)
    if jc:
        jc_el = make_element("jc", {"w:val": jc})
        pPr.append(jc_el)
    if spacing_after or line:
        sp_attrs = {}
        if spacing_after:
            sp_attrs["w:after"] = str(spacing_after)
        if line:
            sp_attrs["w:line"] = str(line)
            sp_attrs["w:lineRule"] = "auto"
        spacing = make_element("spacing", sp_attrs)
        pPr.append(spacing)
    p.append(pPr)

def add_run(p, text, bold=False, italic=False, font_size=24, font_family="Times New Roman", color=None):
    r = make_element("r")
    rPr = make_element("rPr")
    
    rFonts = make_element("rFonts", {"w:ascii": font_family, "w:hAnsi": font_family, "w:cs": font_family})
    rPr.append(rFonts)
    
    if bold:
        rPr.append(make_element("b"))
        rPr.append(make_element("bCs"))
    if italic:
        rPr.append(make_element("i"))
        rPr.append(make_element("iCs"))
    if font_size:
        rPr.append(make_element("sz", {"w:val": str(font_size)}))
        rPr.append(make_element("szCs", {"w:val": str(font_size)}))
    if color:
        rPr.append(make_element("color", {"w:val": color}))
        
    r.append(rPr)
    t = make_element("t")
    t.text = text
    if text.startswith(" ") or text.endswith(" "):
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    p.append(r)
    return r

def add_page_break(p):
    r = make_element("r")
    br = make_element("br", {"w:type": "page"})
    r.append(br)
    p.append(r)

def create_table(rows_data, col_widths=None):
    tbl = make_element("tbl")
    tblPr = make_element("tblPr")
    
    # Table borders
    tblBorders = make_element("tblBorders")
    for border_name in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        b = make_element(border_name, {"w:val": "single", "w:sz": "4", "w:space": "0", "w:color": "CCCCCC"})
        tblBorders.append(b)
    tblPr.append(tblBorders)
    
    # Centered table
    tblPr.append(make_element("jc", {"w:val": "center"}))
    tblPr.append(make_element("tblW", {"w:w": "9600", "w:type": "dxa"}))
    tbl.append(tblPr)
    
    if col_widths:
        tblGrid = make_element("tblGrid")
        for w in col_widths:
            tblGrid.append(make_element("gridCol", {"w:w": str(w)}))
        tbl.append(tblGrid)
        
    for r_idx, row in enumerate(rows_data):
        tr = make_element("tr")
        trPr = make_element("trPr")
        if r_idx == 0:
            trPr.append(make_element("tblHeader"))
        tr.append(trPr)
        
        for c_idx, cell_text in enumerate(row):
            tc = make_element("tc")
            tcPr = make_element("tcPr")
            if col_widths and c_idx < len(col_widths):
                tcPr.append(make_element("tcW", {"w:w": str(col_widths[c_idx]), "w:type": "dxa"}))
            if r_idx == 0:
                shd = make_element("shd", {"w:val": "clear", "w:color": "auto", "w:fill": "EAEAEA"})
                tcPr.append(shd)
            tc.append(tcPr)
            
            p = make_element("p")
            add_pPr(p, jc="left" if c_idx == 1 or c_idx == 0 else "center", spacing_after=60, line="240")
            is_bold = (r_idx == 0)
            add_run(p, str(cell_text), bold=is_bold, font_size=20, font_family="Times New Roman")
            tc.append(p)
            tr.append(tc)
        tbl.append(tr)
    return tbl

def build_dissertation_document():
    # Read text from markdown
    md_path = PROJECT_ROOT / "outputs" / "reports" / "thesis_weather_model_integration" / "FULL_DISSERTATION_TEXT_INTEGRATED.md"
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    doc = make_element("document")
    body = make_element("body")
    doc.append(body)

    # 1. Title Page
    p_title = make_element("p")
    add_pPr(p_title, jc="center", spacing_after=200, line="360")
    add_run(p_title, "SHRI RAMDEOBABA COLLEGE OF ENGINEERING & MANAGEMENT, NAGPUR\n", bold=True, font_size=28)
    add_run(p_title, "DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING\n", bold=True, font_size=24)
    add_run(p_title, "SESSION 2026–2027 | B.TECH (HONOURS) MAJOR PROJECT\n\n\n", bold=False, font_size=22, italic=True)
    body.append(p_title)

    p_proj = make_element("p")
    add_pPr(p_proj, jc="center", spacing_after=300, line="400")
    add_run(p_proj, "MULTIMODAL AI PLATFORM FOR CONVECTIVE NOWCASTING AND LANDSLIDE RISK PREDICTION\n", bold=True, font_size=32, color="1A365D")
    body.append(p_proj)

    p_by = make_element("p")
    add_pPr(p_by, jc="center", spacing_after=300, line="280")
    add_run(p_by, "Submitted in partial fulfillment of the requirements for the award of degree of\n", italic=True, font_size=22)
    add_run(p_by, "Bachelor of Technology (Honours)\n\n", bold=True, font_size=24)
    add_run(p_by, "Submitted by:\n", bold=True, font_size=22)
    add_run(p_by, "Nilam Bhojwani (Roll No. AIML-B-33)\n", bold=False, font_size=22)
    add_run(p_by, "Moli Agrawal (Roll No. CSE-B-35)\n", bold=False, font_size=22)
    add_run(p_by, "Grisha Lala (Roll No. CSE-A-53)\n", bold=False, font_size=22)
    add_run(p_by, "Shloak Nioding (Roll No. CSE-C-34)\n\n", bold=False, font_size=22)
    add_run(p_by, "Under the Guidance of:\n", bold=True, font_size=22)
    add_run(p_by, "Dr. Amit Pimpalkar, Assistant Professor\n", bold=True, font_size=24)
    add_run(p_by, "Department of Computer Science & Engineering\n", font_size=22)
    add_page_break(p_by)
    body.append(p_by)

    # 2. Certificate Page
    p_cert = make_element("p")
    add_pPr(p_cert, jc="center", spacing_after=200)
    add_run(p_cert, "CERTIFICATE\n", bold=True, font_size=28)
    body.append(p_cert)

    p_cert_text = make_element("p")
    add_pPr(p_cert_text, jc="both", spacing_after=200, line="300")
    add_run(p_cert_text, "This is to certify that the project work entitled \"Multimodal AI Platform for Convective Nowcasting and Landslide Risk Prediction\" is a bona fide work carried out by Nilam Bhojwani, Moli Agrawal, Grisha Lala, and Shloak Nioding in partial fulfillment of the requirements for the award of Bachelor of Technology (Honours) in Computer Science & Engineering / AI & ML from Shri Ramdeobaba College of Engineering and Management, Nagpur, during the academic session 2026–2027.\n\n\n\n", font_size=22)
    body.append(p_cert_text)

    p_sig = make_element("p")
    add_pPr(p_sig, jc="both", spacing_after=200)
    add_run(p_sig, "Dr. Amit Pimpalkar                             Head of Department\n", bold=True, font_size=22)
    add_run(p_sig, "Project Guide                                 Department of CSE\n", font_size=20)
    add_page_break(p_sig)
    body.append(p_sig)

    # 3. Parse and Insert Markdown Sections
    lines = md_text.split("\n")
    i = 0
    in_table = False
    table_lines = []

    while i < len(lines):
        line = lines[i].strip()
        
        # Table detection
        if line.startswith("|") and line.endswith("|"):
            in_table = True
            table_lines.append(line)
            i += 1
            continue
        elif in_table:
            # Table ended, process it
            in_table = False
            # Parse table_lines
            rows = []
            for t_line in table_lines:
                # Skip divider line like | :--- | :---: |
                if re.match(r"^\|[\s\:\-\|]+$", t_line):
                    continue
                cells = [c.strip() for c in t_line.split("|")[1:-1]]
                # clean cells of markdown bold/latex
                clean_cells = []
                for c in cells:
                    c_clean = c.replace("**", "").replace("$\\mathbf{", "").replace("}$", "").replace("$", "")
                    c_clean = c_clean.replace("\\pm", "+/-").replace("\\ge", ">=").replace("\\le", "<=")
                    c_clean = c_clean.replace("\\Delta", "Delta").replace("\\text{", "").replace("}", "")
                    c_clean = c_clean.replace("<br>", " ")
                    clean_cells.append(c_clean)
                rows.append(clean_cells)
            if rows:
                col_count = len(rows[0])
                widths = [int(9600 / col_count)] * col_count
                tbl_el = create_table(rows, widths)
                body.append(tbl_el)
                # Add spacer paragraph after table
                p_sp = make_element("p")
                add_pPr(p_sp, spacing_after=120)
                body.append(p_sp)
            table_lines = []

        # Heading 1
        if line.startswith("# ") and not line.startswith("## "):
            h_text = line[2:].strip()
            # If title already done, treat as major section
            p_h1 = make_element("p")
            add_pPr(p_h1, jc="left", spacing_after=160, line="300")
            add_run(p_h1, h_text, bold=True, font_size=28, color="1A365D")
            body.append(p_h1)
        # Heading 2 / Chapter
        elif line.startswith("## "):
            h_text = line[3:].strip()
            p_h2 = make_element("p")
            # If chapter heading, page break before
            if "CHAPTER" in h_text or "ABSTRACT" in h_text or "REFERENCES" in h_text:
                add_page_break(p_h2)
            add_pPr(p_h2, jc="left", spacing_after=140, line="280")
            add_run(p_h2, h_text, bold=True, font_size=24, color="2B6CB0")
            body.append(p_h2)
        # Heading 3
        elif line.startswith("### "):
            h_text = line[4:].strip()
            p_h3 = make_element("p")
            add_pPr(p_h3, jc="left", spacing_after=100, line="260")
            add_run(p_h3, h_text, bold=True, font_size=22, color="2D3748")
            body.append(p_h3)
        # Code block
        elif line.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            p_code = make_element("p")
            add_pPr(p_code, jc="left", spacing_after=100, line="240")
            add_run(p_code, "\n".join(code_lines), font_size=18, font_family="Courier New", color="333333")
            body.append(p_code)
        # Paragraph text
        elif line:
            # Clean math and markdown formatting
            p_text = line
            p_text = p_text.replace("**", "").replace("$\\mathbf{", "").replace("}$", "").replace("$", "")
            p_text = p_text.replace("\\pm", "+/-").replace("\\ge", ">=").replace("\\le", "<=")
            p_text = p_text.replace("\\Delta", "Delta").replace("\\text{", "").replace("}", "")
            p_text = p_text.replace("\\approx", "~").replace("\\times", "x").replace("\\in", "in")
            p_text = p_text.replace("\\mathbb{E}", "E").replace("\\mathbb{I}", "I")
            p_text = p_text.replace("\\mathcal{L}", "L").replace("\\sigma", "sigma")
            p_text = p_text.replace("\\alpha", "alpha").replace("\\lambda", "lambda").replace("\\tau", "tau")
            
            p_body = make_element("p")
            add_pPr(p_body, jc="both", spacing_after=120, line="276")
            add_run(p_body, p_text, font_size=22, font_family="Times New Roman")
            body.append(p_body)
            
        i += 1

    # Section properties: margins 1 inch (1440 dxa)
    sectPr = make_element("sectPr")
    pgSz = make_element("pgSz", {"w:w": "12240", "w:h": "15840"}) # Letter
    pgMar = make_element("pgMar", {"w:top": "1440", "w:right": "1440", "w:bottom": "1440", "w:left": "1440", "w:header": "720", "w:footer": "720", "w:gutter": "0"})
    sectPr.append(pgSz)
    sectPr.append(pgMar)
    body.append(sectPr)

    # Serialize document.xml
    doc_xml_bytes = ET.tostring(doc, encoding="utf-8", xml_declaration=True)

    # Package into DOCX by copying template non-document files
    print(f"Opening template from {TEMPLATE_DOCX}...")
    with zipfile.ZipFile(TEMPLATE_DOCX, "r") as zin:
        with zipfile.ZipFile(OUT_DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == "word/document.xml":
                    zout.writestr("word/document.xml", doc_xml_bytes)
                else:
                    zout.writestr(item, zin.read(item.filename))

    print(f"Successfully assembled polished dissertation DOCX at: {OUT_DOCX}")
    print(f"File size: {OUT_DOCX.stat().st_size} bytes")

if __name__ == "__main__":
    build_dissertation_document()
