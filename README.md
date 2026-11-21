# Bete Engine

Motor geometric determinist pentru detectarea segmentelor desenate de mână și interfață web locală.

## 1. Instalare

Necesită Python 3.10+.

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
\`\`\`

## 2. Pornește interfața locală

Din directorul repo-ului:

\`\`\`bash
python bete_local.py
\`\`\`

Serverul pornește pe \`http://127.0.0.1:8765\` și încearcă să deschidă browserul implicit (Safari pe macOS). Dacă pagina nu se deschide automat, copiază adresa în browser.

Oprește serverul cu \`Ctrl+C\`.

## 3. Folosire

1. Încarcă o fotografie PNG/JPG.
2. Opțional, introdu calibrarea în pixeli/cm.
3. Apasă **Analizează fotografia**.
4. Vezi numărul de segmente certificate, ambigue și respinse.
5. Inspectează cele trei imagini de debug și descarcă JSON-ul.

## 4. Linie de comandă

Motorul poate fi folosit și fără interfață:

\`\`\`bash
python bete_engine.py fotografie.jpg --out-dir debug
python bete_engine.py fotografie.jpg --out-dir debug --px-per-cm 120
\`\`\`

## Date și confidențialitate

Interfața folosește exclusiv serverul Python pornit pe calculator. Fotografiile și rezultatele fiecărei sesiuni sunt păstrate în \`.bete_local/runs/\`, folder ignorat de Git. Nu există upload către un serviciu extern.

Limita implicită pentru o fotografie este 25 MB. Serverul ascultă implicit doar pe \`127.0.0.1\`, nu pe întreaga rețea.
