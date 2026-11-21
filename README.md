# Bete Engine

Motor geometric determinist pentru detectarea bețelor, cu două pagini de analiză locală.

## Instalare

Necesită Python 3.10+.

\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
\`\`\`

## Pornește interfața web locală

Din directorul repo-ului:

\`\`\`bash
python bete_local.py
\`\`\`

Serverul pornește la \`http://127.0.0.1:8765\` și încearcă să deschidă browserul implicit. Dacă pagina nu se deschide automat, copiază adresa în Safari.

Oprește serverul cu \`Ctrl+C\`.

## Cele două pagini

### Pagina 1 — motorul nostru

Rulează \`bete_engine.py\`. Reunește fragmente consecutive din același traseu atunci când forma combinată rămâne dreaptă sau ușor curbată. Nu mai respinge segmentele pe baza raportului lungime/grosime; fragmentele liniare foarte scurte sunt păstrate ca ambigue.

### Pagina 2 — codul atașat

Rulează \`bete_engine_page2.py\`. Fișierul este copiat exact din codul furnizat, fără modificări. Aplică extracția în Faza 1 și clasificarea relațională în Faza 2.

Ambele pagini folosesc aceeași fotografie selectată în interfață, dar fiecare analiză are propriul director de rezultate și propriul JSON.

## Utilizare

1. Alege Pagina 1 sau Pagina 2.
2. Încarcă o fotografie PNG/JPG.
3. Opțional, introdu calibrarea în pixeli/cm.
4. Apasă butonul de rulare.
5. Inspectează imaginile de debug și segmentele/bețele extrase.
6. Descarcă JSON-ul paginii respective.

## Linie de comandă

Motorul Paginii 1:

\`\`\`bash
python bete_engine.py fotografie.jpg --out-dir debug
\`\`\`

Motorul Paginii 2:

\`\`\`bash
python bete_engine_page2.py fotografie.jpg --out-dir debug
\`\`\`

## Date locale

Interfața folosește exclusiv serverul Python pornit pe calculator. Fotografiile, imaginile de debug și rezultatele sunt păstrate în \`.bete_local/runs/\`, folder ignorat de Git. Nu există upload către servicii externe.

Limita implicită este de 25 MB per fotografie. Serverul ascultă implicit numai pe \`127.0.0.1\`.
