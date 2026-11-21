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

### Pagina 1 — extracție nedistructivă

Rulează \`bete_engine.py\`.

Pagina 1 urmărește să detecteze toate segmentele liniare vizibile, fără reguli care aleg un câștigător la forme F/T/X/L și fără să elimine ramurile necoliniare. La fiecare joncțiune, motorul unește doar brațele cu direcții aproape opuse, ca să reconstruiască o linie continuă; orice braț neîmperecheat rămâne băț separat.

Exemple geometrice urmărite:
- F: trunchiul și cele două bare sunt păstrate ca trei bețe.
- f: tija și bara mică sunt păstrate separat; cârligul curbat rămâne în stratul curbelor.
- X: cele două trasee opuse sunt păstrate ca două bețe, fără verdict de exclusivitate.
- π: bara de sus și cele două picioare sunt păstrate ca trei bețe.

Fragmentele consecutive din același traseu sunt reunite când forma combinată rămâne dreaptă sau ușor curbată. Nu există filtru bazat pe raportul lungime/grosime. Pixelii izolați sunt raportați separat ca pete, iar curbele sunt afișate gri.

### Pagina 2 — codul atașat v3

Rulează \`bete_engine_page2.py\`. Fișierul este copia exactă a codului atașat în conversație, fără editări. Are 1.473 linii și 59.916 bytes UTF-8. Include extracția și clasificarea relațională din sursa originală și generează patru imagini de debug.

## Utilizare

1. Alege Pagina 1 sau Pagina 2.
2. Încarcă o fotografie PNG/JPG.
3. Opțional, introdu calibrarea în pixeli/cm.
4. Apasă butonul de rulare.
5. Inspectează imaginile de debug și bețele/segmentele extrase.
6. Descarcă JSON-ul paginii respective.

## Linie de comandă

Pagina 1:

\`\`\`bash
python bete_engine.py fotografie.jpg --out-dir debug
\`\`\`

Pagina 2:

\`\`\`bash
python bete_engine_page2.py fotografie.jpg --out-dir debug
\`\`\`

## Date locale

Interfața folosește exclusiv serverul Python pornit pe calculator. Fotografiile, imaginile de debug și rezultatele sunt păstrate în \`.bete_local/runs/\`, folder ignorat de Git. Nu există upload către servicii externe.

Limita implicită este de 25 MB per fotografie. Serverul ascultă implicit numai pe \`127.0.0.1\`.
