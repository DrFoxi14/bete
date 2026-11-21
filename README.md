# Bete Engine

Set de trei motoare geometrice deterministe, cu interfață web locală.

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

## Cele trei pagini

### Pagina 1 — extracție nedistructivă

Rulează \`bete_engine.py\`. Păstrează brațele liniare la joncțiuni, fără reguli de câștigător F/T/X/L și fără filtrul lungime/grosime. Reunește numai brațele geometrice aproape coliniare; brațele neîmperecheate rămân independente.

### Pagina 2 — codul atașat v3

Rulează \`bete_engine_page2.py\`. Fișierul este copia exactă a codului v3 atașat anterior: 1.473 linii, 59.916 bytes UTF-8. Include extracție și clasificare relațională în două faze.

### Pagina 3 — extractor lossless DSS

Rulează \`bete_engine_page3.py\`. Fișierul este copia exactă a ultimului cod atașat: 560 de linii, 19.207 bytes UTF-8.

Pagina 3:
- estimează masca binară a cernelii;
- extrage contururile inter-pixel în bucle exterioare și interioare;
- descompune contururile în segmente digitale drepte (DSS);
- reconstruiește masca exclusiv din DSS;
- calculează XOR între masca inițială și reconstrucție și oprește analiza dacă există pixeli diferiți.

**Interpretarea XOR:** XOR = 0 certifică faptul că lista DSS reconstruiește exact masca binară produsă de binarizare. Nu certifică faptul că binarizarea a detectat absolut toată cerneala vizibilă în fotografia originală. Calitatea măștii și acoperirea reconstrucției sunt două verificări diferite.

Pagina 3 rulează la rezoluția nativă cu \`scale=1.0\`. Câmpul de calibrare pixeli/cm este ascuns pentru această pagină, deoarece parametrul \`scale\` al motorului înseamnă scalarea imaginii, nu calibrare fizică.

## Utilizare

1. Alege Pagina 1, 2 sau 3.
2. Încarcă o fotografie PNG/JPG.
3. Opțional, introdu calibrarea pixeli/cm pentru Paginile 1 și 2.
4. Apasă butonul de rulare.
5. Inspectează imaginile de debug și tabelul rezultat.
6. Descarcă JSON-ul analizei respective.

## Linie de comandă

\`\`\`bash
python bete_engine.py fotografie.jpg --out-dir debug
python bete_engine_page2.py fotografie.jpg --out-dir debug
python bete_engine_page3.py fotografie.jpg --out-dir debug
\`\`\`

Pentru Pagina 3 se poate seta opțional scalarea imaginii:

\`\`\`bash
python bete_engine_page3.py fotografie.jpg --out-dir debug_dss --scale 1.0
\`\`\`

## Date locale

Interfața folosește exclusiv serverul Python pornit pe calculator. Fotografiile, imaginile de debug și JSON-urile sunt păstrate în \`.bete_local/runs/\`, folder ignorat de Git. Nu există upload către servicii externe.

Limita implicită este de 25 MB per fotografie. Serverul ascultă implicit numai pe \`127.0.0.1\`.
