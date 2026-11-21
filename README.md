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

### Pagina 3 — geometrie pură, Stratul 0 + Stratul 1

Rulează bete_engine_page3.py. Fișierul este copia exactă a noului cod atașat: 1.376 de linii, 51.671 bytes UTF-8.

**Stratul 0 — graniță exactă:**
- estimează masca binară a cernelii;
- extrage buclele contururilor inter-pixel;
- descompune fiecare buclă în segmente digitale drepte (DSS);
- reconstruiește masca exclusiv din DSS și verifică XOR = 0.

**Stratul 1 — partiție geometrică a cernelii:**
- grupează segmentele DSS în pereți;
- caută perechi de pereți opuși pentru a estima axele și lățimea benzilor;
- identifică primitive banda_dreapta, banda_curba și pata_compacta;
- marchează drept jonctiune pixelii revendicați de cel puțin două benzi drepte;
- atribuie fiecare pixel al măștii unei primitive și verifică să nu rămână pixeli neacoperiți sau atribuiți în afara cernelii.

Motorul verifică automat două condiții distincte: reconstrucția DSS (layer0_dss_xor_error_pixels = 0) și acoperirea măștii de către primitive (layer1_uncovered_ink_pixels = 0). Dacă partiția nu este exactă, analiza se oprește cu eroare.

**Limitele verificării:** aceste verificări se referă la masca binară produsă de binarizare. Ele nu dovedesc că binarizarea a capturat fiecare urmă vizibilă în fotografia originală și nici că eticheta geometrică a fiecărei primitive coincide întotdeauna cu intenția de scriere. Acoperirea pixelilor și identificarea semantică a fiecărui băț sunt probleme diferite.

Pagina 3 rulează la rezoluția nativă cu scale=1.0. Câmpul de calibrare pixeli/cm este ascuns pentru această pagină, deoarece scale înseamnă scalarea imaginii, nu calibrare fizică. Produce patru imagini de debug și un JSON cu sumar, primitive și joncțiuni.

## Utilizare

1. Alege Pagina 1, 2 sau 3.
2. Încarcă o fotografie PNG/JPG.
3. Opțional, introdu calibrarea pixeli/cm pentru Paginile 1 și 2.
4. Apasă butonul de rulare.
5. Pe Pagina 3, inspectează cele patru imagini, tabelul primitivelor și verificările celor două straturi.
6. Descarcă JSON-ul analizei respective.

## Linie de comandă

\`\`\`bash
python bete_engine.py fotografie.jpg --out-dir debug
python bete_engine_page2.py fotografie.jpg --out-dir debug
python bete_engine_page3.py fotografie.jpg --out-dir debug
\`\`\`

Pentru Pagina 3 se poate seta opțional scalarea imaginii:

\`\`\`bash
python bete_engine_page3.py fotografie.jpg --out-dir debug_geo --scale 1.0
\`\`\`

## Date locale

Interfața folosește exclusiv serverul Python pornit pe calculator. Fotografiile, imaginile de debug și JSON-urile sunt păstrate în \`.bete_local/runs/\`, folder ignorat de Git. Nu există upload către servicii externe.

Limita implicită este de 25 MB per fotografie. Serverul ascultă implicit numai pe \`127.0.0.1\`.
