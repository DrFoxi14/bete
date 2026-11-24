# Bete Engine

Set de trei pagini geometrice deterministe, cu Pagina 3 organizată în trei straturi și interfață web locală.

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

Pagina 3 rulează la rezoluția nativă cu scale=1.0. Câmpul de calibrare pixeli/cm este ascuns pentru această pagină, deoarece scale înseamnă scalarea imaginii, nu calibrare fizică.

### Stratul 2 — topologie și organizare geometrică

Modul separat: geometry_layer2.py. Acesta nu primește fotografia ca intrare analitică și nu recalculează pixeli. Primește exclusiv JSON-ul *_geometry_layer1.json produs de Pagina 3 după Straturile 0 + 1.

**Poarta obligatorie înainte de rulare:**
- layer0_dss_xor_error_pixels trebuie să fie 0;
- layer1_uncovered_ink_pixels trebuie să fie 0;
- guaranteed_100pct_coverage trebuie să fie true;
- lista de primitive și numărul lor trebuie să fie valide.

Dacă oricare condiție nu este îndeplinită, geometry_layer2.py oprește execuția. Serverul local aplică aceeași poartă și nu apelează Stratul 2 înainte ca Straturile 0 + 1 să fi terminat cu succes.

Stratul 2 calculează axa dominantă, grile periodice, bare transversale, poduri paralele, bețe ancorate, molecule geometrice, coridoare și rânduri aliniate. La final verifică dacă fiecare primitive_id din Stratul 1 apare exact o dată: fie într-o singură grilă, fie într-o singură moleculă. Duplicatele sau ID-urile necontabilizate opresc execuția.

Produce trei imagini debug și un JSON separat *_geometry_layer2.json. În interfață, Pagina 3 rulează automat Straturile 0 + 1, validează rezultatul, apoi rulează Stratul 2 și Geometry Layer 3A v5.1. Stratul 3A primește exclusiv JSON-urile validate ale Straturilor 1 și 2 și execută verificările stricte de metrică locală, graf de contacte/proximități și audit count == 1. Sunt disponibile separat descărcarea JSON-urilor Straturilor 1, 2 și 3A. Motorul local este `geometry_layer3a_v51.py`.

Comanda manuală pentru Stratul 2:
python geometry_layer2.py debug_geo/poza_geometry_layer1.json --out-dir debug_l2

Fotografia originală poate fi folosită numai ca fundal vizual al imaginilor debug; deciziile geometrice ale Stratului 2 sunt calculate din primitivele din JSON. Câmpul de calibrare pixeli/cm este ascuns pentru această pagină, deoarece scale înseamnă scalarea imaginii, nu calibrare fizică. Produce patru imagini de debug și un JSON cu sumar, primitive și joncțiuni.

## Utilizare

1. Alege Pagina 1, 2 sau 3.
2. Încarcă o fotografie PNG/JPG.
3. Opțional, introdu calibrarea pixeli/cm pentru Paginile 1 și 2.
4. Apasă butonul de rulare.
5. Pe Pagina 3, inspectează cele șapte imagini, primitivele din Stratul 1 și rapoartele topologice din Stratul 2.
6. Descarcă JSON-ul analizei respective.

## Linie de comandă

\`\`\`bash
python bete_engine.py fotografie.jpg --out-dir debug
python bete_engine_page2.py fotografie.jpg --out-dir debug
python bete_engine_page3.py fotografie.jpg --out-dir debug_geo
python geometry_layer2.py debug_geo/fotografie_geometry_layer1.json --out-dir debug_l2
\`\`\`

Pentru Pagina 3 se poate seta opțional scalarea imaginii:

\`\`\`bash
python bete_engine_page3.py fotografie.jpg --out-dir debug_geo --scale 1.0
\`\`\`

## Date locale

Interfața folosește exclusiv serverul Python pornit pe calculator. Fotografiile, imaginile de debug și JSON-urile sunt păstrate în \`.bete_local/runs/\`, folder ignorat de Git. Nu există upload către servicii externe.

Limita implicită este de 25 MB per fotografie. Serverul ascultă implicit numai pe \`127.0.0.1\`.
