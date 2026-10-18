# Revisione degli schemi dei tool e delle valutazioni

La scansione strutturale copre i 275 moduli Python di produzione: dimensioni
delle funzioni e accessi protetti. La lettura manuale approfondisce il diff,
schemi e conversione degli argomenti, caricamento delle capability, cataloghi,
scope e ritorni delle skill, provider, valutatore e test collegati.
Non equivale a una verifica esaustiva di ogni comportamento della repository.

## Problemi corretti

| Problema | Correzione e responsabilità |
| --- | --- |
| La costruzione degli schemi condivisa con i cataloghi apparteneva a un modulo specifico di Chat Completions | `AnnotationSchemaBuilder` risiede nei modelli; anche i relativi diagnostici appartengono al dominio dei modelli. Import aggiornati direttamente, senza moduli di compatibilità |
| Le descrizioni generali del catalogo usavano XML manuale, quelle dei parametri il renderer condiviso | `SkillCatalog` costruisce un solo albero XML e usa il renderer condiviso. Carriage return e caratteri non rappresentabili sono preservati in entrambe le sezioni |
| Il mapper degli output strutturati modificava anche oggetti letterali in default, esempi, enum, costanti ed estensioni | La visita dei soli nodi di schema è condivisa in `json_schema`; il mapper mantiene la responsabilità delle regole strict del provider |
| Due schemi costruiti dalle stesse annotazioni condividevano metadati annidati mutabili | Il builder copia i metadati; il chiamante può modificare il risultato senza modificare annotazioni e altri schemi |
| Il nome `has_parameter_descriptions` confondeva disponibilità dei metadati e visibilità verso il provider | `exposes_parameter_descriptions` descrive la politica effettiva; le copie conservano firma, annotazioni e comportamento |
| Il controllo del catalogo completo ricostruiva manualmente il rendering della skill nel benchmark | Entrambe le condizioni usano `attach_catalog` e `render_instructions`; il benchmark sceglie soltanto dove mettere la documentazione |
| Un output `.jsonl` poteva sovrapporre report XML e traccia API | Il runner richiede un percorso `.xml` prima di creare file o contattare il provider |
| Un provider che falliva la verifica iniziale lasciava aperto il client | `LocalProvider` gestisce il proprio ciclo di vita, espone `close` e supporta il context manager; runner e nuovo test live usano questa interfaccia |
| Un intero troppo grande in un campo float interrompeva il valutatore | Il confronto assegna un check fallito senza interrompere la misura |

Le regressioni su XML, valori letterali degli schemi, isolamento dei metadati,
collisione dei report, chiusura dopo inizializzazione fallita e overflow del
punteggio sono state riprodotte prima delle correzioni.

## Principi verificati

| Principi | Riscontro |
| --- | --- |
| Responsabilità, dipendenze e incapsulamento | Annotazioni nei modelli, politica di visibilità in `Tool`, composizione nel caricatore, XML nel catalogo e nel renderer, regole strict nei mapper, connessioni nel provider. I chiamanti usano le API pubbliche |
| Nomi, funzioni e livello di astrazione | Operazioni esplicite per copie, visita dei nodi, allegato dei cataloghi e chiusura delle risorse. Nessun nuovo flag che selezioni operazioni diverse |
| Side effect e comando/query | Estrazione e rimozione delle descrizioni non modificano la sorgente; allegato dei cataloghi e scrittura dei report sono operazioni esplicite |
| Duplicazione | Stesso builder di annotazioni, stessa visita dei nodi e stesso renderer XML; nessuna duplicazione delle regole strict nel catalogo |
| Errori e null | Errori di trasporto separati dagli errori del modello; null di business preservati, null delle opzioni runtime equivalenti ai default documentati; campi mancanti non riparati dal valutatore |
| Test e refactoring | Test deterministici separati dai 14 scenari di misura; suite completa e live; refactoring guidati da difetti riproducibili |
| Semplicità e cambiamenti minimi | Nessuna gerarchia nuova, nessun file di soli alias; i dispatcher più lunghi mantengono una responsabilità coerente e non vengono suddivisi solo per ridurne la lunghezza |

Gli accessi protetti individuati dalla scansione appartengono alla stessa classe,
alle sue copie o ai suoi override; `_replace` è l'API pubblica delle tuple URL.
Non sono emersi accessi di produzione che richiedessero nuovi getter per
nascondere l'implementazione di un'altra classe.

## Contratti e limiti

- Il benchmark usa deliberatamente gli argomenti grezzi dell'SDK: la normale
  chiamata del provider converte alcuni valori prima dell'esecuzione e
  nasconderebbe errori di compilazione. Questa scelta resta confinata al runner.
- Report e progressi strutturati usano XML; le tracce API conservano JSON.
- L'unificazione del renderer cambia l'indentazione dei cataloghi e la forma
  degli elementi vuoti. Le precedenti misure sono conservate come evidenza
  storica, distinguendole dalle esecuzioni della presentazione corrente.
- Il workflow esplorativo che riapre una skill già completata rimane un problema
  osservato: si riproduce anche ripristinando le descrizioni complete. Non è
  stata identificata una causa nel codice tale da giustificare una correzione
  del routing. Il test live mirato non misura questo comportamento.
- Una ripetizione dei 14 casi non dimostra equivalenza di qualità né una
  differenza statisticamente stabilita fra le rappresentazioni.

## Verifica

- Suite offline completa: **3340 test superati**, 74 casi opzionali saltati.
- Live: **3 test superati**; compilazione da catalogo per Responses e Chat
  Completions, più conservazione di fatti con output strutturato.
- Ruff lint e formato: superati su `src`, `tests` ed `evaluations`.
- Pyright: nessun errore o warning su produzione ed `evaluations`.
- Benchmark sulla versione finale: **28 richieste completate**, 14 scenari in
  entrambe le condizioni, nessun errore di infrastruttura. Punteggi medi
  **97,80/100** con descrizioni negli schemi e **96,02/100** con documentazione
  nelle skill; rispettivamente 13/14 e 12/14 call esatte. Esiti, limiti e
  [report completo](test-evidence/tool-argument-compilation-reviewed-2026-10-18.xml)
  restano consultabili senza interpretare il risultato come soglia CI.
- Diff finale: verificato, inclusi nuovi file e migrazione degli import.
