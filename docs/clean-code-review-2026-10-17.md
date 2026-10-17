# Revisione di new-file e dei confini condivisi

La scansione strutturale copre i 269 moduli Python di produzione: dimensioni
delle funzioni, accessi protetti e struttura dei moduli. La lettura manuale
approfondisce creazione/modifica/cancellazione, parsing, prerequisiti di lettura,
tool e hook, mapper dei provider, cache e registrazione degli output.
La revisione non equivale a una certificazione di ogni percorso eseguibile
dell'intera repository.

## Problemi corretti

| Problema osservato | Correzione e responsabilità |
| --- | --- |
| Gli adapter new-file e delete-file orchestrano direttamente parsing, filesystem e risultati parziali | I servizi `create_resources()` e `delete_resources()` possiedono i batch; gli adapter delegano mantenendo firma e ciclo degli hook |
| Il parser di creazione mescola limiti del documento e parsing dei singoli blocchi | `CreationDocument` separa validazione della sorgente, confini e parsing degli elementi; i corpi rimangono letterali |
| Creazione, modifica e cancellazione ripetono il confronto dell'identità del genitore aperto | Un'unica query nel modulo filesystem confronta directory corrente e descriptor; ogni operazione mantiene il proprio errore e il proprio lock |
| Una creazione concorrente di file produce `filesystem_error`, mentre una directory produce `file_already_exists` | Il mapper condiviso riconosce `FileExistsError`; eliminato l'adattamento speciale di mkdir |
| Un percorso con surrogate produce errori diversi o avvia un'ispezione inutile | `validate_file_path()` rifiuta il percorso con `invalid_path` prima del filesystem e del recupero automatico in tutti i tool |
| Dopo il rifiuto, gli stessi argomenti invalidi fanno fallire l'export della history e i mapper dei provider | `ToolCall.arguments_json()` possiede la serializzazione degli argomenti; entrambi i mapper usano l'API pubblica. L'export usa lo stesso encoder JSON con escape Unicode e conserva anche chiavi annidate invalide |
| Il contenuto invalido di un nuovo file suggerisce di cambiare la codifica di un file esistente | La validazione/codifica della creazione produce `invalid_content`; gli errori di codifica degli edit conservano il loro significato |
| Gli errori sui limiti del documento non indicano il limite o come recuperare | Riportano il massimo configurato, confermano l'assenza di creazioni e suggeriscono batch più piccoli |
| Il nuovo test accede alla coda privata del provider fittizio | La coda è gestita tramite `SequenceProvider.queue_responses()` |

Le regressioni su collisioni, percorsi e serializzazione riproducevano i
fallimenti prima delle correzioni. Il test sul contenuto invalido riproduceva
anche il suggerimento di recupero incoerente.

## Principi verificati

| Principi | Riscontro |
| --- | --- |
| Responsabilità, incapsulamento e dipendenze | Framing nei modelli, orchestrazione nei servizi, effetti nel filesystem, serializzazione delle call in ToolCall. I chiamanti usano API pubbliche; nessuna nuova gerarchia |
| Nomi, funzioni piccole, argomenti e livello di astrazione | Operazioni esplicite, servizi separati per create/delete, validazione del contenuto separata dal commit; nessun flag che selezioni operazioni diverse |
| Side effect e separazione comando/query | Parsing e confronti non scrivono; creazione, commit, cancellazione e accodamento delle risposte hanno nomi espliciti |
| Duplicazione e configurazione | Confronto del genitore e classificazione degli errori condivisi; grammatica, limiti e messaggi nei rispettivi config |
| Errori e null | Errori locali non cancellano i successi; gli errori globali precedono le scritture. Esiti tipizzati nel parser, collezioni vuote quando appropriate, nessun silenziamento dei difetti trovati |
| Test, bug fix e refactoring | Regressioni prima delle correzioni, suite completa e live; comportamento pubblico conservato durante gli spostamenti di responsabilità |
| Commenti, codice morto e formattazione | Commenti sulle ragioni delle scelte; rimosse conversioni speciali e verifiche duplicate; formatter, lint e controllo dei tipi |
| Semplicità, cambiamenti minimi e prestazioni | Nessun framework generico per i batch, nessuna classe priva di stato necessaria solo per raggruppare funzioni, nessuna ottimizzazione senza misura |

I metodi lunghi nel crawler, nei dispatcher di tipi e nella logica formale
sono stati valutati rispetto alla loro responsabilità: la sola lunghezza
non giustifica uno spostamento o una gerarchia aggiuntiva. Gli accessi protetti
di produzione trovati dalla scansione appartengono alla stessa classe o alle
copie che costruisce; non è emersa una ragione per cambiarli in questa revisione.

## Contratti e limiti intenzionali

- La nuova firma è `new-file(document)`. La creazione singola resta disponibile
  nell'API filesystem interna; il minimo configurato resta di due caratteri,
  contando anche gli spazi.
- I batch sono ordinati e senza rollback globale; ogni file viene pubblicato
  esclusivamente. Il testo delle directory viene ignorato con warning.
- I prefissi dei marcatori sono riservati anche nei corpi: questo protocollo
  non rappresenta letteralmente quei delimitatori e non è un documento XML.
- I risultati strutturati e gli output recuperati restano XML; argomenti e
  history esportata restano JSON. Gli escape Unicode cambiano la rappresentazione
  delle stringhe JSON, senza modificarne il valore decodificato.
- Percorsi assoluti interni alla workroot restano ammessi per lettura e creazione;
  delete mantiene il proprio contratto più restrittivo di percorsi relativi.
- I lock sono cooperativi: processi esterni possono ancora intervenire dopo
  l'ultimo controllo d'identità. Nessuna transazione globale è stata introdotta.

## Verifica

- Suite completa: **3127 test superati**, 72 casi opzionali saltati, con
  `.venv/bin/pytest -n 4 -q --tb=short`.
- Suite live modificata: **16 test superati** sull'ultima versione con
  `.venv/bin/pytest -n 0 --run-live tests/live/test_file_editing_live.py`.
- Ruff lint: superato su `src` e `tests`.
- Ruff formato: tutti i 475 file conformi; corretto anche un rientro preesistente
  in un test del protocollo interattivo.
- Pyright: nessun errore o warning.
- Diff finale, inclusi i nuovi file: esaminato; `git diff --cached --check` superato.

Il primo tentativo live si fermava prima dei tool perché il provider rifiutava
la connessione. Dopo l'avvio del provider i casi sono passati e sono stati
ripetuti sulla versione definitiva prima del commit.
