# Revisione del codice e delle responsabilità

Revisione del 2026-10-13. La scansione strutturale copre i 233 moduli Python di
produzione e comprende funzioni estese, accessi con underscore, descrizioni dei
tool, configurazione e controlli statici. La lettura manuale approfondita riguarda
file processing, protocollo interattivo, tool, memoria, Bash e subagenti. Sono
state preservate le modifiche già presenti e integrate quelle comparse durante
la revisione, inclusa la disambiguazione dei separatori tramite snapshot.

## Correzioni e responsabilità

| Problema | Soluzione |
| --- | --- |
| Il servizio confrontava la rappresentazione completa degli snapshot degli alias | `EditApproval.shares_snapshot()` usa `FileSnapshot.matches()`, testo e codifica; percorso simbolico e access time non invalidano lo stesso contenuto e la stessa identità |
| La preparazione degli edit ricostruiva i dettagli dell'approvazione | Validazione, preparazione della sostituzione e commit passano dai metodi pubblici di `EditApproval` |
| Lettura e cancellazione ripetevano il confronto di inode e timestamp | `FileIdentity` possiede il confronto dei metadati stabili |
| Delete non controllava le capacità richieste della piattaforma | La stessa verifica delle operazioni relative a descriptor è condivisa con create/edit, selezionando le operazioni effettivamente necessarie |
| Il selettore del testo decideva se un file esistesse | La selezione lavora soltanto sul testo; esistenza e autorizzazione appartengono al filesystem e al servizio di approvazione |
| Il commit conservava un ramo di cancellazione senza chiamanti | Il commit pubblica soltanto contenuto; la cancellazione resta in `delete_file()` |
| Il parser combinava tutte le fasi in un metodo | `EditDocument` separa validazione, prenotazione degli ID e scelta del separatore, preservando i corpi letterali e la disambiguazione autorizzata |
| Il contratto modificava direttamente i campi delle call autorizzate | `_AuthorizedEditCall.with_snapshot()` possiede l'aggiornamento dello stato immutabile |
| Il minimo di new-file era ripetuto e l'annotazione Pydantic non compariva nello schema | Un unico limite configurato alimenta runtime, schema supportato dal mapper, descrizione e istruzioni della skill |
| La rinomina del tool dei subagenti lasciava il nome originale nei messaggi di errore | Il messaggio usa `get_name()`, verificato tramite l'API pubblica `with_name()` |
| Descrizioni, guidance, avvisi e diagnostica erano sparsi nel codice operativo | I testi risiedono nei config del rispettivo dominio; i nomi e i limiti dinamici derivano dagli stessi settings usati dal runtime |

## Principi controllati

| Principi | Riscontro |
| --- | --- |
| Responsabilità, moduli e incapsulamento | Modelli di identità, approvazione, framing e stato interattivo possiedono le rispettive operazioni; i servizi chiamano le loro API |
| Chiarezza, nomi e semplicità | Metodi con azioni esplicite, nessuna gerarchia aggiunta; descrizioni statiche sono campi di configurazione |
| Funzioni piccole e livello di astrazione | Il servizio orchestra approvazioni e commit; encoding e limiti della sostituzione restano nell'approvazione; parsing diviso per fase |
| Side effect e separazione comando/query | Confronti e selezioni non modificano stato; pubblicazione, cancellazione e copie dello stato interattivo sono esplicite |
| Duplicazione e configurazione | Identità, prerequisiti, capacità della piattaforma, minimo di creazione e testo condivisi nei rispettivi domini |
| Confini esterni e dipendenze | HTTP e filesystem restano nei loro adapter; non è stata aggiunta dipendenza Pydantic allo schema dei nuovi file |
| Errori e null | Errori applicativi restano distinti dai fallimenti di esecuzione; il contenuto del commit non è più opzionale |
| Codice morto, commenti e formato | Rimossi ramo e parametro obsoleti; Ruff applicato all'intera produzione e ai test; diff privo di errori di whitespace |
| Testabilità e regressioni | Test deterministici per alias, piattaforma, configurazione dei limiti e messaggi, rinomina dei tool e descrizioni coerenti con i nomi configurati |
| Cambiamenti contenuti e prestazioni | Refactoring dei confini osservati e spostamento dei testi; nessuna ottimizzazione senza misure |

Gli accessi con underscore rimasti nella scansione sono interni alla stessa
classe, alle copie che essa costruisce, o agli hook di ereditarietà. Non sono
stati introdotti accessi del chiamante allo stato privato di altri servizi.
Le funzioni estese nei dispatcher di tipi, nel crawler e nel DSL formale sono
state valutate nel loro contesto: la lunghezza da sola non giustifica un'altra
classe. La revisione non certifica automaticamente ogni scelta architetturale
di ogni modulo.

Il vocabolario dei protocolli, le chiavi dei risultati e gli input letterali dei
test restano distinti dal testo di presentazione configurabile. Framing e
behavior condivisi dell'editing sono centralizzati in `FILE_EDIT_FORMAT`.
Risultati strutturati e metadata continuano a usare XML; gli argomenti del
provider e gli export della history mantengono JSON.

## Validazione

- Suite completa: **2635 test non live superati** con `pytest -n 4 -q -m 'not live'`.
- Dopo l'ultima correzione del narrowing di tipo: **495 test mirati superati**.
- Ruff lint e formato: superati sull'intero `src` e `tests`.
- Pyright: nessun errore o warning.
- `git diff --check`: superato.

Le regressioni su piattaforma, alias e nome registrato riproducevano i rispettivi
fallimenti prima delle correzioni. La regressione sul limite di creazione ha
individuato anche l'assenza di `minLength` nello schema precedente.
Nessun test live è stato eseguito; il comportamento del server remoto non è
stato verificato in questa revisione.
