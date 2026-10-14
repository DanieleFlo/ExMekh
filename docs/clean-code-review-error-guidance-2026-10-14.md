# Revisione clean code dei messaggi di errore — 14 ottobre 2026

La revisione copre il diff del task sui messaggi di errore e un controllo dei
255 moduli Python di produzione per accessi allo stato privato di altri oggetti
e funzioni estese. Sono stati riletti anche orchestrazione, hook, cache, accesso
ai file, renderer XML e adapter dei provider. Il controllo automatico orienta
la lettura; non dimostra che ogni possibile comportamento della repository sia
corretto o che tutti i principi siano soddisfatti in ogni modulo.

## Responsabilità e incapsulamento

| Componente | Responsabilità verificata o corretta |
| --- | --- |
| `ToolDiagnosticError` | Possiede causa, suggerimento e contesto; espone proprietà in sola lettura, snapshot e `add_details()` per registrare osservazioni. |
| `ToolArgumentError` | Costruisce le diagnostiche di binding e preserva una causa già identificata quando aggiunge il contesto dell'invocazione. |
| `Tool` | Valida la firma, applica gli hook e isola gli input dell'invocazione da quelli del chiamante e dalle mutazioni della funzione. |
| `ToolCallExecutor` | Orchestra l'esecuzione e converte gli errori al confine runtime; distingue la validazione delle opzioni dall'avvio del manager. |
| `WebNavigationError` | Classifica lo stato HTTP osservato senza ricavarlo dal testo del provider. |
| `_BrowserCapture` | Possiede policy di navigazione, errore delle azioni e pagina osservata; espone controllo degli errori e copie degli esiti. |
| Lifecycle del crawler | Avvia e chiude il browser, conservando l'errore primario e le osservazioni anche quando falliscono cattura o pulizia. |
| `workspace.resolve_work_root()` | Applica la stessa validazione della workroot a ispezione e scrittura. |
| Renderer XML condiviso | Converte i risultati strutturati al confine di presentazione senza cambiare record e protocolli JSON. |

I chiamanti usano le operazioni esposte dalle classi diagnostiche; non modificano
il dizionario interno dell'errore. Il catalogo dei suggerimenti è una mappa in
sola lettura. I dettagli sono copiati sia in ingresso sia in uscita, anche per
liste e dizionari annidati. Gli accessi privati rilevati nell'audit generale erano
interni alla stessa classe, chiamate a `super()` o API specifiche già previste;
non sono stati riscritti indiscriminatamente.

## Comportamenti illogici corretti

- Un `TypeError` o `ValueError` nell'avvio del manager non viene più attribuito
  agli argomenti del modello. Anche limite di capacità e chiamata già attiva
  mantengono la famiglia di errore di esecuzione.
- Un codice diagnostico sconosciuto riceve indicazioni generali sul fallimento,
  senza sostenere che l'input sia sbagliato.
- Gli hook e le funzioni non modificano più gli argomenti annidati del chiamante.
  Gli input vengono congelati alla creazione dell'operazione; l'hook successivo
  e il record conservano quelli effettivamente passati, prima delle mutazioni
  della funzione. `execute()` diretto mantiene la normale semantica Python.
- Un HTTP 404, 401/403, 429 o 5xx viene riconosciuto anche se il crawler dichiara
  successo. Gli errori successivi non cancellano l'errore dell'azione né gli
  effetti già osservati. Una chiusura fallita conserva la causa primaria;
  se è l'unico problema viene segnalata esplicitamente.
- `sqrt(4, 2)` e altre chiamate con arità errata vengono rifiutate. Il secondo
  argomento non può diventare involontariamente un'opzione interna di SymPy;
  solo `log` accetta una base opzionale.
- La workroot assente o non valida produce lo stesso codice nei percorsi di
  lettura e scrittura, evitando un suggerimento di creazione del file errato.
- Il test del timeout del trasporto interattivo usa un orologio controllato:
  il ramo verificato non dipende più dalla velocità del computer.

## Verifica dei principi

| Principi | Esito della revisione |
| --- | --- |
| Correttezza, comprensione prima della modifica, bug fix | I problemi sono stati riprodotti con test di regressione prima delle correzioni; sono stati controllati i confini di propagazione e gli effetti parziali. |
| Responsabilità singola, incapsulamento, dipendenze e confini esterni | La causa nasce nel dominio che la rileva, il runtime orchestra e il renderer presenta. Le classi espongono operazioni per i propri dati. SDK e filesystem restano nei rispettivi confini. |
| Nomi, funzioni piccole, livello di astrazione e argomenti | I nuovi metodi descrivono l'operazione. Classificazione HTTP, binding e chiusura sono separati dall'orchestrazione. Le firme pubbliche dei tool continuano a rappresentare lo schema del provider. |
| Effetti espliciti e separazione comandi/query | `add_details()` registra contesto; le proprietà restituiscono copie. Commit parziali, azioni web e fallimenti di consegna conservano il loro stato e indicano come recuperarlo. |
| Duplicazione, semplicità e cambiamenti minimi | Validazione della workroot e diagnostiche sono condivise. I refactoring fuori dal diff riguardano bug concreti; non sono state aggiunte gerarchie o adapter senza necessità. |
| Gestione degli errori e valori opzionali | Le famiglie compatibili sono mantenute, le cause sono esplicite e i dettagli assenti sono omessi. Una pulizia fallita viene registrata, senza sostituire l'errore primario. |
| Commenti, codice morto, formattazione e coerenza | Sono stati rimossi suggerimenti inutilizzati e corretta la docstring che definiva atomiche le azioni web. Testi e formati seguono config e renderer esistenti. |
| Test, refactoring sicuro e revisione finale | Sono stati aggiunti test mirati per mutazioni annidate, classificazione, effetti e lifecycle; eseguita l'intera suite, oltre ai controlli statici e al diff. |

Le funzioni più lunghe individuate nel controllo generale non richiedevano tutte
un'estrazione: dispatcher coerenti e confini del lifecycle possono mantenere
una responsabilità unica. La lunghezza da sola non giustifica nuovi livelli di
indirezione. Le eccezioni intenzionali di formato e comportamento sono descritte
in [Tool Error Guidance](17-tool-error-guidance.md).

## Validazione e limiti

- Suite completa: **2874 test passati, 55 saltati**, con copertura di istruzioni
  e rami al **100%** nel perimetro configurato (`main.py` è escluso).
- Ruff lint e format, Pyright e `git diff --check` passati.
- Build offline di wheel e archivio sorgenti riuscita. Il comando segnala
  il range del backend `uv_build` rispetto alla versione di `uv` installata;
  il backend configurato completa entrambe le build.
- Test eseguiti in WSL/Linux. Test live e codice nativo Windows non eseguiti;
  browser e provider sono verificati tramite fixture controllate.

Le due modifiche ai test già presenti prima del task sono conservate e lasciate
fuori dal commit. Il file `todo.md`, ignorato da Git, resta aggiornato localmente.
