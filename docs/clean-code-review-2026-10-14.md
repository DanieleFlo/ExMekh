# Revisione del codice del 14 ottobre 2026

La revisione ha incluso il percorso dei grandi output e un audit dei sorgenti
Python dell'intera repository. Sono stati conservati protocolli e firme pubbliche;
i cambiamenti estesi fuori dal runtime riguardano soprattutto la collocazione dei
testi nei config del relativo dominio.

## Responsabilità e interfacce

| Componente | Responsabilità |
| --- | --- |
| `ToolOutputPresentation` | Identità, corpo, header e allegati di una vista |
| `ToolOutputPreparer` | Budget, suddivisione e manifest del contenuto grande |
| `OutputFileWorkspace` | Risoluzione nella workroot e API di storage del nodo |
| `OutputFileSession` | Scrittura atomica, identità dei file e durata della sessione |
| `ToolOutputRuntime` | Preparazione e pubblicazione comune a lettura, ricerca e recorder |
| `ToolExecutionManager` | Sincronizzazione, ricerca del lavoro e API di stato versionate |
| `_Execution` | Transizioni dello stato mutabile e conferme della versione corrente |
| `ToolOutputQuery` | Validazione, filtri e paginazione della ricerca |
| `ToolCallOutcome` | Valore, esecuzione, durata e stato di una chiamata |
| `ExecutionLaunch` | Chiamata originaria, operazione, policy e proprietà del lavoro |
| `FileReadAccess` | Unica dipendenza opzionale dei tool file per i limiti degli artefatti |

I chiamanti non raggiungono il session store passando manualmente il node ID.
Usano `NodeCache.write_output_files()` e `output_file_read_limit()`. Le classi
messaggio e risultato espongono le operazioni di sostituzione e interpretazione;
il runtime usa questi metodi per preservare allegati, flag e metadati.

`FileIdentity` è condivisa tra ispezione e cache come concetto di filesystem;
la cache non importa l'implementazione del servizio di ispezione. Sono stati
rimossi un config duplicato per due codici di modifica dei file, deleghe di
lettura nel preparer e un metodo di snapshot non utilizzato.

## Comportamenti corretti

- Una lettura parziale non conferma un finale completato durante la lettura.
- Un annullamento confermato durante l'esportazione non pubblica la precedente
  risposta completata e rende nuovamente consegnabile il nuovo stato.
- La conclusione del worker non conferma la consegna prima della preparazione
  della risposta; un'esportazione fallita può essere ritentata.
- I flag di errore di consegna vengono mantenuti anche per risultati
  preformattati e per le viste salvate di un lavoro completato.
- La ricerca prepara anche record salvati precedenti alla policy, con e senza
  un manager, senza confermare letture o mutare i record JSON.
- Un record con `external_output` deve avere `result=None`; la conversione è
  esplicita e non scarta dati silenziosamente nel costruttore.
- File esportati modificati o sostituiti non vengono riutilizzati.
- Tokenizer base e Qwen condividono la validazione dei limiti dei blocchi.
- Le istruzioni dell'exit tool usano il renderer XML condiviso e conservano
  caratteri come `<` e `&` come testo.

La suddivisione resta limitata a **una singola riga fisica**. Un output multilinea
grande produce un solo file; ciascun blocco di una riga singola rispetta il limite
configurato, predefinito a 5000 token.

## Configurazione dei testi e delle costanti

Diagnostiche, log, descrizioni e rappresentazioni sono nei config di agenti,
modelli, provider, runtime, cache, reasoning, tokenizer, web, interfacce e utility.
Testi dei tool dimostrativi sono in `CapabilityText`; budget e diagnostiche di
esportazione in `ToolOutputSettings` e `ToolOutputText`. Il formato dello storage
è in `CacheFormat`. I due codici delle precondizioni di edit sono in
`FileEditFormat`, usato sia dal servizio sia dal percorso interattivo.

Sono stati centralizzati anche regex XML, lunghezza delle anteprime e soglie
delle euristiche per il contenuto web. I nomi XML/JSON, selettori delle API e
vocabolari del protocollo restano nel codice: sono definizioni del formato, non
messaggi per l'utente. Le docstring descrivono il codice e non vengono trattate
come configurazione della presentazione.

Un test AST controlla tutti i moduli di produzione per impedire diagnostiche e
frasi inglesi inserite direttamente fuori da `config`. Un test comportamentale
verifica anche che una diagnostica configurata venga effettivamente utilizzata.
Il controllo automatico integra la revisione; non dimostra da solo tutti i
principi di clean code o identifica ogni possibile testo dinamico.

## Funzioni con molti argomenti

I dati correlati sono stati raggruppati in `ToolOutputPresentation`,
`ToolOutputQuery`, `ToolCallOutcome` ed `ExecutionLaunch`. La preparazione prende
una vista, il renderer della ricerca prende servizio e query, l'avvio interno
prende una richiesta e la chiusura della skill prende il proprio risultato.

Le firme dei tool esposte al provider restano individuali perché costituiscono
lo schema degli argomenti. `ToolExecutionManager.start()` conserva la firma
precedente come adapter verso `launch()`. Le API di cache precedenti restano
compatibili; `ToolCallRecord` è già il tipo da usare nel percorso interno.
Costruttori con collaboratori distinti e opzioni keyword di presentazione non
sono stati racchiusi indiscriminatamente in contenitori senza responsabilità.

## Verifica

La suite è stata eseguita con `pytest -n 4 -m 'not live'`: **2717 test passati**. Nessun test live è stato
avviato. Sono inclusi test deterministici con eventi per le condizioni di
concorrenza e processi locali per crash e pulizia dello storage.

Ruff lint/format, Pyright e `git diff --check` sono passati. Il codice dei
lock nativi Windows non è stato eseguito: le prove dello storage usano WSL/Linux.
