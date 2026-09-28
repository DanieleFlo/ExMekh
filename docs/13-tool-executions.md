# Esecuzioni dei tool

Il runtime del nodo gestisce i tool di lavoro con un protocollo comune. Gli
argomenti aggiunti allo schema del modello sono consumati dall'harness: la
funzione Python continua a ricevere soltanto i propri argomenti applicativi.

## Modalità e limiti

| Argomento | Default | Comportamento |
| --- | --- | --- |
| `execution_mode` | `"wait"` | Attende il risultato oppure restituisce il controllo allo scadere dell'attesa. `"background"` restituisce subito il controllo. |
| `wait_seconds` | `60` | Attesa iniziale, maggiore di zero e al massimo 300 secondi. Non è un timeout del lavoro. |

Se il lavoro termina durante l'attesa, il modello riceve il risultato normale.
Altrimenti riceve `status: running`, il `tool_call_id` originario, l'`output_id`
canonico, tempi, eventuale output parziale e un messaggio che il lavoro continua.
Lo schema strict richiede entrambi i campi ma accetta `null` per usare i default;
gli schemi non strict permettono di ometterli. Il runtime ammette al massimo
16 lavori attivi per nodo. Non impone una scadenza al lavoro.

Esempio di argomenti generati dal modello per un tool `analyze`:

```json
{"path": "report.csv", "execution_mode": "background"}
```

L'esecuzione e gli hook avvengono in un worker. Cache, history, eventi e nuove
inferenze vengono gestiti dal loop del nodo. I tool di lavoro devono quindi
essere adatti all'esecuzione in un thread; due invocazioni in background possono
lavorare contemporaneamente. Le chiamate `run()` sullo stesso nodo restano
sequenziali.

## Operazioni sincrone

Apertura e uscita delle skill, tool di lettura e controllo dell'harness e
`call_sub_agent` restano sincroni. `call_sub_agent`, anche se sostituito tramite
override, attende il completamento del figlio senza il limite di 60 secondi e
non espone `execution_mode`.

Per disattivare esplicitamente la gestione in un tool applicativo:

```python
from qs.core.agents.tools import deep_tool
from qs.core.models.tool import Tool

@deep_tool(description="Perform an operation inline", background_enabled=False)
def operation(work_root, value: str) -> str:
    return value

inline_tool = Tool(lambda value: value, name="inline", background_enabled=False)
```

Il protocollo richiede `NodeRuntime`. Le invocazioni Python dirette e il solo
`Client` non creano un gestore delle esecuzioni: nelle integrazioni che eseguono
i tool autonomamente, usare `background_enabled=False` o gestire il protocollo
esplicitamente. I nomi `execution_mode` e `wait_seconds` sono riservati quando
la gestione è abilitata.

## Attesa e annullamento

Il tool globale è:

```text
control_tool_execution(tool_call_id, action="wait", wait_seconds=60)
```

- `wait` attende fino al limite indicato e si sblocca subito al completamento.
  Restituisce stato, tempi e ID dell'output, senza ripetere il contenuto.
- `cancel` richiede l'annullamento e restituisce `cancellation_supported` e
  `cancelled`. Un tool Python generico non supporta la terminazione forzata.
  I tool specializzati possono fornire un callback che conferma l'annullamento.
- Un ID sconosciuto produce un errore recuperabile. Due lavori attivi con la
  stessa call ID vengono rifiutati. Se un ID viene riutilizzato dopo il
  completamento, il controllo seleziona l'unico lavoro ancora attivo; più
  esecuzioni terminate con lo stesso ID sono ambigue e i loro output vanno
  distinti tramite l'ID canonico.

Ogni controllo ha una nuova call ID e indica quella originaria negli argomenti.
La call originaria riceve un solo tool result: il completamento non aggiunge un
secondo risultato alla stessa chiamata.

Se il callback conferma l'annullamento mentre il worker termina, lo stato finale
resta `cancelled`: l'output del lavoro interrotto non viene esposto come un
risultato completato. Se il callback restituisce `False`, un eventuale
completamento naturale conserva il proprio risultato.

## Lettura e output incrementale

`read_tool_output(output_id, cursor=None)` sostituisce il precedente nome
`read_saved_tool_output`. Il nome nei selettori e negli override di
`DefaultCapabilities` deve essere aggiornato. L'argomento `output_id` conserva
il suo significato.

La lettura funziona da ogni livello dello stesso nodo, anche dopo la
compattazione di una skill. Il campo XML `<type>` resta `Tool` o `DeepSkill`;
il nuovo `<response-type>` indica:

| Tipo di risposta | Significato |
| --- | --- |
| `running` | Lavoro attivo, senza nuovi dati parziali disponibili. |
| `partial` | Dati parziali disponibili; il lavoro è ancora attivo. |
| `final` | Risultato completo, anche quando il valore segnala un esito negativo. |
| `failed` | L'esecuzione o un hook ha sollevato un'eccezione. |
| `cancelled` | Un callback ha confermato l'annullamento. |

Per i lavori gestiti l'header include anche stato, call ID, timestamp di avvio,
timestamp finale quando presente e secondi trascorsi. I record precedenti
senza metadati di esecuzione vengono letti come risultati finali.

Se il tool offre una lettura incrementale, iniziare con `cursor=0` e riutilizzare
il `<next-cursor>` restituito. Il cursore è opaco al modello: il tool specializzato
decide come interpretarlo. Ometterlo consente la lettura completa. Al termine,
una lettura con cursore contiene i dati residui e il risultato finale; una
lettura senza cursore restituisce il risultato completo. Un tool senza callback
incrementale segnala che il cursore non è supportato. Per record soltanto
salvati il cursore è un offset di caratteri nel contenuto renderizzato.

## Hook e tool specializzati

`BeforeTool` vede gli argomenti applicativi, la workroot e una copia della
history al momento della chiamata. Può modificarli o rispondere al posto del
tool. `AfterTool` filtra il risultato completo. Ciascuna fase viene eseguita una
sola volta per lavoro; attese e riletture non rieseguono gli hook del lavoro.
Gli hook dei tool globali, se configurati tramite override, si applicano alle
loro singole chiamate.

Un tool specializzato può sovrascrivere
`Tool.create_execution(arguments, *, call_id, work_root, history)` e restituire
`ToolExecutionOperation(run, cancel=None, read_partial=None)`. La factory viene
chiamata nel worker dopo Before e la validazione degli argomenti. `run` restituisce
il risultato; `cancel` deve essere rapido e restituire un booleano che conferma
la cessazione del lavoro; `read_partial(cursor)` deve leggere senza bloccare e
restituire `(testo, prossimo_cursore)`, con cursore intero non decrescente.
Buffer e callback devono gestire l'accesso concorrente del worker e del loop.

Gli output incrementali devono essere già filtrati dal tool specializzato.
Con un `AfterTool` configurato, il runtime trattiene i dati parziali e pubblica
soltanto il risultato finale filtrato: un hook che lavora sul risultato completo
non viene ripetuto su frammenti e l'output grezzo non può aggirarlo. Finché
Before non ha autorizzato e creato l'operazione, non sono disponibili né output
parziali né callback di annullamento.

## Consegna automatica e livello di esecuzione

Quando un lavoro termina, il runtime accoda la lettura automatica. Al primo
confine della history che permette una coppia completa di messaggi, prima della
prossima inferenza, inserisce una chiamata assistant a `read_tool_output` e il
suo tool result, con una nuova call ID. Più consegne vengono serializzate.

La consegna richiede la stessa sessione conversazionale del nodo e la stessa
catena di chiamate che hanno aperto le skill. Le posizioni delle aperture
distinguono anche call ID riutilizzate. Uscire e rientrare nella stessa skill
crea una nuova istanza: il risultato precedente resta leggibile manualmente
ma non viene inserito automaticamente nel nuovo livello. Una lettura manuale
del risultato finale evita una seconda consegna. Escludere `read_tool_output`
da `DefaultCapabilities` disabilita anche l'inserimento automatico.

Il meccanismo funziona anche fra turni di una history persistente. Con
`persistent_history=False` la nuova `run()` crea una sessione conversazionale
diversa; può leggere gli output precedenti ma non riceve le vecchie consegne
automatiche. I worker non modificano la history e non avviano inferenze.

Durante un ritorno nativo da una skill, le letture automatiche vengono aggiunte
allo stesso gruppo della chiamata di apertura e la risposta di ritorno resta
ultima. Il provider riceve così una transizione valida prima di riprendere
l'inferenza nel livello proprietario, anche con più risultati pronti.

## Durata, cache e riavvio

Stato e risultati restano in memoria per la durata del `NodeRuntime`. L'ID
canonico resta costante: quando il loop osserva il completamento, aggiorna
atomicamente il record della cache. Se il lavoro termina dopo il ritorno di
`run()`, la vista in memoria è già leggibile; il salvataggio finale avviene al
successivo passaggio del loop. Gli output salvati seguono la durata della
[sessione di cache](09-tool-call-cache.md#sessione-directory-e-durata-dei-dati).

`cache_result=False` conserva stato e risultato soltanto nella vista temporanea
del runtime, senza scriverli su disco. Le letture di un altro nodo non possono
accedervi. La ricerca dei risultati salvati elenca soltanto i record della cache.

Una nuova inizializzazione o il riavvio non riprende i vecchi lavori. I thread
sono daemon e non impediscono la chiusura del processo; questo non garantisce
l'annullamento di risorse esterne create da un tool. Un record recuperato che
indicava ancora `running` viene mostrato con stato `unavailable`, risposta
`failed` e un messaggio esplicito, senza presentarlo come risultato finale.
I risultati terminati già salvati conservano i propri metadati.

Gli eventi `TOOL_START` e `TOOL_END` descrivono il tentativo di chiamata: per un
lavoro in background, `TOOL_END` può accompagnare una risposta `running`.
La consegna automatica produce gli eventi della nuova lettura nel loop.

## Responsabilità delle classi

| Classe | Responsabilità |
| --- | --- |
| `Node` | Interfaccia pubblica per eseguire l'agente, controllare i lavori e leggere gli output. |
| `NodeRuntime` | Ciclo conversazionale, persistenza e consegna dei risultati al livello proprietario. |
| `Tool` | Argomenti di esecuzione, firma pubblica, hook e creazione dell'operazione. |
| `ToolCallExecutor` | Esecuzione delle capability e transizioni delle skill. |
| `MessageHistory` | Modifica dei messaggi e dei batch, conservando risposte e metadati del provider. |
| `ToolExecutionManager` | Worker, attese, accesso concorrente e registro degli stati. |
| `ExecutionControlTool` | Contratto del tool globale di attesa e annullamento. |
| `ToolOutputReader` | Lettura degli output attivi o salvati, cursori, header XML e conferma della consegna. |

Il runtime usa operazioni pubbliche della history per inserire chiamate automatiche;
non ricostruisce i messaggi né interpreta la struttura interna dei batch. Gli adapter
dei provider chiedono a `Tool` quali argomenti appartengono all'esecuzione e come
normalizzarli. L'executor restituisce i metadati dell'esecuzione in un unico
snapshot tipizzato, senza duplicare call ID e stato in campi indipendenti.

I controlli sono disponibili anche dal codice chiamante:

```python
status = node.control_tool_execution(tool_call_id, action="wait", wait_seconds=10)
output = node.read_tool_output(output_id)
```

Queste operazioni richiedono un nodo inizializzato e non aggiungono messaggi alla
history. Una lettura finale conferma la consegna e impedisce il successivo
inserimento automatico dello stesso risultato. Gli hook dei tool globali si
applicano alle chiamate eseguite dal modello.

## Verifica

```shell
uv run pytest tests/core/tool_calls tests/core/agents/node tests/core/providers -q
uv run ruff check src tests
```

I test usano eventi deterministici per attesa, completamento, annullamento,
filtri, output incrementale, isolamento e confini delle skill; non richiedono
un provider esterno.

Per le prove con modello e processi reali:

```shell
uv run pytest --run-live tests/live/test_tool_executions_live.py -v -rs
```

La suite copre anche il minuto predefinito effettivo, processi contemporanei,
isolamento dei subagenti e risultati fra turni persistenti. Valuta gli invarianti
del runtime e gli output effettivamente consegnati al modello, lasciando libera
la formulazione della risposta finale. La mancata esecuzione di uno scenario
specifico viene segnalata separatamente. Vedi [i test live](07-cli-and-testing.md#live-tool-execution-checks).
