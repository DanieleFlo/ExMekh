# Cache dei risultati dei tool

Questa cache permette di rileggere e cercare i risultati delle chiamate anche
quando la cronologia di una skill viene compattata. Non evita nuove esecuzioni:
chiamare nuovamente un tool esegue nuovamente la sua funzione. Per gli ID generati
dal tokenizer vedi [Tokenizer-based IDs](08-tokenizer-ids.md); per la cronologia
vedi [DeepAgent & Hierarchical Execution](05-deep-agent-and-execution.md).

`NodeRuntime` decide quali risultati salvare: solo tool ed exit tool. Le
istruzioni delle deep skill, incluse quelle del relativo exit tool, rimangono nella cronologia. Il runtime
riceve una `NodeCache` dalla configurazione al primo salvataggio e la riutilizza;
la ricrea se la sessione viene chiusa. L'esecutore non accede allo storage.

## Responsabilità

- `CacheManager` espone l'accesso alle sessioni condivise.
- `SessionPool` identifica una sessione con processo, identità del tokenizer,
  fingerprint del vocabolario e directory assoluta. La creazione è sincronizzata.
- `SessionCache` possiede directory temporanea, lock, contatore e stato dei nodi.
  Chiuderla invalida le viste e rimuove i suoi file.
- `NodeCache` limita letture, scritture e stato a un nodo.
- `ToolCallStore` implementa un solo formato di lettura e scrittura atomica.
- `ToolCallRecord` contiene metadati e una copia JSON dei dati ricevuti.
  `ToolCallData` è un alias di compatibilità dello stesso modello.

## Primo esempio: salvare, leggere e cercare

Questo esempio è autonomo: non richiede un provider, una chiamata al modello o
il download di un tokenizer. La sessione elimina i propri file all'uscita dal
blocco `with`.

```python
from pathlib import Path

from qs.core.cache import SessionCache, ToolCallRecord
from qs.core.tool_calls import ToolCallSearch

with SessionCache(Path(".qs/examples")) as session:
    cache = session.node("analysis")
    cache.store(ToolCallRecord(
        name="read_file",
        id="output-001",
        tool_call_id="call-001",
        args={"path": "README.md"},
        result={"text": "Project documentation"},
        skill_path="/review",
    ))

    record = cache.get("output-001")
    assert record is not None
    assert record.result == {"text": "Project documentation"}

    search = ToolCallSearch(cache)
    page = search.search(query="documentation", skill_path="/review", limit=10)
    assert page["total"] == 1
    assert page["next_offset"] is None
    assert session.node("another-node").get_all() == []
```

`get()` restituisce `None` per un record assente; `get_all()` e `list_ids()`
elencano i record validi ordinati per ID. `set_state()`, `get_state()`,
`has_state()` e `delete_state()` gestiscono invece stato Python in memoria,
isolato per nodo: questo stato non viene serializzato nei record.

## Sessione, directory e durata dei dati

Il layout predefinito del tokenizer e della cache runtime è:

```text
.qs/tokenizers/<identity-hash>/
    vocabulary.json
    temp/session-<unique-suffix>/
        counter.json
        tool_calls/<encoded-node-id>/<encoded-record-id>.json
```

Il vocabolario è riutilizzabile fra processi; la directory `session-*` è
temporanea. `provider.get_session()` restituisce la sessione del tokenizer e può
inizializzarne il vocabolario al primo accesso. Due provider condividono la
sessione solo se coincidono processo, identità, fingerprint e directory risolta.

Ogni `NodeRuntime` genera un ID con nome HDS e UUID: due nodi con lo stesso HDS
non condividono i risultati. Le successive `run()` dello stesso runtime
mantengono la cache anche con `persistent_history=False`, che azzera soltanto
lo stato conversazionale. Una nuova `Node.initialize()` crea un nuovo runtime
e un nuovo ID; i dati precedenti restano nella sessione fino alla sua chiusura.

`SessionCache.close()` elimina i file e invalida le viste `NodeCache` esistenti.
`CacheManager.close_all()` chiude le sessioni gestite dal pool, anche quelle dei
tokenizer; non chiude le sessioni costruite direttamente come nell'esempio.
Il runtime acquisisce una nuova vista al salvataggio successivo se la precedente
è stata chiusa. La pulizia automatica alla normale uscita del processo non è
garantita in caso di terminazione forzata: i vecchi file non vengono ripresi
come sessione attiva.

## Identificatori e dati

| Campo JSON | Significato |
| --- | --- |
| `name` | Nome del tool eseguito, anche quando è un exit tool. |
| `id` | Chiave del record nello storage del nodo. |
| `tool_call_id` | ID della chiamata del provider, oppure `null` per record manuali. |
| `args` | Copia JSON degli argomenti. |
| `result` | Risultato della funzione convertito in JSON, prima della formattazione XML. |
| `skill_path` | Percorso logico della skill; `/` indica lo scope radice. |
| `thinking` | Testo esposto dal provider nel turno della chiamata, oppure `null`. |

Nel runtime `id` corrisponde all'ID dell'header XML, quando disponibile; altrimenti
è un UUID. `tool_call_id` conserva l'ID originale del provider. Riutilizzare una
call ID non sovrascrive quindi i risultati precedenti. Chiamare direttamente
`store` con lo stesso ID di record sostituisce invece quel record atomicamente.

La distinzione è rilevante per `record_call()`: se non si passa `id`, questo
helper usa la call ID del provider come ID del record. Chiamate manuali ripetute
con la stessa call ID possono quindi sostituire il record. La generazione di
un ID distinto per ciascun risultato appartiene al runtime.

`skill_path` è il percorso logico della skill al momento della chiamata, non la
directory di lavoro del filesystem. Per un exit tool viene catturato prima
dell'uscita. Anche gli argomenti vengono copiati prima di eseguire il tool.

Gli ID dei nodi e dei record sono dati opachi: vengono codificati quando non sono
nomi portabili. La codifica evita traversal, nomi riservati di Windows e collisioni
dovute a maiuscole o sostituzione di caratteri. I vecchi filename dei record sono
ancora leggibili, verificando l'ID completo nel JSON; i vecchi record senza path
usano `/`. Gli elenchi escludono file corrotti o incoerenti.

Oggetti non JSON vengono rappresentati con `repr`, valori non finiti con stringhe
e cicli con `<circular reference>`. La cache non ricostruisce oggetti Python.

## Ricerca e archivi separati

`ToolCallSearch(node_cache).as_tool()` crea `search_saved_tool_outputs`, un tool
di sola lettura da registrare esplicitamente nell'integrazione che lo utilizza.
Accetta `query`, `skill_path` esatto, `offset` e `limit` (1–100); restituisce
`records`, `total` e `next_offset`. Cerca solo nel nodo fornito, mai negli altri.

| Parametro | Default | Comportamento |
| --- | --- | --- |
| `query` | `""` | Sottostringa senza distinzione fra maiuscole e minuscole nel JSON completo del record, inclusi metadati e thinking. Vuota seleziona tutti i record. |
| `skill_path` | `None` | Corrispondenza esatta; `/review` non include `/review/details`. |
| `offset` | `0` | Numero intero non negativo di risultati da saltare. |
| `limit` | `20` | Dimensione della pagina, intero fra 1 e 100. |

`total` conta tutti i record corrispondenti prima della paginazione.
`next_offset` è `null` alla fine; altrimenti può essere passato alla chiamata
successiva. L'ordinamento è per ID, non cronologico. Ogni ricerca legge l'intero
archivio del nodo: non esiste un indice full-text e pagine successive non sono
uno snapshot stabile se nel frattempo vengono scritti nuovi record.

Per esporre lo schema a un `Client`, l'integrazione può costruire
`Tools([ToolCallSearch(cache).as_tool()])` usando `qs.core.models.tools.Tools`.
L'esecuzione delle chiamate restituite dal modello resta a carico
dell'integrazione. `Node` non registra automaticamente questo tool e non espone
attualmente un accessor pubblico per la propria cache runtime. Creare
`provider.get_session().node("main")` non seleziona la cache di un nodo con HDS
`main`, il cui ID include un UUID. Gli esempi manuali usano un ID controllato
dall'applicazione proprio per rendere esplicito questo collegamento.

`ToolCallReader` e `ToolCallSaver` restano disponibili per archivi con directory
esplicita. La costruzione tramite identità del tokenizer risolve un archivio
separato: non seleziona automaticamente una sessione o un nodo attivo.

Per conservare un risultato oltre la durata di una sessione, salvarlo in una
directory esplicita gestita dall'applicazione:

```python
from pathlib import Path

from qs.core.tool_calls import ToolCallReader, ToolCallSaver

archive = Path(".qs/saved-outputs")
saver = ToolCallSaver(directory=archive)
saver.record(
    name="summarize",
    id="report-001",
    args={"topic": "cache"},
    result={"summary": "Node-scoped tool outputs"},
    skill_path="/review",
)

reader = ToolCallReader(directory=archive)
assert reader.get("report-001").result["summary"] == "Node-scoped tool outputs"
assert reader.read("missing") is None
```

`reader.get()` solleva `FileNotFoundError` se il record non esiste o non è valido;
`read()` restituisce `None`. Gli archivi espliciti non vengono rimossi da
`CacheManager.close_all()`. Per esportare record già presenti si può passare
ciascun `ToolCallRecord` a `saver.save(record)`.

## Errori e concorrenza

Gli errori di salvataggio risalgono al chiamante. Il runtime applica prima la
risposta alla cronologia, mantenendola coerente con un tool già eseguito, ed
emette comunque l'evento di fine. Non riesegue automaticamente il tool.

Il lock della sessione coordina storage, contatore e chiusura. Dopo un errore di
scrittura del contatore l'ID prenotato resta consumato, per non riutilizzarlo se
la sostituzione del file era già avvenuta. Queste garanzie non rendono concorrente
l'esecuzione di più `run` sullo stesso `NodeRuntime`.

| Situazione | Effetto |
| --- | --- |
| Il tool solleva un'eccezione | Nessun record di successo viene salvato; l'errore risale al chiamante. |
| Lo storage fallisce dopo l'esecuzione | La risposta è già applicata alla cronologia; la funzione può avere già prodotto effetti esterni. |
| Un record non è leggibile o è incoerente | Viene escluso dalla lettura e dagli elenchi. |
| Una vista appartiene a una sessione chiusa | Le operazioni sulla vista sollevano `RuntimeError`. |

L'evento `TOOL_END`, quando lo streaming è attivo, segnala la fine del tentativo
di esecuzione anche in caso di errore; non è da solo una conferma di salvataggio.


## Thinking del turno di chiamata

`thinking` contiene il testo restituito dal provider nella risposta che ha
chiamato il tool. Non viene normalizzato o sostituito con il thinking della
risposta finale. Più tool call nello stesso turno condividono quel testo.
Il valore è `null` quando il provider non lo espone; i vecchi record senza
questo campo vengono letti con `thinking=None`.

## Verifica

Per verificare storage, isolamento, serializzazione e ricerca senza un modello:

```shell
uv run pytest tests/core/cache tests/core/tool_calls -q
```

Per includere l'integrazione con il runtime e i tokenizer:

```shell
uv run pytest tests/core/agents/node tests/core/tokenizers -q
```

I batch misti di skill e tool hanno limitazioni note nella continuazione della
cronologia; vedi [la guida al runtime](05-deep-agent-and-execution.md#6-runtime-and-saved-tool-outputs).

I test live dedicati si eseguono con:

```shell
uv run pytest --run-live tests/live/test_tool_cache_live.py -v
```

Usano `BASE_URL`, `API_KEY` e `QS_LIVE_MODEL`, come gli altri test live, e l'API
Responses. Coprono streaming e non streaming, tool ordinari, exit tool,
path annidati, rilettura del JSON, thinking del turno esatto e isolamento
tra nodi. I test del tool ordinario richiedono un backend che restituisca
thinking non vuoto; gli altri verificano la corrispondenza esatta anche quando
il provider restituisce `null`. Il vocabolario degli ID è locale al test:
le chiamate al modello sono reali e non servono download del tokenizer.
