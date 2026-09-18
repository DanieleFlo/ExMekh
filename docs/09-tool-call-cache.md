# Cache dei risultati dei tool

`NodeRuntime` decide quali risultati salvare: solo tool ed exit tool. Le
istruzioni di deep skill ed exit skill rimangono nella cronologia. Il runtime
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

## Identificatori e dati

Nel runtime `id` corrisponde all'ID dell'header XML, quando disponibile; altrimenti
è un UUID. `tool_call_id` conserva l'ID originale del provider. Riutilizzare una
call ID non sovrascrive quindi i risultati precedenti. Chiamare direttamente
`store` con lo stesso ID di record sostituisce invece quel record atomicamente.

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

`ToolCallReader` e `ToolCallSaver` restano disponibili per archivi con directory
esplicita. La costruzione tramite identità del tokenizer risolve un archivio
separato: non seleziona automaticamente una sessione o un nodo attivo.

## Errori e concorrenza

Gli errori di salvataggio risalgono al chiamante. Il runtime applica prima la
risposta alla cronologia, mantenendola coerente con un tool già eseguito, ed
emette comunque l'evento di fine. Non riesegue automaticamente il tool.

Il lock della sessione coordina storage, contatore e chiusura. Dopo un errore di
scrittura del contatore l'ID prenotato resta consumato, per non riutilizzarlo se
la sostituzione del file era già avvenuta. Queste garanzie non rendono concorrente
l'esecuzione di più `run` sullo stesso `NodeRuntime`.


## Thinking del turno di chiamata

`thinking` contiene il testo restituito dal provider nella risposta che ha
chiamato il tool. Non viene normalizzato o sostituito con il thinking della
risposta finale. Più tool call nello stesso turno condividono quel testo.
Il valore è `null` quando il provider non lo espone; i vecchi record senza
questo campo vengono letti con `thinking=None`.

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
