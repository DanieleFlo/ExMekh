# Output testuali grandi

Il runtime misura il corpo testuale della risposta con il tokenizer del nodo,
dopo gli hook e la rappresentazione XML condivisa. Con il limite predefinito di
5000 token, un corpo di 5000 token resta nella risposta; oltre il limite viene
scritto in file temporanei. Header, argomenti del protocollo e allegati
multimodali conservano il loro percorso attuale.

La policy copre tool ordinari e runtime, testo preformattato, diagnostiche,
output parziali/finali, consegne automatiche e ritorni delle skill, anche dopo
la composizione degli `items`. Le istruzioni che aprono una skill sono contesto
di esecuzione e rimangono visibili; non sono risultati di lavoro.

## File e suddivisione

Un output su più righe viene conservato in **un solo file**, anche oltre il
limite. Soltanto un output con una singola riga fisica viene diviso in blocchi
da al massimo il limite configurato. Il taglio può cadere dentro una parola:
non vengono aggiunti separatori o newline. CR, LF e CR/LF usano le convenzioni
di `SourceLines`; una newline finale non crea una seconda riga fisica.

I blocchi mantengono tutti i caratteri originali: concatenandoli in ordine si
ricostruisce il corpo esatto della risposta. Il tokenizer Qwen usa gli offset
del testo originale e verifica i singoli blocchi, senza ritokenizzare ogni
volta tutto il suffisso. Un carattere Unicode può occupare più token: il limite
viene rispettato senza dividerlo o sostituirlo; un budget troppo piccolo per
rappresentare un carattere produce un errore esplicito.

Il messaggio XML sostitutivo elenca **tutti** i file, nell'ordine di lettura,
con path relativo alla workroot, token e righe per file, conteggi totali e limite.
ID canonico, nome, stato, tempi e allegati del risultato restano disponibili.
I manifest possono superare il limite quando contengono moltissimi path:
questa eccezione evita esportazioni ricorsive e conserva l'elenco completo.

## Configurazione

`ToolOutputSettings` in `config/tool_output.py` espone:

| Campo | Default | Variabile di ambiente |
| --- | --- | --- |
| `max_tokens` | `5000` | `EXMEKH_LIB_TOOL_OUTPUT_MAX_TOKENS` |
| `directory` | `.exmekh_lib/tool_outputs` | `EXMEKH_LIB_TOOL_OUTPUT_DIR` |

Il limite deve essere un intero positivo. La directory deve essere relativa
alla workroot, diversa dalla workroot stessa, senza `..` o reindirizzamenti.
Le impostazioni vengono lette all'inizializzazione del runtime. Per esempio:

```shell
EXMEKH_LIB_TOOL_OUTPUT_MAX_TOKENS=5000
EXMEKH_LIB_TOOL_OUTPUT_DIR=.exmekh_lib/tool_outputs
```

Il layout separa sessioni, nodi e versioni del contenuto:

```text
<workroot>/.exmekh_lib/tool_outputs/
  session-<uuid>/<node-hash>/<output-and-content-hash>/0001.txt
```

Una rilettura dello stesso risultato riutilizza i file. La stessa call ID con
un risultato differente non sovrascrive i file precedenti. I parziali esportati
sono snapshot; il risultato finale può avere un riferimento diverso.

## Cache, lettura e durata

`ToolCallRecord.external_output` conserva una struttura tipizzata e serializzata
in JSON; il campo `result` è `null`, senza una copia del testo originale.
`read_tool_output`, ricerca e `items` restituiscono riferimenti, senza aprire i
file o riassumere il contenuto originale. `ToolOutputRuntime` pubblica la stessa
vista per risposte immediate, letture, ricerca e persistenza. Il manager conserva
lo stato e verifica la versione pubblicata, senza scrivere file. Le letture di record grandi precedenti alla policy
li convertono al nuovo riferimento.

Per recuperare il testo, usare `inspect-file` e poi `read-file` sui path elencati,
aprendo `file-processing` quando necessario. Le normali verifiche di ispezione,
hash e history continuano a valere. I tool file del runtime accettano soltanto
gli artefatti registrati dello stesso nodo/sessione. Per questi file il limite
ordinario di byte può essere esteso alla loro dimensione registrata, così un
output multilinea oltre 8 MiB rimane leggibile.

L'identità, la dimensione e la versione del file vengono registrate al momento
della pubblicazione. Un file modificato, sostituito, rimosso o trasformato in
symlink non viene riutilizzato e non riceve un limite di lettura esteso.

I risultati di `read-file` misurano il testo selezionato, escludendo numeri di
riga e framing XML. Questo permette di leggere un blocco di esattamente 5000
token senza riesportarlo per il solo costo della presentazione. Per un file
multilinea, scegliere intervalli più piccoli se il testo selezionato supera
il budget. Il cursore di `read_tool_output` non taglia il manifest di un output
già esportato; per nuovi dati parziali mantiene il cursore del tool di origine.

Con `cache_result=False` non viene creato un record JSON, ma il testo grande
viene comunque materializzato temporaneamente: è una distinzione intenzionale
fra cache dei risultati e consegna tramite file. Le esecuzioni gestite restano
leggibili dal runtime; un tool sincrono senza cache conserva i path nella sua
risposta, senza aggiungere una nuova vista recuperabile per ID.

`SessionCache` possiede anche questi file e li rimuove alla chiusura. All'avvio,
il runtime elimina le sessioni abbandonate nella directory configurata, usando
lock di processo sulle singole sessioni e sulla pulizia. Un nuovo client o HDS
non elimina file di una sessione ancora attiva. Dopo un crash il lock viene
rilasciato dal sistema operativo e la pulizia successiva può rimuovere i residui.
Un riavvio non ripristina il contenuto temporaneo o i lavori precedenti.

La scrittura pubblica la directory dei blocchi soltanto quando tutti sono
completi. Un errore di conteggio o scrittura produce `tool_output_export_failed`
in XML, senza consegnare il testo enorme o promettere file incompleti. Una
lettura finale fallita non conferma la consegna automatica del risultato.

Pubblicazione e conferma di lettura sono associate alla versione dello stato
effettivamente letta. Un completamento durante una lettura parziale rimane da
consegnare; un annullamento confermato durante l'esportazione obbliga a preparare
la nuova risposta `cancelled`. La conferma di annullamento riapre la consegna
del nuovo stato anche se una precedente versione risultava già consegnata.

`ToolOutputPresentation` raggruppa identità, testo, allegati e budget della vista.
`OutputFileWorkspace` usa le API pubbliche di `NodeCache`, che applica lo scope
del nodo. I tool file dipendono soltanto da `FileReadAccess`; non conoscono il
preparer o l'implementazione della cache. `ToolCallRecord.externalized()` è la
conversione esplicita: un record con riferimento e testo originale insieme viene
rifiutato, evitando una cancellazione nascosta nel costruttore.

## Verifica

```shell
uv run pytest tests/core/tool_calls/test_output_preparer.py \
  tests/core/cache/test_output_file_store.py \
  tests/core/agents/node/test_large_tool_outputs.py \
  tests/core/tokenizers/test_tokenizers.py
uv run ruff check src tests
uv run pyright
```

I test includono soglie, Unicode/BPE, fedeltà del testo, accesso tramite file,
hook, modalità di esecuzione, cache disabilitata e processi reali per crash e
pulizia concorrente. Non richiedono un provider remoto.
