# Esecuzioni dei tool

Il runtime del nodo gestisce i tool di lavoro con un protocollo comune. Gli
argomenti aggiunti allo schema del modello sono consumati dall'harness: la
funzione Python continua a ricevere soltanto i propri argomenti applicativi.

## Formato delle risposte

XML è lo standard dei risultati strutturati presentati al modello, sia in
modalità `wait` sia in `background`. Il runtime antepone l'header `<header>`
e converte dizionari, liste e valori scalari non testuali in un corpo `<tool-result>`.
La stessa conversione vale per `control_tool_execution`, la consegna automatica
e `read_tool_output`, anche quando legge un record già salvato.

I nomi dei campi del risultato restano quelli Python (per esempio
`tool_call_id` e `output_id`); i metadati nell'header conservano i nomi esistenti
con trattini. Oggetti e array usano `type="object"` e `type="array"`, gli
elementi delle liste sono `<item>`. I tipi scalari sono `string`, `integer`,
`number`, `boolean` e `null`; booleani e null restano distinti da stringhe vuote.
Le chiavi non valide come nomi XML usano `<field><key>…</key><value>…</value></field>`.
Il testo usa escaping XML; caratteri non rappresentabili in XML sono codificati
in base64 con `encoding="base64"` (UTF-8 con `surrogatepass`).

Header, cronologia, anteprime di ricerca e diagnostiche di lettura usano lo stesso
renderer XML per preservare caratteri di controllo e ritorni a capo. I nomi degli
argomenti non validi come nomi XML restano attributi `name` di nodi `<arg>`;
se contengono caratteri non rappresentabili, l'attributo usa base64 e aggiunge
`name-encoding="base64"`. I valori strutturati degli argomenti nell'header
mantengono il formato JSON compatto.

Esempio del corpo restituito all'avvio di un lavoro ancora attivo (tempi omessi):

```xml
<tool-result type="object">
  <status type="string">running</status>
  <tool_call_id type="string">call-example</tool_call_id>
  <output_id type="string">output-example</output_id>
  <message type="string">Tool execution is still running.</message>
</tool-result>
```

Un risultato di `read-web-page` espone campi come `<title>`, `<status_code>` e
`<markdown>` nello stesso corpo. `output_mode="markdown"` sceglie il contenuto
estratto dalla pagina: non il formato del messaggio del runtime. Il Markdown
rimane testo all'interno del campo XML e il contenuto web resta non attendibile.

Le stringhe prodotte direttamente dai tool, i risultati preformattati, i dati
incrementali e le diagnostiche di lettura restano testo. I ritorni Python e i
record della cache rimangono strutturati; lo storage, gli argomenti delle tool
call e gli export della history continuano a usare JSON. Chi analizza il corpo
strutturato delle risposte deve leggere XML anziché usare `json.loads`.

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

La history conserva gli argomenti della chiamata generata dal provider. La cache
conserva invece gli argomenti applicativi scelti da `BeforeTool`, congelati prima
di avviare il lavoro, anche quando il lavoro fallisce. La regola vale per tool
sincroni, `wait` e `background`; gli argomenti di controllo restano nel protocollo
dell'harness. Un record ancora in attesa dell'hook può mostrare gli argomenti
originari e viene aggiornato quando il risultato è disponibile. Le modifiche
successive apportate dal tool a liste o dizionari di input non cambiano la copia
salvata degli argomenti.

## Operazioni sincrone

Apertura e uscita delle skill, tool di lettura e controllo dell'harness e
`call_sub_agent` restano sincroni. `call_sub_agent`, anche se sostituito tramite
override, attende il completamento del figlio senza il limite di 60 secondi e
non espone `execution_mode`.

Per disattivare esplicitamente la gestione in un tool applicativo:

```python
from exmekh_lib.core.agents.tools import deep_tool
from exmekh_lib.core.models.tool import Tool


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

## Steering esterno

`DeepAgent.steer(user_message) -> None` e `Node.steer(user_message) -> None`
accodano indicazioni nella `run()` attiva senza avviare un secondo loop. La
risposta finale viene restituita dalla `run()` originale. Il frontend può
ricevere input su un thread distinto da quello che esegue `run()`; il metodo è
utilizzabile anche da un handler degli eventi, senza attendere la risposta.

Il messaggio viene copiato e accodato prima di sbloccare l'attesa. Si sbloccano
sia l'attesa iniziale dei tool gestiti sia `control_tool_execution(...,
action="wait")`. Finché i messaggi sono pendenti, le ulteriori attese del manager
restituiscono lo stato disponibile senza bloccare, comprese eventuali altre
chiamate nello stesso batch e controlli Python concorrenti. Dopo il consumo dei
messaggi, le attese successive tornano al comportamento ordinario.

Lo sblocco non termina il lavoro e non modifica lo stato in `completed` o
`cancelled`: il risultato segue il percorso già previsto per un'attesa scaduta,
con XML, output ID canonico, eventuali dati parziali filtrati e consegna finale.
Se il lavoro è già terminato quando il runtime acquisisce lo stato, conserva il
risultato finale. Hook, output e cache non vengono duplicati.

Solo il loop modifica la history. Completa il batch, esegue le letture
automatiche disponibili e inserisce i messaggi nell'ordine di accettazione al
confine della successiva richiesta al provider. Un messaggio arrivato dopo quel
confine attende la risposta dell'inferenza in corso. Gli input già consegnati
durante una skill restano nel suo transcript archiviato e non vengono ripetuti
nel livello chiamante dopo la compattazione. Anche le diagnostiche
`ErrorUserMessage` restano locali alla skill. Uno steering ancora in coda viene
consegnato dopo il completamento del `return`, che termina con il risultato
della skill. Se il batch contiene altre skill o uscite differite, ogni ritorno
viene completato prima di eseguire la successiva chiamata. La risposta
intermedia viene richiesta senza tool e senza streaming, poi viene sostituita
dal batch già in attesa.

Gli steering inseriti a `depth > 0` vengono anche salvati nella cache del nodo
con nome `steering_message`, tipo `UserMessage` e un output ID generato con lo
stesso meccanismo dei tool. L'ID compare in `get_execution_history`, in
`search_tool_outputs` e nei `used-tools` di tutte le skill contenitrici. Dopo
la compattazione, `read_tool_output(output_id)` recupera il testo e ripassa al
provider le immagini e i file originali. La cache conserva i byte degli allegati
e l'ordine del contenuto misto; una modifica successiva degli oggetti inviati
non cambia il messaggio salvato.

La consegna iniziale resta un messaggio utente. Alla rilettura, il testo usa il
normale risultato del tool con header XML; gli allegati viaggiano in un messaggio
utente associato all'output, dopo tutte le risposte del batch. Questo permette
a entrambi i mapper OpenAI di inviare contenuti multimodali senza interrompere
le coppie tool call/result. Gli steering inseriti in root non creano record:
la profondità considerata è quella della consegna, anche quando il messaggio
è stato accodato prima dell'apertura o durante la chiusura di una skill.
I record seguono durata e isolamento della cache del nodo.

Un'inferenza attiva e i tool con `background_enabled=False` non vengono
interrotti. `call_sub_agent` resta sincrono: lo steering su `DeepAgent` riguarda
la history del nodo radice e viene elaborato dopo il ritorno del figlio. Per
indirizzare esplicitamente un figlio attivo si usa il suo `Node.steer()`.

Senza una `run()` attiva, `steer()` solleva `RuntimeError`, anche quando rimangono
lavori in background. Le `run()` concorrenti sullo stesso nodo sono rifiutate.
Accettazione e chiusura del turno sono sincronizzate: una richiesta accettata
viene elaborata prima del ritorno normale di `run()`. Se il turno fallisce,
l'eccezione viene propagata, le indicazioni ancora pendenti vengono eliminate e
le attese vengono ripristinate; nessuna indicazione pendente passa al turno
successivo. I messaggi già inseriti seguono la policy della history.

## Lettura e output incrementale

`read_tool_output(output_id, cursor=None, summary_mode="none")` sostituisce il precedente nome
`read_saved_tool_output`. Il nome nei selettori e negli override di
`DefaultCapabilities` deve essere aggiornato. L'argomento `output_id` conserva
il suo significato.

La lettura funziona da ogni livello dello stesso nodo, anche dopo la
compattazione di una skill. Il campo XML `<type>` è `Tool`, `DeepSkill` o
`UserMessage` per lo steering salvato;
il nuovo `<response-type>` indica:

| Tipo di risposta | Significato |
| --- | --- |
| `running` | Lavoro attivo, senza nuovi dati parziali disponibili. |
| `partial` | Dati parziali disponibili; il lavoro è ancora attivo. |
| `final` | Risultato completo, anche quando il valore segnala un esito negativo. |
| `failed` | L'esecuzione o un hook ha sollevato un'eccezione. |
| `cancelled` | Un callback ha confermato l'annullamento. |

Per i lavori gestiti sia la risposta immediata sia le letture usano lo stesso
header con argomenti, stato, call ID, timestamp di avvio, timestamp finale quando
presente e secondi trascorsi nel campo `<elapsed-seconds>`. Durante un lavoro
in background il tempo viene ricalcolato con un orologio monotono a ogni lettura;
dopo completamento, errore o annullamento resta fermo. Le risposte ordinarie dei
tool sincroni riportano la durata della chiamata, inclusi gli hook; i ritorni delle
skill riportano il tempo dall'apertura, incluse inferenze e lavoro annidato.
Queste durate si conservano nella cache e nelle letture dopo il riavvio.
Una rilettura conserva il tempo dell'output originario. I record precedenti
senza metadati di esecuzione vengono letti come risultati finali, senza inventare
tempi mancanti. Gli output preformattati senza header conservano il loro formato
originale, senza aggiungere un header solo per la durata. Gli errori di
validazione del runtime usano il renderer XML condiviso e dichiarano
`response-type=failed`. Un risultato preformattato sincrono mantiene i suoi metadati
indipendentemente dal nome registrato del tool. Un'esecuzione gestita aggiunge i
propri metadati nell'envelope esterno.

Gli output ordinari e i ritorni delle skill conservano sempre un header e un ID
leggibile: se il provider non può generare un ID valido, viene usato un UUID.
I risultati diretti conclusi e i ritorni delle skill dichiarano
`response-type=final`, come le letture dei rispettivi output salvati.
Le eccezioni dei tool sincroni producono un risultato con
`error_kind=tool_execution_failed`, stato `failed` e durata misurata, senza
interrompere il loop. Gli errori di binding/validazione prima della funzione
sono distinti (`invalid_tool_arguments`); una `TypeError` o `ValueError` sollevata
dal corpo del tool resta un errore di esecuzione. I suggerimenti opzionali
`argument_guidance` appartengono alla definizione del tool. Gli errori mantengono
la policy di cache del tool; anche gli errori recuperabili dell'exit tool hanno
un ID e metadati leggibili. Gli errori dell'infrastruttura fuori dall'invocazione,
come la scrittura della cache, continuano a propagarsi al chiamante.
Per lavori gestiti, anche gli stati running/failed/cancelled hanno lo stesso
contenuto strutturato nelle risposte immediate e nelle riletture; solo i metadati
temporali di un lavoro ancora attivo continuano ad aggiornarsi.
Gli override di tool caricati e runtime conservano entrambi la scelta esplicita
di esecuzione sincrona della definizione originale.
Un errore nell'annotazione opzionale `thinking` omette solo quell'annotazione.
Il contenuto di `PreformattedToolResult` viene preservato; un'esecuzione gestita
lo include nel proprio envelope per aggiungere stato e ID canonico.
Un `PreformattedToolResult` può associare un `UserMessage` al testo tramite
`attachments`: il provider lo riceve dopo tutte le risposte del batch.
La lettura finale di un lavoro gestito conserva lo stesso messaggio associato.
Quando `cache_result=True`, la cache salva anche il suo contenuto ordinato e
i byte degli allegati, così una rilettura senza il manager li ripristina.
Gli errori nei contenuti multimodali salvati vengono segnalati come errori
recuperabili di lettura.

L'esecuzione diretta e quella gestita usano entrambe l'operazione esposta da
`Tool.create_execution()`. Gli hook applicano le modifiche agli argomenti prima
di creare l'operazione e filtrano il risultato una sola volta. Per i tool
specializzati, la call ID e il contesto di esecuzione vengono conservati anche
quando la gestione in background è disabilitata.

Se il tool offre una lettura incrementale, iniziare con `cursor=0` e riutilizzare
il `<next-cursor>` restituito. Il cursore è opaco al modello: il tool specializzato
decide come interpretarlo. Ometterlo consente la lettura completa. Al termine,
una lettura con cursore contiene i dati residui e il risultato finale; una
lettura senza cursore restituisce il risultato completo. Un tool senza callback
incrementale segnala che il cursore non è supportato. Per record soltanto
salvati il cursore è un offset di caratteri nel contenuto renderizzato;
anche oltre la fine, `next-cursor` non torna indietro.

Se il canale dei progressi fallisce, `read_tool_output` riporta la diagnostica
in `partial-output-error` nell'header XML e restituisce lo stato disponibile.
Il guasto non cambia lo stato del lavoro e non nasconde un risultato finale o
l'errore dell'esecuzione. Non viene avanzato un cursore senza dati validi.
Un tool che non supporta i cursori continua a rifiutare le letture con cursore.

## Riassunti delle letture

Il parametro opzionale `summary_mode` accetta:

| Valore | Contenuto restituito |
| --- | --- |
| `none` (default) | Output completo, senza chiamate aggiuntive al modello. |
| `brief` | Da 3 a 5 punti, con al massimo due frasi per punto. |
| `detailed` | Solo le informazioni fondamentali, con i dettagli essenziali per usare il risultato. |

Sotto `RUNTIME.tool_output_summary_min_tokens` (300 token per default) il corpo
viene restituito integralmente in ogni modalità. A partire dalla soglia, `brief`
e `detailed` usano i rispettivi prompt in `RuntimeSettings`, un client dedicato
con lo stesso provider e modello del nodo, senza tool, history del nodo,
streaming o reasoning (`reasoning_effort_override="none"`). Ogni lettura ha un
client isolato dalle altre letture.

Il riassunto sostituisce soltanto il corpo dopo un header XML valido e aggiunge
`<summary-mode>` all'header. Testo senza header e header malformati restano
invariati. Le aggiunte di metadati sostituiscono il campo già presente senza
creare copie duplicate. Stato, ID, durata, diagnostica, cursore e allegati
restano disponibili; cache e risultato originario non vengono modificati.
Con un cursore si riassume il contenuto della lettura corrente, e `next-cursor`
continua a riferirsi all'output originario. I byte degli allegati non vengono
inviati al riassuntore: vengono ripristinati dal normale percorso di lettura.
Se il conteggio dei token o il modello fallisce, oppure manca una risposta
testuale o la risposta contiene chiamate a tool inattese, si registra la
diagnostica e si restituisce l'output integrale. Le annotazioni del thinking
usano lo stesso servizio di riassunto con client isolato, mantenendo la propria
soglia e cache per turno.
Una lettura finale riassunta conferma la consegna come una lettura integrale.

```python
summary = node.read_tool_output(output_id, summary_mode="brief")
original = node.read_tool_output(output_id)  # output o riferimenti ai file temporanei
```

Gli output oltre il budget configurato restituiscono riferimenti a file anche
con `summary_mode="none"`; il testo originale si legge con i tool file. I
riassunti non aprono automaticamente questi file. Vedi
[Output testuali grandi](16-large-tool-outputs.md) per soglia, suddivisione,
durata, lettura e pulizia concorrente.

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

## Tool Bash

La definizione del tool si trova in `src/exmekh_lib/capabilities/tools/bash/main.py`,
accanto agli altri tool dell'harness in `capabilities/tools`. Il runtime
fornisce workroot e gestore dei lavori al momento della costruzione del nodo.

`DefaultCapabilities` include il tool globale `bash(command, execution_mode,
wait_seconds)`, disponibile a ogni livello del nodo. Ogni chiamata avvia una
nuova istanza di Bash nella `workroot`, anche se un'altra chiamata Bash è ancora
attiva. Serve l'eseguibile `bash` nel `PATH`: se manca, l'esecuzione termina con
stato `failed` e un messaggio esplicito. Non viene usata un'altra shell.

Il risultato finale contiene `command`, `cwd`, `status: exited`, `exit_code`,
`stdout`, `stderr`, `started_at` e `finished_at`. Un exit code diverso da zero è
un risultato valido con stato di esecuzione `completed`. Un errore di avvio o
lettura produce invece `failed`. Per l'output incrementale usare
`read_tool_output(output_id, cursor=0)` e poi il `next-cursor` restituito:
i frammenti sono etichettati `[stdout]` o `[stderr]`. La lettura finale con
cursore aggiunge l'eventuale frammento residuo e il risultato strutturato;
senza cursore restituisce il risultato completo.

`control_tool_execution(tool_call_id, action="cancel")` interrompe il processo e
i suoi figli quando applicabile. Su Windows usa `psutil` per terminare l'albero;
su POSIX invia `SIGKILL` al gruppo di processi. Se l'annullamento non riesce,
il tool di controllo segnala un errore e il lavoro mantiene il proprio stato.
Gli hook Before possono cambiare il comando o rispondere senza avviare Bash;
gli hook After possono filtrare il risultato. Quando è presente un hook After,
l'output parziale grezzo resta nascosto, anche nelle letture e nelle attese.

## Consegna automatica e livello di esecuzione

Quando un lavoro termina, il runtime accoda la lettura automatica. Al primo
confine della history che permette una coppia completa di messaggi, prima della
prossima inferenza, inserisce una chiamata assistant a `read_tool_output` e il
suo tool result, con una nuova call ID. Più consegne vengono serializzate.
Se la history termina con una risposta testuale dell'assistente, la chiamata
automatica viene associata a quella risposta, conservandone contenuto e
metadati senza creare due turni assistant consecutivi.

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

L'header XML del ritorno include `<background-tools type="array">` quando
restano tool in esecuzione nella skill o nelle sue discendenti, anche se queste
sono già state chiuse. Ogni `<item type="object">` contiene `name`,
`tool-call-id`, `output-id`, `skill-path` e `status` (`running`), così il livello
superiore può leggere o controllare il lavoro usando gli ID originari.
La lista viene ricalcolata dal manager a ogni uscita, in ordine di avvio;
include anche lavori senza cache e attese scadute in modalità `wait`.
Sono esclusi tutti i lavori terminati, falliti o annullati, anche se il loro
output finale non è ancora stato letto. Letture parziali, controlli e ricerche
non rimuovono un lavoro ancora attivo dalla lista. Le altre invocazioni e i
lavori dei livelli antenati restano fuori dal riepilogo della skill restituita.
Questi metadati descrivono lo stato al momento del ritorno e non modificano
il risultato Python o il record salvato della skill.

## Durata, cache e riavvio

Stato e risultati restano in memoria per la durata del `NodeRuntime`. L'ID
canonico resta costante: quando il loop osserva il completamento, aggiorna
atomicamente il record della cache. Se il lavoro termina dopo il ritorno di
`run()`, la vista in memoria è già leggibile; il salvataggio finale avviene al
successivo passaggio del loop. Gli output salvati seguono la durata della
[sessione di cache](09-tool-call-cache.md#sessione-directory-e-durata-dei-dati).

`cache_result=False` conserva stato e risultato soltanto nella vista temporanea
del runtime, senza scriverli su disco. Le letture di un altro nodo non possono
accedervi.
La ricerca `search_tool_outputs` include anche questi output temporanei e i
lavori in background, con la stessa visibilità per nodo della lettura. Le
anteprime non confermano la consegna finale e non modificano la cache; gli
output sono distinti per stato, tipo di risposta e origine `saved` o `runtime`.

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
| `TurnControl` | Ammissione esclusiva delle run, coda degli steer, chiusura atomica e rilascio delle attese. |
| `Tool` | Argomenti di esecuzione, firma pubblica, hook e creazione dell'operazione. |
| `DeepTool` | Collegamento dell'entrypoint decorato al workroot e alle dipendenze runtime, conservando metadati e hook. |
| `ToolCallExecutor` | Esecuzione delle capability e transizioni delle skill. |
| `MessageHistory` | Modifica dei messaggi e dei batch, conservando risposte e metadati nelle sostituzioni. |
| `ToolExecutionManager` | Worker, attese, accesso concorrente e registro degli stati. |
| `ExecutionSnapshot` | Vista dello stato e metodi pubblici per ottenere contenuto, allegati e risultato salvabile. |
| `ExecutionControlTool` | Contratto del tool globale di attesa e annullamento. |
| `ToolOutputReader` | Accesso agli output attivi o salvati, lettura incrementale e conferma della consegna. |
| `ToolOutputView` | Rappresentazione degli output: contenuto, allegati ripristinati, anteprime, metadati XML e stato dei record salvati. |
| `ToolCallSearch` | Ricerca filtrata e paginata degli output salvati e del runtime, senza confermare la consegna. |

Il runtime usa operazioni pubbliche della history per inserire chiamate automatiche;
la history identifica il batch corrente e conserva le risposte già presenti.
Il runtime non ricostruisce i messaggi né interpreta la struttura interna dei batch. `TurnControl`
gestisce la sincronizzazione tramite i metodi pubblici del manager; un'attesa già
rilasciata resta tale anche se il loop riattiva subito le attese successive. Gli adapter
dei provider chiedono a `Tool` quali argomenti appartengono all'esecuzione e come
normalizzarli. L'executor restituisce i metadati dell'esecuzione in un unico
snapshot tipizzato, senza duplicare call ID e stato in campi indipendenti.
Ricerca, lettura e composizione dei risultati delle skill usano la vista pubblica
degli output; la rappresentazione dei risultati e lo stato dei record salvati
sono interpretati nelle classi che ne hanno la responsabilità.

I tool specializzati derivano da `DeepTool`: il costruttore comune applica i
metadati di `@deep_tool` e nasconde gli argomenti runtime dalla firma pubblica.
Ogni tool mantiene nella propria cartella l'esecuzione e la presentazione che
gli appartengono. Il lettore degli output separa l'accesso al manager dalla
formattazione XML e conferma la consegna dopo una lettura finale riuscita.

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
