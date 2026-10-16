# Compattazione della history

La compattazione è disponibile su un nodo già inizializzato. Con `DeepAgent`
il nodo è **obbligatorio** e deve appartenere alla sua gerarchia:

```python
agent.compact_history(node=agent.root_node)
agent.compact_history(node=child_node)

# Se si dispone già del nodo:
child_node.compact_history()
```

Non esiste una selezione implicita della radice, né una ricerca per nome.
Il servizio non aggiunge un tool invocabile dal modello. La selezione automatica
è una policy dell'applicazione, installabile separatamente su ciascun nodo.

## Policy automatica e osservazione

```python
from exmekh_lib.core.history_compaction import HistoryCompactionPolicy
from exmekh_lib.core.models.request_budget import RequestBudget

# policy implementa il contratto HistoryCompactionPolicy dell'applicazione.
agent.root_node.set_history_compaction_policy(policy)
# Se l'applicazione conosce la capienza e sa stimare il prompt completo:
budget = RequestBudget(capacity=26000, count_tokens=count_request_tokens)
agent.root_node.set_history_compaction_policy(policy, budget=budget)
child_node.set_history_compaction_policy(child_policy)
child_node.set_history_compaction_policy(None)  # disabilita il trigger
```

`before_inference(request)` osserva una richiesta completa e decide se tentare
la compattazione; non viene invocato quando non esistono porzioni sostituibili.
`after_compaction(request)` riceve il contesto già applicato e può riarmare la
policy senza prendere una seconda decisione alla stessa frontiera. La libreria
non decide soglie né capienza: appartengono all'applicazione e ai metadati server.
Le inferenze interne di scoring e riassunto non attraversano la policy.

Con un `RequestBudget`, il runtime controlla ogni richiesta completa prima
dell'inferenza. Il superamento del budget forza un tentativo di compattazione
anche se la policy è disarmata o restano solo messaggi protetti. Dopo la
compattazione e la consegna dello steering, controlla nuovamente la richiesta:
se non entra, propaga `ContextBudgetExceededError` senza inviarla al provider.
Il limite del prompt riserva alla generazione il minore fra 1024 token e un
ottavo della capienza. L'applicazione fornisce lo stimatore, includendo template,
tool e schema strutturato; senza capienza nota non viene inventato un limite.

Un errore di compattazione automatica viene registrato e propagato con la sua
causa: l'inferenza si ferma e conserva la history originale per un nuovo tentativo.
La cancellazione viene propagata. Dopo una sostituzione,
il runtime ricostruisce la richiesta e consegna lo steering accettato durante il
riassunto prima di riprendere l'inferenza.

`DeepAgentHistoryCompactionEvent` espone copie della history applicata,
dell'albero delle skill e delle porzioni originali sostituite, correlate all'ID
della chiamata sintetica. L'applicazione può conservare una vista completa senza
reinserirla nel contesto del modello. Il fallimento di un osservatore opzionale
viene registrato senza annullare una sostituzione già applicata.

`CompactionResult.replacement_for(original)` preserva l'identità dei messaggi
trattenuti, anche se il calcolo è stato eseguito su un checkpoint indipendente.
Il proprietario della history usa questo metodo senza interpretare le regole
della mappa degli indici, e può ancora rimuovere input rifiutati dopo il riassunto.

## Esecuzione e concorrenza

Quando il nodo è inattivo, la chiamata esegue e applica la compattazione prima
di ritornare. Durante una `run()`, accoda la richiesta e ritorna subito: soltanto
il loop la applica, dopo il completamento del batch e prima della successiva
inferenza o del ritorno finale. Le richieste pendenti vengono accorpate.

La richiesta sblocca un'attesa gestita; il lavoro continua in background senza
annullamento, riesecuzione o nuovi risultati per la call originale. Le inferenze
e i tool sincroni non vengono interrotti. Lo steering accettato nel frattempo viene
consegnato normalmente. Il nodo resta esclusivo durante la compattazione; gli altri
nodi mantengono la propria history e il proprio coordinamento.

Gli errori immediati vengono propagati da `compact_history()`; quelli di una
richiesta accodata vengono propagati dalla `run()` originale. Una run fallita
scarta le richieste ancora pendenti. Una richiesta durante la chiusura già
iniziata viene rifiutata. Con history non persistente, la successiva run mantiene
il normale comportamento di reset.

## Valutazione e riassunti

Una chiamata strutturata riceve una copia della history completa, con system
e tools invariati, seguita da un messaggio temporaneo contenente istruzioni e
catalogo JSON compatto. Il catalogo è input interno temporaneo, non un risultato
di tool: evitare i tag XML ripetuti lascia spazio allo schema dei voti quando
la conversazione è vicina al limite. I dati e le anteprime restano gli stessi.
Questi messaggi non vengono aggiunti alla conversazione del nodo; i risultati
della compattazione e dei tool mantengono il renderer XML.

Se lo scoring non entra nel budget, i corpi degli output sostituibili vengono
salvati su file e sostituiti con riferimenti più brevi nella copia temporanea,
partendo dai più grandi. Questo può riguardare anche output inferiori alla
soglia ordinaria di 5000 token. System, user e istruzioni delle skill attive
restano protetti. Le anteprime e i riassunti usano il contenuto originale; il
risultato conserva i metadati dei file e l'archivio conserva i messaggi originali.
I manifest già esterni non vengono esportati ricorsivamente.

Se il catalogo e lo schema restano troppo grandi, i candidati vengono valutati
in lotti con campi obbligatori distinti. Ogni richiesta viene verificata prima
dell'invio. Anche i riassunti troppo grandi vengono suddivisi senza troncare
il testo sorgente e poi riassunti insieme; la mancata riduzione interrompe
l'operazione. Un errore di esportazione, scoring o riassunto conserva la history
di ingresso. Se sono i messaggi protetti a occupare tutto il contesto, si ferma
esplicitamente anziché tagliarli o inviare una richiesta fuori budget.

Il catalogo contiene nome, ID canonico, posizione univoca della risposta,
conteggio dei token e primi 20 + ultimi 20 token del corpo. Con al massimo
40 token il corpo compare interamente, senza sovrapposizioni. Gli header non
consumano questa anteprima.

Si includono anche gli output dichiarati nei `used-tools` delle skill già chiuse,
risolvendoli soltanto nel nodo corrente. Riferimenti ricorsivi vengono visitati
una sola volta per risultato visibile; record estranei non entrano nel catalogo.
Gli output non più disponibili restano riferimenti espliciti, senza riassunti
inventati. Le copie salvate dei messaggi user non vengono riassunte come
output discendenti.

Lo schema viene costruito per la history specifica: ogni candidato corrisponde
a un campo obbligatorio, per esempio `entry_3`, con voto intero fra 1 e 100.
Non ci sono array di voti né ID da riscrivere; i campi aggiuntivi sono vietati.
Il catalogo associa ciascun `score_field` al suo output. Il codice conserva
questa corrispondenza e valida nuovamente la risposta prima di applicarla.
Una posizione distinta identifica ciascuna risposta, anche quando call ID o
ID canonici sono riutilizzati. Una tool call nella risposta strutturata fa
fallire l'operazione.

Dopo aver raccolto tutti i lotti, `normalize_scores()` abbassa i voti dello stesso
offset, senza scendere sotto 1. La configurazione di sistema
`config/history_compaction.py` imposta `summary_fraction=0.10`,
`minimum_summary_candidates=2` e `summary_score_threshold=10`. Il numero obiettivo
è il 10% degli output totali, arrotondato per eccesso e con minimo due candidati
quando disponibili. Se i candidati sopra soglia sono già entro questo numero,
i voti restano invariati. Tutti i pari merito al confine sono mantenuti, anche
quando fanno superare il 10%; se non si escluderebbe alcun candidato, i voti
restano invariati. I voti inferiori a 10 non vengono promossi per raggiungere
il minimo. La tabella seguente usa i voti normalizzati; alcuni output brevi
possono ancora conservare il corpo originale senza una chiamata di riassunto.

| Voto | Rappresentazione |
| --- | --- |
| 1–9 | Nome e ID, con metadati di recupero e stato |
| 10–49 | Riassunto con prompt di una frase breve |
| 50–79 | Corpo originale sotto 100 token; altrimenti prompt di 3–5 frasi |
| 80–94 | Corpo originale sotto 200 token; altrimenti riassunto dettagliato |
| 95–100 | Corpo originale sotto 5000 token; altrimenti riassunto delle informazioni essenziali |

Le soglie scelgono quando riassumere il **testo originale**. Non impongono
limiti al testo generato: non vengono aggiunti `max_tokens`, troncamenti,
controlli sul numero di frasi o un budget complessivo della history compatta.
Un riassunto più lungo del suggerimento del prompt viene conservato.
La compattazione può quindi non ridurre la dimensione finale.

Valutazione e riassunti richiedono `reasoning_effort_override="none"`.
I riassunti riusano `TextSummarizer`, con client isolato, prompt dedicato e
nessun tool. Temperatura e altre impostazioni mantengono i default del provider.
Prompt e valori del protocollo sono in `config/history_compaction.py`.

## Avanzamento transitorio

`HistoryCompactor(..., on_progress=callback)` pubblica snapshot immutabili
`CompactionProgress`: preparazione/valutazione, piano dei riassunti, output
completati e conclusione, anche in caso di errore o interruzione. Gli output
discendenti sono contati come candidati distinti. Un errore dell'osservatore
opzionale viene registrato senza interrompere la compattazione.

Il tempo trascorso comprende tutta l'operazione. La stima residua usa soltanto
la durata dei riassunti effettivi, compreso quello dello stato dell'assistant
quando presente; riferimenti e corpi trattenuti senza inferenza non alterano la
media. Prima del primo riassunto completato la stima non è disponibile. Il tempo
già speso nel riassunto in corso viene sottratto dalla previsione residua.

Il runtime inoltra `DeepAgentCompactionProgressEvent` con identità del nodo,
posizione della skill e ID dell'operazione. Questi eventi non entrano nella
history del modello né nell'archivio delle risposte originali.

## Ricostruzione e protezioni

Ogni intervallo sostituibile diventa una coppia sintetica assistant/tool
`compact_history`, con call ID nuova e risposta XML attraverso il renderer
condiviso. La risposta conserva anche un riassunto delle decisioni e del lavoro
pendente espresso dall'assistant. Testo originale, Markdown, caratteri di
controllo e contenuti strutturati rispettano le regole di escaping del renderer.
Gli allegati dei tool restano disponibili tramite il normale percorso dei
messaggi multimodali.

System e messaggi user restano invariati, inclusi ordine, nomi e allegati.
Un batch incompleto, una risposta orfana o una corrispondenza call/result errata
vengono rifiutati prima dell'inferenza di valutazione.

I batch che aprono skill ancora attive vengono conservati interamente, inclusi
eventuali risultati fratelli: questo evita di spezzare il protocollo di apertura.
Le istruzioni non sono candidati. Le ancore degli execution tracker vengono
riallineate senza modificare scope ID, profondità o posizioni del registro delle
esecuzioni. Anche il batch finale richiesto da un ritorno nativo ancora pendente
rimane intatto.

I blocchi già prodotti dalla compattazione vengono conservati: richieste
successive agiscono sul nuovo contesto senza riassumere ripetutamente gli stessi
riassunti. Questa scelta intenzionale può mantenere più coppie sintetiche nello
stesso intervallo user quando si aggiunge nuovo lavoro.

La history sostitutiva viene validata prima di applicarla. Se il provider fallisce
o la history cambia durante la preparazione, il runtime non applica il risultato.
Cache, registro delle esecuzioni, consegne dei lavori in background e relativi
acknowledgement non vengono riscritti dalla compattazione.

## Output su file e prerequisiti

Le anteprime e i riassunti degli output su file usano il contenuto reale,
ricostruito nell'ordine dei file attraverso `OutputFileWorkspace`. Il servizio
può leggere soltanto artefatti registrati e invariati del medesimo nodo/sessione.
File mancanti o modificati fanno fallire l'operazione, lasciando disponibile
la history precedente.

I riferimenti sono mantenuti anche con `cache_result=False`, attraverso metadati
tipizzati del `ToolMessage`, senza salvare una copia del corpo o del riassunto in
cache. Le parti di un output grande non vengono reinserite integralmente nel
contesto: i path rimangono nel risultato compattato per il recupero tramite i tool
file. Un risultato più recente con lo stesso ID non sostituisce lo snapshot già
visibile nella history.

Un messaggio sintetico di compattazione non è una prova di ispezione o lettura
completa del file. La normale verifica di provenienza richiede nuovamente i
prerequisiti prima di leggere o modificare quando le call originali non sono
più visibili. La durata e la pulizia dei file non cambiano.

La preparazione può leggere integralmente gli output su disco e i riassunti
senza budget richiedono che il relativo input rientri nel contesto del provider.
Con un budget, `RequestBudget.split_text()` suddivide il testo senza perdite
verificando ogni richiesta completa, compreso il costo del prompt e del wrapping.
`TextSummarizer` coordina i riassunti dei frammenti e la loro riduzione successiva;
non deduce la capienza sottraendo costi che potrebbero non essere additivi.

## Servizio standalone

```python
from exmekh_lib.core.history_compaction import HistoryCompactor

result = HistoryCompactor(provider).compact(history_object, tools=tools)
new_history = result.history
print(result.original_message_count, result.compacted_message_count)
```

`history_object` è un `MessageHistory` e non viene modificato. Il risultato espone
anche `index_map` (posizioni vecchie → nuove, incluso il confine finale) e
`output_count`. Il chiamante che possiede lo stato coordina l'applicazione.

Per una history con skill aperte, il chiamante standalone deve fornire
`protected_indices` con le aperture da preservare. Per cache e file può fornire
`output_lookup` limitato al nodo e il suo `OutputFileWorkspace`. Il runtime
configura questi dati automaticamente; non espone gli interni del tracker.

## Prefill e validazione

Inviare la stessa history non implica necessariamente rielaborarla sul server.
La valutazione mantiene system, tools e messaggi precedenti come prefisso e
aggiunge il catalogo in coda. Il formato strutturato, `tool_choice` e il cambio
di reasoning possono però influenzare il prompt reso dal backend: non viene
promesso un cache hit. I campi dello schema cambiano insieme ai candidati della
history: sui backend che inseriscono lo schema nel prompt questo può cambiare
anche il prefisso riutilizzabile.

Dopo la sostituzione, la successiva inferenza deve normalmente elaborare il
contesto dalla prima modifica in avanti. Il riuso effettivo dipende da backend,
modello, template e disponibilità della KV cache; i test offline non misurano
il prefill del server.

Test deterministici coprono soglie, risposte strutturate errate, riassunti lunghi,
file, isolamento dei nodi, skill annidate, steering e lavori in background.

## Prove live del workflow

I test live usano il tokenizer reale e una history sintetica con log ripetitivi,
più messaggi user e un'approvazione pendente. Voti e riassunti possono cambiare:
non vengono confrontati con un testo atteso. Si verificano copertura dei voti,
corrispondenza call/result, preservazione di system/user, riduzione del contesto
inline, recupero dello stato pendente e prosecuzione con un vero tool.

La compattazione standalone viene provata con Chat Completions e Responses,
con ripetizioni indipendenti e file senza record in cache. Le prove di continuazione
agentica usano Responses: il server richiede questo endpoint per il protocollo
`agentic`. Coprono compattazione esplicita di un nodo figlio e richiesta durante
skill annidate, con successiva uscita da entrambi i livelli e mantenimento della
history compattata:

```shell
uv run pytest -n 0 --run-live tests/live/test_history_compaction_live.py tests/live/test_history_compaction_agent_live.py -q -s
```

Ogni caso salva nella propria directory temporanea l'osservazione JSON con
history precedente/successiva, conteggi dei token, voti e tempi delle chiamate.
Questi test controllano il workflow; non impongono uguaglianza dei riassunti,
voti prefissati o un numero massimo di token generati.

La qualifica del 15 ottobre 2026 sul server configurato, con
`Qwen3.8-27B-UD-Q4_K_S`, ha superato gli otto scenari dopo le correzioni e
le ripetizioni dei casi interessati. Nelle quattro prove inline la history
sintetica è passata da 6259 token a 949–1007, mantenendo checkpoint, coda e
stato di approvazione. Sono passate anche le due prove su file senza cache,
la continuazione del nodo figlio con un tool reale e la compattazione durante
due skill annidate, preservata dopo entrambe le uscite.

Questa qualifica verifica il comportamento del client e del modello nel workflow;
non misura il riuso della KV cache o il costo del prefill sul server.

## Responsabilità dopo la revisione

`HistoryCompactor` coordina la preparazione; `CompactionScorer` possiede la
valutazione e la validazione dei voti; `CompactionPresenter` possiede riassunti e
presentazione dei blocchi. Il runtime del nodo resta l'unico responsabile
dell'applicazione e del riallineamento degli scope.

`CompactionContext` possiede soltanto la copia temporanea dello scoring.
`ToolOutputPreparer.shorten()` decide se un riferimento su file riduce davvero
il costo; `OutputCandidate.with_presentation()` aggiorna la rappresentazione
senza perdere il corpo originale. I chiamanti non ne ricostruiscono i campi.

`ProviderRequest.with_event_handler()` adatta l'osservazione preservando il tipo
della richiesta, le opzioni, l'handler interattivo e la conversazione già
espansa. I wrapper devono usare questa API: ricostruire una richiesta a partire
da `get_messages()` espanderebbe nuovamente i companion degli allegati dei tool.

I metadati dei record in cache sono usati soltanto quando il contenuto coincide
con lo snapshot visibile. Una lista `used-tools` esplicitamente vuota esclude
i riferimenti precedenti. Le letture degli output su file passano dall'API del
nodo fino al proprietario dei file, che verifica la stessa identità prima e
dopo la lettura. Metadati file incompatibili con il manifest sono un errore,
senza sostituzione della history.

Il lettore pubblico `XmlHeader.parse_content()` conserva il corpo esatto ed
espone campi decodificati e riferimenti senza esporre il parser XML ai chiamanti.
Distingue metadati assenti da una lista esplicitamente vuota.
