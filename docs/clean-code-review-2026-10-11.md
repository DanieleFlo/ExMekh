# Revisione del codice e delle responsabilità

Revisione del 2026-10-11 delle modifiche della chat e delle parti collegate:
provider, runtime dei nodi, hook, esecuzioni in background, cache, history,
ispezione e modifica dei file, presentazione XML e caricamento delle capability.
L'analisi strutturale iniziale dell'intera repository comprende 229 moduli Python e
1361 funzioni. Le verifiche automatiche non certificano da sole ogni scelta
architetturale: la revisione manuale si concentra sui confini e sui flussi indicati.

## Problemi corretti

| Problema | Responsabilità e comportamento risultanti |
| --- | --- |
| L'adapter HTTP manipolava liste, dizionari di autorizzazione, exchange consumati e call rifiutate | `InteractiveContract` possiede lo stato di una singola inferenza; l'adapter usa i suoi metodi pubblici |
| Il ciclo di inferenza mescolava controllo, preparazione dello snapshot, parsing e composizione delle interruzioni | Fasi distinte per pausa/ripresa, risultato interrotto e completamento, con cleanup comune |
| Il limite negoziato dello snapshot era un campo condiviso del provider | Il limite è locale alla singola chiamata, senza interferenze tra richieste concorrenti |
| Il runtime del nodo configurava direttamente il gestore concreto dell'executor | L'executor configura il proprio handler ed espone le richieste di ispezione tramite metodi |
| I confronti di snapshot replicavano dettagli di inode, device, timestamp e byte | `FileSnapshot.matches()` incapsula il confronto completo e rifiuta snapshot parziali |
| La cache sincrona conservava gli input post-hook, mentre wait/background conservavano quelli originari | Operation e snapshot espongono gli argomenti effettivi; il recorder usa la stessa regola in tutte le modalità |
| I record degli output runtime non ancora salvati ricostruivano gli input originali | Lo stesso accessor dello snapshot alimenta anche i record runtime e aggiorna i record salvati |
| Gli input post-hook contenevano riferimenti a collezioni modificabili dal lavoro | L'operation possiede uno stato privato; gli argomenti sono copiati prima dell'esecuzione e gli accessor restituiscono copie indipendenti |
| Il record pubblico di un'interruzione poteva divergere dal suo testo XML già costruito | `InterruptedMessage` possiede il record e ne espone una copia, anche per export e cache |
| Il decorator accettava metadata interattivi invalidi e una policy background che il costruttore rifiutava | Una validazione condivisa applica gli stessi errori immediati a entrambe le API di definizione dei tool |
| Lo schema delle azioni browser non esponeva campi, tipi e limiti richiesti dal runtime; la descrizione aveva un limite scritto a mano | `BrowserAction` descrive i campi pubblici; schema e descrizione usano il limite configurato, con almeno un'azione |
| Le annotazioni rinviate di `TypedDict` rendevano obbligatori campi `NotRequired` nello schema e nella coercizione | Un helper comune interpreta i qualificatori risolti, mantenendo la semantica dei campi ereditati |
| Il test live completo poteva ripetere azioni vuote senza limite e non chiudeva il provider in caso di errore | Budget di 24 turni, traccia delle chiamate reali e cleanup del provider anche in caso di fallimento |
| Più prove indipendenti riusavano contesti server fissi, con conflitto dopo un cambio di schema | Agent, macro, cache e subagenti ricevono contesti nuovi tramite il parametro pubblico di `Node`; restano stabili fra i turni dello stesso nodo |
| Le prove live della cache attraversavano due campi privati del nodo e ripetevano nomi dei tool nei prompt | I record si leggono dalle API pubbliche di sessione e `ToolCallReader`; la ricerca usa il tool registrato, i nomi provengono da configurazione e HDS |
| I prompt e i filtri live di subagenti ed esecuzioni conservavano altri nomi di controllo scritti a mano | Sono usati i nomi configurati anche in queste prove, lasciando distinti i nomi delle capability di esempio |
| Un test Bash assumeva che l'ora UTC di sistema fosse sempre crescente | Si verificano timestamp UTC validi e durata monotona non negativa; non si alterano gli orari reali del processo |

La history continua a rappresentare la chiamata originaria del provider. Questa
distinzione preserva la corrispondenza tra call/reply e la provenienza dei report
di ispezione; non si ricostruisce la chiamata originaria dagli argomenti post-hook.

## Principi controllati

| Principio | Verifica e decisione |
| --- | --- |
| Responsabilità ed encapsulation | Stato del protocollo, snapshot, operation e binding dell'handler appartengono agli oggetti che li possiedono |
| Funzioni piccole e livello di astrazione | Il ciclo interattivo orchestra fasi separate; i dettagli di validazione non sono ricostruiti dal chiamante |
| Nomi e interfacce esplicite | Metodi distinti per accettazione, pausa, rifiuto, completamento, argomenti da salvare e confronto di snapshot |
| Separazione query/comandi | Accessor e confronti non modificano stato; registrazione di reply, acceptance e commit rendono esplicita la mutazione |
| Duplicazione e configurazione | Rimane la definizione centralizzata dei marcatori; è stato centralizzato anche il filtro dello scope dei comandi di controllo |
| Confini esterni | HTTP/SSE resta nel transport; contratto e snapshot non dipendono da un formato interno dell'SDK |
| Errori e null | Le eccezioni ai confini degli hook diventano esiti del runtime; i cleanup rilanciano il guasto originario; l'handler obbligatorio ha un controllo esplicito |
| Coerenza dei tool | Stessi hook, input effettivi, cache, XML e stato di esecuzione per sincrono/wait/background; `ok=False` applicativo resta distinto da un errore di esecuzione |
| Side effect e concorrenza | Nessun lavoro anticipato; gli input sono congelati prima del lavoro; il limite negoziato è locale alla chiamata |
| Semplicità e modifiche contenute | Riutilizzati i modelli e i renderer esistenti, con test sui comportamenti osservabili |

Le funzioni più estese fuori dalla modifica includono il dispatch dei tipi nel
value coercer, il ciclo del crawler e la sequenza atomica del filesystem. Sono
già suddivise in helper di dominio o mantengono un ordine operativo necessario.
La lunghezza è un indicatore da leggere nel contesto, non una soglia che richiede
automaticamente nuove classi. Gli accessi con underscore segnalati dall'analisi
nel codice di produzione sono operazioni sulla stessa classe e `_replace` delle
tuple standard, non accessi del chiamante alla rappresentazione privata di altri
servizi.

## Validazione

La regressione della cache riproduceva quattro fallimenti prima della correzione:
successo/errore nelle modalità wait e background; la modalità sincrona passava.
I sei casi passano dopo la correzione, con verifica della history originaria.
Altri due casi verificano i record runtime salvati/non salvati e la provenienza
degli argomenti nell'header della lettura; la lettura conserva i filtri del tool.
Il test esistente sugli input mutati dal tool continua a passare. Una regressione
separata verifica l'immutabilità del record di interruzione negli export e nella
cache. Quattro ulteriori casi verificano i metadata interattivi invalidi e la
policy background nelle due API di definizione: tre riproducevano un fallimento
prima della correzione. Due casi verificano lo schema browser nelle due API;
altri due verificano i campi opzionali con annotazioni rinviate e l'arresto del
test macro al budget, con la sua traccia. La suite completa passa 2482 test, con
47 test live esclusi dall'esecuzione ordinaria. Ruff e Pyright passano.
Una ripetizione ha riprodotto il test Bash fragile durante un arretramento reale
dell'orologio WSL: `finished_at` precedeva `started_at` di circa mezzo secondo.
La suite completa passa nuovamente dopo aver corretto quell'assunzione del test;
il comportamento del processo e i suoi timestamp restano invariati.

I risultati live sono registrati separatamente nel
[report della suite restante](test-evidence/clean-code-live-2026-10-11.xml), nel
[report file editing](test-evidence/clean-code-live-file-editing-2026-10-11.xml)
e nella [prova del browser da cache vuota](test-evidence/clean-code-live-browser-bootstrap-2026-10-11.xml).
La prova file editing passa edit, new e l'attacco di 30 secondi (due tentativi,
una call completa validata); falliscono remove e i due test isolati del marcatore.
Remove emette un testo simile al marcatore, senza il suo carattere iniziale;
il test isolato ordinario emette newline nel marcatore e lo schema enum termina
con `incomplete_generation`. Le tracce sono conservate, senza normalizzazione
automatica degli argomenti. Il browser da cache vuota passa.
La prima prova completa di deep-thinking è stata interrotta dopo una sequenza
di `interact-web-page(actions=[])`, rifiutata dal runtime.
[L'osservazione registrata prima del cleanup della cache](test-evidence/clean-code-browser-empty-actions-2026-10-11.json)
e il [report interrotto](test-evidence/clean-code-live-macro-interrupted-2026-10-11.xml)
sono conservati. Questo ha motivato la verifica e la correzione del contratto
delle azioni e il budget diagnostico; il ciclo non viene conteggiato come successo.
Un successivo tentativo ha ricevuto `409` sul contesto fisso `main`, poiché il
registro dei tool aveva già lo schema precedente. Il
[report del conflitto](test-evidence/clean-code-live-registry-conflict-2026-10-11.xml)
è conservato: il controllo del server è corretto, il difetto era nell'isolamento
delle fixture. La prova con contesti nuovi non aggira né resetta quel registro.
Nel tentativo con contesti isolati passano i due controlli agentic e i sette
controlli web. Il modello emette azioni browser strutturate e le chiamate di
calcolo e logica nella
[traccia macro](test-evidence/clean-code-macro-model-trace-2026-10-11.json), ma
la risposta successiva ha stato `incomplete`: il workflow non è qualificato.
La traccia comprende le chiamate completate; non è la risposta wire incompleta
e non ne dimostra la causa.

La suite restante termina con 9 pass, 11 fallimenti, 19 errori e 2 skip in
642,82 secondi. Oltre al fallimento macro, 29 casi non completano la verifica:
la lettura di testo riceve una disconnessione e i successivi tentativi ricevono
`Connection refused`. Il processo DS4 iniziale, PID 3833, era terminato; il nuovo
server ha PID 68399. L'utente ha confermato di averlo riavviato manualmente:
questa interruzione dell'ambiente non viene classificata come crash del server
o regressione del client. Il controllo dei processi ha confermato che i test
interrotti precedenti erano terminati e che c'era una sola suite live attiva;
l'altro processo Python era il formatter di VS Code. Le richieste della suite
contribuivano al carico della GPU, usata dal server.

I 31 casi rimasti sono stati rieseguiti in sequenza sul server riavviato,
includendo anche i due controlli di recupero dopo allegati rifiutati. Il
[report di ripresa](test-evidence/clean-code-live-recovery-2026-10-11.xml)
registra 23 pass, 6 fallimenti e 2 skip in 1007,73 secondi. Passano le letture
tramite tool di testo, immagine e PDF, i round trip e la history del client,
le sei combinazioni di testo e immagini nelle due API, la memoria dei subagenti,
la cache, wait/background con e senza streaming e l'output incrementale.
Passano anche entrambi i recuperi dopo allegati rifiutati, senza corrompere la
history. I sei fallimenti richiedono invece la lettura di file allegati nativi:
DS4 li rifiuta esplicitamente con `only text and images are supported`. Questo
limite dell'endpoint resta distinto dalla lettura tramite i tool, che convertono
i contenuti in testo e immagini supportati. I due skip richiedono la foto reale
indicata da `EXMEKH_LIB_LIVE_IMAGE_PATH`, non disponibile nell'ambiente.

La rimozione degli accessi privati nelle prove della cache è verificata
separatamente nel
[report finale mirato](test-evidence/clean-code-live-final-2026-10-11.xml):
passano tutti e tre i casi. La nuova prova macro fallisce con un errore diverso:
`context_length_exceeded`, 57.643 token rispetto ai 26.000 configurati nel
server. La [traccia finale](test-evidence/clean-code-macro-final-model-trace-2026-10-11.json)
registra la navigazione da statistics a math in modalità AST. I limiti di
contenuto web sono applicati in caratteri e nodi; non garantiscono che la history
serializzata, con XML e contenuti accumulati nella stessa skill, rientri nella
finestra del modello. Non esiste una riduzione automatica della history attiva
basata sulla finestra negoziata del server. È un limite del flusso completo da
considerare in una modifica specifica della gestione del contesto, distinto
dai difetti dei marcatori e non risolto aumentando il budget dei turni. La prova
macro resta non qualificata: non sono stati troncati messaggi né trasformato il
rifiuto in successo. Un warning di Python segnala inoltre l'uso di `fork()` da
un processo con più thread durante quel workflow; non è stato osservato un
deadlock e la scelta del metodo di multiprocessing non è stata modificata.

Il [report consolidato](test-evidence/clean-code-live-consolidated-2026-10-11.xml)
contiene l'ultimo risultato di ciascuno dei 47 casi: **35 pass, 10 fallimenti,
2 skip, nessun errore di setup**. È l'unione di prove sequenziali separate,
non una singola esecuzione ininterrotta; ogni caso conserva il proprio report
di origine. I dieci fallimenti sono remove e i due controlli del marcatore,
sei letture di allegati nativi non supportati e il workflow macro oltre il
contesto disponibile. Tutti i processi dei test sono terminati e il runner
temporaneo è stato rimosso.
Le anomalie storiche del server sono conservate nel
[report interattivo](interactive-edit-server-anomalies.md); una regressione
deterministica del client non dimostra che quelle anomalie siano risolte.
