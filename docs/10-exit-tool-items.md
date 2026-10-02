# Output selezionati con `exit_agent`

`exit_agent(text, items)` conclude la skill corrente. `text` contiene il risultato
per il chiamante; `items` è una lista di zero o più ID di output completi da
inserire nel messaggio che torna alla cronologia principale. Una lista vuota
restituisce solo `text`.

Ogni ID deve identificare un output salvato nel nodo corrente, prodotto durante
**questa invocazione** della skill o in una sua skill discendente. Il runtime
accetta l'ID dell'output mostrato come `saved-output-id` e, quando univoco,
l'ID della chiamata del tool. Preferire `saved-output-id`: una call ID riutilizzata
può identificare più output. I tool di memoria non salvano le proprie risposte
e quindi non sono selezionabili. Una skill precedente con lo stesso nome non
appartiene all'invocazione corrente.

## Posizionamento e formato

Un segnaposto `{{id}}` in `text` colloca l'output in quel punto. Il record deve
comparire anche in `items`; testo e lista possono usare i due ID diversi dello
stesso record. Lo stesso segnaposto può comparire più volte. Gli output di
`items` senza segnaposto vengono aggiunti dopo il testo nell'ordine della lista,
separati da una riga vuota. Lo stesso record non può comparire due volte in
`items`, neanche usando un ID per ciascun alias.

Ogni output usa un blocco XML:

```xml
<tool-output id="alpha_beta_gamma" name="read_data" type="Tool"><content>A &lt; B &amp; C</content></tool-output>
```

`id` è sempre l'ID canonico dell'output. `type` è `Tool` o `DeepSkill`.
Il contenuto è completo: per le deep skill è il testo finale, per gli altri
tool è la stessa rappresentazione letta da `read_tool_output`. Caratteri
XML speciali vengono escapati; se il contenuto contiene caratteri non ammessi
da XML 1.0, `<content encoding="base64">` conserva tutti i byte UTF-8.

Esempi, supponendo che `alpha_beta_gamma` sia l'output della chiamata
`call-42`:

```text
exit_agent(text="Sintesi", items=[])
exit_agent(text="Dettagli: {{alpha_beta_gamma}}", items=["call-42"])
exit_agent(text="Sintesi", items=["alpha_beta_gamma"])
```

La prima chiamata restituisce solo la sintesi. La seconda inserisce il blocco
XML dopo `Dettagli:`; la terza lo aggiunge in coda.
`text="Dettagli: {{alpha_beta_gamma}}", items=[]` è invalido perché il segnaposto non è
elencato. Anche `items=["call-42", "alpha_beta_gamma"]` è invalido perché
seleziona due volte lo stesso output.

## Errori correggibili

Un ID inesistente, non salvato, fuori scope o ambiguo, un duplicato, un
segnaposto non elencato o malformato e argomenti `text`/`items` non validi
producono un messaggio di errore come risultato del tool. Il messaggio indica
la causa e una forma corretta con placeholder, ad esempio
`items=["<id-output>"]` e `text="… {{<id-output>}} …"`. La skill rimane aperta,
la cronologia non viene compattata e il modello può richiamare `exit_agent`.
Un errore interno o di storage continua invece a propagarsi.

## Istruzioni inserite nei prompt

Il runtime aggiunge `harness-output-guidance` al system prompt di **ogni** HDS.
La parte relativa alla selezione degli output è:

```text
<harness-output-guidance>
Quando concludi una skill, inserisci in items solo gli ID degli output completi che il chiamante deve leggere subito. Preferisci il saved-output-id mostrato dalla cronologia; l'ID della chiamata è accettato se non è ambiguo. Se il risultato servirà in una skill successiva, comunica nel text soltanto l'ID e chiedi di recuperarlo con read_tool_output: duplicare l'output nel messaggio consuma contesto. Usa items=[] quando basta il testo.
</harness-output-guidance>
```

Il blocco comprende anche le modalità di esecuzione, il controllo dei lavori
e le regole di lettura e consegna automatica, descritte in
[Esecuzioni dei tool](13-tool-executions.md).

La guida è nel system prompt perché vale per l'intera esecuzione del nodo,
anche quando un HDS applicativo sostituisce quello incluso nel progetto. La
selezione dell'output completo è utile quando il chiamante deve leggerlo subito.
Quando il dato servirà in una skill successiva, passare soltanto l'ID evita di
occupare la cronologia principale con testo che può essere recuperato dalla
cache. La preferenza per `saved-output-id` evita l'ambiguità delle call ID.

Le istruzioni specifiche di `exit_agent` sono in
`src/exmekh_lib/capabilities/exit-tool/finish/instructions.md` e vengono mostrate
quando si apre una skill. Spiegano scope, segnaposti, ordine degli output e
correzione degli errori vicino al momento in cui il modello usa il tool.
