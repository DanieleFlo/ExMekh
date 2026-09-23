Quando il lavoro della skill corrente è completo, chiama {{!exit_agent}} con `text` e `items`.
`text` è il risultato utile al chiamante. `items` contiene zero o più ID di output salvati
durante questa invocazione della skill o nelle sue skill discendenti. Usa `items=[]`
quando non serve includere output completi.

Preferisci il `saved-output-id` di `get_execution_history`. Puoi usare anche l'ID
della chiamata, se identifica un solo output. Un tool di memoria senza output salvato
non può essere incluso. Non ripetere lo stesso output in `items`.

Per collocare un output nel testo usa `{{id-output}}` e inserisci il relativo ID
anche in `items`. L'ID nel testo e quello nella lista possono essere i due ID dello
stesso output. Gli output elencati ma senza segnaposto vengono aggiunti alla fine,
nell'ordine di `items`, come blocchi XML `<tool-output>`.

Se `exit_agent` restituisce un errore, la skill resta aperta: correggi gli ID o i
segnaposto indicati e richiama il tool. La chiamata valida conclude solo la skill
corrente e restituisce il controllo al livello precedente.
