
TODO list:
<!-- - Valutare il codice per refactoring verso clean code. Valutare i tipi delle funzioni e il senso di alcuni metodi. Valutarte le classi con i loro metodi per refactoring. Valutare introduzione di bug e gestione multi nodo. -->
- Implementare un tool che conseta di navigare fra i output dei tool salvati, cercandoli in base a query o in base a dove sono stati eseguiti.
  - Tool per sapere lo stato attuale dell'harness, tipo riassunto di cosa è stato fatto funzioni chiamate, anche nelle sotto skill e tool chiamati. Deve dire solo il nome e l'ID di ciò che è stato chiamato, e quando, deve anche dire indirettamente se è recuperabile dalla memoria; stile history. Non richiede parametri.
  - Tool per la lettura dei tool salvati dalla memoria. Deve richiedere dei parametri non obbligatori. Il suo scopo è di fornire la lista di tool recuperabili con un indicazione del contenuto. I parametri servono a sforgliare la lista, cercare per tipo o per dove sono stati utilizzati. 
  Devo capire se questo tool può anche fornire tutto il contenuto se selezionato singolarmente o divere questo compito in un tool apposito. Oppure divere i compiti anche con il tool precedente.


- Skill di deep thininking
  - Con skill per la ricerca web
  - Skill per i calcoli o tool generici
  - Conservazione del thinking riassunto.

- Quando una skill risale nell'header inseriamo anche il thinking ma limitato nella lunghezza.

- Tool per la comunicazione fra nodi (HDS)
  - Test annessi

- Scrittura finale dei prompt per l'harness
- nel sing dei tool callback per funzione prima del tool, che può anche sostiturire il risultato. Funzione da eseguire dopo con lo stesso metodo di callback.