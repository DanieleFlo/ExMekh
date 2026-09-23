
TODO list:
<!-- - Valutare il codice per refactoring verso clean code. Valutare i tipi delle funzioni e il senso di alcuni metodi. Valutarte le classi con i loro metodi per refactoring. Valutare introduzione di bug e gestione multi nodo. -->
- [x] Tool specifici per questo harness che aiutano il comdello nella comprensione di ciò che succede nonstante la compattazione all'uscita delle skill.
  - Tool per sapere lo stato attuale dell'harness, tipo riassunto di cosa è stato fatto funzioni chiamate, anche nelle sotto skill e tool chiamati. Deve dire solo il nome e l'ID di ciò che è stato chiamato, e quando, deve anche dire indirettamente se è recuperabile dalla memoria; stile history. Non richiede parametri.
  - Tool per la lettura dei tool salvati dalla memoria. Deve richiedere dei parametri non obbligatori. Il suo scopo è di fornire la lista di tool recuperabili con un indicazione del contenuto. I parametri servono a sforgliare la lista, cercare per tipo o per dove sono stati utilizzati. 
  Devo capire se questo tool può anche fornire tutto il contenuto se selezionato singolarmente o divere questo compito in un tool apposito. Oppure divere i compiti anche con il tool precedente.
  - Implementa i test per una copertura completa anche live.
  - Aggiornare la documentazione
  - Rileggi il codice scritto e se necessario fai un refactoring per renderlo più chiaro e aderente al Clean Code.
  - Committa il codice.

- [x] Sub agents
  - [x] Tool per comunicare con i sub-agent
  - [x] HDS sub agents mock per test e test live
  - [x] Aggiornare la documentazione
  - [x] commit codice

- [ ] Skill di deep thininking
  - Con skill per la ricerca web
  - Skill per i calcoli o tool generici
  - Conservazione del thinking riassunto.

- [ ] Nell header finale di risposta della chiama della skill, inseriamo anche il thinking che è stato generato alla chiamata del tool di uscita ma limitato nella lunghezza 100 token. Lo inseriamo come un altro elemento oltre: nome, id ecc.

- [ ] Finalizzare l'exti tool, va deciso il comportamento degli items
  - Il modello deve inserire negli items nessuno, 1 o più id di tool che vuole mandare nella history principale.
  - Il modello può inserire solo ID di items utilizzati in quella sub skill o inferiori.
  - Il modello può inserire nel testo {{id_tool}} per inidicare la posizione di dove vuole venga inserito quel testo che deve esse anche presente nella lista con lo stesso ID.
  - Bisogna spiegare bene nelle istruzioni questo funzionamento. Non va bene: descrizione con {{id}} e lista vuota. Se viene inserito nella lista allora il testo verrà appeso alla fine, nell'ordine della lista.
  - Il testo viene inserito in formato XML.
  - Bisgona prevedere messaggi dei errore da dare al modello quando sbaglia, ogni tipo di errore deve essere diverso perché deve contentere dei consigli su come evitarlo.
  - Bisgona dire al modello che non sepre conviene mandare il testo sopra perché riempe il contesto, se questo testo deve essere utilizzato poi in una sub skill conviene dire solo l'id della risposta da utilizzare in modo da non sporacare il contesto ed utilizzare l'harness al meglio (se vedi altre strategie utili inseriscile nelle istruzioni).

- Passiamo da limitazioni per caratteri a limitazioni per token fruttando il tokenizer presente.

- [ ] Manipolazione file
  - Tool analisi
  - Tool di lettura
  - Tool di edit/scrittura

- [ ] Conversione id tool call in id tool generato dai 3 token se sbaglia, devono funzionare entrambi, ma bisogna continuare a suggerire al modello quello generato dal tokenizer.

- [ ] Scrittura finale dei prompt per l'harness
- [ ] nel sing dei tool callback per funzione prima del tool, che può anche sostiturire il risultato. Funzione da eseguire dopo con lo stesso metodo di callback.
