# Verifica live del 14 ottobre 2026

Esito finale dei **60 casi live attuali**: **52 passati, 6 falliti, 2 saltati**.
L'endpoint configurato usa `Qwen3.8-27B-UD-Q4_K_S` su localhost:8080.
I sei fallimenti riguardano allegati file nativi, rifiutati dal server con HTTP
400 e `Unsupported content type: only text and images are supported`.
La libreria propaga `ProviderRequestRejectedError`; i test di lettura degli
allegati restano falliti e le loro asserzioni non sono state allentate.

I due casi con una foto reale sono saltati perché manca
`EXMEKH_LIB_LIVE_IMAGE_PATH`. Le altre configurazioni opzionali sono state
attivate nella singola esecuzione, usando lo stesso endpoint e modello: recupero
dopo rifiuti, audit dei tool e installazione del browser da una cache vuota.
Il file `.env` e la configurazione del server non sono stati modificati.

## Cosa è stato verificato

| Area | Esito |
| --- | --- |
| Scope e schema dei tool | Passati i controlli di invocazione annidata e del tipo stringa. |
| Web pubblico | Ricerca, lettura, cache, compilazione del form, navigazione sulla stessa origine e blocco dei redirect passati. |
| Stati HTTP | Passati cinque nuovi casi reali: 401, 403, 404, 429 e 503, con causa, stato, URL e suggerimento coerenti. |
| Modifiche dei file | Tutti i 14 casi passati, inclusi alias, più risorse, stress avversariale, rifiuto del secondo edit e annullamento al cambio di scope. |
| Lettura dei file | Testo selezionato, immagine e pagine PDF consegnati al provider; contenuti e allegati recuperati dalla cache verificati. |
| Memoria e messaggi | Passati i round trip dei tool, i messaggi misti con immagini/testo, la memoria dei sub-agent e gli output salvati. |
| Processi reali | Passati attesa predefinita, background, consegna automatica e letture incrementali. |
| Recupero dopo rifiuto | Passati entrambi i controlli API: storia preservata dopo i rifiuti e richiesta successiva completata dallo stesso client. |
| Audit completo | Passati instradamento dei sette tool reali, media 5, stati logici `consistent`, `valid`, `found`, HTTP 200 ed effetto della navigazione. |
| Allegati nativi | Sei fallimenti, su tre disposizioni dei messaggi attraverso entrambe le API, per il rifiuto esplicito del server. |

## Correzioni dei test

Il verificatore confronta le scritture con gli argomenti effettivamente emessi,
prima di giudicare il comportamento del software. Controlla indipendentemente
unicità letterale, confini delle righe, sovrapposizioni e selezioni multiple di
un file vuoto. La verifica avversariale usa lo stesso oracolo indipendente,
anziché il selettore di produzione. Gli edit parziali vengono interpretati
secondo i file dichiarati come già committati; scritture non dichiarate falliscono.

I casi di file non dipendono più dalla scelta, uscita o riapertura di una skill
da parte del modello. Le letture controllano direttamente il contenuto selezionato
e gli allegati arrivati al client, oltre alla cache. Un'ispezione facoltativa di
un'immagine non invalida una lettura corretta. I rami richiesti devono essere
esercitati: una copertura mancante non diventa uno skip attribuito al modello.

L'audit precedente accumulava richieste fino a superare il contesto del server:
26.432 token contro 26.000. La fixture aggiornata dichiara la sequenza di chiamate
e restringe i nomi consentiti, mantenendo invariato il registro dei tool.
I parametri comprendono i default ottenuti dalla firma pubblica. Le risposte e
gli argomenti restituiti restano reali e vengono eseguiti senza sostituzioni.
Gli after hook osservano i risultati originali prima della presentazione;
le asserzioni verificano i risultati dei tool, non soltanto che siano stati chiamati.
Le chiamate interne vengono lette dall'API `skill_executions`.

Sono stati aggiunti controlli offline che iniettano ricevute incoerenti e risposte
con argomenti diversi da quelli suggeriti, verificano le fixture e proteggono
storia, registro stabile e checkpoint di ritorno. Non sono state necessarie
modifiche al codice di produzione per questa verifica.

## Riproduzione e artefatti

```shell
.venv/bin/pytest -n 0 --run-live tests/live -vv --tb=short -o junit_family=xunit1
.venv/bin/pytest -n 4 -m 'not live' --cov=exmekh_lib --cov-report=term:skip-covered
```

Per attivare tutti i controlli eseguiti qui, impostare anche
`EXMEKH_LIB_LIVE_DEEP_THINKING_MODEL` al modello configurato,
`EXMEKH_LIB_LIVE_REJECTION_URL` all'endpoint configurato e
`EXMEKH_LIB_LIVE_BROWSER_BOOTSTRAP=1`. Le foto richiedono la fixture indicata
da `EXMEKH_LIB_LIVE_IMAGE_PATH`.

I report e i trace generati sono in
`.exmekh_lib/live-verification-20261014/`, ignorata da Git.
`final-summary.xml` usa il renderer XML condiviso; `final-summary.json` è
l'aggregato strutturato interno. I report JUnit conservano il formato di pytest.
Il conteggio prende l'ultimo esito per ciascun identificatore di test, dopo le
correzioni e le riesecuzioni mirate: non rappresenta una singola esecuzione tutta
verde. L'insieme aggregato è stato confrontato con la raccolta attuale di pytest
e copre esattamente i 60 casi live.

La suite offline finale passa **2896 test**, con copertura di istruzioni e rami
al **100%** nel perimetro configurato. Ruff lint/format, Pyright e controllo del
diff passati. Le prove sono state eseguite in WSL/Linux; restano due skip per
la foto assente e sei incompatibilità esplicite dell'endpoint con i file nativi.
