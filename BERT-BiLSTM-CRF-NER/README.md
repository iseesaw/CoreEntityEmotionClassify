### BERT + BiLSTM + CRF
- Only NER
    + Input
    + Output
        * labels => [O N-B N-I]
        * (Only Named Entity in labels, and then judge sentiment)
- NER + Sentiment Analyze
    + Output
        * lables => [O P-B P-I N-B N-I]
        * (Positive or Negtive in labels as well)
