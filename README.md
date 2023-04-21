# Préambule général

## La modularité à l'ère des notebooks

La doctrine invite à placer l'ensemble de ses *imports* en début de notebook et à avoir une approche séquentielle, consistant à faire dépendre l'état initial d'une cellule à l'état produit par les précédentes.

L'expérience pratique nous a convaincus de ne pas adhérer à cette doctrine.

Dans les notebooks présentés ci-après, l'essentiel des cellules de code sont pensées pour une exécution autonome, sans dépendre de l'exécution préalable de cellules précédentes.

Quand de telles chaînes de dépendances existent, elles sont courtes et localisées à une section élémentaire du notebook.

Cela fait que chaque cellule possède (à quelque exceptions près, cf. ces micro-séquences locales), tous les imports et toutes les intructions de chargement qui permettent de l'exécuter indépendamment de ce qui précède.

Exemple
-------

```Python
from home_credit.load import get_pos_cash_balance
data = get_pos_cash_balance().copy()
data.MONTHS_BALANCE = -data.MONTHS_BALANCE
pivoted = data.pivot(
    index=["SK_ID_CURR", "SK_ID_PREV"],
    columns="MONTHS_BALANCE",
    values=["NAME_CONTRACT_STATUS", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF"]
)
display(pivoted)
```

Un mécanisme de cache fait que si `get_pos_cash_balance` a été appelée dans une cellule précedente, ou que vous exécutez une seconde fois cette cellule, le chargement du fichier n'aura pas lieu une seconde fois, et que votre temps d'attente en sera considérablement réduit.

