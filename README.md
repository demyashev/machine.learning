# Отчет по machine.learning
Третье задание из [курса](https://github.com/Ba-Ski/AI-in-IS-course) по машинному обучению.

## Код
Повторяющийся код вынесен в отдельные функции. А именно:

> def classify(X, y, clf, vectorizer)

    # Преобразование массива строк в структуру bag of words
    X_vector = vectorizer.fit_transform(X)

    # Обучение
    clf.fit(X_vector, y)

    # Оценка
    score = cross_val_score(clf, X_vector, y, cv=3)
    print('Accuracy: ', end='')
    print(score)
    print('Mean accuracy: ', end='')
    print(score.mean())

Преобразовывает массив строк письма в структуру Bag of words, обучает классификатор, производит проверку результатов с использованием кросс-валидации, выводим результаты:
* Accuracy - точность
* Mean accuracy - усредненная точность

> def compare(email_str0, email_str1, clf, vectorizer)

    # Классификация исходного письма
    Z = []
    Z.append(email_str0)
    Z_vector = vectorizer.transform(Z)
    label = clf.predict(Z_vector)[0]
    print('Predicted label: ', end='')
    print(label)

    # Классификация измененного письма
    Z = []
    Z.append(email_str1)
    Z_vector = vectorizer.transform(Z)
    label = clf.predict(Z_vector)[0]
    print('New label: ', end='')
    print(label)

Сравниваем письма, строим прогноз, выводим результаты.
* Predicted label - было
* New label - стало

## Выполнение задания
### Наивный Байес [пункт 0-2]
Читаем письма, с помощью наивного байесовского классификатора проводим обучение и оценку.

Получаем результат:

    Accuracy: [0.97983294 0.96901352 0.95111182]
    Mean accuracy: 0.9666527593717619

### Отравления Байеса [пункт 3]
Меняем письмо:

![Различие между оригинальным письмом (слева) и измененым (справа)](docs/diff.letters.jpg "Различие между оригинальным письмом (слева) и измененым (справа)")
*Различие между оригинальным письмом (слева) и измененым (справа)*

На первом (исходном) письме, обученный классификатор считает письмо спамом. Второе (измененное) тоже спам. Расстояние Левинштейна равно `82`

    First label: 0
    Edit distance: 82
    Predicted label: 0
    New label: 0

### Замена extract_email_text на load [пункт 4]

Загружаем письма с помощью метода `load()`,  а не `extract_email_text()`, проводим классификацию, сравниваем письма.

    Accuracy: [0.97820207 0.97295147 0.95497036]
    Mean accuracy: 0.9687079683156293
    Predicted label: 0
    New label: 0

Классификатор, после замены функции, отнес модифицированное письмо к спаму.

### Биграммы [пункт 5]

**CountVectorizer** с параметрами инициализации `ngram_range=(2, 2)`

    Accuracy: [0.98838504 0.98174224 0.97497912]
    Mean accuracy: 0.9817021344353768
    Predicted label: 0
    New label: 0

Точность повысилась, оба письма - спам.

**TfidfVectorizer**

    Accuracy: [0.97143994 0.97171838 0.9761327 ]
    Mean accuracy: 0.9730970052068706
    Predicted label: 0
    New label: 0

Точность меньше, чем при `CountVectorizer(ngram_range=(2, 2))`, но выше, чем у `байесовского`.

В обоих случаях "отравленное" письмо отнесли к спаму.

### Случайный лес [пункт 6-7]

    Accuracy: [0.98281623 0.98257757 0.97748518]
    Mean accuracy: 0.9809596590451125
    Predicted label: 0
    New label: 0

Усредненная точность выше `байесовского`, `TfidfVectorizer`, но чуть меньше `CountVectorizer` с биграммами. Оба письма - спам.

## Ссылки

* [Задание](https://github.com/Ba-Ski/AI-in-IS-course/tree/master/Homework%203)
* [Dataset](https://plg.uwaterloo.ca/~gvcormac/treccorpus07/)
