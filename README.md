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
### Наивный Байес [Пункт 0-2]
Читаем письма, с помощью наивного байесовского классификатора проводим обучение и оценку.
    
    X, y = read_email_files()
    vectorizer = CountVectorizer()
    clf = MultinomialNB()
    classify(X, y, clf, vectorizer)

Получаем результат:

    Accuracy: [0.97983294 0.96901352 0.95111182]
    Mean accuracy: 0.9666527593717619

### Отравления Байеса [пункт 3]
Меняем письмо:

![Различие между оригинальным письмом (слева) и измененым (справа)](docs/diff.letters.jpg "Различие между оригинальным письмом (слева) и измененым (справа)")
*Различие между оригинальным письмом (слева) и измененым (справа)*

    filename = 'inmail.4'
    email_str0 = email_read_util.extract_email_text(os.path.join(DATA_DIR, filename))
    email_str1 = email_read_util.extract_email_text(os.path.join(filename))
    ind = X.index(email_str0)
    print('First label: ', end='')
    print(y[ind])
    print('Edit distance: ', end='')
    print(edit_distance(email_str0, email_str1))
    compare(email_str0, email_str1, clf, vectorizer)

На первом (исходном) письме, обученный классификатор считает письмо спамом. Второе (измененное) тоже спам. Расстояние Левенштейна равно `82`

    First label: 0
    Edit distance: 82
    Predicted label: 0
    New label: 0

### Замена extract_email_text на load [пункт 4]

Загружаем письма с помощью метода `load()`,  а не `extract_email_text()`, проводим классификацию, сравниваем письма.

    X, y = read_email_files_load()
    classify(X, y, clf, vectorizer)
    email_str0 = email_read_util.load(os.path.join(DATA_DIR, filename))
    email_str1 = email_read_util.load(os.path.join(filename))
    compare(email_str0, email_str1, clf, vectorizer)

Классификатор, после замены функции, отнес модифицированное письмо к спаму.

    Accuracy: [0.97820207 0.97295147 0.95497036]
    Mean accuracy: 0.9687079683156293
    Predicted label: 0
    New label: 0

### Биграммы [пункт 5]

**CountVectorizer** с параметрами инициализации `ngram_range=(2, 2)`
    
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    classify(X, y, clf, vectorizer)
    compare(email_str0, email_str1, clf, vectorizer)

Точность повысилась, оба письма - спам:

    Accuracy: [0.98838504 0.98174224 0.97497912]
    Mean accuracy: 0.9817021344353768
    Predicted label: 0
    New label: 0

**TfidfVectorizer**

    vectorizer = TfidfVectorizer()
    classify(X, y, clf, vectorizer)
    compare(email_str0, email_str1, clf, vectorizer)
    
В обоих случаях "отравленное" письмо отнесли к спаму:

    Accuracy: [0.97143994 0.97171838 0.9761327 ]
    Mean accuracy: 0.9730970052068706
    Predicted label: 0
    New label: 0

Точность меньше, чем при `CountVectorizer(ngram_range=(2, 2))`, но выше, чем у `байесовского`.

### Случайный лес [пункт 6-7]

    clf = RandomForestClassifier()
    classify(X, y, clf, vectorizer)
    compare(email_str0, email_str1, clf, vectorizer)

Усредненная точность выше `байесовского`, `TfidfVectorizer`, но чуть меньше `CountVectorizer` с биграммами. Оба письма - спам:

    Accuracy: [0.98281623 0.98257757 0.97748518]
    Mean accuracy: 0.9809596590451125
    Predicted label: 0
    New label: 0


## Ссылки

* [Задание](https://github.com/Ba-Ski/AI-in-IS-course/tree/master/Homework%203)
* [Dataset](https://plg.uwaterloo.ca/~gvcormac/treccorpus07/)
