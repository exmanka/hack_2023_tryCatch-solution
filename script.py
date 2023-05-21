# Нужные библиотеки и предобученные модели
import pickle
import joblib
import nltk
import numpy as np
import pandas as pd
# import catboost
from catboost import CatBoostClassifier
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

with open('columns_action.pkl', 'rb') as inp:
    columns_action = pickle.load(inp)

model1 = keras.models.load_model('model_mlp.h5')

model2 = CatBoostClassifier()
model2.load_model('model_catboost.cbm')


def find_class_swap(df, columns_action, model1):
    # Удаление ненужного столбца ID
    df.drop('id', axis=1, inplace=True)
    df_to_show = df.copy()

    # Инициализация инструментов NLTK
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

    # Удаление стоп-слов
    stop_words = set(stopwords.words('russian'))

    # Инициализация лемматизатора
    lemmatizer = WordNetLemmatizer()

    # Функция для обработки текста
    def process_text(text):
        # Приведение текста к нижнему регистру
        # print(text)
        text = text.lower()
        # Токенизация текста
        tokens = nltk.word_tokenize(text, language='russian')
        # Удаление стоп-слов и пунктуации
        filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        # Лемматизация токенов
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        # Соединение токенов обратно в текст
        processed_text = ' '.join(lemmatized_tokens)
        return processed_text

    # Используем наш Series columns_action, содержащий в себе предобученные модели для изменения датафрейма
    # 0 — колонка будет дропнута
    for col_name, action in zip(columns_action.index, columns_action.values):
        if action == 0:
            df.drop(col_name, axis=1, inplace=True)
        elif type(action) == type(OneHotEncoder()):
            df[col_name] = action.transform(df[[col_name]]).toarray().tolist()
        elif type(action) == type(LabelEncoder()):
            df[col_name] = action.transform(df[col_name])
        elif type(action) == type(CountVectorizer()):
            df[col_name] = action.transform(df[col_name].apply(process_text)).toarray().tolist()

    # Формируем из датафрейма с нестандартной размерностью нормальный тензор
    df_np = np.array(df)
    vector_len = len(np.hstack([df_np[0][j] for j in range(len(df_np[0]))]))
    x = np.empty([df_np.shape[0], vector_len])

    for i in range(len(x)):
        x[i] = np.hstack([df_np[i][j] for j in range(len(df_np[i]))])

    # Используем предобученную модель НС
    # pred = model1.predict(x)
    pred = model2.predict(x)

    # Преобразуем вероятности в классы
    pred_1 = np.array([1 if prob > 0.5 else 0 for prob in np.ravel(pred)])

    # Возвращаем датафрейм, содержащий в себе элементы с предсказанием изменения класса
    ans = df_to_show[pred_1 == 1]
    ans['Тип обращения итоговый'] = np.where(ans['Тип обращения на момент подачи'] == 'Запрос', 'Инцидент', 'Запрос')
    ans['Тип переклассификации'] = np.where(ans['Тип обращения на момент подачи'] == 'Запрос', 1, 2)

    return ans

# Пример работы
df = pd.read_csv('test.csv')
find_class_swap(df, columns_action, model1)