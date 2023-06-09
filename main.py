import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# CSV 파일 불러오기
df = pd.read_csv('Dataset.csv', encoding='utf-8', delimiter='\t')

# 테스트 데이터 파일 경로
test_data_file = './unlabeled/unlabeled_comments_1.txt'

# 테스트 데이터 읽어오기
with open(test_data_file, 'r', encoding='utf-8') as file:
    test_data = [line.strip() for line in file]

# stopwords = [
#     '은', '는', '이', '가', '을', '를', '에', '에서', '의', '과', '와', '으로', '로', '과정에서', '하는', '된', '그', '되는',
#     # Add more stopwords as needed
# ]
# # 불용어 제거 -> 0.7025
# df['content'] = df['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

# 구두점 제거
# import string
# punctuation = string.punctuation
# df['content'] = df['content'].apply(lambda x: ''.join([word for word in x if word not in (punctuation)]))

# 특징 엔지니어링: 댓글의 길이, 단어의 평균 길이, 단어 수 추가
# df['comment_length'] = df['content'].apply(lambda x: len(x))
# df['average_word_length'] = df['content'].apply(lambda x: np.mean([len(word) for word in x.split()]))
# df['word_count'] = df['content'].apply(lambda x: len(x.split()))

# 악플과 비악플로 분리
df_malicious = df[df['lable'] == 0]
df_non_malicious = df[df['lable'] == 1]

# 악플 데이터에서 자주 등장하는 단어 추출
vectorizer_malicious = CountVectorizer()
X_malicious = vectorizer_malicious.fit_transform(df_malicious['content'])
malicious_words = vectorizer_malicious.get_feature_names_out()

# 비악플 데이터에서 자주 등장하는 단어 추출
vectorizer_non_malicious = CountVectorizer()
X_non_malicious = vectorizer_non_malicious.fit_transform(df_non_malicious['content'])
non_malicious_words = vectorizer_non_malicious.get_feature_names_out()

# 가장 자주 등장하는 단어들을 합친 후 토크나이저에 추가
# malicious_words와 non_malicious_words의 길이를 맞춤
max_word_count = min(len(malicious_words), len(non_malicious_words))
malicious_words = list(set(malicious_words))[:max_word_count]
non_malicious_words = list(set(non_malicious_words))[:max_word_count]

# 두 리스트 결합
common_words = list(set(malicious_words + non_malicious_words))

# 입력 텍스트와 레이블 분리
X = df['content']  # 입력 텍스트
y = df['lable']  # 악플 여부(0: 악플, 1: 일반 텍스트)

# 정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

# 시퀀스 패딩
max_sequence_length = 100  # 시퀀스 최대 길이 설정
X = pad_sequences(X, maxlen=max_sequence_length)

# 레이블 인코딩
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_sequence_length))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# 모델 평가
print('\n테스트 정확도: %.4f' % (model.evaluate(X_test, y_test)[1]))

# # 테스트 데이터 정수 인코딩
# test_sequences = tokenizer.texts_to_sequences(test_data)
# test_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length)

# # 테스트 데이터 예측
# predictions = model.predict(test_sequences)
# predicted_labels = [1 if pred >= 0.5 else 0 for pred in predictions]

# f = open('output.txt', 'w', encoding='utf-8')

# # 예측 결과 출력
# for text, label in zip(test_data, predicted_labels):
#     if label == 1:
#         f.write(f"텍스트: {text}\n예측 결과: 악플\n")
#     else:
#         f.write(f"텍스트: {text}\n예측 결과: 일반 텍스트\n")

# f.close()