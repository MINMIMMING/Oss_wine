from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# 1. 데이터셋 불러오기
wine = load_wine()

X = wine.data
y = wine.target

print("데이터 크기:", X.shape)
print("타겟 크기:", y.shape)
print("클래스 이름:", wine.target_names)

# 2. 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=4
)

print("훈련 데이터 크기:", X_train.shape)
print("테스트 데이터 크기:", X_test.shape)

# 3. KNN 모델 학습
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)

# 4. 테스트 데이터 예측 및 정확도 평가
y_pred = knn.predict(X_test)

score = metrics.accuracy_score(y_test, y_pred)
print("정확도:", score)
print("정확도(%):", round(score * 100, 2), "%")

# 5. 새로운 데이터 예측
new_data = [
    [13.0, 2.0, 2.3, 18.0, 100.0, 2.5, 2.0, 0.3, 1.5, 5.0, 1.0, 3.0, 1000.0],
    [12.0, 1.5, 2.0, 20.0, 90.0, 1.8, 1.2, 0.4, 1.0, 3.0, 0.9, 2.5, 600.0]
]

new_pred = knn.predict(new_data)

print("새 데이터 예측 결과:")
print(wine.target_names[new_pred[0]])
print(wine.target_names[new_pred[1]])
