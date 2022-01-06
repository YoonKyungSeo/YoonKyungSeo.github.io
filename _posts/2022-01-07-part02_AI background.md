---
layout: single
title: "D&A study part02_AI background"
---

# 01. 인공지능 > 머신러닝 > 딥러닝

![image.png](attachment:image.png)

### 1) 인공지능
- **인공지능이란?**  
인간의 지능으로 할 수 있는 사고 학습, 자기 개발 등을 컴퓨터가 할 수 있도록 하는 방법을 연구하는 컴퓨터 공학 및 정보 기술의 한 분야로, 컴퓨터가 인간의 지능적인 행동을 모방할 수 있도록 하는 것으로 머신러닝과 딥러닝이 모두 포함되는 기술이다.

### 2) 머신러닝
- **머신러닝이란?**  
기계 학습이라고도 불리며, 인공지능의 한 분야로 컴퓨터가 학습할 수 있도록 하는 알고리즘과 기술을 개발하는 분야이다. 최근의 머신러닝은 정형데이터를 이용하여 예측 도는 분류를 할 때에 사용된다.  
  
  
- **머신러닝 구분**  
    ① 지도학습  
    - x와 y 데이터가 모두 주어졌을 때 x로 y를 예측하는 학습 방법으로 x를 '독립변수' 혹은 Feature라고 하며, y를 '종속변수', '반응변수', '타깃변수'라 한다.
    - 지도학습 내에서도 `회귀(Regression) 문제`와 `분류(Classification) 문제` 두 가지 종류로 나뉜다.
        - 회귀 모델 : y가 실수형 값을 가질 때
        - 분류 모델 : y가 명목형 변수, 즉, 특정 Class를 가질 때  
  
  ② 비지도학습
    - 지도학습에서의 x와 y 변수가 모두 존재하는 것이 아닌 x 변수만 존재하는 것으로 데이터는 제공하되 명확한 답은 제공하지 않는 학습 방법이다.
    - 명확한 답이 없기 때문에 독립변수로 새로운 Feature를 찾아내거나 군집화 하는 등의 데이터 내의 새로운 패턴을 찾아내는 것에 초점을 맞춘다.
    - 대표적인 예 : 군집화(Clustering), 차원 축소법(Dimension Reduction)  
    
  ③ 강화학습  
    - 수많은 시뮬레이션을 통해 컴퓨터가 현재 상태에서 어떤 행동을 취해야 먼 미래의 보장을 최대로 할 것인지를 학습하는 알고리즘이다.  
    - 현재 상태(state), 행동(action), 보상(reward), 다음 상태(next state)가 있어야 함. 즉, 일련의 애피소드가 있어 시뮬레이션된 연속적인 데이터의 값이 존재해야 한다.
    - 대표적인 예 : 현재 바둑판에서 어떤 수를 둬야 먼 미래에 이길 수 있을지에 대한 학습이 진행되어 수업이 많은 경우이 수를 학습하기 위해 강화학습과 딥러닝을 결합한 심층 강화학습이 이용괸 알파고  

# 02. 인공신경망

### **1) 퍼셉트론(Percetron)**  
   - 최초의 인공지능 모형
   - 다수의 입력으로부터 하나의 결과를 내보내는 알고리즘으로 선형분류(Linear Classifier) 모형의 형태를 띤다.  
  ![image.png](attachment:image.png)
  <center> <font color=grey> [출처]https://untitledtblog.tistory.com/27  </font> </center>
  
    - Input과 Weight가 선형 결합 형태를 띠는 것을 볼 수 있다.
    - 선형 결합의 값에 특정 임계값의 초과 여부를 판단하는 함수를 적용하는데, 이 출력값이 0보다 크면 1, 작으면 -1(혹은 0)으로 결과값을 내보내는 분류 모형을 `퍼셉트론`이라 한다.
    - 퍼셉트론은 선형 분류 모형 형태를 띠기 때문에 선형 문제만 풀 수 있다.
    
※ '활성화 함수(Activation Function)' : 임계값의 초과 여부를 판단하는 함수  
  → 기본적인 Activation Function은 Step Function으로 Input값이 0 이상이면 1, 아니면 0을 출력하는 함수이다.

### - **Step function(계단 함수) 구현**


```python
# 인수 x는 실수(부동소수점)만 받아들임
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
    
# 계단 함수는 0이나 1을 출력하는 함수이기 때문에 y를 bool 배열로 생성하여 이를 int형으로 변경해줌
def step_function(x):
    y = x > 0
    return y.astype(np.int)
```

#### - **AND, OR, XOR**
① AND 게이트  
  
|$$x_1$$|$$x_2$$|$$y$$|
|---|---|---|
|0|0|0|
|1|0|0|
|0|1|0|
|1|1|1|

② NAND 게이트  
  
|$$x_1$$|$$x_2$$|$$y$$|
|---|---|---|
|0|0|1|
|1|0|1|
|0|1|1|
|1|1|0|

③ OR 게이트  
  
|$$x_1$$|$$x_2$$|$$y$$|
|---|---|---|
|0|0|0|
|1|0|1|
|0|1|1|
|1|1|1|

### - **퍼셉트론 구현 (AND 게이트)**


```python
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <=theta:
        return 0
    elif tmp> theta:
        return 1
```


```python
AND(0,0)  # 0을 출력
AND(1,0)  # 0을 출력
AND(0,1)  # 0을 출력
AND(1,1)  # 1을 출력
```

### - **가중치와 편향 도입 (AND 게이트)**


```python
def AND(x1, x2):
    x = np.array([x1, x2])      # 입력
    w = np.array([0.5, 0.5])  # 가중치
    b = -0.7                  # 편향
    tmp = np.sum(x*w) _ b
    if tmp <=0:
        return 0
    else:
        return 1
```

### - **가중치와 편향 도입 (NAND 게이트, OR 게이트)**


```python
def NAND(x1, x2):
    x = np.array([x1, x2])      # 입력
    w = np.array([-0.5, -0.5])  # 가중치
    b = 0.7                     # 편향
    tmp = np.sum(x*w) _ b
    if tmp <=0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])      # 입력
    w = np.array([0.5, 0.5])    # 가중치
    b = -0.2                    # 편향
    tmp = np.sum(x*w) _ b
    if tmp <=0:
        return 0
    else:
        return 1
```

### - **XOR 게이트**  
|$$x_1$$|$$x_2$$|$$y$$|
|---|---|---|
|0|0|0|
|1|0|1|
|0|1|1|
|1|1|0|  
  
- XOR 게이트는 퍼셉트론으로 구현할 수 없기 때문에 퍼셉트론을 층을 쌓아 `다층 퍼셉트론(MLP; Multi Layer Perceptron)`을 사용하여 구현해야 한다.  
  
- 다음은 NAND의 출력을 $s_1$, OR의 출력을 $s_2$로 하여 XOR을 만든 진리표이다.  

|$$x_1$$|$$x_2$$||$$s_1$$|$$s_2$$||$$y$$|
|---|---||---|---||---|
|0|0||1|0||0|
|1|0||1|1||1|
|0|1||1|1||1|
|1|1||0|1||0|

### - **XOR 게이트 구현**   


```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
```


```python
XOR(0, 0)   # 0을 출력
XOR(1, 0)   # 1을 출력
XOR(0, 1)   # 1을 출력
XOR(1, 1)   # 0을 출력
```

### **2) MLP(Multi Layer Perceptron)**
  - 퍼셉트론이 비션형 분류 문제는 풀지 못한다는 한계점을 극복하기 위해 여러 Layer를 쌓아올린 현태로 구성되어 있는 모델
  - 여러개의 퍼셉트론 조합과 그것들의 재조합으로 비선형적인 모형을 만들어낸 것으로 딥러닝의 기본 구조가 되는 신경망(Neural network)을 의미한다.  
  
  ![image.png](attachment:image.png)
  <center> <font color=grey> [출처] http://www.tcpschool.com/deep2018/deep2018_deeplearning_intro </font> </center>
     
     - Input 1개, Hidden 2개, Output 1개로 총 4개의 Layer로 연결되어 있는 MLP
     - 각 원은 `노드(Node)`라고 부르며, Input node의 수는 Input Data의 변수의 수, Hidden Layer와 Hidden node의 수는 사용자가 지정하는 하이퍼파라미터이다.
     - Output Node의 수는 풀고자 하는 문제에 따라 달라지는데 회귀 분석의 경우 1, 분류의 경우 Class의 수만큼이 된다.
         - 회귀에서 사용하는 일반적인 활성 함수 : 항등 함수
         - 분류에서 사용하는 일반적인 활성 함수 : softmax function
     

### - **softmax 함수 구현**  


```python
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```

     
#### - **Feed Forward (순전파)**  
Input에서 Weight와 Hidden을 거쳐 Output을 내보내는 과정  
  
  
#### - **Back Propagatoin (역전파)**  
  - Feed Foward를 이용해 우리가 얻고자 하는 예측값인 Output을 계산하고, 모델의 예측값과 실제 값의 차이(Error)를 계산하여 이 Error를 바탕으로 신경망의 Weight를 업데이트하는 과정  
  - Feed Foward와 Back Propagation을 반복하면서 Weight를 업데이트하며 점차 신경망의 Output이 실제값에 가까워지면서 모델 학습이 일어난다.
  - Feed Forward와 Back Propagation을 반복하면서 학습하는데, 이 반복하는 횟수를 `Epoch`라고 한다.  
  
  
#### - **활성함수**
어떤 신호를 받아 이를 적절히 처리해 출력해주는 함수로 신경망에서는 
  - Sigmoid Function(시그모이드)
    - 입력값이 0 이하이면 0.5 이하의 값을 출력하고, 0 이상이면 0.5 이상의 값을 출력하는 함수로, 즉, 입력값에 대해 0부터 1 사이로 Scaling 해주는 것으로 이해할 수 있다.
    - $S(x) = \frac{1}{1+e^{-x}}$
    

### - **sigmoid 함수 구현**


```python
def sigmoid(x):
    return 1 / (1+np.exp(-x))
```

#### - **Gradient Descent Method(기울기 경사 하강법)**
  - 가장 간단한 선형 회귀 모형을 가정하여 손실 함수를 MSE로 설정하고 MSE가 감소하도록 회귀계수를 추정한다면, MSE는 이차함수의 형태이기 때문에 미분하여 기울기가 0이 되는 지점을 찾을 수 있다. 즉, MSE가 최소가 되는 지점을 쉽게 찾을 수 있다.
  - MLP(신경망)의 경우 많은 Hidden Layer로 인해 모델이 복잡하여 한번에 기울기가 최소가 되는 지점을 찾기 힘들기 때문에 Feed Forward와 Back Propagation을 번갈아가며 학습을 진행하여 학습데이터에 대한 MSE를 줄여간다.
  - 이 과정에서 모든 데이터를 한 번에 학습하면 너무 많은 연산을 필요로 하는 컴퓨팅 문제가 발생하기 때문에 한번에 계산하지 않고 데이터를 쪼개서 학습한다.
    - 예를 들어 전체 데이터가 1,000개라 하면, 100개씩 쪼개 Feed Foward와 Back Propagation을 10번 반복한다. 이 한 과정을 `'Epoch'`이라 하고, 100개의 데이터를 `'Mini-Batch'`라 하며, 100의 크기에 대해서는 `'Batch Size'`라 한다. 이렇게 Gradient Descent Method하는 방법을 `'Stochastic Gradient Descent(SGD)'`라고 부르며, 이렇게 Gradient Descent 해주는 것을 통틀어 `'Optimizer'`라고 한다.  
    
    
#### - **손실 함수**   
  - 신경망 학습에서 사용하는 지표로, 손실함수가 작아지는 방향으로 모델이 학습된다.  
    
  ① MSE (평균 제곱 오차)
  
  ② Cross Entropy Error (교차 엔트로피 오차)  
      ②-1 Mini-Batch cross entropy error (미니 배치 교차 엔트로피 오차) : 데이터의 일부만을 학습에 사용
   

### - **손실함수 구현**  


```python
# MSE
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# cross entropy error
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# mini-batch cross entropy error
### num1
def cross_entropy_error(y, t):
    if y,ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y_size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

### num2
def cross_entropy_error(y, t):
    if y,ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y_size)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    # np.arange(batch_size)는 0부터 batch_size-1까지 배열을 생성
    # y[np.arange(batch_size), t]는 각 데이터의 정답 레이블에 해당하는 신경망의 출력을 추출
```

#### - **신경망 모형의 단점**  
① 과적합  
  
② Gradient Vanishing Problem  
- 기울기가 사라지는 현상으로, 예를 들면 sigmoid 함수를 미분한 값의 최대값은 0.25로 미분한 값을 계속해서 곱하게 되면 그 값이 0에 수렴하게 되어 Layer가 깊어질수록 Wight의 Gradient는 큰 변화가 없어지고 Weight의 변화도 거의 일어나지 않게 된다.  

### - **신경망 구현**


```python
# 입력층에서 1층으로 신호 전달
X = np.array([1.0, 0.5])
W1 = np.array([[0.1,0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1


# 활성 함수 통과
Z1 = sigmoid(A1)


# 1층에서 2층으로 신호 전달
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2


# 활성 함수 통과
Z2 = sigmoid(A2)


# 2층에서 출력층으로 신호 전달
def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array(0.1, 0.2)

A3 = np.dot(Z2, W3) + B3

Y = identity_function(A3)  # 혹은 Y = A3
```


```python
## 정리

def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array(0.1, 0.2)
    
    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    
    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
```
