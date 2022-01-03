```python
import torch
import numpy as np
```

# Scalar
- 상수값 (차원이 없음)
- `torch.tensor`, `torch.FloatTensor` 함수를 통해 정의할 수 있음


```python
scalar1 = torch.tensor([1.])
print(scalar1)

scalar2 = torch.tensor([3.])    
print(scalar2)
```

    tensor([1.])
    tensor([3.])
    

- 사칙연산 수행 가능
- `torch` 모듈의 내장 함수를 통해서 사칙연산 수행 가능


```python
# 덧셈
add_scalar = scalar1 + scalar2
print(add_scalar)

# torch 모듈 사용
torch.add(scalar1, scalar2)
```

    tensor([4.])
    




    tensor([4.])




```python
# 뺄셈
sub_scalar = scalar1 - scalar2
print(sub_scalar)

# torch 모듈 사용
torch.sub(scalar1, scalar2)
```

    tensor([-2.])
    




    tensor([-2.])




```python
# 곱셈
mul_scalar = scalar1 * scalar2
print(mul_scalar)

# torch 모듈 사용
torch.mul(scalar1, scalar2)
```

    tensor([3.])
    




    tensor([3.])




```python
# 나눗셈
div_scalar = scalar1 / scalar2
print(div_scalar)

# torch 모듈 사용
torch.div(scalar1, scalar2)
```

    tensor([0.3333])
    




    tensor([0.3333])



# Vector
- scalar의 배열 (1차원)
- `torch.tensor`, `torch.FloatTensor`를 통해 벡터를 정의하는 것 또한 가능


```python
vector1 = torch.tensor([1., 2., 3.])  # numpy로도 정의할 수 있음 = np.array([1., 2., 3.])
print(vector1)

vector2 = torch.tensor([4., 5., 6.])
print(vector2)
```

    tensor([1., 2., 3.])
    tensor([4., 5., 6.])
    


```python
# 차원
vector1.ndim
```




    1




```python
# shape
vector1.shape
```




    torch.Size([3])



- 사칙연산 수행 가능
- `torch` 모듈의 내장 함수를 통해서 사칙연산 수행 가능


```python
# 덧셈
add_vector = vector1 + vector2
print(add_vector)

# torch 모듈 사용
torch.add(vector1, vector2)
```

    tensor([5., 7., 9.])
    




    tensor([5., 7., 9.])




```python
# 뺄셈
sub_vector = vector1 - vector2
print(sub_vector)

# torch 모듈 사용
torch.sub(vector1, vector2)
```

    tensor([-3., -3., -3.])
    




    tensor([-3., -3., -3.])




```python
# 곱셈
mul_vector = vector1 * vector2
print(mul_vector)

# torch 모듈 사용
torch.mul(vector1, vector2)
```

    tensor([ 4., 10., 18.])
    




    tensor([ 4., 10., 18.])




```python
# 나눗셈
div_vector = vector1 / vector2
print(div_vector)

# torch 모듈 사용
torch.div(vector1, vector2)
```

    tensor([0.2500, 0.4000, 0.5000])
    




    tensor([0.2500, 0.4000, 0.5000])




```python
# 내적 _ torch 모듈로만 가능
torch.dot(vector1, vector2)
```




    tensor(32.)



## ※ Broadcasting
- `pytorch`에서 제공하는 기능
- 두 tensor의 크기가 다를 때 자동적으로 사이즈를 맞춰주는 기능
- 자동적으로 사용되기 때문에 주의해야 할 필요가 있음


```python
# Vector + scalar
v1 = torch.FloatTensor([[1, 2]])
s1 = torch.FloatTensor([3]) # 3 -> [[3,3]]
print(v1 + s1)
```

    tensor([[4., 5.]])
    


```python
# 2 x 1 Vector + 1 x 2 Vector
v1 = torch.FloatTensor([[1, 2]])  # (1,2) -> (2,2)
v2 = torch.FloatTensor([[3], [4]]) # (2,1) -> (2,2)
print(v1 + v2)
```

    tensor([[4., 5.],
            [5., 6.]])
    

# Matrix
- 행렬 : 2개 이상의 벡터로 구성된 선형 대수의 기본 단위 (2차원)
- 2D Tensor |t| = (batch size, dim) = batch size * dimension
- `torch.tensor`, `torch.FloatTensor`를 통해 행렬 정의 가능


```python
matrix1 = torch.tensor([[1., 2.], [3., 4.]])  # numpy로도 정의할 수 있음 = np.array([[1., 2.],[3., 4.]])
print(matrix1)

matrix2 = torch.tensor([[5., 6.], [7., 8.]])
print(matrix2)
```

    tensor([[1., 2.],
            [3., 4.]])
    tensor([[5., 6.],
            [7., 8.]])
    


```python
# 차원
matrix1.ndim
```




    2




```python
# shape
matrix1.shape
```




    torch.Size([2, 2])




```python
# 평균 (전체 Element의 평균)
print('행렬의 평균 :', matrix1.mean())

# dimension 0의 평균
print('dim 0 평균 :', matrix1.mean(dim=0))

# dimension 1의 평균 (-1로 해도 결과는 같음)
print('dim 1 평균 :', matrix1.mean(dim=1))
```

    행렬의 평균 : tensor(2.5000)
    dim 0 평균 : tensor([2., 3.])
    dim 1 평균 : tensor([1.5000, 3.5000])
    


```python
# 덧셈(전체 Element의 합계)
print('행렬의 합계 :', matrix1.sum())

# dimension 0의 합계
print('dim 0 합계 :', matrix1.sum(dim=0))

# dimension 1의 합계 (-1로 해도 결과는 같음)
print('dim 1 합계 :', matrix1.sum(dim=1))
```

    행렬의 합계 : tensor(10.)
    dim 0 합계 : tensor([4., 6.])
    dim 1 합계 : tensor([3., 7.])
    


```python
# 최대값(전체 Element에서의 최대값)
print('행렬의 최대값 :', matrix1.max())
print(' ')

# dimension 0의 최대값
print(matrix1.max(dim=0))
print('Max (최대값) :', matrix1.max(dim=0)[0])
print('Argmax (최대값의 인덱스) :', matrix1.max(dim=0)[1])
print(' ')

# dimension 1의 최대값 (-1로 해도 결과는 같음)
print('dim 1 최대값 :', matrix1.max(dim=1))
```

    행렬의 최대값 : tensor(4.)
     
    torch.return_types.max(
    values=tensor([3., 4.]),
    indices=tensor([1, 1]))
    Max (최대값) : tensor([3., 4.])
    Argmax (최대값의 인덱스) : tensor([1, 1])
     
    dim 1 최대값 : torch.return_types.max(
    values=tensor([2., 4.]),
    indices=tensor([1, 1]))
    

- 사칙연산 수행 가능
- `torch` 모듈의 내장 함수를 통해서 사칙연산 수행 가능


```python
# 덧셈
sum_matrix = matrix1 + matrix2
print(sum_matrix)

# torch 모듈 사용
torch.add(matrix1, matrix2)
```

    tensor([[ 6.,  8.],
            [10., 12.]])
    




    tensor([[ 6.,  8.],
            [10., 12.]])




```python
# 뺄셈
sub_matrix = matrix1 - matrix2
print(sub_matrix)

# torch 모듈 사용
torch.sub(matrix1, matrix2)
```

    tensor([[-4., -4.],
            [-4., -4.]])
    




    tensor([[-4., -4.],
            [-4., -4.]])



- `*` 와 `/`의 경우, 행렬의 같은 위치에 존재하는 값들끼리 연산이 수행됨 (element-wise operation)


```python
# 곱셈
mul_matrix = matrix1 * matrix2
print(mul_matrix)

# torch 모듈 사용
torch.mul(matrix1, matrix2)
```

    tensor([[ 5., 12.],
            [21., 32.]])
    




    tensor([[ 5., 12.],
            [21., 32.]])




```python
# 나눗셈
div_matrix = matrix1 / matrix2
print(div_matrix)

# torch 모듈 사용
torch.div(matrix1, matrix2)
```

    tensor([[0.2000, 0.3333],
            [0.4286, 0.5000]])
    




    tensor([[0.2000, 0.3333],
            [0.4286, 0.5000]])



- 행렬 곱을 위해서는 `@` 연산 기호나 `torch.matmul` 함수를 사용하여야 함


```python
# 행렬곱 (= 내적)
matmul_matrix = matrix1 @ matrix2
print(matmul_matrix)

# torch 모듈 사용
torch.matmul(matrix1, matrix2)
```

    tensor([[19., 22.],
            [43., 50.]])
    




    tensor([[19., 22.],
            [43., 50.]])



# Tensor
- Vector를 1차원, Matrix를 2차원이라고 한다면, **Tensor는 3차원 이상의 배열을 의미**
  - 그냥 Vector, Matrix, Tensor를 전부 Tensor로 지칭하기도 함
- 3D Tensor
  - Vision |t| = (batch size, width, height) = batch size * width * height
  - NLP |t| = (batch size, length, dim) = batch size * length * dimension


```python
tensor1 = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
print(tensor1)

tensor2 = torch.tensor([[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]])
print(tensor2)
```

    tensor([[[1., 2.],
             [3., 4.]],
    
            [[5., 6.],
             [7., 8.]]])
    tensor([[[ 9., 10.],
             [11., 12.]],
    
            [[13., 14.],
             [15., 16.]]])
    

- Element-wise 사칙연산


```python
# 덧셈
sum_tensor = tensor1 + tensor2
print(sum_tensor)

# torch 모듈 사용
torch.add(tensor1, tensor2)
```

    tensor([[[10., 12.],
             [14., 16.]],
    
            [[18., 20.],
             [22., 24.]]])
    




    tensor([[[10., 12.],
             [14., 16.]],
    
            [[18., 20.],
             [22., 24.]]])




```python
# 뺄셈
sub_tensor = tensor1 - tensor2
print(sub_tensor)

# torch 모듈 사용
torch.sub(tensor1, tensor2)
```

    tensor([[[-8., -8.],
             [-8., -8.]],
    
            [[-8., -8.],
             [-8., -8.]]])
    




    tensor([[[-8., -8.],
             [-8., -8.]],
    
            [[-8., -8.],
             [-8., -8.]]])




```python
# 곱셈
mul_tensor = tensor1 * tensor2
print(mul_tensor)

# torch 모듈 사용
torch.mul(tensor1, tensor2)
```

    tensor([[[  9.,  20.],
             [ 33.,  48.]],
    
            [[ 65.,  84.],
             [105., 128.]]])
    




    tensor([[[  9.,  20.],
             [ 33.,  48.]],
    
            [[ 65.,  84.],
             [105., 128.]]])




```python
# 나눗셈
div_tensor = tensor1 / tensor2
print(div_tensor)

# torch 모듈 사용
torch.div(tensor1, tensor2)
```

    tensor([[[0.1111, 0.2000],
             [0.2727, 0.3333]],
    
            [[0.3846, 0.4286],
             [0.4667, 0.5000]]])
    




    tensor([[[0.1111, 0.2000],
             [0.2727, 0.3333]],
    
            [[0.3846, 0.4286],
             [0.4667, 0.5000]]])




```python
# 행렬곱
matmul_tensor = tensor1 @ tensor2
print(matmul_tensor)

# torch 모듈 사용
torch.matmul(tensor1, tensor2)
```

    tensor([[[ 31.,  34.],
             [ 71.,  78.]],
    
            [[155., 166.],
             [211., 226.]]])
    




    tensor([[[ 31.,  34.],
             [ 71.,  78.]],
    
            [[155., 166.],
             [211., 226.]]])



## ※ View (= Reshape)
- numpy에서 reshape와 같음
- 즉, 재배열 하는 것과 같아 원하는 형태로 재배열 할 수 있음
- -1을 변동이 심한 배치사이즈에서 사용하여 shape를 변경할 수 있음


```python
print(tensor1.shape)
```

    torch.Size([2, 2, 2])
    


```python
print(tensor1.view([-1, 2]))
print(tensor1.view([-1, 2]).shape)

print(tensor1.view([-1, 1, 2]))
print(tensor1.view([-1, 1, 2]).shape)
```

    tensor([[1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.]])
    torch.Size([4, 2])
    tensor([[[1., 2.]],
    
            [[3., 4.]],
    
            [[5., 6.]],
    
            [[7., 8.]]])
    torch.Size([4, 1, 2])
    

## ※ Squeeze & Unsqueeze
- squeeze : dimension의 element의 갯수가 1인 경우 해당 demension을 없애줌
- unsqueeze : 내가 원하는 dimension에 1을 넣어줌


```python
vector1 = torch.FloatTensor([[0],[1],[2]])
print(vector1)
print(vector1.shape)
```

    tensor([[0.],
            [1.],
            [2.]])
    torch.Size([3, 1])
    


```python
print(vector1.squeeze())
print(vector1.squeeze().shape)
```

    tensor([0., 1., 2.])
    torch.Size([3])
    


```python
# dimension을 지정하여 squeeze 함수를 실행할 수도 있음
print(vector1.squeeze(dim=0).shape)
print(vector1.squeeze(dim=1).shape)
```

    torch.Size([3, 1])
    torch.Size([3])
    


```python
vector2 = torch.FloatTensor([0, 1, 2])
print(vector2)
print(vector2.shape)
```

    tensor([0., 1., 2.])
    torch.Size([3])
    


```python
# dimension 0에 1을 넣어라
print('---- dim 0 ----')
print(vector2.unsqueeze(0))         # vector2.view(1,-1) 과 같음
print(vector2.unsqueeze(0).shape)

# dimension 1에 1을 넣어라
print('---- dim 1 ----')
print(vector2.unsqueeze(1))         # -1로 해도 결과 같음
print(vector2.unsqueeze(1).shape)
```

    ---- dim 0 ----
    tensor([[0., 1., 2.]])
    torch.Size([1, 3])
    ---- dim 1 ----
    tensor([[0.],
            [1.],
            [2.]])
    torch.Size([3, 1])
    

## ※ Type Casting
- tensor의 type을 변경해줌


```python
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
print(lt.float())
```

    tensor([1, 2, 3, 4])
    tensor([1., 2., 3., 4.])
    


```python
bt = torch.ByteTensor([True, False, False, True])
print(bt)
print(bt.long())
print(bt.float())
```

    tensor([1, 0, 0, 1], dtype=torch.uint8)
    tensor([1, 0, 0, 1])
    tensor([1., 0., 0., 1.])
    

## ※ Concatenate & Stacking
- tensor를 이어 붙여줌


```python
tensor1 = torch.FloatTensor([[1, 2], [3, 4]])
tensor2 = torch.FloatTensor([[5, 6], [7, 8]])

tensor3 = torch.FloatTensor([1, 4])
tensor4 = torch.FloatTensor([2, 5])
tensor5 = torch.FloatTensor([3, 6])
```


```python
print(torch.cat([tensor1, tensor2], dim=0))
print(torch.cat([tensor1, tensor2], dim=1))
```

    tensor([[1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.]])
    tensor([[1., 2., 5., 6.],
            [3., 4., 7., 8.]])
    


```python
print(torch.stack([tensor3, tensor4, tensor5]))
print(torch.stack([tensor3, tensor4, tensor5], dim=1))
print(torch.cat([tensor3.unsqueeze(0), tensor4.unsqueeze(0), tensor5.unsqueeze(0)], dim=0))
```

    tensor([[1., 4.],
            [2., 5.],
            [3., 6.]])
    tensor([[1., 2., 3.],
            [4., 5., 6.]])
    tensor([[1., 4.],
            [2., 5.],
            [3., 6.]])
    

## ※ Ones and Zeros
- 같은 사이즈의 1 또는 0으로만 이루어진 tensor를 생성
- 같은 device에 tensor를 선언


```python
tensor1 = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(tensor1)
```

    tensor([[0., 1., 2.],
            [2., 1., 0.]])
    


```python
print(torch.ones_like(tensor1))
print(torch.zeros_like(tensor1))
```

    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    

## ※ In-place Operation
- 새로운 변수에 할당하지 않고 연산을 수행


```python
print(tensor1.mul(2.)) # 곱하기 2를 해라
print(tensor1)
```

    tensor([[0., 2., 4.],
            [4., 2., 0.]])
    tensor([[0., 1., 2.],
            [2., 1., 0.]])
    


```python
print(tensor1.mul_(2.))
print(tensor1)
```

    tensor([[0., 2., 4.],
            [4., 2., 0.]])
    tensor([[0., 2., 4.],
            [4., 2., 0.]])
    
