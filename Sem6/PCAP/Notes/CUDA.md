## Calculations of global threadID
### 1D Grid/1D Block
![[Pasted image 20260211114448.png]]
$$
threadId = (blockIdx.x * blockDim.x) + threadIdx.x
$$
#### Notation : <<<blockDim.x, threadDIm.x>>>
### 1D Grid/2D Block
![[Pasted image 20260211114743.png]]
$$
threadIdx = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.y * threadDim.x) + threadIdx.x
$$
#### Notation : <<<blockDim.x, (threadIdx.y, threadDIm.x)>>>
### 2D Grid/1D Block
![[Pasted image 20260211121251.png]]
$$
blockIdx = (grid)
$$