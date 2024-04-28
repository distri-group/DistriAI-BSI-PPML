# Introduction



Private Set Intersection (PSI) is an interactive computational protocol enabling two parties, namely the sender and the receiver, to determine the intersection of their respective private sets $X$ and $Y$, each with a predetermined size. The key feature of PSI is that the receiver exclusively gains knowledge of $X\cap Y$ from the interaction while the sender remains oblivious. Labeled PSI denotes a variant where the sender associates a label $l_i$ with each item $x_i$ in its set, and the receiver is intended to acquire $\{(x_i, l_i):x_i\in Y\}$ upon completion of the protocol. It is worth noting that labeled PSI is equivalent to Private Information Retrieval (PIR) based on keywords.



# Our Contributions



Our protocol can be viewed as a successor to FHE-based labeled PSI protocols.



- We introduce the concept of a collision-free hash map, which serves as a mapping between two sets, ensuring uniqueness and collision avoidance. Furthermore, we outline the ideal functionality of a collision-free hash map and provide a concrete implementation.

- We propose CFHM-APSI, an enhanced labeled PSI protocol offering malicious security guarantees. Our approach improves upon previous methods by substituting cuckoo hashing with CFHM, which resolves the data expansion issue stemming from the use of multiple hash functions in cuckoo hashing. Theoretically, the computational efficiency of constructing interpolation polynomials in the PSI process increases several-fold compared to the original protocol.



## CFHM (Collision-free hash map)



The sender and receiver hash the items in their sets into two hash tables using a predetermined deterministic hash function, such as cuckoo hashing. Subsequently, they only need to execute a PSI for each bin. To ensure accurate bin-wise comparisons leading to the correct intersection, the sender must insert each of its items into the table using all $h$ hash functions $H_1,...,H_h$. Consequently, the actual number of items inserted into the table is $hN_x$, which is $h$ times the original set size. This phenomenon, known as the data expansion problem, elevates the average degree of the interpolation polynomials from $B=N_x/(m\times \alpha)$ to $\hat B=hN_x/(m\times \alpha)$, significantly reducing the efficiency of the entire protocol.



CFHM facilitates the insertion of items into a fixed table by setting $B=[m]$, where $m$ denotes the table size. We provide a concrete realization of CFHM as follows: Assuming $hash()$ represents a hash function, we define $hash_s(x)$ as $hash(s||x)$. Subsequently, CFHM can be implemented by iteratively attempting different $s$ values until collision-free conditions are met. By replacing cuckoo hash with CFHM, the data expansion problem is effectively mitigated, as the actual number of items to be inserted by the sender remains consistent with the original dataset.



Receiver inputs set $Y\subset\{0,1\}^\sigma$ of size $N_y$; sender inputs set $X\subset\{0,1\}^{\sigma}$ of size $N_x$; $\lambda$ denotes the computational security parameter. The receiver outputs $Y\cap X$ together with its corresponding

labels; the sender outputs $\bot$.



### PreProcess Stage



1. [Generate Parameters]:



   (a) The sender generates OPRF secret key $k=OPRF.KeyGen(1^\lambda)$.



   (b) The receiver generates FHE key $(sk,p\hat{k})=FHE.KeyGen(parms)$ and CFHM parameter $para=CFHM_Gen(Y,m)$.



   (c) The receiver sends $pk$ and para to the sender.



2. [OPRF]:



   (a) The sender updates its set to be $X^\prime=\{OPRF.Cal(k,x):x\in X\}$.



   (b) The receiver performs the interactive OPRF protocol using its set $Y$ in a random order as private input. The sender uses the secret key $k$ as its private input. At the end of OPRF protocol, the receiver learns $OPRF.Cal(k,\dot{y})$ for $y\in Y$ and sets $Y^\prime=\{OPRF.Cal(k,y):y\in Y\}$.



3. [CFHM]:



   (a) The sender inserts all $x\in X^{\prime}$ into the Set $\mathcal{B}[CFHM(x,para)]$.



   (b) The recciver inserts all $y\in Y^{\prime}$ into the Set $\mathcal B(CFHM(y,para))$, and pads each empty bin with a dummy item.



4. [Construct Polynomial]:



   (a) For each set $\mathcal B[i]$, the sender splits it into $\alpha$ subsets, which are denoted as $\mathcal{B}[i,1],\cdots,\mathcal{B}[i,\alpha].$



   (b) For each set $B[i,j]$, the sender constructs a połynomial $S_{i,j}$ satisfying $S_{i,j}(x)=0$ for $x\in\mathcal{B}[i,j]$.



   (c) For the labels associated with each set $\mathcal{B}[i,j]$, the sender constnacts a polynomial $P_{ij}$ such that $P_{ij}(x)=l$ for $x\in\mathcal{B}[i,j]$ where $l$ is the label associated with $x$.



### Query Stage



5. [EncryptY]:



   (a) For cach $y\in Y^\prime$, the receiver computes $c_y=FHE.Encrypt(y,pk)$ and forwards the ciphertext set $C$ to the sender.



6. [Intersect]:



   (a) For cach $S_iy(x)$, $i\in[m]$, $j\in[\alpha]$, the sender homomorphically evaluates $u_{i,j,y}=S_{i,j}(c_{y})$, where $c_y$ ranges over $C.$



   (b) For each $P_ij(x),i\in[m],j\in[\alpha]$, the sender homomorphically evaluates $v_{i,j,y}=P_{i,j}(c_y)$, where $c_y$ ranges over $C.$ 



   (c) The sender sends all $u_{i,j}$ and $v_{i,j}$ back to the receiver. 



7. [Decrypt and get result]:



   (a) The sender computes $\widehat u_{i,j,y}=FHE.Decrypt(u_{i,j,y},sk)$ and $\widehat v_{i,j,y}=FHE.Decrypt(v_{i,j,y},sk)$.



   (b) If $\hat{u}_{i,j,y}=0$,the sender outputs $y$ and the corresponding label $\hat{v}_{i,j,y}$