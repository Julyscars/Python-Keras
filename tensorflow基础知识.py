a# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:30:46 2017

@author: jk
"""




1.矩阵操作

1.1矩阵生成
这部分主要将如何生成矩阵，包括全０矩阵，全１矩阵，随机数矩阵，常数矩阵等


tf.ones | tf.zeros  
  
tf.ones(shape,type=tf.float32,name=None)   
tf.zeros([2, 3], int32)   
用法类似，都是产生尺寸为shape的张量(tensor)  
  
sess = tf.InteractiveSession()  
x = tf.ones([2, 3], int32)  
print(sess.run(x))  
#[[1 1 1],  
# [1 1 1]]  


tf.ones_like | tf.zeros_like  
  
tf.ones_like(tensor,dype=None,name=None)   
tf.zeros_like(tensor,dype=None,name=None)   
新建一个与给定的tensor类型大小一致的tensor，其所有元素为1和0  
  
tensor=[[1, 2, 3], [4, 5, 6]]   
x = tf.ones_like(tensor)   
print(sess.run(x))  
#[[1 1 1],  
# [1 1 1]]  


tf.fill  
  
tf.fill(shape,value,name=None)   
创建一个形状大小为shape的tensor，其初始值为value  
  
print(sess.run(tf.fill([2,3],2)))  
#[[2 2 2],  
# [2 2 2]]  


tf.constant  
  
tf.constant(value,dtype=None,shape=None,name=’Const’)  
创建一个常量tensor，按照给出value来赋值，可以用shape来指定其形状。value可以是一个数，也可以是一个list。  
如果是一个数，那么这个常亮中所有值的按该数来赋值。   
如果是list,那么len(value)一定要小于等于shape展开后的长度。赋值时，先将value中的值逐个存入。不够的部分，则全部存入value的最后一个值。  
  
a = tf.constant(2,shape=[2])  
b = tf.constant(2,shape=[2,2])  
c = tf.constant([1,2,3],shape=[6])  
d = tf.constant([1,2,3],shape=[3,2])  
  
sess = tf.InteractiveSession()  
print(sess.run(a))  
#[2 2]  
print(sess.run(b))  
#[[2 2]  
# [2 2]]  
print(sess.run(c))  
#[1 2 3 3 3 3]  
print(sess.run(d))  
#[[1 2]  
# [3 3]  
# [3 3]]  


tf.random_normal | tf.truncated_normal | tf.random_uniform  
  
tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)  
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)  
tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None)  
这几个都是用于生成随机数tensor的。尺寸是shape   
random_normal: 正太分布随机数，均值mean,标准差stddev   
truncated_normal:截断正态分布随机数，均值mean,标准差stddev,不过只保留[mean-2*stddev,mean+2*stddev]范围内的随机数  
random_uniform:均匀分布随机数，范围为[minval,maxval]  
  
sess = tf.InteractiveSession()  
x = tf.random_normal(shape=[1,5],mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)  
print(sess.run(x))  
#===>[[-0.36128798 0.58550537 -0.88363433 -0.2677258 1.05080092]]  


tf.get_variable  
  
get_variable(name, shape=None, dtype=dtypes.float32, initializer=None,  
regularizer=None, trainable=True, collections=None,  
caching_device=None, partitioner=None, validate_shape=True,  
custom_getter=None):  
  
如果在该命名域中之前已经有名字=name的变量，则调用那个变量；如果没有，则根据输入的参数重新创建一个名字为name的变量。在众多的输入参数中，有几个是我已经比较了解的，下面来一一讲一下  
  
name: 这个不用说了，变量的名字   
shape: 变量的形状，[]表示一个数，[3]表示长为3的向量，[2,3]表示矩阵或者张量(tensor)  
dtype: 变量的数据格式，主要有tf.int32, tf.float32, tf.float64等等  
initializer: 初始化工具，有tf.zero_initializer, tf.ones_initializer, tf.constant_initializer, tf.random_uniform_initializer, tf.random_normal_initializer, tf.truncated_normal_initializer等  

1.2 矩阵变换

tf.shape  
  
tf.shape(Tensor)   
Returns the shape of a tensor.返回张量的形状。但是注意，tf.shape函数本身也是返回一个张量。而在tf中，张量是需要用sess.run(Tensor)来得到具体的值的。  
  
labels = [1,2,3]  
shape = tf.shape(labels)  
print(shape)  
sess = tf.InteractiveSession()  
print(sess.run(shape))  
# >>>Tensor("Shape:0", shape=(1,), dtype=int32)  
# >>>[3]  


tf.expand_dims  
  
tf.expand_dims(Tensor, dim)   
为张量+1维。官网的例子：’t’ is a tensor of shape [2]   
shape(expand_dims(t, 0)) ==> [1, 2]   
shape(expand_dims(t, 1)) ==> [2, 1]   
shape(expand_dims(t, -1)) ==> [2, 1]  
  
sess = tf.InteractiveSession()  
labels = [1,2,3]  
x = tf.expand_dims(labels, 0)  
print(sess.run(x))  
x = tf.expand_dims(labels, 1)  
print(sess.run(x))  
#>>>[[1 2 3]]  
#>>>[[1]  
# [2]  
# [3]]  


tf.pack  
  
tf.pack(values, axis=0, name=”pack”)   
Packs a list of rank-R tensors into one rank-(R+1) tensor  
将一个R维张量列表沿着axis轴组合成一个R+1维的张量。  
  
# 'x' is [1, 4]  
# 'y' is [2, 5]  
# 'z' is [3, 6]  
pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]] # Pack along first dim.  
pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]  


tf.concat  
  
tf.concat(concat_dim, values, name=”concat”)   
Concatenates tensors along one dimension.   
将张量沿着指定维数拼接起来。个人感觉跟前面的pack用法类似  
  
t1 = [[1, 2, 3], [4, 5, 6]]  
t2 = [[7, 8, 9], [10, 11, 12]]  
tf.concat(0, [t1, t2])   
#==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]  
tf.concat(1, [t1, t2])   
#==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]  


tf.sparse_to_dense  
  
稀疏矩阵转密集矩阵   
定义为：  
  
def sparse_to_dense(sparse_indices,  
output_shape,  
sparse_values,  
default_value=0,  
validate_indices=True,  
name=None):  
  
  
几个参数的含义：   
sparse_indices: 元素的坐标[[0,0],[1,2]] 表示(0,0)，和(1,2)处有值  
output_shape: 得到的密集矩阵的shape   
sparse_values: sparse_indices坐标表示的点的值，可以是0D或者1D张量。若0D，则所有稀疏值都一样。若是1D，则len(sparse_values)应该等于len(sparse_indices)  
default_values: 缺省点的默认值  


tf.random_shuffle  
  
tf.random_shuffle(value,seed=None,name=None)   
沿着value的第一维进行随机重新排列  
  
sess = tf.InteractiveSession()  
a=[[1,2],[3,4],[5,6]]  
x = tf.random_shuffle(a)  
print(sess.run(x))  
#===>[[3 4],[5 6],[1 2]]  


tf.argmax | tf.argmin  
  
tf.argmax(input=tensor,dimention=axis)   
找到给定的张量tensor中在指定轴axis上的最大值/最小值的位置。  
  
a=tf.get_variable(name='a',  
shape=[3,4],  
dtype=tf.float32,  
initializer=tf.random_uniform_initializer(minval=-1,maxval=1))  
b=tf.argmax(input=a,dimension=0)  
c=tf.argmax(input=a,dimension=1)  
sess = tf.InteractiveSession()  
sess.run(tf.initialize_all_variables())  
print(sess.run(a))  
#[[ 0.04261756 -0.34297419 -0.87816691 -0.15430689]  
# [ 0.18663144 0.86972666 -0.06103253 0.38307118]  
# [ 0.84588599 -0.45432305 -0.39736366 0.38526249]]  
print(sess.run(b))  
#[2 1 1 2]  
print(sess.run(c))  
#[0 1 0]  


tf.equal  
  
tf.equal(x, y, name=None):   
判断两个tensor是否每个元素都相等。返回一个格式为bool的tensor  
  
tf.cast  
  
cast(x, dtype, name=None)   
将x的数据格式转化成dtype.例如，原来x的数据格式是bool，那么将其转化成float以后，就能够将其转化成0和1的序列。反之也可以  
  
a = tf.Variable([1,0,0,1,1])  
b = tf.cast(a,dtype=tf.bool)  
sess = tf.InteractiveSession()  
sess.run(tf.initialize_all_variables())  
print(sess.run(b))  
#[ True False False True True]  


tf.matmul  
  
用来做矩阵乘法。若a为l*m的矩阵，b为m*n的矩阵，那么通过tf.matmul(a,b) 结果就会得到一个l*n的矩阵  
不过这个函数还提供了很多额外的功能。我们来看下函数的定义：  
  
matmul(a, b,  
transpose_a=False, transpose_b=False,  
a_is_sparse=False, b_is_sparse=False,  
name=None):  
  
可以看到还提供了transpose和is_sparse的选项。   
如果对应的transpose项为True，例如transpose_a=True,那么a在参与运算之前就会先转置一下。  
而如果a_is_sparse=True,那么a会被当做稀疏矩阵来参与运算。  


tf.reshape  
  
reshape(tensor, shape, name=None)   
顾名思义，就是将tensor按照新的shape重新排列。一般来说，shape有三种用法：   
如果 shape=[-1], 表示要将tensor展开成一个list   
如果 shape=[a,b,c,…] 其中每个a,b,c,..均>0，那么就是常规用法   
如果 shape=[a,-1,c,…] 此时b=-1，a,c,..依然>0。这表示tf会根据tensor的原尺寸，自动计算b的值。  
官方给的例子已经很详细了，我就不写示例代码了  
  
# tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]  
# tensor 't' has shape [9]  
reshape(t, [3, 3]) ==> [[1, 2, 3],  
[4, 5, 6],  
[7, 8, 9]]  
  
# tensor 't' is [[[1, 1], [2, 2]],  
# [[3, 3], [4, 4]]]  
# tensor 't' has shape [2, 2, 2]  
reshape(t, [2, 4]) ==> [[1, 1, 2, 2],  
[3, 3, 4, 4]]  
  
# tensor 't' is [[[1, 1, 1],  
# [2, 2, 2]],  
# [[3, 3, 3],  
# [4, 4, 4]],  
# [[5, 5, 5],  
# [6, 6, 6]]]  
# tensor 't' has shape [3, 2, 3]  
# pass '[-1]' to flatten 't'  
reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]  
  
# -1 can also be used to infer the shape  
# -1 is inferred to be 9:  
reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],  
[4, 4, 4, 5, 5, 5, 6, 6, 6]]  
  
# -1 is inferred to be 2:  
reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],  
[4, 4, 4, 5, 5, 5, 6, 6, 6]]  
  
# -1 is inferred to be 3:  
reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],  
[2, 2, 2],  
[3, 3, 3]],  
[[4, 4, 4],  
[5, 5, 5],  
[6, 6, 6]]]  

2. 神经网络相关操作

tf.nn.embedding_lookup  
  
embedding_lookup(params, ids, partition_strategy=”mod”, name=None,  
validate_indices=True):  
  
简单的来讲，就是将一个数字序列ids转化成embedding序列表示。   
假设params.shape=[v,h], ids.shape=[m], 那么该函数会返回一个shape=[m,h]的张量。用数学来表示，就是  
  
ids=[i1,i2,…,im]params=⎡⎣⎢⎢⎢⎢⎢w11,w21,⋯,wh1w12,w22,⋯,wh2⋮w1v,w2v,⋯,whv⎤⎦⎥⎥⎥⎥⎥res=⎡⎣⎢⎢⎢⎢⎢⎢w1i1,w2i1,…,whi1w1i2,w2i2,…,whi2⋮w1im,w2im,…,whim⎤⎦⎥⎥⎥⎥⎥⎥  
  
那么这个有什么用呢？如果你了解word2vec的话，就知道我们可以根据文档来对每个单词生成向量。单词向量可以进一步用来测量单词的相似度等等。那么假设我们现在已经获得了每个单词的向量，都存在param中。那么根据单词id序列ids,就可以通过embedding_lookup来获得embedding表示的序列。  


tf.trainable_variables  
  
返回所有可训练的变量。   
在创造变量(tf.Variable, tf.get_variable 等操作)时，都会有一个trainable的选项，表示该变量是否可训练。这个函数会返回图中所有trainable=True的变量。  
tf.get_variable(…), tf.Variable(…)的默认选项是True, 而 tf.constant(…)只能是False  
  
import tensorflow as tf  
from pprint import pprint  
  
a = tf.get_variable('a',shape=[5,2]) # 默认 trainable=True  
b = tf.get_variable('b',shape=[2,5],trainable=False)  
c = tf.constant([1,2,3],dtype=tf.int32,shape=[8],name='c') # 因为是常量，所以trainable=False  
d = tf.Variable(tf.random_uniform(shape=[3,3]),name='d')  
tvar = tf.trainable_variables()  
tvar_name = [x.name for x in tvar]  
print(tvar)  
# [<tensorflow.python.ops.variables.Variable object at 0x7f9c8db8ca20>, <tensorflow.python.ops.variables.Variable object at 0x7f9c8db8c9b0>]  
print(tvar_name)  
# ['a:0', 'd:0']  
  
sess = tf.InteractiveSession()  
sess.run(tf.initialize_all_variables())  
pprint(sess.run(tvar))  
#[array([[ 0.27307487, -0.66074866],  
# [ 0.56380701, 0.62759042],  
# [ 0.50012994, 0.42331111],  
# [ 0.29258847, -0.09185416],  
# [-0.35913971, 0.3228929 ]], dtype=float32),  
# array([[ 0.85308731, 0.73948073, 0.63190091],  
# [ 0.5821209 , 0.74533939, 0.69830012],  
# [ 0.61058474, 0.76497936, 0.10329771]], dtype=float32)]  


tf.gradients  
  
用来计算导数。该函数的定义如下所示  
  
def gradients(ys,  
xs,  
grad_ys=None,  
name="gradients",  
colocate_gradients_with_ops=False,  
gate_gradients=False,  
aggregation_method=None):  


tf.trainable_variables  
  
返回所有可训练的变量。   
在创造变量(tf.Variable, tf.get_variable 等操作)时，都会有一个trainable的选项，表示该变量是否可训练。这个函数会返回图中所有trainable=True的变量。  
tf.get_variable(…), tf.Variable(…)的默认选项是True, 而 tf.constant(…)只能是False  
  
import tensorflow as tf  
from pprint import pprint  
  
a = tf.get_variable('a',shape=[5,2]) # 默认 trainable=True  
b = tf.get_variable('b',shape=[2,5],trainable=False)  
c = tf.constant([1,2,3],dtype=tf.int32,shape=[8],name='c') # 因为是常量，所以trainable=False  
d = tf.Variable(tf.random_uniform(shape=[3,3]),name='d')  
tvar = tf.trainable_variables()  
tvar_name = [x.name for x in tvar]  
print(tvar)  
# [<tensorflow.python.ops.variables.Variable object at 0x7f9c8db8ca20>, <tensorflow.python.ops.variables.Variable object at 0x7f9c8db8c9b0>]  
print(tvar_name)  
# ['a:0', 'd:0']  
  
sess = tf.InteractiveSession()  
sess.run(tf.initialize_all_variables())  
pprint(sess.run(tvar))  
#[array([[ 0.27307487, -0.66074866],  
# [ 0.56380701, 0.62759042],  
# [ 0.50012994, 0.42331111],  
# [ 0.29258847, -0.09185416],  
# [-0.35913971, 0.3228929 ]], dtype=float32),  
# array([[ 0.85308731, 0.73948073, 0.63190091],  
# [ 0.5821209 , 0.74533939, 0.69830012],  
# [ 0.61058474, 0.76497936, 0.10329771]], dtype=float32)]  


tf.gradients  
  
用来计算导数。该函数的定义如下所示  
  
def gradients(ys,  
xs,  
grad_ys=None,  
name="gradients",  
colocate_gradients_with_ops=False,  
gate_gradients=False,  
aggregation_method=None):  
  
虽然可选参数很多，但是最常使用的还是ys和xs。根据说明得知，ys和xs都可以是一个tensor或者tensor列表。而计算完成以后，该函数会返回一个长为len(xs)的tensor列表，列表中的每个tensor是ys中每个值对xs[i]求导之和。如果用数学公式表示的话，那么 g = tf.gradients(y,x)可以表示成  
  
gi=∑j=0len(y)∂yj∂xig=[g0,g1,...,glen(x)]  

[python] view plain copy
tf.clip_by_global_norm  
  
修正梯度值，用于控制梯度爆炸的问题。梯度爆炸和梯度弥散的原因一样，都是因为链式法则求导的关系，导致梯度的指数级衰减。为了避免梯度爆炸，需要对梯度进行修剪。  
先来看这个函数的定义：  
  
def clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None):  
  
  
  
  
输入参数中：t_list为待修剪的张量, clip_norm 表示修剪比例(clipping ratio).  
  
函数返回2个参数： list_clipped，修剪后的张量，以及global_norm，一个中间计算量。当然如果你之前已经计算出了global_norm值，你可以在use_norm选项直接指定global_norm的值。  
  
那么具体如何计算呢？根据源码中的说明，可以得到   
list_clipped[i]=t_list[i] * clip_norm / max(global_norm, clip_norm),其中  
global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))  
  
如果你更熟悉数学公式，则可以写作   
  
Lic=Lit∗Ncmax(Nc,Ng)Ng=∑i(Lit)2−−−−−−−√  
  
其中，   
Lic和Lig代表t_list[i]和list_clipped[i]，   
Nc和Ng代表clip_norm 和 global_norm的值。   
其实也可以看到其实Ng就是t_list的L2模。上式也可以进一步写作   
  
Lic={Lit,(Ng<=Nc)Lit∗NcNg,(Ng>Nc)Ng=∑i(Lit)2−−−−−−−√  
  
也就是说，当t_list的L2模大于指定的Nc时，就会对t_list做等比例缩放  


tf.nn.dropout  
  
dropout(x, keep_prob, noise_shape=None, seed=None, name=None)  
按概率来将x中的一些元素值置零，并将其他的值放大。用于进行dropout操作，一定程度上可以防止过拟合   
x是一个张量，而keep_prob是一个（0,1]之间的值。x中的各个元素清零的概率互相独立，为1-keep_prob,而没有清零的元素，则会统一乘以1/keep_prob, 目的是为了保持x的整体期望值不变。  
  
sess = tf.InteractiveSession()  
a = tf.get_variable('a',shape=[2,5])  
b = a  
a_drop = tf.nn.dropout(a,0.8)  
sess.run(tf.initialize_all_variables())  
print(sess.run(b))  
#[[ 0.28667903 -0.66874665 -1.14635754 0.88610041 -0.55590457]  
# [-0.29704338 -0.01958954 0.80359757 0.75945008 0.74934876]]  
print(sess.run(a_drop))  
#[[ 0.35834879 -0.83593333 -1.43294692 1.10762548 -0. ]  
# [-0.37130421 -0. 0. 0.94931257 0.93668592]]  

3.普通操作
[python] view plain copy
tf.linspace | tf.range  
  
tf.linspace(start,stop,num,name=None)   
tf.range(start,limit=None,delta=1,name=’range’)   
这两个放到一起说，是因为他们都用于产生等差数列，不过具体用法不太一样。   
tf.linspace在[start,stop]范围内产生num个数的等差数列。不过注意，start和stop要用浮点数表示，不然会报错  
tf.range在[start,limit)范围内以步进值delta产生等差数列。注意是不包括limit在内的。  
  
sess = tf.InteractiveSession()  
x = tf.linspace(start=1.0,stop=5.0,num=5,name=None) # 注意1.0和5.0  
y = tf.range(start=1,limit=5,delta=1)  
print(sess.run(x))  
print(sess.run(y))  
#===>[ 1. 2. 3. 4. 5.]  
#===>[1 2 3 4]  


tf.assign  
  
assign(ref, value, validate_shape=None, use_locking=None, name=None)  
tf.assign是用来更新模型中变量的值的。ref是待赋值的变量，value是要更新的值。即效果等同于 ref = value  
简单的实例代码见下  
  
sess = tf.InteractiveSession()  
  
a = tf.Variable(0.0)  
b = tf.placeholder(dtype=tf.float32,shape=[])  
op = tf.assign(a,b)  
  
sess.run(tf.initialize_all_variables())  
print(sess.run(a))  
# 0.0  
sess.run(op,feed_dict={b:5.})  
print(sess.run(a))  
# 5.0  

4.规范化

tf.variable_scope  
  
简单的来讲，就是为变量添加命名域  
  
with tf.variable_scope("foo"):  
with tf.variable_scope("bar"):  
v = tf.get_variable("v", [1])  
assert v.name == "foo/bar/v:0"  
  
函数的定义为  
  
def variable_scope(name_or_scope, reuse=None, initializer=None,  
regularizer=None, caching_device=None, partitioner=None,  
custom_getter=None):  
  
各变量的含义如下：   
name_or_scope: string or VariableScope: the scope to open.  
reuse: True or None; if True, we Go into reuse mode for this scope as well as all sub-scopes; if None, we just inherit the parent scope reuse. 如果reuse=True, 那么就是使用之前定义过的name_scope和其中的变量，  
initializer: default initializer for variables within this scope.  
regularizer: default regularizer for variables within this scope.  
caching_device: default caching device for variables within this scope.  
partitioner: default partitioner for variables within this scope.  
custom_getter: default custom getter for variables within this scope.  


tf.get_variable_scope  
  
返回当前变量的命名域，返回一个tensorflow.Python.ops.variable_scope.VariableScope变量。  
1，tensorflow的基本运作

为了快速的熟悉TensorFlow编程，下面从一段简单的代码开始：

import tensorflow as tf
 #定义‘符号’变量，也称为占位符
 a = tf.placeholder("float")
 b = tf.placeholder("float")

 y = tf.mul(a, b) #构造一个op节点

 sess = tf.Session()#建立会话
 #运行会话，输入数据，并计算节点，同时打印结果
 print sess.run(y, feed_dict={a: 3, b: 3})
 # 任务完成, 关闭会话.
 sess.close()
1
2
3
4
五
6
7
8
9
10
11
12
其中tf.mul（a，b）函数便是tf的一个基本的算数运算，接下来介绍跟多的相关函数。

2，TF函数

TensorFlow 将图形定义转换成分布式执行的操作, 以充分利用可用的计算资源(如 CPU 或 GPU。一般你不需要显式指定使用 CPU 还是 GPU, TensorFlow 能自动检测。如果检测到 GPU, TensorFlow 会尽可能地利用找到的第一个 GPU 来执行操作.
并行计算能让代价大的算法计算加速执行，TensorFlow也在实现上对复杂操作进行了有效的改进。大部分核相关的操作都是设备相关的实现，比如GPU。下面是一些重要的操作/核：
1
2
3
操作组	操作
数学	添加，子，Mul，Div，Exp，日志，更多，更少，平等
排列	Concat，切片，错开，常数，等级，形状，随机播放
矩阵	MatMul，MatrixInverse，MatrixDeterminant
神经元网络	SoftMax，Sigmoid，ReLU，Convolution2D，MaxPool
检查点	保存，恢复
队列和同步	排队，出列，互斥量，互斥量释放
流量控制	合并，切换，回车，离开，NextIteration
TensorFlow的算术操作如下：

操作	描述
tf.add（x，y，name = None）	求和
tf.sub（x，y，name = None）	减法
tf.mul（x，y，name = None）	乘法
tf.div（x，y，name = None）	除法
tf.mod（x，y，name = None）	取模
tf.abs（x，name = None）	求绝对值
tf.neg（x，name = None）	取负（y = -x）。
tf.sign（x，name = None）	返回符号y = sign（x）= -1如果x <0; 0如果x == 0; 1如果x> 0。
tf.inv（x，name = None）	取反
tf.square（x，name = None）	计算平方（y = x * x = x ^ 2）。
tf.round（x，name = None）	舍入最接近的整数
＃ 'a'是[0.9，2.5，2.3，-4.4] 
tf.round的（a）==> [1.0，3.0，2.0，-4.0]
tf.sqrt（x，name = None）	开根号（y = \ sqrt {x} = x ^ {1/2}）。
tf.pow（x，y，name = None）	幂次方
＃张量'x'是[[2,2]，[3，3]] 
＃张量'y'是[[8,16]，[2,3]] 
tf.pow（x，y）= => [[256,65536]，[9,27]]
tf.exp（x，name = None）	计算ê的次方
tf.log（x，name = None）	计算日志，一个输入计算ê的LN，两输入以第二输入为底
tf.maximum（x，y，name = None）	返回最大值（x> y？x：y）
tf.minimum（x，y，name = None）	返回最小值（x <y？x：y）
tf.cos（x，name = None）	三角函数余弦
tf.sin（x，name = None）	三角函数正弦
tf.tan（x，name = None）	三角函数棕褐色
tf.atan（x，name = None）	三角函数CTAN
张量操作张量转换

数据类型转换铸造
操作	描述
tf.string_to_number 
（string_tensor，out_type = None，name = None）	字符串转为数字
tf.to_double（x，name ='ToDouble'）	转为64位浮点类型-float64
tf.to_float（x，name ='ToFloat'）	转为32位浮点类型-float32
tf.to_int32（x，name ='ToInt32'）	转为32位整型-int32
tf.to_int64（x，name ='ToInt64'）	转为64位整型-int64
tf.cast（x，dtype，name = None）	将x或者x.values转换为dtype 
＃张量a为[1.8,2.2]，dtype = tf.float 
tf.cast（a，tf.int32）==> [1,2]＃dtype = tf.int32
形状操作Shapes and Shaping
操作	描述
tf.shape（输入，名称=无）	返回数据的形状
＃'t'是[[[ 1,1,1 ]，[2,2,2]]，[[3,3,3]，[4,4,4]] 
形状（t） ==> [2，2，3]
tf.size（输入，名称=无）	返回数据的元素数量
＃ 'T'是[[[1，1，1]，[2，2，2]]，[[3，3,3]，[4,4，4]]]] 
大小（ t）==> 12
tf.rank（输入，名称=无）	返回张量的秩
注意：此排名不同于矩阵的秩，
张量的秩表示一个张量的需要索引数目来唯一表示任何一个元素
也。就是通常所说的“秩序”，“度”或”为ndims” 
＃“T是[[1,1,1,2,2,2]]，[[3,3,3]，[4,4,4]] 
张形't'的形状是[2， 2，3] 
rank（t）==> 3
tf.reshape（张量，形状，名称=无）	改变张量的形状
＃张量't'是[1,2,3,4,5,6,7,8,9] 
张量't'的形状[9] 
reshape（t，[3，3]）= => 
[[1,2,3]，
[4,5,6]，
[7,8,9]] 
＃如果shape有元素[-1]，表示在该维度打平至一维
＃-1自动推导得到9：
reshape（t，[2，-1]）==> 
[[ 1,1,1,2,2,3,3,3 ]，[ 4,4,4,5 
， 5，5，6，6，6]]
tf.expand_dims（输入，暗淡，名称=无）	插入维度1进入一个张量中
＃该操作要求-1-input.dims（）
＃'t'是shape [2] 
shape（expand_dims（t，0））==> [1,2] 
形状expand_dims（t，1））==> [2,1] 
shape（expand_dims（t，-1））==> [2,1] <= dim <= input.dims（）
切片与合并（切片与连接）
操作	描述
tf.slice（input_，begin，size，name = None）	对张量进行切片操作
其中size [i] = input.dim_size（i） - begin [i] 
该操作要求0 <= begin [i] <= begin [i] + size [i] <= Di for i in [ 0，n] 
＃'input'是
＃[[[1,1,1]，[2,2,2]]，[[3,3,3]，[4,4,4]]，[[5 ，[5，5]，[ 
6，6，6 ]] tf.slice（输入，[1,0,0 ]，[ 1,1，3]）==> [[[3，3，3]] ] 
tf.slice（input，[1,0,0]，[1,2,3]）==> 
[[[ 3，3，3 ]，
[ 
4,4,4 ]] tf.slice（input ，[1,0,0]，[2,1,3]）==> 
[[[ 3，3，3 ]]，
[[5,5,5]]]
tf.split（split_dim，num_split，value，name ='split'）	沿着某一维度将张量分离为num_split tensors 
＃'value'是一个具有形状的张量[ 5，30 ] 
＃沿着维度1将'值'拆分成3个张量
split0，split1，split2 = tf.split（1,3，值）
tf.shape（split0）==> [5,10]
tf.concat（concat_dim，values，name ='concat'）	沿着某一维度连结张量
t1 = [[1,2,3]，[4,5,6]] 
t2 = [[7,8,9]，[10,11,12]] 
tf.concat（0 ，[t1，t2]）==> [[1,2,3]，[4,5,6]，[7,8,9]，[10,11,12]] 
tf.concat（1， t1，t2]）==> [[ 
1,2,3，7，8，9，4，5，6，10，11，12 ]] 如果想沿着张量一新轴连结打包，那么可以：
tf.concat（axis，[tf.expand_dims（t，axis）for t in 
tensors ]）等同于tf.pack（tensors，axis = axis）
tf.pack（values，axis = 0，name ='pack'）	将R的张量打包为一个等级 - （R + 1）的张量
＃'x'是[1，4]，'y'是[2,5]，'z'是[3，6] 
（[x，y，z]）=> [[1,4]，[2,5]，[3,6]] 
＃沿着第一维pack 
pack（[x，y，z]，axis = 1）=> [[1,2,3]，[4,5,6]] 
等价于tf.pack（[x，y，z]）= np.asarray（[x，y，z]）
tf.reverse（张量，变暗，名称=无）	沿着某维度进行序列反转
其中暗淡为列表，元素为bool型，尺寸等于秩（张量）
＃张量't' 
[[[[0,1,2,3]，
＃[4,5,6 ，7]，

＃[8,9,10,11]]，
＃[[12,13,14,15]，
＃[16,17,18,19]，
＃[20,21,22,23]] ]] 
＃张量't'形状是[1,2,3,4] 
＃'dims'是[False，False，False，True] 
reverse（t，dims）==> 
[[[[ 3,2,1 ，
[0，7，6，5，4，11，10，9，8 
]，
[[15,14,13,12]，
[19,18,17,16]，
[23,22 ，21,20]]]]
tf.transpose（a，perm = None，name ='转置'）	调整张量的维度顺序
按照列表调用张量的顺序，
如定义，则perm为（n-1 ... 0）
＃'x'[[1 2 3]，[4 5 6]] 
tf.transpose （x）==> [[1 4]，[2 5]，[3 6]] 
＃等价于
tf.transpose（x，perm = [1,0]）==> [[1 4]，[2 5 ]，[3 6]]
tf.gather（params，indices，validate_indices = None，name = None）	合并索引指数所指示PARAMS中的切片
tf.gather
tf.one_hot 
（指数，深度，on_value =无，off_value =无，
axis =无，dtype =无，名称=无）	index = [0，2，-1，1] 
depth = 3 
on_value = 5.0 
off_value = 0.0 
axis = -1 
＃然后输出为[4 x 3]：
output = 
[5.0 0.0 0.0] // one_hot（0） 
[0.0 
（1）
[0.0 5.0 0.0] // one_hot（2）[0.0 0.0 0.0] // one_hot（-1）[0.0 5.0 0.0] // one_hot（1）
矩阵相关运算

操作	描述
tf.diag（对角线，名称=无）	返回一个给定对角值的对角张量
＃ '对角线'是[1，2，3，4] 
tf.diag（对角线）==> 
[[1，0，0，0] 
[0，2,0 ，0] 
[0,0,3,0] 
[0,0,0,4]]
tf.diag_part（输入，名称=无）	功能与上面相反
tf.trace（x，name = None）	求一个2维张量足迹，即对角值的对角线之和
tf.transpose（a，perm = None，name ='转置'）	调整张量的维度顺序
按照列表调用张量的顺序，
如定义，则perm为（n-1 ... 0）
＃'x'[[1 2 3]，[4 5 6]] 
tf.transpose （x）==> [[1 4]，[2 5]，[3 6]] 
＃等价于
tf.transpose（x，perm = [1,0]）==> [[1 4]，[2 5 ]，[3 6]]
tf.matmul（a，b，transpose_a = False，
transpose_b = False，a_is_sparse = False，
b_is_sparse = False，name = None）	矩阵相乘
tf.matrix_determinant（输入，名称=无）	返回方阵的行列式
tf.matrix_inverse（输入，伴随=无，名称=无）	求方阵的逆矩阵，伴随矩阵为真时，计算输入共轭矩阵的逆矩阵
tf.cholesky（输入，名称=无）	对输入方阵的Cholesky分解，
即把一个对称正定的矩阵表示成一个下三角矩阵大号和其转置的乘积的分解A = LL ^ T
tf.matrix_solve（matrix，rhs，adjoint = None，name = None）	求矩阵
为[M，M]，rhs的形为[M，K]，输出为[M，K]
复数操作

操作	描述
tf.complex（真实，形象，名称=无）	将两实数转换为复数形式
＃张量'real'是[ 2.25，3.25 ] 
张量imag是[ 
4.75,5.75 ] tf.complex（real，imag）==> [[2.25 + 4.75j]，[3.25 + 5.75j ]]
tf.complex_abs（x，name = None）	计算复数的绝对值，即长度。
＃tensor'x '是[[-2.25 + 4.75j]，[-3.25 + 5.75j]] 
tf.complex_abs（x）==> [5.25594902,6.60492229]
tf.conj（输入，名称=无）	计算共轭复数
tf.imag（输入，名称=无）
tf.real（输入，名称=无）	提取复数的虚部和实部
tf.fft（输入，名称=无）	计算一维的离散傅里叶变换，输入数据类型为complex64
归约计算（还原）

操作	描述
tf.reduce_sum（input_tensor，reduction_indices = None，
keep_dims = False，name = None）	计算输入张量元素的和，或者安照reduction_indices指定的轴进行求和
＃'x'是[[ 1,1,1 ] 
＃[ 
1,1,1 ]] tf.reduce_sum（x）==> 6 
tf 
（x，1）==> [ 2，2] tf.reduce_sum（x，1）==> [3，3] 
tf.reduce_sum（x，1，keep_dims = True）==> [[ 3]，[3]] 
tf.reduce_sum（x，[0，1]）==> 6
tf.reduce_prod（input_tensor，
reduction_indices = None，
keep_dims = False，name = None）	计算输入张量元素的乘积，或者安照reduction_indices指定的轴进行求乘积
tf.reduce_min（input_tensor，
reduction_indices = None，
keep_dims = False，name = None）	求张量中最小值
tf.reduce_max（input_tensor，
reduction_indices = None，
keep_dims = False，name = None）	求张量中最大值
tf.reduce_mean（input_tensor，
reduction_indices = None，
keep_dims = False，name = None）	求张量中平均值
tf.reduce_all（input_tensor，
reduction_indices = None，
keep_dims = False，name = None）	对张量中各个元素求逻辑'与' 
＃' x'is 
＃[[True，True] 
＃[False，False]] 
tf.reduce_all（x）==> False 
tf.reduce_all（x，0）==> [False，False] 
tf.reduce_all（x，1）==> [True，False]
tf.reduce_any（input_tensor，
reduction_indices = None，
keep_dims = False，name = None）	对张量中各个元素求逻辑 '或'
tf.accumulate_n（输入，形状=无，
tensor_dtype =无，名称=无）	计算一系列张量的和
张量'a'是[[1,2]，[3,4]] 
张量b是[[ 
5,0 ]，[ 0,6 ]] tf.accumulate_n（[a，b， a]）==> [[7,4]，[6,14]]
tf.cumsum（x，axis = 0，exclusive = False，
reverse = False，name = None）	求累积和
tf.cumsum（[a，b，c]）==> [a，a + b，a + b + c] 
tf.cumsum（[a，b，c]，exclusive = True）==> [0，a，b + c] 
tf.cumsum（[a，b，c]，reverse = True）==> [a + b + c，b + c，c] 
tf.cumsum（[a，b， c]，exclusive = True，reverse = True）==> [b + c，c，0]
分割（分割）

操作	描述
tf.segment_sum（data，segment_ids，name = None）	根据segment_ids的分段计算各个片段的和
其中segment_ids为一个尺寸与数据第一维相同的张量
其中id为int型数据，最大id不大于size 
c = tf.constant（[[1,2,3,4 ]，[-1，-2，-3，-4]，[5,6,7,8]]）
tf.segment_sum（c，tf.constant（[0，0，1]））
==> [ [0 0 0] 
[5 6 7 8]] 
上面例子分为[0,1]两个id，对相同id的数据相应数据进行求和，
并放入结果的相应id中，
且segment_ids只升不不降
tf.segment_prod（data，segment_ids，name = None）	根据segment_ids的分段计算各个片段的积
tf.segment_min（data，segment_ids，name = None）	根据segment_ids的分段计算各个片段的最小值
tf.segment_max（data，segment_ids，name = None）	根据segment_ids的分段计算各个片段的最大值
tf.segment_mean（data，segment_ids，name = None）	根据segment_ids的分段计算各个片段的平均值
tf.unsorted_segment_sum（data，segment_ids，
num_segments，name = None）	与tf.segment_sum函数类似，
不同在于segment_ids中编号顺序可以是无序的
tf.sparse_segment_sum（data，indices，
segment_ids，name = None）	输入进行稀疏分割求和
c = tf.constant（[[1,2,3,4]，[-1，-2，-3，-4]，[5,6,7,8]）
＃Select两行，一段。
tf.constant（[0，1]），tf.constant（[0，0]））
==> [[0 0 0 0]] 
对原数据的索引为[0,1] tf.sparse_segment_sum 位置的进行分割，
并按照segment_ids的分组进行求和
序列比较与索引提取（Sequence Comparison and Indexing）

操作	描述
tf.argmin（输入，维度，名称=无）	返回输入最小值的索引指数
tf.argmax（输入，维度，名称=无）	返回输入最大值的索引指数
tf.listdiff（x，y，name = None）	返回X，Y中不同值的索引
tf.where（输入，名称=无）	返回布尔型张量中为True的位置
＃'input'张量是
＃[[True，False] 
＃[True，False]] 
＃'input'有两个'True'，那么输出两个坐标值。
＃'input'的等级为2，所以每个坐标为具有两个维度。
where（input）==> 
[[0,0]，
[1,0]]
tf.unique（x，name = None）	返回一个元组tuple（y，idx），y为x的列表的唯一化数据列表，
idx为x数据对应y元素的索引
＃tensor'x '为[ 1,1,2,4,4,4 ，
7,8,8 ] y，idx = unique（x）
y ==> [ 
1,2,4,7,8 ] idx ==> [0,0,1,2,2,3，4， 4]
tf.invert_permutation（x，name = None）	置换x数据与索引的关系
＃张量x为[ 
3,4,0,2,1 ] invert_permutation（x）==> [2,4,3,0,1]
神经网络（神经网络）

激活函数（Activation Functions）
操作	描述
tf.nn.relu（功能，名称=无）	整流函数：max（features，0）
tf.nn.relu6（功能，名称=无）	以6为阈值的整流函数：min（max（features，0），6）
tf.nn.elu（功能，名称=无）	elu函数，exp（特征）-1如果<0，否则特征为
指数线性单位（ELUs）
tf.nn.softplus（功能，名称=无）	计算softplus：log（exp（features）+ 1）
tf.nn.dropout（x，keep_prob，
noise_shape = None，seed = None，name = None）	计算辍学，keep_prob为保持概率
noise_shape为噪声的形状
tf.nn.bias_add（value，bias，data_format = None，name = None）	对值一加偏置量
此函数为tf.add的特殊情况，偏压仅为一维，
函数通过广播机制进行与值求和，
数据格式可以与值不同，返回为与值相同格式
tf.sigmoid（x，name = None）	y = 1 /（1 + exp（-x））
tf.tanh（x，name = None）	双曲线切线激活函数
卷积函数（卷积）
操作	描述
tf.nn.conv2d（input，filter，
strides ，padding，use_cudnn_on_gpu = None，data_format = None，name = None）	在给定的4D输入与滤波器下计算2D卷积
输入形状为[batch，height，width，in_channels]
tf.nn.conv3d（input，filter，strides，padding，name = None）	在给定的5D输入与滤波器下计算3D卷积
输入形状为[batch，in_depth，in_height，in_width，in_channels]
池化函数（池）
操作	描述
tf.nn.avg_pool（value，ksize，strides，padding，
data_format ='NHWC'，name = None）	平均方式池化
tf.nn.max_pool（value，ksize，strides，padding，
data_format ='NHWC'，name = None）	最大值方法池化
tf.nn.max_pool_with_argmax（输入，ksize，strides，
padding，Targmax = None，name = None）	返回一个二维元组（输出，argmax），最大值池，返回最大值及其相应的索引
tf.nn.avg_pool3d（输入，ksize，strides，
padding，name = None）	3D平均值池
tf.nn.max_pool3d（输入，ksize，strides，
padding，name = None）	3D最大值池
数据标准化（归一化）
操作	描述
tf.nn.l2_normalize（x，dim，epsilon = 1e-12，name = None）	对维度dim进行L2范式标准化
output = x / sqrt（max（sum（x ** 2），epsilon））
tf.nn.sufficient_statistics（x，axes，shift = None，
keep_dims = False，name = None）	与计算均值方差状语从句：有关的完全统计量
报道查看4维元组，*元素个数，*元素总和，*元素的平方和，*移查询查询结果
参见算法介绍
tf.nn.normalize_moments（counts，mean_ss，variance_ss，shift，name = None）	基于完全统计量计算均值和方差
tf.nn.moments（x，轴，shift = None，
name = None，keep_dims = False）	直接计算均值与方差
损失函数（损失）
操作	描述
tf.nn.l2_loss（t，name = None）	输出= sum（t ** 2）/ 2
分类函数（分级）
操作	描述
tf.nn.sigmoid_cross_entropy_with_logits 
（logits，targets，name = None）*	计算输入logits，targets的交叉熵
tf.nn.softmax（logits，name = None）	计算softmax 
softmax [i，j] = exp（logits [i，j]）/ sum_j（exp（logits [i，j]））
tf.nn.log_softmax（logits，name = None）	logsoftmax [i，j] = logits [i，j] - log（sum（exp（logits [i]）））
tf.nn.softmax_cross_entropy_with_logits 
（logits，labels，name = None）	计算logits和labels的softmax交叉熵
logits，标签必须为相同的shape和数据类型
tf.nn.sparse_softmax_cross_entropy_with_logits
（logits，labels，name = None）	计算logits和标签的SOFTMAX交叉熵
tf.nn.weighted_cross_entropy_with_logits 
（logits，targets，pos_weight，name = None）	与sigmoid_cross_entropy_with_logits（）相似，
但给正向样本损失加了权重pos_weight
符号嵌入（曲面嵌入）
操作	描述
tf.nn.embedding_lookup 
（params，ids，partition_strategy ='mod'，
name = None，validate_indices = True）	根据索引ids查询嵌入列表params中的张量值
如果len（params）> 1，id将会安照partition_strategy策略进行分割
1，如果partition_strategy为“mod”，
id所分配到的位置为p = id％len params）
比如有13个ids，分为5个位置，那么分配方案为：
[[ 0,5,10 ]，[ 1,6,11 ]，[ 2,7,12 ]，[3,8]， [ 
4,9 ]] 2，如果partition_strategy为“div”，那么分配方案为：
[[0,1,2]，[3,4,5]，[6,7,8]，[9,10] ，[11，12]]
tf.nn.embedding_lookup_sparse（params，
sp_ids，sp_weights，partition_strategy ='mod'，
name = None，combiner ='mean'）	对给定的IDS和权重查询嵌入
1，sp_ids为一个N×M个的稀疏张量，
Ñ为批量大小，男为任意，数据类型的int64 
2，sp_weights的形状与sp_ids的稀疏张量权重，
浮点类型，若为无，则权重为全'1'
循环神经网络（Recurrent Neural Networks）
操作	描述
tf.nn.rnn（cell，inputs，initial_state = None，dtype = None，
sequence_length = None，scope = None）	基于RNNCell类的实例单元建立循环神经网络
tf.nn.dynamic_rnn（cell，inputs，sequence_length = None，
initial_state = None，dtype = None，parallel_iterations = None，
swap_memory = False，time_major = False，scope = None）	基于RNNCell类的实例细胞动态建立神经循环网络
与一般回归神经网络不同的是，函数该会根据输入侧动态展开
报道查看（输出，状态）
tf.nn.state_saving_rnn（cell，inputs，state_saver，state_name，
sequence_length = None，scope = None）	可储存调试状态的RNN网络
tf.nn.bidirectional_rnn（cell_fw，cell_bw，inputs，
initial_state_fw = None，initial_state_bw = None，dtype = None，
sequence_length = None，scope = None）	双向RNN，返回一个3元组tuple 
（outputs，output_state_fw，output_state_bw）
- tf.nn.rnn简单介绍 - 
cell：一个RNNCell实例
输入：一个shape为[batch_size，input_size]的张量
initial_state：为RNN的状态设置初值，可选
sequence_length：制定输入的每一个序列的长度， size为[batch_size]，值范围为[0，T）的int型数据
其中T为输入数据序列的长度
@ 
@针对输入批次中序列长度不同，所设置的动态计算机制
@的
（行，州）（b，t）=？（b，t），状态（b，t-1））的单元（零（cell.output_size），状态（b，sequence_length（b）-1）

求值网络（评价）
操作	描述
tf.nn.top_k（输入，k = 1，排序=真，名称=无）	返回前ķ大的值及其对应的索引
tf.nn.in_top_k（预测，目标，k，名称=无）	返回判断是否目标索引的预测的相应值
是否在在预测前ķ个位置中，
返回数据类型为布尔类型，LEN与预测同
监督候选采样网络（Candidate Sampling）
对于有巨大量的多分类与多标签模型，如果使用全连接SOFTMAX将会占用大量的时间与空间资源，所以采用候选采样方法仅使用一小部分类别与标签作为监督以加速训练。

操作	描述
信报	
tf.nn.nce_loss（权重，偏差，输入，标签，num_sampled，
num_classes，num_true = 1，sampled_values = None，
remove_accidental_hits = False，partition_strategy ='mod'，
name ='nce_loss'）	返回噪声对比的训练损失结果
tf.nn.sampled_softmax_loss（权重，偏差，输入，标签，
num_sampled，num_classes，num_true = 1，sampled_values = None，
remove_accidental_hits = True，partition_strategy ='mod'，
name ='sampled_softmax_loss'）	返回样本softmax的训练损失
参考 - Jean et al。，2014第3部分
候选人取样器	
tf.nn.uniform_candidate_sampler（true_classes，num_true，
num_sampled，unique，range_max，seed = None，name = None）	通过均匀分布的采样集合
返回三元组tuple 
1，sampled_candidates候选集合
。2，期望的true_classes个数，为浮点值
3，期望的sampled_candidates个数，为浮点值
tf.nn.log_uniform_candidate_sampler（true_classes，num_true，
num_sampled，unique，range_max，seed = None，name = None）	通过登录均匀分布的采样集合，返回三元元组
tf.nn.learned_unigram_candidate_sampler 
（true_classes，num_true，num_sampled，unique，
range_max，seed = None，name = None）	在根据训练过程中学习到的分布状况进行采样
报道查看三元元组
tf.nn.fixed_unigram_candidate_sampler（true_classes，num_true，
num_sampled，unique，range_max，vocab_file =“，
distortion = 1.0，num_reserved_ids = 0，num_shards = 1，
shard = 0，unigrams =（），seed = None，name = None）	基于所提供的基本分布进行采样
保存与恢复变量

操作	描述
类tf.train.Saver（保存和恢复变量）	
tf.train.Saver .__ init __（var_list = None，reshape = False，
sharded = False，max_to_keep = 5，
keep_checkpoint_every_n_hours = 10000.0，
name = None，restore_sequentially = False，
saver_def = None，builder = None）	创建一个存储器Saver 
var_list定义需要存储和恢复的变量
tf.train.Saver.save（sess，save_path，global_step = None，
latest_filename = None，meta_graph_suffix ='meta'，
write_meta_graph = True）	保存变量
tf.train.Saver.restore（sess，save_path）	恢复变量
tf.train.Saver.last_checkpoints	列出最近未删除的检查点文件名
tf.train.Saver.set_last_checkpoints（last_checkpoints）	设置检查点文件名列表
tf.train.Saver.set_last_checkpoints_with_time（last_checkpoints_with_time）	设置检查点文件名列表和时间戳
