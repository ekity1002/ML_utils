import tensorflow as tf

#変数と定数
tf.Variable('Helo', tf.string)
tf.constant([1,2,3,4,5])

# テンソルは shape, data type を持つ
tf.Variable([1,2], dtype=tf.float32) #float32型で持つ

tf.Variable([1,2,3], shape=(3, 1)) #shape指定

# shapeを未知にしておくこともできる
tf.Variable([1,2,3,4], shape=tf.TensorShape(None))

# 文字列
tf.Variable('Elephant', dtype=tf.string)

# すべて同じ値
tf.constant(-1, shape=[2,2])

############## 演算
tf.add([1,2], [3,4])
tf.multiply(x,y) #要素積

tf.square(5) #25

tf.reduce_sum([1,2,3]) #->6, shape=はなし（整数）

#reshape
tf.reshape(x, (5,5))

#キャスト
tf.cast(x, tf.float32)

############# numpyとの互換性
np.multipy(a,b)
tf.multiply(numpy, 3)

a.numpy() #->numpy型で返す


