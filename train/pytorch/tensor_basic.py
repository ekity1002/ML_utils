from __future__ import print_function
import torch

################# テンソル生成
x = torch.empty(5,3)
torch.rand(5,3)
# torch.rand
# torch.empty
# torch.zeros
# torch.randn_like()
# x.new_ones() #xを置き換えて生成
x = x.new_ones(5, 3, dtype=torch.double)

# テンソルサイズ
print(x.size())
print(x.shape)

################# テンソル演算
torch.add(x, y)

# 出力先のテンソルを out 引数に指定することができる
result = torch.empty_like(x)
torch.add(x,y, out=result)

# リサイズ view関数をつかう
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.shape, z.shape)

# inplace 処理
# メソッドの後に _ をつけると、変数に上書きされる
print(y)
print(y.add(x)) 
print(y) #y は変化なし
y.add_(x)

# すべてのxの要素とyの要素が等しいか
x.eq(y).all()



################## 値の取り出し・変換
# item() : 要素一つのテンソル(スカラーテンソル)から値を取り出す
print(x[0,0]) # torch.tensor型
print(x[0,0].item()) # float

# numpy との接続
## Char Tensor を覗いて相互に変換可能
## CPU上にある場合、 torch tensor と numpy Array はメモリ上の同じ領域に配置され変換できる
a = torch.ones(5)
b = a.numpy() # numpy() で numpy 配列へ
# aと b はメモリ上の同じ領域を参照している
# aの参照先を書き換えると b の値も変わる
a.add_(1)
print(a)
print(b)

# CUDAテンソル
if torch.cuda.is_available():
    device = torch.device('cuda')
    y = torch.ones_like(x, device=device) #GPU上に直接テンソル作成
    x = x.to(device) # CPUテンソルに to メソッドを使って GPUにすることもできる
    z = x + y
    print(z)
    #print(type(z))
    print(z.to('cpu', torch.double)) # CPU上にもどし、同時にデータタイプも変更する


#################  自動微分
"""
autograd パッケージで提供
define-by-run(データを流しながらモデルを構築) する
* requires_grad 属性に True が指定された場合、autogradによってすべての操作が追跡される
* 操作後、.backward() を呼び出すと すべての操作に対する勾配が自動で計算される
* tensor に対する勾配は .grad 属性に蓄積される
* 追跡を停止する場合は .detach() を呼び出す。その後 このてんそるは追跡されない
""" 
x = torch.ones(2,2, requires_grad=True)
y = x + 2
print(y)
y.grad_fn #計算履歴を grad_fn属性として持つ

# backwardするテンソルがスカラーでない場合、backward に引数を指定する必要がある
# xの値を vにしたときの 勾配を計算
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

# autogradが必要ない場合 with no_gradでくくる
# -> 学習中にモデル構築に使用した テンソルに演算などがOKになる
with torch.no_grad():
    print((x**2).requires_grad)

# tensor.detach() で autograd されないtensor を返却する
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
