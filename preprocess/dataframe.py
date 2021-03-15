import pandas as pd
def concat_row(df, columns, sep=','):
    """DFのカラムの各行を結合する
    テキストカラムの結合など
    """
    def f(row):
        # row : pd.Series
        l = [val for val in row]
        return sep.join(l)
    out_df = df.loc[:,columns].apply(f, axis=1)
    return out_df


############ 値の変換 #############
def convert_ex():
    pd.Series([1,1,2]).map({1: 100}) # 1-> 100 に変換

    # map で各要素に関数を適応
    def converter(x):
        if x < 10:
            return 1000
        return x
    pd.Series([1,200, 2]).map(converter)

def replace():
    # dtypeがobjectになってるのでfloatに直す
    size_info[column_name] = size_info[column_name].replace('', np.nan)

def astype():
    df[column] = df[column].astype(float) 
    

############ 条件絞り込み #############
# ある条件で絞った値を取り出すときは、条件を一度変数に代入するとわかりやすい
def condition_ex():
    idx = series == 19
    series[idx]

    # 条件を絞り込む方法　いくつか
    series > 100 #100より大きいindex
    series.isin([19, 20]) #isin
    series.isnull() #isnull
    series.astype(str).str.contains('foo') #文字列として評価したときに foo を含むか
    series == train['object_id'] #他のカラムと値が一致しているか

def extract_regex_text_ex():
    """str.extract 正規表現抽出"""
    # 正規表現を使ってサイズを抽出
    size_info = input_df['sub_title'].str.extract(r'(\d*|\d*\.\d*)(cm|mm)') 
    size_info = size_info.rename(columns={0: column_name, 1: 'unit'})



############ groupby #############
def groupby_ex():
    group = train.groupby('principal_maker')
    group.size() #各グループの要素の数
    group['sub_title'].nunique() #グループごとのユニークなタイトルの数
    group['dating_sorting_date'].agg(['min', 'max', 'mean','median','std','var','sum','skew', #基本統計量
                                    pd.DataFrame.kurt #尖度 これは文字列がない
                                    np.size, #グループごとの要素数
                                    pd.Series.nunique, #グループごとのユニーク数
    ])

    # groupごとのcumsum
    df.groupby(['name', 'day']).sum() \
    .groupby(level=0).cumsum().reset_index()

    #関数の適応
    # 元データとのdfff, ratio なども加えると良さそう
    def my_func(x):
        return max(x) - min(x)
    
    print(grouped.agg(my_func))

    ############ string #############
    sub_title_length = train['sub_title'].str.len().rename('subtitle_length')


####### hone-hot用展開 ######
# 2つのカテゴリ変数：crosstab
# 出現回数30以上に絞る
def crosstab():
    vc = person_df['name'].value_counts()
    use_names = vc[vc > 30].index

    # isin で 30 回以上でてくるようなレコードに絞り込んでから corsstab を行なう
    # 行数が多すぎるのを防ぐため
    idx = person_df['name'].isin(use_names)
    _use_df = person_df[idx].reset_index(drop=True)
    pd.crosstab(_use_df['object_id'], _use_df['name'])

###### merge,join ######
def left_join(left, right, on=OBJECT_ID):
    """joinで増えたcolumnsだけ返す
    NOTE :
    right の onに指定するカラムに重複がある場合は drop_duplicate しないと右側のdfにnan行ができてしまうので注意。
    """
    if isinstance(left, pd.DataFrame):
        left = left[on]
    return pd.merge(left, right, on=on, how='left').drop(columns=[on])
