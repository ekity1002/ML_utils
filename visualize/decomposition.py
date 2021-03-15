
# import umap
# mapper = umap.UMAP(n_components=50)
# features = create_input_array_umap(df)
# embedding = mapper.fit_transform(features)
# embedding.shape
# sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=train['log_likes'], ) #hueの値で色付けして表示
# plt.show()


def create_input_array_umap(input_df, column):
    """umap用の配列作成
    df: 入力DF
    column 使用するからむ
    """
    features = []
    for array in input_df[column]:
        features.append(array)
    return pd.DataFrame(features).values.astype(np.float32)

def umap(self, df):
    mapper = umap.UMAP(n_components=self.umap_components)
    features = create_input_array_umap(df, 'description_feature')
    embedding = mapper.fit_transform(features)
    # sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=train['log_likes'], ) #hueの値で色付けして表示
    # plt.show()        
    return pd.DataFrame(embedding)


def show_tsne(z):
    """ TSNE
    z: tf-difしたテキストなどの特徴量 np.array. size=(row, feature_dif)
    """
    from MulticoreTSNE import MulticoreTSNE as TSNE

    # tsneおそいのでちょっと時間かかる
    with Timer(prefix='run tsne'):
        tsne = TSNE(n_jobs=-1)
        embedding = tsne.fit_transform(z)

    # わかりやすさのため１０を亭としたlog変換
    train['log_likes'] = np.log10(train['likes'] + 1)

    bin_labels = pd.cut(train['log_likes'], bins=5)
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(*embedding.T, c=bin_labels.cat.codes, s=20, alpha=.8, cmap='cividis')
    ax.grid()
