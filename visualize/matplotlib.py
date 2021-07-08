def seaborn_snipets():
    # y 範囲設定
    ax[0].set(ylim=(0, 40000))
    
    # title
    ax[0].set_title('no06-10 のコンテンツごとのクリック数')
    ax[1].set_title('no06-10 のコンテンツごとの承諾率')

    # axis rotation 回転
    ax[0].xaxis.set_tick_params(rotation=45)
    ax[1].xaxis.set_tick_params(rotation=45)

    plt.tight_layout()
    plt.show()


    plt.ylim(0, 40000)
    ax[0].set(ylim=(0, 40000))
