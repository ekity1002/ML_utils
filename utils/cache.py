import inspect


def param_to_name(params: dict, key_sep='_', key_value_sep='=') -> str:
    """
    from  https://github.com/nyk510/vivid/blob/master/vivid/utils.py

    dictを'key=value で連結したstringに変換
    params:
    key_sep: key同士を連結する際に使う文字列
    key_value_sep: それぞれの key/value を連結するのに使う文字列
    """
    sorted_params = sorted(params.items())
    return key_sep.join(map(lambda x: key_value_sep.join(map(str, x)), sorted_params))


def cachable(function):
    """関数の返り値をキャッシュしておくデコレータ
    ex: キャッシュしたい関数にデコレートする
    @cachable
    def read_csv(name):
        if '.csv' not in name:
            name = name + '.csv'
        return pd.read_csv(os.path.join(INPUT_DIR, name))
    """
    attr_name = '__cachefile__'

    def wrapper(*args, **kwargs):
        force = kwargs.pop('force', False)
        call_args = inspect.getcallargs(function, *args, **kwargs)
        print(call_args)
        arg_name = param_to_name(call_args)
        print('arg_name', arg_name)
        name = attr_name + arg_name
        print(name)

        use_cache = hasattr(function, name) and not force

        if use_cache:
            cache_object = getattr(function, name)
        else:
            print('run')
            cache_object = function(*args, **kwargs)
            setattr(function, name, cache_object)
        return cache_object

    return wrapper
