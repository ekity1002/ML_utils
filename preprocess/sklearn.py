from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import category_encoders as ce

def build_preprocessor(train, config): 
    """
    前処理変換器構築
    train: pd.DataFrame
    config: dict
    """
    num_col = [col for col in train.columns if col in config['cols_definition']['numerical_col']]
    cat_col = [col for col in train.columns if col in config['cols_definition']['categorical_col']]
    
    # 変換器作成
    categorical_transformer = Pipeline(steps=[
        ('labelenc', (ce.OrdinalEncoder(cols=cat_col, drop_invariant=False))),
        #('te', (ce.TargetEncoder(cols=cat_col, drop_invariant=False, smoothing=100)))
        ])

    num_transformer = Pipeline(steps=[('std', (StandardScaler())),
        #('te', (ce.TargetEncoder(cols=cat_col, drop_invariant=False, smoothing=100)))
        ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cat_col),
            ('num', num_transformer, num_col),
        ],
        remainder="drop")
    return preprocessor
