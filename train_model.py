import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

categorical = ['event', 'importance']
num = ['surprise', 'minutes_from_release', 'pct_change_since_release','pre_news_return_pct','pre_news_volatility','return_last_5m_pct','high_since_release_pct','low_since_release_pct','volume_ratio_since_release']

def load_data(csv_path="XGBoost_Episodic_Data.csv"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['target_return_to_end_pct'])
    df = df[df['minutes_from_release']!=60].copy()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df

def add_episode_id(df):
    minutes_since_news = pd.to_timedelta(df['minutes_from_release'], unit='m')
    df['release_time'] = df['timestamp'] - minutes_since_news

    return df

def split_by_time(df, test_frac=0.2):
    unique_times = df['release_time'].drop_duplicates().sort_values().reset_index(drop=True)
    cut_position = int(len(unique_times) * (1 - test_frac))
    cutoff = unique_times.iloc[cut_position]

    train_df = df[df['release_time'] < cutoff].copy()
    test_df = df[df['release_time'] >= cutoff].copy()

    return train_df, test_df

def get_X_y(df):
    X = df[categorical + num]
    y = df['target_direction']

    return X, y

def build_model():
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)),
    ])

    return model

def train_and_score(train_df, test_df):
    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    print("Always-buy baseline:", y_test.mean())

    return model

df = load_data()
df = add_episode_id(df)
train_df, test_df = split_by_time(df)
model = train_and_score(train_df, test_df)