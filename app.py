import streamlit as st
import numpy as np
import pandas as pd
import datetime as datetime
import pandas_datareader
import plotly.graph_objects as go
import sklearn
import sklearn.linear_model
import sklearn.model_selection
from PIL import Image
import yfinance as yf

st.title("AIで株価予測アプリ")
st.write('AIを使って、株価を予測してみましょう。')

image = Image.open('stock_predict.png')
st.image(image, use_column_width=True)

st.write('※あくまでAIによる予測です（参考値）。こちらのアプリによる損害や損失は一切補償しかねます。')

st.header("株価銘柄のティッカーシンボルを入力してください。")
stock_name = st.text_input("例:AAPL, FB, SFTBY（大文字・小文字どちらでも可）", "AAPL")

# 大文字にする
stock_name = stock_name.upper()
# ティッカーシンボルから会社名（英語表記）に変換
stock_name_full = yf.Ticker(str(stock_name))
stock_name_full = stock_name_full.info['longName']

link = 'https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_usequity_list.html'
st.markdown(link)
st.write('ティッカーシンボルについては上のリンク（SBI証券）をご参照ください。')

try:

    df_stock = pandas_datareader.data.DataReader(stock_name, 'yahoo', '2021-01-01')

    st.header(stock_name_full + "2021年1月1日から現在までの価格（USD）")
    st.write(df_stock)

    st.header(stock_name_full + "終値と14日間平均（USD)")
    df_stock['SMA'] = df_stock['Close'].rolling(window=14).mean()
    df_stock2 = df_stock[['Close','SMA']]
    st.line_chart(df_stock2)

    st.header(stock_name_full + "値動き（USD)")
    df_stock['change']= (((df_stock['Close']-df_stock['Open'])) / (df_stock['Open']) *100)
    st.line_chart(df_stock['change'].tail(100))

    fig = go.Figure(
        data = [go.Candlestick(
            x = df_stock.index,
            open = df_stock['Open'],
            high = df_stock['High'],
            low = df_stock['Low'],
            close = df_stock['Close'],
            increasing_line_color = 'green',
            decreasing_line_color = 'red',
            )
        ]
    )

    st.header(stock_name_full + "キャンドルスティック")
    st.plotly_chart(fig)

    df_stock['label'] = df_stock['Close'].shift(-30)

    st.header(stock_name_full + '1ヶ月後を予測しよう（USD）')

    def stock_predict():
        # ラベルとSMAを削除したデータをXに代入
        X = np.array(df_stock.drop(['label','SMA'], axis=1))

        # 取りうる値の大小が著しく異な宇特徴量を入れると結果が悪くなるため、
        # 平均を引いて標準偏差で割って、スケーリングする。
        X = sklearn.preprocessing.scale(X)

        # 予測に使用する過去30日間のデータ
        predict_data = X[-30:]
        # 過去30日間を除いた入力データ
        X = X[:-30]

        # 正解ラベル
        y = np.array(df_stock['label'])
        # 過去30日間を除いた正解ラベル
        y = y[:-30]

        # 訓練データ80%　検証データ20%に分ける
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

        # 訓練用データを用いて学習する。
        model = sklearn.linear_model.LinearRegression()
        model.fit(X_train, y_train)

        # 検証用データを用いて検証してみる
        accuracy = model.score(X_test, y_test)

        # 小数点第一位で四捨五入
        st.write(f'正答率は{round((accuracy) * 100,1)}%です。')

        # accuracyより信頼度を表示
        if accuracy > 0.75:
            st.write('信頼度：高')
        elif accuracy > 0.5:
            st.write('信頼度：中')
        else:
            st.write('信頼度：低')
        st.write('オレンジの線（Predict)が予測値です。')

        # 検証データを用いて検証してみる。
        predicted_data = model.predict(predict_data)
        df_stock['Predict'] = np.nan
        last_date = df_stock.iloc[-1].name
        one_day = 86400
        next_unix = last_date.timestamp() + one_day

        # 予測をグラフ化
        for data in predicted_data:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += one_day
            df_stock.loc[next_date] = np.append([np.nan]*(len(df_stock.columns)-1), data)

        df_stock['Close'].plot(figsize=(15,6), color="green")
        df_stock['Predict'].plot(figsize=(15,6), color="orange")

        df_stock3 = df_stock[['Close', 'Predict']]
        st.line_chart(df_stock3)

    if st.button('予測する'):
        stock_predict()
except:
    st.error(
        "エラーがおきているようです。"
    )
st.write('Copyright @ 2022 Tomiyuki Yoshikawa. All Rights Reserved.')