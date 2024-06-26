import os
import re

import pandas as pd

import streamlit as st


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["prompt", "response_a", "response_b"]:
        df[col + "_null"] = df[col].apply(
            lambda x: x.replace("null,", "nullnull,").replace(",null", ",nullnull")
        )
        df[col + "_list"] = df[col + "_null"].apply(
            lambda x: re.split(r"(\",\"|null,\"|\",null)", x[1:-1])
        )
        df[col + "_len"] = df[col + "_list"].apply(lambda x: len(x))
        # \n\nを改行に変換
        df[col + "_list"] = df[col + "_list"].apply(
            lambda x: [
                y.replace("\\n", "\n")
                .replace("null,", "null")
                .replace('",null', "null")
                .replace('",null', "null")
                .replace('""', "")
                .replace('","', "")
                for y in x
            ]
        )

    def get_winner(x):
        if x["winner_model_a"] == 1:
            return "model a wins"
        elif x["winner_model_b"] == 1:
            return "model b wins"
        else:
            return "draw"

    df["winner"] = df.apply(get_winner, axis=1)
    return df


def display(df: pd.DataFrame, id: int, response_type: str, model_name: str):
    # dataframeの行を選択
    row = df.iloc[id]
    # for i, row in df.iterrows():
    st.markdown(
        f"## Model\n### {row[model_name]} ({model_name})\n\n## Winner\n### {row['winner']}"
    )
    prompt = row["prompt_list"]
    response = row[response_type]
    for j, (p, res) in enumerate(zip(prompt, response)):
        if j == 0:
            p = p[1:] if p != "null" else p
            res = res[1:] if res != "null" else res
        if j == len(prompt) - 1:
            p = p[:-1] if p != "null" else p
            res = res[:-1] if res != "null" else res
        st.write(f"Turn {j+1}")
        with st.chat_message("user"):
            st.write(p)
        with st.chat_message("assistant"):
            st.write(res)


train_data = pd.read_csv("data/train.csv")
train_data = preprocess_data(train_data)

if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit App", page_icon=":rocket:")
    st.title("Streamlit App")

    # tmpフォルダからファイル名を選択する
    # file_name = st.selectbox("Select a file", os.listdir("data"))
    # conv_data = pd.read_csv(f"data/{file_name}")
    # train_data = preprocess_data(train_data)
    # conv_a or conv_bを選択する
    response_type = st.selectbox(
        "Select a response type", ["response_a_list", "response_b_list"]
    )
    # 機能を追加することで表示するデータを変更することができる
    is_null_only = st.checkbox("Show only including null data")
    if is_null_only:
        data = null_contain_df = train_data[
            train_data["response_a"].str.contains("null,")
            | train_data["response_a"].str.contains(",null")
            | train_data["response_b"].str.contains("null,")
            | train_data["response_b"].str.contains(",null")
        ]
    else:
        data = train_data
    row = st.number_input("Select a row", min_value=0, max_value=len(data) - 1)
    display(
        data,
        row,
        response_type,
        "model_a" if response_type == "response_a_list" else "model_b",
    )
