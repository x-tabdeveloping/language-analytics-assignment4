from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px


def main():
    print("Loading results")
    emotions = pd.read_json("results/emotions.jsonl", orient="records", lines=True)

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    print("Extracting emotion frequencies per season")
    # Extracting emotion distributions per season
    per_season = emotions.groupby("Season")["emotion_label"].value_counts()
    per_season = per_season.reset_index("Season")
    # Extracting relative frequencies
    per_season = (
        per_season.groupby("Season").apply(lambda s: s / np.sum(s)).reset_index()
    )
    per_season = per_season.rename(columns={"count": "rel_freq"})
    # Mapping Season name to just number
    per_season["Season"] = per_season["Season"].map(lambda s: int(s.split(" ")[-1]))

    print("Creating bar plot with frequencies.")
    fig = px.bar(
        per_season,
        y="emotion_label",
        x="rel_freq",
        color="emotion_label",
        facet_row="Season",
        template="plotly_white",
        category_orders={
            "emotion_label": ["neutral", "anger", "surprise", "disgust", "fear", "joy"]
        },
    )
    fig = fig.update_traces(showlegend=False)
    fig = fig.update_yaxes(title="")
    fig = fig.update_layout(
        width=800, height=1200, xaxis_title="Relative Frequency of Emotion in Season"
    )
    fig.write_image(out_dir.joinpath("emotions_per_season_bar.png"), scale=2)

    print("Creating line plot with frequencies.")
    fig = px.line(
        per_season,
        y="rel_freq",
        x="Season",
        color="emotion_label",
        # facet_row="season",
        template="plotly_white",
        category_orders={
            "emotion_label": ["neutral", "anger", "surprise", "disgust", "fear", "joy"]
        },
    )
    fig = fig.update_layout(
        width=800, height=1200, yaxis_title="Relative Frequency of Emotion in Season"
    )
    fig = fig.update_traces(line=dict(width=4))
    fig.write_image(out_dir.joinpath("emotion_per_season_line.png"), scale=2)

    print("Calculating season distributions per emotion.")
    per_emotion = (
        emotions.groupby("emotion_label")["Season"].value_counts().reset_index()
    )

    print("Creating Pie plots.")
    fig = px.pie(
        per_emotion,
        values="count",
        names="Season",
        facet_col="emotion_label",
        facet_col_wrap=3,
        category_orders={"Season": [f"Season {i+1}" for i in range(8)]},
    )
    fig = fig.update_layout(width=800, height=800)
    fig.write_image(out_dir.joinpath("season_per_emotion_pie.png"), scale=2)


if __name__ == "__main__":
    main()
