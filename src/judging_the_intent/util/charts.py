import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


if __name__ == "__main__":
    datasets = ["corpus-subsamples/clueweb09/en/trec-web-2009",
                "corpus-subsamples/clueweb09/en/trec-web-2010",
                "corpus-subsamples/clueweb09/en/trec-web-2011",
                "corpus-subsamples/clueweb09/en/trec-web-2012",
                "corpus-subsamples/clueweb12/trec-web-2013",
                "corpus-subsamples/clueweb12/trec-web-2014"]
    models = ["mistral:7b-instruct-v0.3-q4_0",
              "llama3.1:8b-instruct-q4_K_M"]

    avg_ndcg_total_h = []
    avg_ndcg_total_i = defaultdict(list)
    avg_ndcg_total_no_i = defaultdict(list)

    for dataset in tqdm(datasets):
        for model in models:
            rank_output_path = Path(__file__).parent.parent.parent.parent.joinpath("trec-web", "rank-output")
            human_performance = pd.read_csv(Path(rank_output_path).joinpath(
                f"{model.replace(':', '-')}-{dataset.replace('/', '-')}-human-gt.tsv"),
                sep="\t")
            llm_performance = pd.read_csv(Path(rank_output_path).joinpath(
                f"{model.replace(':', '-')}-{dataset.replace('/', '-')}-llm-gt-intent.tsv"),
                sep="\t")
            llm_performance_si = pd.read_csv(Path(rank_output_path).joinpath(
                f"{model.replace(':', '-')}-{dataset.replace('/', '-')}-llm-gt-no-intent.tsv"),
                sep="\t")

            if len(avg_ndcg_total_h) < len(datasets):
                avg_ndcg_total_h.append(round(human_performance["ndcg_cut.10"].mean(), 2))

            avg_ndcg_total_i[model].append(round(llm_performance["ndcg_cut.10"].mean(), 2))
            avg_ndcg_total_no_i[model].append(round(llm_performance_si["ndcg_cut.10"].mean(), 2))


    datasets = ("2009", "2010", "2011", "2012", "2013", "2014")
    models = {
        # Value is nDCG for each dataset
        'Human': avg_ndcg_total_h,
        'Llama3.1': avg_ndcg_total_no_i["llama3.1:8b-instruct-q4_K_M"],
        'Llama3.1-I': avg_ndcg_total_i["llama3.1:8b-instruct-q4_K_M"],
        'Mistral-I': avg_ndcg_total_i["mistral:7b-instruct-v0.3-q4_0"],
        'Mistral': avg_ndcg_total_no_i["mistral:7b-instruct-v0.3-q4_0"],
    }

    color_map ={
        'Human': "#DC602E",
        'Llama3.1': "#D7B49E",
        'Llama3.1-I': "#B8D5B8",
        'Mistral-I': "#05A8AA",
        'Mistral': "#b51963",
    }

    x = np.arange(len(datasets))  # the label locations
    width = 0.18  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(12, 4), dpi=600)

    for attribute, measurement in models.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, color=color_map[attribute], label=attribute)
        ax.bar_label(rects, padding=3, fontsize="small")
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Avg. nDCG@10')
    ax.set_title('Average nDCG@10 Comparing Human and LLM Judgments')
    ax.set_xticks(x + (width * 2))
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper left', ncols=5)
    ax.set_yticks(np.arange(0, 0.6, 0.1))
    plt.margins(x=0)

    chart_output_dir_path = Path(__file__).parent.parent.parent.parent.joinpath("trec-web", "rank-output", "images")
    chart_output_dir_path.mkdir(exist_ok=True)
    chart_output_path = Path(chart_output_dir_path).joinpath("chart.png")

    plt.savefig(chart_output_path)
    plt.close()