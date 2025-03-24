from datasets import load_dataset

import pydantic
from typing import Literal
import datasets
import tqdm
import asyncio
import pandas as pd


class ReasoningData(pydantic.BaseModel):
    input: str
    output: str
    category: Literal["code", "math", "science", "chat", "safety"]
    generator: Literal[
        "DeepSeek-R1, Qwen-2.5-72B-Instruct",
        "DeepSeek-R1, Qwen-2.5-32B-Instruct",
        "Mixtral-8x22B-Instruct-v0.1",
        "DeepSeek-R1",
    ]


def filter_reasoning_data() -> datasets.DatasetDict:
    ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset-v1")

    filtered_ds_dict = datasets.DatasetDict(
        {
            split: dataset.filter(
                lambda batch: [reason == "on" for reason in batch["reasoning"]],
                batched=True,
                batch_size=100_000,
            )
            for split, dataset in ds.items()
        }
    )
    return filtered_ds_dict


def process_dataset(ds):
    return ReasoningData(
        input=ds["input"],
        output=ds["output"],
        category=ds["category"],
        generator=ds["generator"],
    )


async def process_split(split, dataset):
    tasks = []
    for ds in dataset:
        task = asyncio.create_task(asyncio.to_thread(process_dataset, ds))
        tasks.append(task)

    results = []
    for task in tqdm.tqdm(
        asyncio.as_completed(tasks), desc=f"For {split}", total=len(tasks)
    ):
        result = await task
        results.append(result)

    return results


async def get_dataset(filtered_ds_dict: datasets.DatasetDict) -> list[ReasoningData]:
    reasoning_data: list[ReasoningData] = []
    for split, dataset in filtered_ds_dict.items():
        results = await process_split(split, dataset)
        reasoning_data.extend(results)
    return reasoning_data


def read_data(filename: str = "reasoning_data.parquet") -> list[ReasoningData]:
    df = pd.read_parquet(filename)
    reasoning_dataset = [
        ReasoningData(**row) for row in tqdm.tqdm(df.to_dict("records"))
    ]
    return reasoning_dataset


if __name__ == "__main__":
    reasoning_data = asyncio.run(get_dataset())
    list_of_dicts = [item.model_dump() for item in tqdm.tqdm(reasoning_data)]
    dataset = datasets.Dataset.from_list(list_of_dicts)
    dataset.to_parquet("reasoning_data.parquet")
