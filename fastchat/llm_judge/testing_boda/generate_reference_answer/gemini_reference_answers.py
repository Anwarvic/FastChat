import os

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from pathlib import Path
from threading import Lock
from pydantic import BaseModel

# add path
import sys

print(f"Current working directory: {Path(__file__).resolve().parent.parent}")
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from infer_gemini import *

# Load environment variables


input_path = "/home/abdelrahman.sadallah/mbzuai/FastChat/fastchat/llm_judge/data/vicuna_ar_extended/question.jsonl"
load_dotenv()
gemini_key = os.getenv("GEMINI_KEY_ANWAR")
assert gemini_key is not None, "Gemini API key not found in environment variables."

PROMPT_TEMPLATE = """
{text}
"""


def load_dataset(path):

    # Load dataset
    df = pd.read_json(path, lines=True)

    # get only the rows with question_id from 71 to 80
    df = df[(df["question_id"] >= 71) & (df["question_id"] <= 80)].reset_index(
        drop=True
    )

    logger.info(f"Loaded dataset with {len(df)} entries.")
    return df


def prepare_prompt(df: pd.DataFrame) -> list[dict]:
    """Prepare the prompt for each entry in the dataset."""
    # create the prompt column
    df["prompt"] = df.apply(
        lambda row: PROMPT_TEMPLATE.format(text=row["turns"][0]), axis=1
    )
    logger.info(f"Prepared prompts for {len(df)} entries.")
    df = df[["question_id", "prompt"]]
    # Convert DataFrame to list of dictionaries
    list_json = df.to_dict(orient="records")

    return list_json


class DataEntry(BaseModel):
    answer: str


def postprocess_output(path) -> list[DataEntry]:
    """Post-process the output data into a list of DataEntry objects."""

    # path = str(path).replace(".jsonl", "_postprocessed.jsonl")
    # load input dataset
    in_df = load_dataset(input_path)
    # convert the output to JSON format
    out_df = pd.read_json(path, lines=True)

    # Merge the input and output dataframes on the 'id' column
    merged_df = in_df.merge(out_df, on="question_id", how="inner")

    # Add columns from the output dataframe to the input dataframe
    for col in out_df.columns:
        if col != "question_id":  # Avoid duplicating the 'id' column
            in_df[col] = merged_df[col]

    # Save the updated input dataframe to jsonl format
    in_df.to_json(path, lines=True, orient="records", force_ascii=False)

    # in_df.to_csv(path.replace(".jsonl", ".csv"))


if __name__ == "__main__":
    # load Gemini client
    gemini = load_gemini(gemini_key)
    # load the dataset
    df = load_dataset(input_path)
    # prepare the prompts
    data = prepare_prompt(df)
    out_jsonl_filepath = Path(
        "/home/abdelrahman.sadallah/mbzuai/FastChat/fastchat/llm_judge/testing_boda/generate_reference_answer/gemini_referece_answers.jsonl"
    )
    # generate content using Gemini
    process_data_with_gemini(
        client=gemini,
        schema=DataEntry,
        data=data,
        id_col="question_id",
        input_col="prompt",
        out_jsonl_filepath=out_jsonl_filepath,
        max_workers=10,  # NOTE: Due to limitation to my subscription, this is the maximum concurrent calls allowed by Gemini.
    )

    # postprocess the output
    postprocess_output(out_jsonl_filepath)
    logger.info("Done postprocessed the output data!")
