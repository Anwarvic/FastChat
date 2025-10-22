import time
import yaml
import json
import logging
from tqdm import tqdm
from google import genai
from pathlib import Path
from threading import Lock
from pydantic import BaseModel
from google.genai.errors import ServerError
from tqdm.contrib.concurrent import thread_map

# add path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

# from utils.uae_knowledge_qa import *
# from utils.uae_knowledge_mcq import *
# from utils.mcq_dream import *
# from utils.zh_dream import *
# from utils.ultrafeedback_binarized import *


# def load_yaml(yaml_filepath: Path) -> dict:
#     if not yaml_filepath.exists():
#         raise FileNotFoundError(
#             f"Credentials file {yaml_filepath} does not exist."
#         )
#     with open(yaml_filepath, "r") as f:
#         return yaml.safe_load(f)

# # load important variables
# credentials_filepath = Path(__file__).resolve().parent.parent.parent / "creds.yaml"
# VARS = load_yaml(credentials_filepath)


# setup logger
log_filepath = Path(__file__).resolve().parent / "logs" / "gemini_sanity_check.log"
log_filepath.parent.mkdir(parents=True, exist_ok=True)
log_format = (
    "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d (%(funcName)s) - %(message)s"
)
# Configure logger
logging.basicConfig(
    level=logging.INFO,  # Capture all log levels
    format=log_format,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler(log_filepath, mode="w", encoding="utf-8"),  # Logs to file
    ],
)
logger = logging.getLogger(__name__)


def load_gemini(api_key: str) -> genai.Client:
    while True:
        try:
            # Attempt to create a Gemini client
            client = genai.Client(api_key=api_key)
            logger.info("Gemini client loaded successfully.")
            return client
        except ServerError as e:
            logger.debug(
                f"Failed to load Gemini client: {e}. Retrying in five seconds..."
            )
            time.sleep(5)
        except Exception as e:
            raise e


def gen_gemini(client, schema: BaseModel, prompt, max_retries=3, retry_delay=5):
    """Generate content using Gemini."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            )
            return response
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    raise RuntimeError("Failed to generate content after multiple attempts.")


def process_entry_wrapper(args):
    entry, client, schema, id_col, input_query, max_retries, retry_delay = args
    for attempt in range(max_retries):
        try:
            response = gen_gemini(client, schema, input_query)
            if response.parsed:
                parsed_output = response.parsed.model_dump()
            elif response.text:
                logger.warning(
                    f"Couldn't get the response in JSON for entry: {entry[id_col]}!"
                )
                parsed_output = {"text": response.text}
            else:
                logger.warning(f"Couldn't get an output from entry: {entry[id_col]}!")
                return None, 0
            parsed_output[id_col] = entry[id_col]
            return parsed_output, response.usage_metadata.total_token_count
        except Exception as e:
            logger.warning(
                f"Retry {attempt + 1}/{max_retries} for entry {entry[id_col]} failed: {e}"
            )
            time.sleep(retry_delay)

    logger.warning(f"Skipping entry {entry[id_col]} after {max_retries} retries.")
    return None, 0


def process_data_with_gemini(
    client,
    schema: BaseModel,
    data: list[dict],
    id_col: str,
    input_col: str,
    out_jsonl_filepath: Path,
    max_retries=3,
    retry_delay=5,
    max_workers=10,
):
    """Generate content using Gemini in parallel using thread_map with retries."""
    if not out_jsonl_filepath.parent.exists():
        out_jsonl_filepath.parent.mkdir(parents=True, exist_ok=True)
    written_ids = set()
    if out_jsonl_filepath.exists():
        with open(out_jsonl_filepath, "r", encoding="utf-8") as fin:
            for line in fin:
                try:
                    obj = json.loads(line)
                    written_ids.add(obj[id_col])
                except json.JSONDecodeError:
                    continue

    entries_to_process = [entry for entry in data if entry[id_col] not in written_ids]
    logger.info(
        f"Skipping {len(written_ids)} rows. "
        + f"Processing {len(entries_to_process)} new entries."
    )

    # Prepare arguments for thread_map
    args_list = [
        (entry, client, schema, id_col, entry[input_col], max_retries, retry_delay)
        for entry in entries_to_process
    ]
    total_tokens = 0
    lock = Lock()
    with open(out_jsonl_filepath, "a", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(args_list), max_workers), desc="Processing"):
            batch_args = args_list[i : i + max_workers]
            results = thread_map(
                process_entry_wrapper,
                batch_args,
                max_workers=max_workers,
                desc=f"Batch {i//max_workers+1}",
            )
            # write out the results
            for result, tokens_used in results:
                if result:
                    with lock:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        total_tokens += tokens_used

            fout.flush()
            logger.info(f"Tokens used so far: {total_tokens}")

    logger.info(f"Finished processing. Output saved to {out_jsonl_filepath}")


if __name__ == "__main__":
    # load Gemini client
    gemini = load_gemini(VARS["gemini_api_key"])
    # load the dataset
    df = load_dataset()
    # prepare the prompts
    data = prepare_prompt(df)
    out_jsonl_filepath = get_output_path()
    # generate content using Gemini
    process_data_with_gemini(
        client=gemini,
        schema=DataEntry,
        data=data,
        id_col="id",
        input_col="prompt",
        out_jsonl_filepath=out_jsonl_filepath,
        max_workers=10,  # NOTE: Due to limitation to my subscription, this is the maximum concurrent calls allowed by Gemini.
    )
    # check if postprocess_output() function is imported
    if "postprocess_output" in globals():
        # postprocess the output
        postprocess_output()
        logger.info("Done postprocessed the output data!")
