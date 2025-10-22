import json
from pathlib import Path
import xlsxwriter

# Input files
questions_file = "/home/abdelrahman.sadallah/mbzuai/FastChat/fastchat/llm_judge/data/vicuna_bench_arabic/question.jsonl"
scores_file = "/home/abdelrahman.sadallah/mbzuai/FastChat/fastchat/llm_judge/data/vicuna_bench_arabic/model_judgment/gpt-4_single.jsonl"
jais_file = "/home/abdelrahman.sadallah/mbzuai/FastChat/fastchat/llm_judge/data/vicuna_bench_arabic/model_answer/jais_8b_LC_all_exp2p2_6.jsonl"
allam_file = "/home/abdelrahman.sadallah/mbzuai/FastChat/fastchat/llm_judge/data/vicuna_bench_arabic/model_answer/allam.jsonl"
output_file = "/home/abdelrahman.sadallah/mbzuai/FastChat/fastchat/llm_judge/testing_boda/merged_results_arabic.xlsx"   # direct Excel file

def load_jsonl(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    # Load all data
    questions = load_jsonl(questions_file)
    scores = load_jsonl(scores_file)
    jais_answers = load_jsonl(jais_file)
    allam_answers = load_jsonl(allam_file)

    # Index data for quick lookup
    q_map = {q["question_id"]: {"question": q["turns"][0], "category": q["category"]}
             for q in questions}

    # Map scores per model
    score_map = {}
    for s in scores:
        qid = s["question_id"]
        model = s["model"]
        if qid not in score_map:
            score_map[qid] = {}
        score_map[qid][model] = s["score"]

    # Answers from models
    jais_map = {a["question_id"]: a["choices"][0]["turns"][0] for a in jais_answers}
    allam_map = {a["question_id"]: a["choices"][0]["turns"][0] for a in allam_answers}

    # Write Excel file
    with xlsxwriter.Workbook(output_file) as workbook:
        worksheet = workbook.add_worksheet("Results")

        # Write header
        headers = ["Question", "Category", "Jais Answer", "Allam Answer", "Jais Score", "Allam Score"]
        for col, header in enumerate(headers):
            worksheet.write(0, col, header)

        # Write data rows
        row = 1
        for qid, qdata in q_map.items():
            question = qdata["question"]
            category = qdata["category"]

            jais_answer = jais_map.get(qid, "")
            allam_answer = allam_map.get(qid, "")

            jais_score = score_map.get(qid, {}).get("jais_8b_LC_all_exp2p2_6", "")
            allam_score = score_map.get(qid, {}).get("allam", "")

            worksheet.write(row, 0, question)
            worksheet.write(row, 1, category)
            worksheet.write(row, 2, jais_answer)
            worksheet.write(row, 3, allam_answer)
            worksheet.write(row, 4, jais_score)
            worksheet.write(row, 5, allam_score)

            row += 1

    print(f"Excel file written to {output_file}")

if __name__ == "__main__":
    main()
