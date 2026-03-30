import csv, json, os, sys, time
from pathlib import Path
import pdfplumber
from openai import OpenAI
import numpy as np


MODEL = 'gpt-5.4'
PAGES_PER_BATCH = 4
MAX_LENGTH = 4000
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5

SYSTEM_PROMPT = """
You are a precise data extractor for social science codebooks.
Your job is to extract every variable and question from the provided codebook text and return a structured JSON.
 
Return ONLY a JSON object with a single key "variables" whose value is an array of variable objects.
 
Each variable object must have:
  - "id": a unique identifier for this item.
  - "name": short variable identifier (e.g. "CSEX", "V23", "Q5A", "State Name", "Identification Number"). Empty string if not present.
  - "type": "variable" (administrative/demographic field) or "question" (survey question posed to respondent).
  - "description": full question text or variable description.
  - "codes": array of {"value": "...", "label": "..."} pairs. Empty array [] if no coded values.
 
Important:
  - Include ALL variables and questions found, even if they have no codes (e.g. open-ended or continuous).
  - Do not include code values in "name" or "description".
  - "id" increments for each new variable or question.
  - Skip administrative and layout text (page headers, footers, titles, section dividers).
  - Preserve the original wording of descriptions.
  - If the page does not contain any variables or questions, return {"variables": []}.
  - Do NOT extract response statistics (e.g. frequency, percentage, mean, median, number of responses etc.). We are only interested in the framing and structure of variables and questions.
  - If the "value" or "label" of a code is blank, then leave it as an empty string e.g. {"value": "", "label": ""}. Do not ignore one if it exists and the other is blank.
  - Pay extra attention to codes. It is most important that you correctly identify code values and labels.

Example:
{
  "variables": [
    {
        "id": 1,
        "name": "FIPSTATE",
        "type": "variable",
        "description": "FIPS State Code",
        "codes": [{"value": "1", "label": "Alabama"}, {"value": "2", "label": "Alaska"}]
    },
    {
        "id": 2,
        "name": "",
        "type": "question",
        "description": "Do you approve or disapprove of the way Bill Clinton is handling foreign affairs?",
        "codes": [{"value": "1", "label": "Approve"}, {"value": "2", "label": "Disapprove"}]
    },
    {
        "id": 3,
        "name": "Income",
        "type": "question",
        "description": "What is your yearly income?",
        "codes": []
    }
  ]
}
"""

USER_TEMPLATE = """
Extract all variables and questions from codebook pages {start}-{end} of {num_pages}.

--- BEGIN CODEBOOK TEXT ---
{text}
--- END CODEBOOK TEXT ---
 
Return only a JSON object with key "variables" containing an array as described.
"""

CONTINUE_TEMPLATE = """

The following variable was incomplete at the end of the previous page and may continue here:
{last_var}
Continue extracting its codes if present, then proceed with new variables. Do not re-emit codes already listed in the carried-over variable — only add newly seen codes.
"""


input_tokens = 0
output_tokens = 0
client = OpenAI()
def call_gpt(text, start_page, end_page, num_pages, last_var):
    user_prompt = USER_TEMPLATE.format(
        start= start_page,
        end= end_page,
        num_pages= num_pages,
        text= text
    )
    syst_prompt = SYSTEM_PROMPT
    if last_var:
        syst_prompt += CONTINUE_TEMPLATE.format(last_var= json.dumps(last_var))

    for attempt in range(1, RETRY_ATTEMPTS+1):
        try:
            response = client.chat.completions.create(
                model= MODEL,
                response_format= {'type': 'json_object'},
                messages= [
                    {'role': 'system', 'content': syst_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            
            if isinstance(parsed, dict):
                parsed = parsed.get("variables", [])
            last_var = parsed[-1] if parsed else {}
            
            return parsed if isinstance(parsed, list) else [], response.usage.prompt_tokens, response.usage.completion_tokens, last_var
        except json.JSONDecodeError as err:
            print(f"    Parse error: {err}")
            return [], 0, 0, {}
        except Exception as err:
            status = getattr(err, 'status_code', None)
            if status in (429, 503):
                print(f"    Limit rate. Retrying ({attempt}/{RETRY_ATTEMPTS})...")
                time.sleep(RETRY_DELAY * attempt)
            else:
                print(f"    API error: {err}")
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(RETRY_DELAY)

    return [], 0, 0, {}


def extract_pages(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text = text.strip()
                text = text[:MAX_LENGTH]
            
            pages.append({'page_num': i+1, 'text': text})
    
    return pages, num_pages


def batch_pages(pages, batch_size):
    return [pages[i : i+batch_size] for i in range(0, len(pages), batch_size)]


def process_pdf(input_tokens, output_tokens, pdf_path, pages_per_batch= PAGES_PER_BATCH):
    pages, num_pages = extract_pages(pdf_path)
    print(f"   {num_pages} pages total, {len(pages)} with extractable text")
    batches = batch_pages(pages, pages_per_batch)
    rows = []

    last_var = {}
    for i, batch in enumerate(batches):
        start_page = batch[0]['page_num']
        end_page = batch[-1]['page_num']

        if all(not page['text'] for page in batch):
            continue
        text = '\n\n'.join([page['text'] for page in batch if page['text']])
        print(f"   Batch {i+1}/{len(batches)}: pages {start_page}-{end_page}...", end= '', flush= True)

        variables, in_tok, out_tok, last_var = call_gpt(text, start_page, end_page, num_pages, last_var)
        input_tokens += in_tok
        output_tokens += out_tok

        batch_rows = 0
        for var in variables:
            if not isinstance(var, dict):
                continue
            var_id = var.get('id') or ''
            var_name = var.get('name') or ''
            var_type = var.get('type') or ''
            description = var.get('description') or ''
            codes = var.get('codes') or []

            if codes:
                for code in codes:
                    rows.append({
                        'id': var_id,
                        'name': var_name,
                        'type': var_type,
                        'description': description,
                        'value': code.get('value', ''),
                        'label': code.get('label', ''),
                        'source_file': Path(pdf_path).name,
                        'page_range': f'{start_page}-{end_page}'
                    })
                    batch_rows += 1
            else:
                rows.append({
                    'id': var_id,
                    'name': var_name,
                    'type': var_type,
                    'description': description,
                    'value': '',
                    'label': '',
                    'source_file': Path(pdf_path).name,
                    'page_range': f'{start_page}-{end_page}'
                })
                batch_rows += 1

        print(f"{len(variables)} variables, {batch_rows} rows")
        time.sleep(0.3)

    print(f"   Complete! {len(rows)} rows extracted")
    GPT4O_INPUT  = 2.50 / 1_000_000
    GPT4O_OUTPUT = 10.00 / 1_000_000
    cost = (input_tokens * GPT4O_INPUT) + (output_tokens * GPT4O_OUTPUT)
    print(f"   Tokens: {input_tokens:,} in / {output_tokens:,} out = ${cost:.4f}")

    return rows, input_tokens, output_tokens


def write_csv(rows, out_path):
    fieldnames = ['id', 'name', 'type', 'description', 'value', 'label', 'source_file', 'page_range']
    with open(out_path, 'w', newline= '', encoding= 'utf-8') as f:
        writer = csv.DictWriter(f, fieldnames= fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_key(path):
    filename = os.path.basename(path)
    parts = filename.split('-')
    x = int(parts[0])
    try:
        y = int(parts[1])
    except ValueError:
        y = 0

    return (x,y)


codebooks_dir = 'intermediate/codebooks'
codebooks_pdfs = [os.path.join(codebooks_dir, pdf) for pdf in os.listdir(codebooks_dir)]
codebooks_pdfs = sorted(codebooks_pdfs, key= sort_key)

test_codebooks = []
while len(test_codebooks) < 5:
    tmp_pdf = codebooks_pdfs[np.random.randint(0, len(codebooks_pdfs)-1)]
    with pdfplumber.open(tmp_pdf) as pdf:
        if len(pdf.pages) < 100 and tmp_pdf not in test_codebooks:
            test_codebooks.append(tmp_pdf)

content = []
for pdf_path in test_codebooks:
    print(pdf_path)
    if not os.path.exists(pdf_path):
        print(f"   File not found: {pdf_path}", file= sys.stderr)
        continue
    rows, input_tokens, output_tokens = process_pdf(input_tokens, output_tokens, pdf_path, pages_per_batch= PAGES_PER_BATCH)
    content.extend(rows)

if content:
    write_csv(content, 'output/codebook_variables.csv')
else:
    print("No variables extracted.")