import spacy,pickle,faiss,re,os
import numpy as np
import polars as pl
from spacy.tokens import Doc
from rapidfuzz import process, fuzz
from tqdm.autonotebook import tqdm 
from collections import Counter  # We'll use Counter for voting

VECTORIZER_PATH = "Feature Extraction/vectorizer/street_vectorizer.pkl"
INDEX_PATH = "Feature Extraction/vectorizer/street_index_ivfpq.faiss"
STREET_LIST_PATH = "Feature Extraction/vectorizer/all_streets_list.pkl"

nlp = spacy.load("Feature Extraction/ner-model/ner-model-best", disable=["parser", "tagger", "lemmatizer"])

columns_to_read = ["city_name", "district_name", "quarter_name", "street_name"]
column_types = {
    "city_name": pl.Categorical,
    "district_name": pl.Categorical,
    "quarter_name": pl.Categorical
}
base_df = pl.read_csv(
    "data/base_df_filtered.csv",
    columns=columns_to_read, 
    schema_overrides=column_types 
)

ALL_CITIES = set(base_df.get_column("city_name").drop_nulls().unique().to_list())
ALL_DISTRICTS = set(base_df.get_column("district_name").drop_nulls().unique().to_list())
ALL_QUARTERS = set(base_df.get_column("quarter_name").drop_nulls().unique().to_list())
ALL_STREETS = set(base_df.get_column("street_name").drop_nulls().unique().to_list())

try:
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(STREET_LIST_PATH, 'rb') as f:
        all_streets_list = pickle.load(f)
    index = faiss.read_index(INDEX_PATH)
    index.nprobe = 10
except FileNotFoundError:
    print("ERROR: Required search files not found. Please run the Phase 1 setup script first.")
    exit()

turkish_map = str.maketrans("ğüşöçıİ", "gusocii")
def normalize(text: str) -> str:
    if not text:
        return ""
    return str(text).lower().translate(turkish_map).replace("i̇","i").strip()

key_to_column_map = {
    'il': 'city_name',
    'ilce': 'district_name',
    'mahalle': 'quarter_name',
    'sokak': 'street_name'
}


# This pattern targets address keywords at the **end** of a string.
# It captures not only the keywords but also the leading whitespace
# and any trailing suffixes like 'NDA, 'NE, etc.
name_extractor_pattern = re.compile(
    r"\s+\b("  # Capture the space before the keyword
    r"MAHALLESİ|MAHALLE|MAH|"
    r"CADDESİ|CADDE|CAD|CD|"
    r"SOKAĞI|SOKAK|SOK|SK|"
    r"BULVARI|BULVAR|BLV|"
    r"MEYDANI|MEYDAN|MEYD|"
    r"APARTMANI|APT|AP|"
    r"SİTESİ|SİTE|SİT"
    r")\b\.?(?:'[A-ZİÖÜÇŞĞ]+)?\s*$",  # Capture optional dot, suffixes like 'NDA, and trailing spaces
    flags=re.IGNORECASE | re.UNICODE
)

def extract_name_part(text):
    """
    Removes address keywords at the end of a component and
    returns only the proper name.
    """
    if not isinstance(text, str):
        return text
        
    # Match the pattern in the text and replace the matched part with a space.
    cleaned_text = name_extractor_pattern.sub("", text)
    
    # Trim any leading/trailing whitespace that may remain.
    return cleaned_text.strip()


def ent2dict(doc: Doc) -> dict:
    """
    Convert entities from a spaCy Doc object into a predefined
    dictionary structure.
    
    Args:
        doc (spacy.tokens.Doc): The processed spaCy document.
        
    Returns:
        dict: A dictionary populated with the entity text.
    """
    entity_data = {
        'il': None,
        'ilce': None,
        'mahalle': None,
        'sokak': None,
        'semt': None, 
        'pk': None,  
        'diger': None
    }
    other_entities = []
    for ent in doc.ents:
        label = ent.label_.lower()
        if label in entity_data:
            entity_data[label] = extract_name_part(normalize(ent.text))
        else:
            other_entities.append(f"{normalize(ent.text)} ({ent.label_})")
    if other_entities:
        entity_data['other'] = other_entities
        
    return entity_data

#trusted_entity_cache = {}
#Caching is not recommended since it takes lots of space when extracting big datasets and causes kernel to crash.
def find_trusted_entity_cached(
    query: str,
    choices: set,
    entity_type: str,
    trust_threshold: float = 90,
    scorer=fuzz.token_set_ratio,
) -> str | None:
    if not query:
        return None

    #cache_key = f"{entity_type}_{query}_{scorer.__name__}"
    # if cache_key in trusted_entity_cache:
    #     return trusted_entity_cache[cache_key]

    if query in choices:
        result = query
    else:
        extracted_match = process.extractOne(query, choices, scorer=scorer)
        result = None
        if extracted_match and extracted_match[1] >= trust_threshold:
            result = extracted_match[0]

    #trusted_entity_cache[cache_key] = result
    return result


# --- NEW, MOST ADVANCED AND COMPREHENSIVE Regex Pattern ---
# Now also recognizes:
# - Slash (/) or hyphen (-) separators
# - Possible whitespace (\s*) around these separators
NUMERIC_STREET_PATTERN = re.compile(r'^\s*\d+(?:\s*[/-]\s*\d+)?\s*\.?\s*$')



# We use vectorizing approach for finding top 50 street match because running 
# fuzzy on +400000 rows of data every address is really slow.
def find_best_street_match(
    query_street: str,
    k_faiss: int = 50,
    score_cutoff: float = 85.0,
    scorer=fuzz.token_set_ratio,
):
    """
    Finds candidates for a street query via FAISS and selects the best one with rapidfuzz.

    Args:
        query_street (str): The street name to search (e.g., "ataürk caddesi").
        k_faiss (int): Number of candidates FAISS should retrieve.
        score_cutoff (float): Minimum acceptable rapidfuzz score.

    Returns:
        tuple: (best_match_str, score) or (None, 0)
    """
    if not query_street or not isinstance(query_street, str):
        return None, 0

    # --- STAGE 1: FILTER (Fast Candidate Retrieval with FAISS) ---
    query_street = normalize(query_street)
    query_street_name = extract_name_part(query_street)

    # --- UPDATED AND MOST ACCURATE CHECK ---
    # Check if the main part of the street matches a numeric format (e.g., "1778" or "1778/7").
    if NUMERIC_STREET_PATTERN.match(query_street_name):
        # If matched, remove any trailing dot to return the clean result.
        # Example: input "1778/7." returns "1778/7".
        final_numeric_street = query_street_name.rstrip(".")
        return final_numeric_street.strip(), 100.0

    query_vector = vectorizer.transform([query_street]).toarray().astype(np.float32)

    distances, indices = index.search(query_vector, k_faiss)
    candidate_streets = [all_streets_list[i] for i in indices[0]]
    best_match = process.extractOne(
        query_street_name,
        candidate_streets,
        scorer=scorer,
    )

    # 5. If the score exceeds our threshold, return the result.
    if best_match and best_match[1] >= score_cutoff:
        return best_match[0], best_match[1]

    # If below threshold or no match found, return None.
    return None, 0

#Final version, overall this works better than the others.
def calculate_match_scores_hypothesis(
    df: pl.DataFrame,
    entity_dict: dict,
    mapping: dict,
    score_cutoff: int = 85,
    return_top: int = 0,
    verbose: int = 0,
) -> pl.DataFrame:
    
    scorers_to_try = [
        fuzz.token_set_ratio,       
        fuzz.ratio, 
        fuzz.token_sort_ratio,     
    ]
    
    # --- STAGE 1: HYPOTHESIS GENERATION ---
    hypotheses = []
    consolidated_found_entities = {"il": [], "ilce": [], "mahalle": [], "sokak": []}
    
    if verbose:
        print("--- STAGE 1: Generating hypotheses for each scorer... ---")
    for scorer in scorers_to_try:
        il = find_trusted_entity_cached(
            entity_dict.get("il"), ALL_CITIES, "il",
            scorer=scorer, trust_threshold=score_cutoff
        )
        ilce = find_trusted_entity_cached(
            entity_dict.get("ilce"), ALL_DISTRICTS, "ilce",
            scorer=scorer, trust_threshold=score_cutoff
        )
        mahalle = find_trusted_entity_cached(
            entity_dict.get("mahalle"), ALL_QUARTERS, "mahalle",
            scorer=scorer, trust_threshold=score_cutoff
        )
        #sokak = find_trusted_entity_cached(entity_dict.get("sokak"), ALL_STREETS, "sokak", scorer=scorer, trust_threshold=score_cutoff)
        sokak, _ = find_best_street_match(
            entity_dict.get("sokak"), scorer=scorer, score_cutoff=score_cutoff + 5
        )
        
        hypothesis = {
            "scorer_name": scorer.__name__,
            "il": il,
            "ilce": ilce,
            "mahalle": mahalle,
            "sokak": sokak,
        }
        
        if any(v for k, v in hypothesis.items() if k != "scorer_name"):
            hypotheses.append(hypothesis)
            if il: consolidated_found_entities["il"].append(il)
            if ilce: consolidated_found_entities["ilce"].append(ilce)
            if mahalle: consolidated_found_entities["mahalle"].append(mahalle)
            if sokak: consolidated_found_entities["sokak"].append(sokak)
            if verbose > 1:
                print(
                    f"  -> Hypothesis for {scorer.__name__}: "
                    f"{ {k: v for k, v in hypothesis.items() if v} }"
                )

    if not hypotheses:
        if verbose:
            print("No scorer could generate a meaningful hypothesis. Falling back.")
        # --- FIX 1: INITIAL FALLBACK MECHANISM ---
        # To guarantee column order, add names first, then scores.
        fallback_data = {}
        for col_name in mapping.values():
            fallback_data[col_name] = None
        for col_name in mapping.values():
            fallback_data[f"{col_name}_score"] = -1.0
        
        fallback_data["Overall_Score"] = -1.0
        return pl.DataFrame([fallback_data])
            

    # --- STAGE 2: EVALUATE EACH HYPOTHESIS SEPARATELY ---
    best_result_from_all_hypotheses = pl.DataFrame()
    if verbose:
        print("\n--- STAGE 2: Testing each hypothesis individually... ---")
    for hypo in hypotheses:
        if verbose > 1:
            print(f"-> Testing hypothesis ({hypo['scorer_name']}):")
        
        temp_df = df
        if hypo.get("il"):
            temp_df = temp_df.filter(pl.col("city_name") == hypo["il"])
        if hypo.get("ilce"):
            temp_df = temp_df.filter(pl.col("district_name") == hypo["ilce"])
        if hypo.get("mahalle"):
            temp_df = temp_df.filter(pl.col("quarter_name") == hypo["mahalle"])
        if hypo.get("sokak"):
            temp_df = temp_df.filter(pl.col("street_name") == hypo["sokak"])
        
        if temp_df.is_empty():
            if verbose > 1:
                print("  No candidate found with this hypothesis.")
            continue
            
        score_expressions = []
        main_score_columns = []
        for dict_key, df_col_name in mapping.items():
            text_value = entity_dict.get(dict_key)
            score_col_name = f"{df_col_name}_score"
            main_score_columns.append(score_col_name)
            
            if text_value and df_col_name in temp_df.columns:
                choices = (
                    temp_df.get_column(df_col_name).cast(pl.String).to_list()
                )
                all_scores = [
                    process.cdist([text_value], choices, scorer=s)[0]
                    for s in scorers_to_try
                ]
                best_scores = np.max(np.array(all_scores), axis=0)
                score_expressions.append(
                    pl.Series(name=score_col_name, values=best_scores)
                )
            else:
                score_expressions.append(pl.lit(0).alias(score_col_name))
        
        hypothesis_result_df = (
            temp_df.with_columns(score_expressions)
            .with_columns(
                pl.mean_horizontal(main_score_columns).alias("Overall_Score")
            )
            .sort("Overall_Score", descending=True)
            .head(1)
        )
        
        if verbose:
            print(
                f"  -> Best result for {hypo['scorer_name']} hypothesis: "
                f"{hypothesis_result_df.get_column('Overall_Score')[0]:.2f} score"
            )
        
        best_result_from_all_hypotheses = pl.concat(
            [best_result_from_all_hypotheses, hypothesis_result_df]
        )

    # --- STAGE 3: SELECTING THE RESULT OF THE BEST HYPOTHESIS ---
    if best_result_from_all_hypotheses.is_empty():
        if verbose:
            print(
                "\nAll hypotheses tested but no match found in the database. "
                "Falling back to the best NER guess."
            )
        fallback_data = {}
        # Step 1: Fill only the name columns first
        for dict_key, df_col_name in mapping.items():
            found_list = consolidated_found_entities.get(dict_key)
            fallback_value = (
                Counter(found_list).most_common(1)[0][0] if found_list else None
            )
            fallback_data[df_col_name] = fallback_value
        
        # Step 2: Then add score columns
        for dict_key, df_col_name in mapping.items():
            fallback_data[f"{df_col_name}_score"] = -1.0
            
        fallback_data["Overall_Score"] = -1.0
        return pl.DataFrame([fallback_data])
        
    if verbose:
        print(
            "\n--- STAGE 3: Selecting the best result among all hypotheses. ---"
        )
    
    final_df = best_result_from_all_hypotheses.sort(
        "Overall_Score", descending=True
    )
        
    return final_df.head(return_top) if return_top > 0 else final_df


def return_cannon(address,score_cutoff=85,return_top=0):
    entity_dict = ent2dict(nlp(address))

    return entity_dict,calculate_match_scores_hypothesis(base_df,entity_dict,key_to_column_map,score_cutoff,return_top)

