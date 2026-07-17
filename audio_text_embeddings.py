"""Generate the text and audio embeddings used by the encoding models.

Text (CamemBERTa v2):
  - cross_attention : the statement tokens re-weighted by attention to the context,
                      concatenated with (statement - context). Captures how the
                      context colours the statement's meaning.
  - statement_only  : the statement encoded on its own (no context).

Audio (openSMILE):
  - opensmile       : Geneva eGeMAPSv02 hand-crafted affective/prosodic functionals.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from transformers import AutoModel, AutoTokenizer
import torchaudio
import torch
import opensmile

MODEL_NAME = "almanach/camembertav2-base"


# ── Text embeddings ─────────────────────────────────────────────────────────

def _load_texts(path):
    texts = {}
    for fname in os.listdir(path):
        if fname.endswith('.txt'):
            scenario = fname.split('_')[-1].split('.')[0]
            with open(os.path.join(path, fname), "r") as f:
                texts.setdefault(scenario, []).append((fname, f.read().strip()))
    return texts


def _load_model():
    model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


def embeddings_cross_attention(contexts_path, statements_path, output_dir, model, tokenizer):
    os.makedirs(output_dir, exist_ok=True)
    contexts = _load_texts(contexts_path)
    statements = _load_texts(statements_path)
    pairs_count = sum(len(contexts[s]) * len(statements[s]) for s in contexts if s in statements)
    print(f"[cross-attention] {pairs_count} pairs to generate")

    pair_count = 0
    for scenario, scenario_contexts in contexts.items():
        if scenario not in statements:
            continue
        for context_file, context_text in scenario_contexts:
            inputs_A = tokenizer(context_text, return_tensors="pt",
                                 truncation=True, max_length=512)
            with torch.no_grad():
                hidden_A = model(**inputs_A).last_hidden_state
                emb_A = hidden_A[:, 1:-1, :].mean(dim=1)  # exclude CLS and SEP

            for statement_file, statement_text in statements[scenario]:
                inputs_B = tokenizer(statement_text, return_tensors="pt",
                                     truncation=True, max_length=512)
                with torch.no_grad():
                    hidden_B = model(**inputs_B).last_hidden_state
                    emb_B_tokens = hidden_B[:, 1:-1, :]  # exclude CLS and SEP
                    attn_scores = torch.softmax(
                        torch.matmul(emb_B_tokens, emb_A.transpose(-1, -2)), dim=1
                    )
                    emb_B_weighted = (emb_B_tokens * attn_scores).sum(dim=1)
                    diff = emb_B_weighted - emb_A
                    embedding = torch.cat([emb_B_weighted, diff], dim=1).numpy()

                fname = f"{os.path.splitext(context_file)[0]}_{os.path.splitext(statement_file)[0]}.npy"
                np.save(os.path.join(output_dir, fname), embedding)
                pair_count += 1
                if pair_count % 10 == 0:
                    print(f"  {pair_count}/{pairs_count}")

    print(f"[cross-attention] Done — {pair_count} embeddings saved to {output_dir}")


def embeddings_statement_only(statements_path, output_dir, model, tokenizer):
    os.makedirs(output_dir, exist_ok=True)
    statements = _load_texts(statements_path)
    all_statements = [(f, t) for stmts in statements.values() for f, t in stmts]
    print(f"[statement-only] {len(all_statements)} statements to generate")

    for i, (statement_file, statement_text) in enumerate(all_statements):
        inputs = tokenizer(statement_text, return_tensors="pt",
                           truncation=True, max_length=512)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
            embedding = hidden[:, 1:-1, :].mean(dim=1).numpy()  # exclude CLS and SEP

        fname = os.path.splitext(statement_file)[0] + ".npy"
        np.save(os.path.join(output_dir, fname), embedding)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(all_statements)}")

    print(f"[statement-only] Done — {len(all_statements)} embeddings saved to {output_dir}")


# ── Audio embeddings (openSMILE / eGeMAPSv02) ───────────────────────────────

def create_audio_embeddings(audio_path, output_dir):
    """Write one *_opensmile.npy per .wav: Geneva eGeMAPSv02 functionals (88-d),
    hand-crafted affective/prosodic features. Filename suffix must match
    dataset.WholeBrainDataset.AUDIO_EMBEDDING_SUFFIX ('_opensmile')."""
    os.makedirs(output_dir, exist_ok=True)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    count = 0
    for filename in os.listdir(audio_path):
        if not filename.endswith('.wav'):
            continue
        y, sr = torchaudio.load(os.path.join(audio_path, filename))
        features = smile.process_signal(signal=y.squeeze().numpy(), sampling_rate=sr)
        np.save(os.path.join(output_dir, filename[:-4] + '_opensmile.npy'), features)
        count += 1
    print(f"[opensmile] {count} audio embeddings saved to {output_dir}")


# ── Paths / driver ──────────────────────────────────────────────────────────

# Defaults mirror analysis_helpers.get_paths(): everything resolves relative to
# the repo, so a fresh clone works without editing this file. Override with the
# flags below if your stimuli live elsewhere.
BASE = Path(__file__).resolve().parent


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text-dir", type=Path, default=BASE / "data" / "text",
                   help="stimuli text root, holding contexts/ and statements/")
    p.add_argument("--audio-dir", type=Path, default=BASE / "data" / "audio",
                   help="stimuli audio root (.wav)")
    p.add_argument("--embeddings-dir", type=Path, default=BASE / "embeddings",
                   help="where the embedding subfolders are written")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model, tokenizer = _load_model()

    embeddings_cross_attention(
        os.path.join(args.text_dir, "contexts"),
        os.path.join(args.text_dir, "statements"),
        model=model, tokenizer=tokenizer,
        output_dir=os.path.join(args.embeddings_dir, "text_cross_attention"),
    )
    embeddings_statement_only(
        os.path.join(args.text_dir, "statements"),
        model=model, tokenizer=tokenizer,
        output_dir=os.path.join(args.embeddings_dir, "text_statement_only"),
    )
    create_audio_embeddings(
        audio_path=args.audio_dir,
        output_dir=os.path.join(args.embeddings_dir, "audio_opensmile"),
    )
