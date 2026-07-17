---
title: Sage
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Ask questions about Apple, Microsoft, and NVIDIA's latest 10-K filings.
---

# Sage

Sage is an AI-native financial research copilot: ask a question in plain
English and get a cited answer pulled from real SEC 10-K filings, not a
model's memorized guess.

This Space is a curated demo pre-loaded with Apple, Microsoft, and NVIDIA's
latest 10-Ks — ask things like:

- "What did Apple say about supply chain risk in its latest 10-K?"
- "Compare how Microsoft and NVIDIA describe AI-related risk factors."
- "What was NVIDIA's revenue for the fiscal year?"

Every answer includes citations back to the specific filing and page it came
from. Retrieval is hybrid (BM25 + vector search) with cross-encoder
reranking before generation, so answers are grounded in the actual retrieved
text rather than the model's general knowledge.

Uploading new documents is disabled on this public demo (`ALLOW_UPLOADS=false`)
— it's a fixed, curated corpus so answer quality stays predictable for
anyone trying it.

See `deploy/huggingface/DEPLOY.md` in the source repo for how this Space is
built and deployed.
