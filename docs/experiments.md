# Experiments

All development comparisons use the same 401 questions, with 375 scorable and 26
unscorable. The reference-document metric averages recall equally across scorable
questions. Missing or ambiguous references are excluded, rather than marked correct.

## First-search feedback

| Experiment, with the same reranker | Recall@5 | Matched baseline |
| --- | ---: | ---: |
| Question-only short expansion | 69.88% | 68.05% |
| Count-aware short expansion | 70.06% | 68.05% |
| Count-aware expansion in the hybrid | 71.90% | 71.82% |
| Missing-information follow-up | 68.62% | 68.05% |

Expansion adds up to three terms to the original query. Counts are the number of
first-round documents matching each original query term under SQLite's tokenizer.
They are not corpus frequencies or relevance labels. The count-aware and
question-only prompts have the same keyword budget. For the hybrid experiment,
only the BM25 branch changes; dense retrieval still uses the original question.

The short count-aware result was 2.01 percentage points above BM25, with an
exploratory 95% paired interval of [+0.43, +3.87]. Its difference from the
question-only control was only +0.18 points, interval [-1.19, +1.56]. A specific
benefit from the counts was not established.

The stronger hybrid's change was +0.07 points, interval [-1.26, +1.43]. Follow-up
search changed recall by +0.57 points, interval [-0.71, +1.92]. Neither established
an improvement. The follow-up model searched again on 154 questions and stopped
on 247; only five excerpts of up to 4,500 characters were supplied. A second list
was merged with the first using equal RRF60, retaining 50 candidates for reranking.
Human review and iterative feedback were used to calibrate the evaluation criteria
for follow-up retrieval proposals.

## Other attempts

The initial prefix-only embedding setup omitted 62.40% of source tokens, so the
comparison was rebuilt on shared passages. Maximum passage scoring outperformed
averaging and summing in the tested retrieval families. A 128-token overlap grew
the index by 17.52% without improving the tested dense/hybrid rankings. Three-fold
BM25 tuning gave no clear gain with reranking (+0.80 points; [-0.27, +2.13]).

## Evaluation boundaries

The hybrid was selected before its first measurement on 99 held-out questions;
later experiments stayed on development questions. Frozen held-out IDs are included
in the package so the same scopes can be used by the evaluation command.
The 99 are now exposed and their reuse is not a fresh test.

Intervals resample groups of questions sharing reference documents. The expansion
refinements are exploratory comparisons; adaptive development and multiple tests
are not independent confirmations. An interval including zero does not establish
equivalence. The references are synthetic and incomplete; an unlabelled retrieved
document is not necessarily irrelevant. These results do not grade final answers.

The current package exposes the selected methods rather than every historical
runner. Single-query execution and model hosting can produce differences from
historical batched runs. The tables record the completed experiments, not new
scores from the source reorganization.
