# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**  
Describe whether you used the rule based model, the ML model, or both.  
Example: “I used the rule based model only” or “I compared both models.”

I used both models. The primary focus is the rule-based model in `mood_analyzer.py` with lexicon + negation logic, plus a simple ML model in `ml_experiments.py` (CountVectorizer + LogisticRegression) for comparison.

**Intended purpose:**  
What is this model trying to do?  
Example: classify short text messages as moods like positive, negative, neutral, or mixed.

Classify short social-media-style text snippets into mood classes: `positive`, `negative`, `neutral`, `mixed`.

**How it works (brief):** 
For the rule based version, describe the scoring rules you created.  
For the ML version, describe how training works at a high level (no math needed).

- Rule-based model: preprocess text, tokenize, match tokens against positive/negative vocabulary, adjust score, handle negation, map to labels (with `mixed` when balanced sentiments appear).
- ML model: vectorize text as bag-of-words with `CountVectorizer`, train `LogisticRegression` on `SAMPLE_POSTS` + `TRUE_LABELS`, predict with fitted model.

## 2. Data

**Dataset description:**  
Summarize how many posts are in `SAMPLE_POSTS` and how you added new ones.

`SAMPLE_POSTS` was extended from the starter 11 examples to 14 examples in `dataset.py`. Added:
- "I love seeing the sun outside",
- "I hate leaving the house on a nice day",
- "Feeling like a million bucks!",
- "This is so boring",
- "Can't wait for the party tonight!",
- "I am always tired but totally stoked that I finished this project",
- "Nothing is working and I feel awful",
- "I'm weirdly thrilled to be so exhausted"

**Labeling process:**  
Explain how you chose labels for your new examples.  
Mention any posts that were hard to label or could have multiple valid labels.

Labels were chosen by manually interpreting sentiment tokens and overall tone.
- "mixed" when both positive and negative tokens co-occur, e.g., “tired but hopeful”, “stressed but proud”.
- "negative" : clearly negative content.
- "positive" : clearly positive content.
Hard cases: “I'm weirdly thrilled to be so exhausted” is mixed due positive ("thrilled") and negative ("exhausted") signals.

**Important characteristics of your dataset:**  
Examples you might include:  

- Contains slang or emojis  
- Includes sarcasm  
- Some posts express mixed feelings  
- Contains short or ambiguous messages

- Contains slang: "stoked", "fire", "sick", "buckets".
- Contains sarcasm example: "I absolutely love getting stuck in traffic" (true label negative).
- Mixed feelings examples.
- Includes short and clearly emotional sentences.

**Possible issues with the dataset:** 
Think about imbalance, ambiguity, or missing kinds of language.

- Small size (14 examples) => unreliable generalization.
- Label distribution not fully balanced.
- Lexicon-dependent; many expressions need explicit entries.

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  
Describe the modeling choices you made.  
Examples:  

- How positive and negative words affect score  
- Negation rules you added  
- Weighted words  
- Emoji handling  
- Threshold decisions for labels

- Preprocessing in `MoodAnalyzer.preprocess`: strip, lower, remove punctuation, normalize repeated letters (`soooo`→`soo`), split tokens.
- Vocabulary in `dataset.py`:
  - `POSITIVE_WORDS`: includes `happy, great, good, love, excited, awesome, fun, chill, relaxed, amazing, hopeful, proud, fire, sick, stoked, thrilled`
  - `NEGATIVE_WORDS`: includes `sad, bad, terrible, awful, angry, upset, tired, stressed, hate, boring, exhausted`
- Sentiment scoring in `score_text`:
  - positive token = +1, negative = -1
  - simple negation with lookahead: not happy → -1,not bad → +1.
- Label logic in `predict_label`:
  - score > 0 → positive
  - score < 0 → negative
  - score == 0 and both positive+negative words occurred → mixed
  - score == 0 else → neutral

**Strengths of this approach:**  
Where does it behave predictably or reasonably well?

- Transparent and explainable with "explain()" outputs.
- Works well for clean lexicon matches and basic negation.
- Mixed emotions can be captured when both sentiment categories appear.

**Weaknesses of this approach:**  
Where does it fail?  
Examples: sarcasm, subtlety, mixed moods, unfamiliar slang.

- Sarcasm not understood (e.g., "I absolutely love getting stuck in traffic" still positive due "love").
- Unknown slang/emojis are ignored unless added explicitly ("this new song is fire" required adding "fire" to positive list).
- Non-literal sentiment depends on many manual patterns.

## 4. How the ML Model Works (if used)

**Features used:**  
Describe the representation.  
Example: “Bag of words using CountVectorizer.”

"CountVectorizer" bag-of-words representation.

**Training data:**  
State that the model trained on `SAMPLE_POSTS` and `TRUE_LABELS`.

Trained on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`.

**Training behavior:**  
Did you observe changes in accuracy when you added more examples or changed labels?

- With a tiny training set, the model quickly achieves 100% training accuracy (overfitting).
- Additional hand-labeled examples changed dataset performance but training accuracy remained perfect on seen data.

**Strengths and weaknesses:**  
Strengths might include learning patterns automatically.  
Weaknesses might include overfitting to the training data or picking up spurious cues.

- Strengths: learns phrase-token associations, handles linguistic patterns without manual lexicon expansion.
- Weaknesses: overfits small training data, could fail on unseen slang and sarcasm similarly if not represented.

## 5. Evaluation

**How you evaluated the model:**  
Both versions can be evaluated on the labeled posts in `dataset.py`.  
Describe what accuracy you observed.

- Rule-based: "python main.py" prints each prediction vs true label and overall accuracy.
- ML: "python ml_experiments.py" trains and evaluates on same sample data.

**Examples of correct predictions (rule-based):**  
Provide 2 or 3 examples and explain why they were correct.

- "I love this class so much": positive (love token).
- "Today was a terrible day": negative (terrible token).
- "I am not happy about this": negative (negation + happy): handled by negation rule.

**Examples of incorrect predictions (rule-based):**  
Provide 2 or 3 examples and explain why the model made a mistake.  
If you used both models, show how their failures differed.

- "I absolutely love getting stuck in traffic" (true negative) predicted positive because token "love" overrides sarcastic meaning.
- "Feeling like a million bucks!" (true positive) predicted neutral before adding "bucks"/explicit mapping.
- "I'm weirdly thrilled to be so exhausted" (true mixed) predicted negative if "thrilled" missing; after addition, predicted mixed.

**ML vs rule-based failure difference:**  
- ML on training set had 1.00 accuracy (no failure on dataset but likely no generalization). Rule-based showed explicit failure cases that are interpretable.

## 6. Limitations

Describe the most important limitations.  
Examples:  

- The dataset is small  
- The model does not generalize to longer posts  
- It cannot detect sarcasm reliably  
- It depends heavily on the words you chose or labeled

- Small dataset: 14 posts, so both models are underpowered and high variance.
- Rule-based model requires vocabulary engineering and misses sarcasm pattern, as shown by "I absolutely love getting stuck in traffic".
- ML model overfits on training set, yield 1.00 in-sample accuracy, but may fail out-of-sample.
- No emoji sentiment mapping currently, so "I'm okay 😊" stays neutral by default.

## 7. Ethical Considerations

Discuss any potential impacts of using mood detection in real applications.  
Examples: 

- Misclassifying a message expressing distress  
- Misinterpreting mood for certain language communities  
- Privacy considerations if analyzing personal messages

- Misclassification risk for emotional support applications (e.g., assuming "mixed" when user signals distress).
- Cultural and dialect bias: dataset uses mostly standard colloquial English; minority dialects or non-English phrases may be misread.
- Privacy: mood analysis of personal text should require consent and robust data handling.

## 8. Ideas for Improvement

List ways to improve either model.  
Possible directions:  

- Add more labeled data  
- Use TF IDF instead of CountVectorizer  
- Add better preprocessing for emojis or slang 
- Use a small neural network or transformer model  
- Improve the rule based scoring method  
- Add a real test set instead of training accuracy only

- Add more labeled data with diverse slang, emoji, and sarcasm to reduce bias.
- Add emoji-to-token conversion in preprocessing.
- Add multi-word pattern rules for sarcasm and intensity ("absolutely love" + negative context, "not so bad").
- For ML: use TF-IDF, cross-validation, separate test set, or transfer learning (small transformer) rather than only train set evaluation.
- Keep a “rule-based + ML fallback” pipeline so lexicon updates can co-exist with learned behavior.


 

