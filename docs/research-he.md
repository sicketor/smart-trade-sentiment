# מחקר: מודל סנטימנט למניות (26/10/2026)

מסמך עבודה שמרכז את מה שנבדק לפני כתיבת `SPEC-he.md`. כל קביעה כאן נבדקה ברשת או בבדיקה חיה, אלא אם כתוב אחרת.

---

## 1. ביקורת על אפיון Gemini

האפיון של Gemini ([שיחה](https://share.gemini.google/3BNFOAjj1VIC)) נקרא במלואו: שני החלקים, הבסיסי ו"5 השכבות".

### מה טוב בו ונשמר
- הפרדה בין מנוע Python לבין ממשק TypeScript.
- ציון רציף ולא רק תווית, והמרה לסולם 7 דרגות.
- שקלול אמינות מקור ודעיכת זמן.
- איחוד כתבות כפולות לפי דמיון סמנטי (all-MiniLM-L6-v2) ודעיכת חדשנות.
- משקל נמוך למשפטים ניטרליים בטקסט ארוך (שקלול לפי אנטרופיה).
- זיהוי קטליזטורים (Guidance, הנפקה מדללת, FDA, תביעות) ומילון Loughran-McDonald.
- NLI ממוקד טיקר נגד "זליגת סנטימנט" בין חברות באותו משפט.
- מדד נפח חריג (Z-score) ומומנטום סנטימנט.

### מה שגוי או חלש, ומה מחליף אותו
| # | בעיה באפיון Gemini | למה זו בעיה | מה באפיון שלנו |
|---|---|---|---|
| 1 | כל המשקלות "ממוספרים ביד": ‎+0.22 ל-Guidance, ‏‎-0.35 להנפקה, ‏0.40/0.35/0.25 בין המודלים, ספי דרגות 0.08/0.25/0.60 | אין שום מדידה מאחוריהם. אי אפשר לדעת אם הם משפרים או מקלקלים | **מטא-מודל נלמד** (רגרסיה לוגיסטית רב-מחלקתית) על סט מתויג, כיול הסתברויות, ומדידה מול תגובת המחיר. הכללים הופכים ל**פיצ'רים** ולא לתוספות קבועות |
| 2 | אין שום מערך בדיקה (Evaluation) | "דיוק ברמת קרנות גידור" בלי מספר אחד שנמדד | סט זהב מתויג, מדדי Macro-F1, ‏MAE על הסולם, ‏ECE לכיול, ותיקוף מול תשואה עודפת (§9 באפיון) |
| 3 | `ProsusAI/finbert` הוא לב המנוע | הרישיון שלו **CC-BY-NC-SA-4.0** (לא מסחרי). נבדק ב-API של Hugging Face | נשאר כמודל אחד מתוך כמה בשלב הפנימי; לפני Paywall מחליפים במודל עם רישיון מתיר או במודל שאימנו בעצמנו |
| 4 | המודל `MoritzLaurer/DeBERTa-v3-small-zeroshot-v2.0` | **לא קיים** ב-Hugging Face (נבדק) | `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` (מהיר, MIT) או `deberta-v3-base-zeroshot-v2.0` (מדויק יותר, MIT) |
| 5 | אין מודל ייעודי לרשתות חברתיות; VADER כללי עושה את העבודה | סלנג של WSB ו-StockTwits ("calls", "🚀", "bagholder") שונה משפת עיתונות | מודלים שאומנו על ציוצים פיננסיים: `FinTwitBERT-sentiment` (MIT), ‏`roberta-base-stocktwits-finetuned` (Apache-2.0), ו-`twitter-roberta-base-sentiment-latest` (CC-BY-4.0) |
| 6 | פיצול פסוקיות לפי "while", "though" וכו' | "while" הוא גם "בזמן ש". פיצול נאיבי שובר משפטים | פיצול רק על מילות ניגוד חד-משמעיות (but, however, despite, although), והחלטת המשקל עוברת למטא-מודל |
| 7 | שיוך לטיקר לפי הופעת הסימול בטקסט בלבד | מפספס "Nvidia" או "Jensen Huang", וטועה בסימולים שהם מילים (AI, ALL, F, ON) | מילון כינויים מ-SEC (`company_tickers.json`) + רשימת סימולים דו-משמעיים שדורשים `$` או שם חברה |
| 8 | "ניטרלי" מערבב "אין דעה" עם "יש מחלוקת" | ‎+0.8 ו-‎-0.8 מתקזזים ל-0 ונראים כמו שקט | מדד **הסכמה** נפרד ודגל "מעורב" |
| 9 | ממוצע משוקלל פשוט | 2 פוסטים ברדיט יכולים להפוך מניה ל"שורית מאוד" | **כיווץ בייסיאני** לאפס כשיש מעט פריטים (§8 באפיון) |
| 10 | Reddit דרך ‎`.json`, ‏Twitter כמקור | ‎`.json` מחזיר 403 בלי OAuth (נבדק); ל-X אין שכבה חינמית מאז 6/2/2026 | Reddit דרך RSS (עובד, נבדק), ‏StockTwits API ציבורי (עובד, נבדק), ‏X לא נכלל |
| 11 | Finnhub ו-Alpha Vantage כמקורות ליבה | מפתח API, מגבלות, שימוש לא מסחרי | אופציונליים בלבד. הליבה עובדת בלי שום מפתח |
| 12 | שרת FastAPI שרץ כל הזמן | שרת קבוע עולה כסף או נרדם | GitHub Actions (חינם) כמו "חדשות חמות", + הרצה לפי דרישה (§10 באפיון) |

---

## 2. המקורות שהמשתמש ביקש לבדוק

| מקור | מה זה | מה לוקחים | מה לא |
|---|---|---|---|
| [BettaFish](https://github.com/666ghj/BettaFish/blob/main/README-EN.md) | מערכת רב-סוכנית לניתוח דעת קהל בסין: סוכני חיפוש, מדיה ותובנות, "פורום" שבו LLM מנחה ויכוח בין הסוכנים, ומחולל דוחות | **רעיונות:** שכבת "ויכוח" בין מודלים (אצלנו: מדד אי-הסכמה ובורר LLM), דוח מובנה מתוך ייצוג ביניים, ריבוי מודלים כולל ML קלאסי | **קוד:** רישיון GPL-2.0 ו**הצהרה שאוסרת שימוש מסחרי**. הסורקים מכוונים לפלטפורמות סיניות (Weibo, ‏Xiaohongshu). המודלים מאומנים על סינית |
| [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | מודלי LLM פיננסיים (LoRA על Llama2 ו-ChatGLM2) | **נתונים:** סט אימון סנטימנט של ~76.8K שורות ב-Hugging Face. **מדדים להשוואה:** v3.3 מגיע ל-F1 של 0.882 על FPB ו-0.903 על TFNS, מול FinBERT עם 0.880 ו-0.733 | **המודלים עצמם:** 7B–13B פרמטרים, דורשים GPU. לא רצים בחינם על CPU |
| [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) | ספריית NLP של Baidu | **רעיון:** UIE, חילוץ "היבט ← סנטימנט" (Aspect-based) | מודלי הסנטימנט (SKEP) לסינית בלבד, ה-Framework הוא PaddlePaddle ולא PyTorch/ONNX. לא נכנס לסטאק |
| [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | RoBERTa-base שאומן על ~124M ציוצים (2018–2021) ו-TweetEval | נכנס כמודל "טון כללי" לרשתות. רישיון CC-BY-4.0 (מסחרי מותר עם קרדיט) | לא פיננסי: "short", "puts" נקראים לא נכון. לכן רק פיצ'ר אחד מתוך כמה |
| [finiteautomata/bertweet-base-sentiment-analysis](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis) | BERTweet על SemEval-2017 (~40K ציוצים) | — | **רישיון לא מסחרי** (pysentimiento) ותוכן לא פיננסי. לא נכנס. cardiffnlp מכסה את אותו תפקיד ברישיון טוב יותר |
| [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) | מודל בסיס קטן (66M), ‏Apache-2.0, ללא ראש סנטימנט | **הבסיס למודל שלנו** (שלב 5): מאמנים אותו מחדש על נתוני StockTwits + תוויות האנסמבל (Distillation). מודל אחד מהיר במקום 5 | לא שימושי כמו שהוא: אין לו ראש סיווג |
| [Google Cloud Natural Language](https://cloud.google.com/natural-language/pricing) | API מסחרי: ‏score ‏(‎-1..1) ו-magnitude | 5,000 יחידות חינם בחודש (יחידה = 1,000 תווים). אפשר כ**קבוצת ביקורת חד-פעמית** בסט הבדיקה | מודל כללי, לא פיננסי, דורש חשבון עם כרטיס אשראי, ומעבר ל-5K משלמים. לא חלק מהריצה השוטפת |

---

## 3. מחקר אקדמי: מה ידוע

- **FinBERT** (Araci, 2019, ‏[arXiv:1908.10063](https://arxiv.org/abs/1908.10063)): ‏BERT שאומן על טקסט פיננסי. בסיס התחום.
- **Loughran & McDonald** (2011, ‏Journal of Finance): מילון פיננסי. מילים "שליליות" במילונים כלליים (tax, cost, liability) ניטרליות בדוחות. נכנס כפיצ'רים.
- **VADER** (Hutto & Gilbert, 2014): כללים ומילון לטקסט קצר; רגיש לסימני קריאה, אותיות גדולות ואימוג'י. נכנס כפיצ'ר עם מילון מורחב לסלנג מסחר.
- **Kirtac & Germano** (2024, ‏Finance Research Letters, ‏[arXiv:2412.19245](https://arxiv.org/abs/2412.19245)): ‏965K כתבות 2010–2023. מודל שפה גדול (OPT) חזה כיוון תשואה ב-74.4%; אסטרטגיית Long-Short עם Sharpe ‏3.05 מול 1.23 למילון Loughran-McDonald. **מסקנה:** מודל מבוסס-הקשר עדיף בהרבה על מילון.
- **Lopez-Lira & Tang** ("Can ChatGPT Forecast Stock Price Movements?", ‏[arXiv:2304.07619](https://arxiv.org/abs/2304.07619), גרסה 6, 10/2025): ציוני LLM על כותרות מתואמים עם תשואת היום הבא; **מודלים קטנים (BERT, ‏GPT-2) לא חזו תשואה**; ההשפעה חזקה יותר במניות קטנות ובחדשות שליליות, ונשחקת ככל שיותר משקיעים משתמשים ב-LLM. **מסקנה לנו:** לסווג סנטימנט זה לא לחזות מחיר. המדד מתאר את השיח; התיקוף מול מחיר הוא בדיקה, לא הבטחה.
- **SEntFiN 1.0** (Sinha et al., ‏[arXiv:2305.12257](https://arxiv.org/abs/2305.12257)): ‏10,753 כותרות עם סנטימנט **לכל ישות**; ב-2,847 מהן יש כמה ישויות, לעיתים בסנטימנט הפוך. ‏RoBERTa/FinBERT הגיעו ל-~94%. **מסקנה:** הסנטימנט שייך לישות ולא לכותרת. נכנס כסט אימון ובדיקה ל-NLI ממוקד הטיקר.
- **FinEntity** (Tang et al., ‏EMNLP 2023): סנטימנט ברמת ישות בחדשות פיננסיות. סט בדיקה נוסף.
- **StockTwits ומחירים** (מחקרים מעורבים): חלק מוצאים כוח ניבוי לטווח קצר (שעה) ובעיקר לסנטימנט שלילי; מחקר על 5 שנות נתונים מצא שהתשואה מנבאת את הסנטימנט ולא להפך. **מסקנה:** רשתות חברתיות הן ערוץ נפרד עם דעיכה מהירה, ושווה להציג גם "תשומת לב" (נפח) ולא רק כיוון.
- **Reddit/WSB** (‏AMCIS 2021, ‏1.1M פוסטים): ראיות לניבוי **תנודתיות** יותר מכיוון. **מסקנה:** מדד "הייפ/תשומת לב" נפרד.
- **LLMs קלים** (‏[arXiv:2512.00946](https://arxiv.org/abs/2512.00946), 12/2025): ‏Qwen3-8B ו-Llama3-8B מובילים בסנטימנט פיננסי גם עם 5% מנתוני האימון. ‏8B על CPU איטי מדי לריצה שוטפת, אבל מתאים כ**בורר** ל-10% מהפריטים הקשים.
- **FinBERT ב-ONNX/int8** (‏[FinNLP 2025](https://aclanthology.org/2025.finnlp-2.6)): קוונטיזציה לחיסכון בזמן ריצה. קיימות גרסאות ONNX מוכנות (`Xenova/finbert`, ~110MB ב-int8).

---

## 4. מקורות נתונים: בדיקה חיה (26/10/2026, מהמחשב של Tom)

| מקור | בקשה | תוצאה |
|---|---|---|
| StockTwits | `api.stocktwits.com/api/2/streams/symbol/AAPL.json` | **200**, ‏30 הודעות אחרונות, בלי מפתח. ~17% מתויגות ע"י המפרסם כ-Bullish/Bearish (**תוויות חינמיות לאימון**) |
| StockTwits Trending | `api.stocktwits.com/api/2/trending/symbols.json` | **200** |
| Google News RSS | `news.google.com/rss/search?q=NVDA+stock+when:1d` | **200** |
| Yahoo Finance RSS | `feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA` | **200** |
| Finviz | `finviz.com/quote.ashx?t=NVDA&p=d` | **200** (אחרי הפניה), טבלת `news-table` קיימת |
| SEC EDGAR | `data.sec.gov/submissions/CIK….json` | **200** עם User-Agent תקני |
| Reddit RSS | `reddit.com/r/wallstreetbets/search.rss?q=NVDA` | **200** בלי OAuth |
| Reddit JSON | `reddit.com/r/…/search.json` | **403**: חסום בלי OAuth |
| Bluesky חיפוש | `public.api.bsky.app/…/searchPosts` | **403**: דורש התחברות (App Password חינמי) |
| Hacker News | `hn.algolia.com/api/v1/search_by_date` | **200** (רלוונטי רק למניות טק) |
| X (Twitter) | — | אין שכבה חינמית לקריאה מאז 6/2/2026 (תשלום לפי שימוש, ~$0.005 לפוסט). **לא נכלל** |

**רישוי:** ‏Yahoo, ‏Reddit (שכבה חינמית), Finnhub ו-Alpha Vantage מוגבלים לשימוש לא מסחרי. אותה החלטה שמופיעה ב-SPEC של הטרמינל §9.4: לפני Paywall צריך החלטה כתובה.

---

## 5. מודלים: רישיון וגודל (נבדק ב-API של Hugging Face)

| מודל | תפקיד | רישיון | הערה |
|---|---|---|---|
| `ProsusAI/finbert` | חדשות, 3 מחלקות | **CC-BY-NC-SA-4.0** | לא מסחרי |
| `yiyanghkust/finbert-tone` | טון דוחות ושיחות ועידה | לא מוצהר | לבדוק מול המחברים לפני שימוש מסחרי |
| `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` | חדשות, 3 מחלקות, 82M | Apache-2.0 | אומן על Financial PhraseBank (שהרישיון שלו NC); הדיוק 98% הוא על תת-סט "הסכמה מלאה" בלבד |
| `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` | NLI ממוקד טיקר, מהיר | MIT | |
| `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | NLI ממוקד טיקר, מדויק | MIT | |
| `StephanAkkerman/FinTwitBERT-sentiment` | ציוצים פיננסיים: NEUTRAL/BULLISH/BEARISH | MIT | אומן על 10M ציוצי מניות וקריפטו |
| `zhayunduo/roberta-base-stocktwits-finetuned` | StockTwits: ‏Bullish/Bearish | Apache-2.0 | 3.2M הודעות, 93.4% על ולידציה |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | טון כללי ברשתות | CC-BY-4.0 | |
| `sentence-transformers/all-MiniLM-L6-v2` | Embeddings לאיחוד כפילויות | Apache-2.0 | 384 ממדים, מהיר מאוד |
| `distilbert-base-uncased` | בסיס למודל שלנו | Apache-2.0 | |
| `finiteautomata/bertweet-base-sentiment-analysis` | — | **לא מסחרי** | לא נכלל |

## 6. סבב מחקר שני: איך מגיעים לדיוק גבוה בחינם
- **מודל גדול מנצח מילונים ו-FinBERT.** ‏Kirtac & Germano (2024): סנטימנט של LLM על חדשות נתן Sharpe ‏3.05 מול 1.23 למילון Loughran-McDonald. ‏arXiv 2403.02185: ‏ChatGPT ‏F1 ‏83.1, ‏FinBERT ‏65.3.
- **זיקוק סוגר את רוב הפער.** באותו מאמר, תלמיד MPNet שאומן על תוויות ה-LLM הגיע ל-77.8. כך גם TinyFinBERT ‏(2409.18999) וזיקוק עם נתונים סינתטיים (LREC 2026). מכאן גישת מורה ← תלמיד (SPEC §5.0).
- **סנטימנט ממוקד ישות.** בכותרות עם כמה חברות, מודל רגיל טועה בכיוון. ‏CLiC-it 2025 ומחקר מ-2025: ‏DeBERTa-v3-base-absa הוביל בכותרות ממוקדות ישות. לכן A1 (absa, ההיבט = שם החברה) והקלט של התלמיד `"{T} | {Company} [SEP] {text}"`.
- **מודלים קטנים לא חוזים תשואות.** ‏Lopez-Lira & Tang ‏v6: היכולת לחזות מופיעה רק במודלים גדולים. ‏arXiv 2609.20198 ‏(2026): אין קשר מובהק בין סנטימנט לתשואה של היום הבא. לכן המדד מוצג כ**מדד שיח**, לא כתחזית, ונבדק מול השוק בכנות (SPEC §9.3).

## 7. סטים ציבוריים לבדיקה ולאימון
| סט | מה | רישיון |
|---|---|---|
| TFNS (`zeroshot/twitter-financial-news-sentiment`) | ~12K ציוצי חדשות פיננסיות, 3 מחלקות | MIT |
| SEntFiN (`tasksource/sentfin`) | ~10K כותרות, סנטימנט לכל ישות | מחקר |
| FinEntity (`yixuantt/FinEntity`) | ישויות + סנטימנט בטקסט פיננסי | מחקר |
| FiQA-SA | כותרות ופוסטים, ציון רציף | מחקר |
| Financial PhraseBank | ~4.8K משפטים מדוחות | CC-BY-NC-SA (בדיקה בלבד) |

## 8. לקחים מהרצה ראשונה של המנוע (26/10/2026)
- **NLI:** ההשערה "This is good news for X stock" נותנת entailment נמוך מאוד (~0.06) גם על "עקפה והעלתה תחזית". ההשערה "The stock of X will go up" נותנת 0.66. הוחלף.
- **תשומת לב:** רוב המקורות מחזירים את N הפריטים האחרונים בלבד, ולכן ספירת פריטים לפי יום תמיד מראה "קפיצה" ב-24 השעות האחרונות (‏z ≈ 8–14 לכל מניה). משתמשים בצפיות ויקיפדיה (30 יום) עד שתהיה היסטוריית תמונות משלנו.
- **EDGAR:** ‏`acceptanceDateTime` הוא UTC אמיתי (השעות מתרכזות ב-20:00–01:00Z = אחרי הסגירה בניו יורק).
- **קידוד:** פידים של RSS חייבים להיקרא כבייטים כדי ש-feedparser יכבד את הקידוד המוצהר; אחרת מרכאות מסולסלות הופכות ל-�.
- **רלוונטיות:** כותרות "Market Report" שמונות את החברה ברשימת שחקנים, ופריטי אגרגטור שלא מזכירים אותה, הגיעו לראש הרשימות. כעת הן מוגבלות ל-0.3, ורשימות "הכי שוריים/דוביים" לוקחות רק פריטים עם רלוונטיות ≥ 0.6.
