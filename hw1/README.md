# Доказательный RAG по ТФКП

Автор: Семенов Вадим. Группа: M05-604.

Python 3.11, поиск на CPU, генерация через OpenAI-совместимый API Qwen3.5-4B. Ревизия модели, thinking, sampling и лимиты зафиксированы в runtime.json. Проверяющие могут заменить endpoint. Для локальной работы на RTX 4080 включены скрипты сервера vLLM.

## Единое окружение Python 3.11

Клиент, PDF-парсер, тесты и сервер Qwen используют одно Conda-окружение tfkp-rag с Python 3.11.18. PyTorch работает с CUDA12.9; TensorFlow проект не использует. Точные Python-зависимости, включая vLLM/PyTorch и транзитивные пакеты, закреплены в requirements.txt. Python, pip, Tectonic и CUDA Toolkit закреплены в configs/environment.yml.

Однократная установка окружения, зависимостей и закреплённых весов:

    bash scripts/setup_qwen_local.sh
    conda activate tfkp-rag

Для уже созданного окружения зависимости можно восстановить отдельно:

    python -m pip install -r requirements.txt
    python -m pip check

Установка рассчитана на Linux x86_64 и NVIDIA RTX4080. Зависимости, модель и токенизатор скачиваются до прогона. Во время ответов сеть используется только для API модели.

## Локальный сервер Qwen на RTX 4080

Запустите сервер в первом терминале:

    conda activate tfkp-rag
    source configs/use_local_qwen.sh
    export LLM_API_KEY=local-qwen-dev
    python scripts/serve_qwen_local.py

Во втором терминале клиент использует то же окружение:

    conda activate tfkp-rag
    source configs/use_local_qwen.sh
    export LLM_API_KEY=local-qwen-dev

API доступен по http://127.0.0.1:8000/v1. Пример ключа предназначен для локального запуска; личные ключи в Git не записываются. При использовании внешнего API задайте LLM_BASE_URL, LLM_MODEL и SUBMISSION_ID. Компоненты запускаются отдельными процессами из одного окружения.

## Индекс и ответы

Для предоставленного корпуса:

    python build_index.py --corpus /path/to/corpus.jsonl --output ./index --config config.json
    python predict.py --index ./index --questions /path/to/questions.jsonl --output ./answers.jsonl --trace ./trace.jsonl --config config.json

Дополнительно оставлен разработанный PDF-парсер. Если корпус нужно получить из учебника:

    python scripts/prepare_corpus.py --pdf /path/to/textbook.pdf --output ./corpus.jsonl --config config.json

Индексатор сохраняет исходные block_id. Reference: BM25 top-6, соседние блоки, упаковка контекста до 3200 input tokens, draft_then_prefill. Первый вызов получает до 2048 output tokens, второй использует остаток общего бюджета 4096. Лимиты на вопрос: ≤4 вызовов, input≤16384, output≤4096 с thinking; на запрос prompt+разрешённый output≤8192. Расход сверяется с API usage, передаются X-Question-ID и X-Submission-ID.

## Проверки

    python -m unittest discover -s tests -q
    python scripts/check_portability.py --output /tmp/portability.json
    python scripts/validate_run.py --questions /path/to/questions.jsonl --corpus /path/to/corpus.jsonl --answers ./answers.jsonl --trace ./trace.jsonl
    python scripts/evaluate.py --questions /path/to/questions.jsonl --gold /path/to/gold.jsonl --corpus /path/to/corpus.jsonl --answers ./answers.jsonl --trace ./trace.jsonl --output ./metrics.json

Для no_context добавьте --no-context. Диагностика с исходными ответами остаётся локальной. scripts/export_call_metadata.py сохраняет usage и ID контекста без текста источников и thinking. Результаты проверки обоснований привязаны к SHA ответов; переносить их на новый запуск нельзя.

Пересчёт опубликованного dev-прогона:

    python scripts/evaluate.py --questions questions/dev_questions.jsonl --gold questions/dev_gold.jsonl --corpus /path/to/corpus.jsonl --answers dev_answers.jsonl --trace dev_trace.jsonl --diagnostics dev_trace.calls.jsonl --review results/content_review_review2_selected.json --output /tmp/dev_metrics.json

## Пять обязательных экспериментов

Конфигурации в configs/submission: no_context.json, bm25.json, without_concise.json (reference), selected_without_finalization.json и selected_without_neighbors.json. Две последние отключают по одному компоненту reference.

    python scripts/run_experiments.py --configs configs/submission/no_context.json configs/submission/bm25.json configs/submission/without_concise.json configs/submission/selected_without_finalization.json configs/submission/selected_without_neighbors.json --index ./index --questions questions/dev_questions.jsonl --gold questions/dev_gold.jsonl --corpus /path/to/corpus.jsonl --runs artifacts/runs

Runner сохраняет ответы и расход каждого запуска; формирование reviewed-таблицы требует отдельной проверки обоснований. scripts/collect_experiments.py собирает CSV только по явно заданному manifest. Исторические и новые запуски автоматически не объединяются.

Основное сравнение — results/qa_prompt_v1_holdout.csv: 40 отложенных вопросов из авторского набора 80. Reference получила Q=0,464, вариант без соседей — Q=0,535833. Старые experiments.csv и корневые dev-ответы относятся к предварительному набору 15 вопросов. Это один прогон, устойчивый выигрыш не доказан.

## Данные и результаты

- own_questions.jsonl и own_gold.jsonl: обязательные 8 авторских вопросов, по 2 на категорию.
- questions/dev_questions.jsonl и dev_gold.jsonl: опубликованный dev-набор.
- questions/expanded_v2: основной авторский набор, 40 dev+40 holdout по 20 темам, аннотации и исходные ссылки. Воспроизведение: python scripts/prepare_expanded_questions.py --directory questions/expanded_v2 --corpus /path/to/corpus.jsonl.
- results/qa_prompt_v1_holdout.csv и results/qa_prompt_v1_holdout/summary.json: итоговая расширенная проверка; методика в docs/QA_PROMPT_V1_RESULTS.md. CSV пилота/dev оставлены отдельно.
- results/review3_index_resources.json: измерения ресурсов индекса; results/formula_validation.csv и textbook*_audit.csv: результаты проверок извлечения.

Индексv5:402 чанка из 298 блоков,1 210 943 байта. Сборка на Ryzen7 5700X:0,0222 с, peak RSS19 288KiB; инициализацияBM25:0,1645 с,42 220KiB. LLM работала наRTX4080 16GiB, RAM31GiB. Границы измерений и ограничения — report.md.

В Git включены рабочий клиент, локальный сервер с установкой окружения, PDF-парсер, необходимые проверки, актуальные данные и итоговые таблицы. Промежуточные конфигурации, архивные версии и полные дополнительные прогоны сохранены локально и исключены из Git-index через .gitignore. PDF, корпус, индекс, веса, окружения, ключи и большие логи не публикуются.
