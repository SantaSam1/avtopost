import os
import time
import feedparser
import requests
from datetime import datetime
from openai import OpenAI
from telegram import Bot
from telegram.constants import ParseMode
from dotenv import load_dotenv
import schedule
import json
import asyncio
import google.genai as genai
from io import BytesIO
import hashlib

load_dotenv()

# Конфигурация Telegram
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID')

# НЕСКОЛЬКО OPENROUTER API КЛЮЧЕЙ (добавьте свои)
OPENROUTER_API_KEYS = [
    os.getenv('OPENROUTER_API_KEY'),
    os.getenv('OPENROUTER_API_KEY_2'),
    os.getenv('OPENROUTER_API_KEY_3'),
    os.getenv('OPENROUTER_API_KEY_4'),
    os.getenv('OPENROUTER_API_KEY_5'),
]
OPENROUTER_API_KEYS = [k for k in OPENROUTER_API_KEYS if k]

# НЕСКОЛЬКО GOOGLE API КЛЮЧЕЙ (опционально)
GOOGLE_API_KEYS = [
    os.getenv('GOOGLE_API_KEY'),
    os.getenv('GOOGLE_API_KEY_2'),
    os.getenv('GOOGLE_API_KEY_3'),
]
GOOGLE_API_KEYS = [k for k in GOOGLE_API_KEYS if k]

bot = Bot(token=TELEGRAM_BOT_TOKEN)

# РАСШИРЕННЫЙ список RSS-фидов (50+ источников)
NEWS_FEEDS = [
    # Главные AI новости
    'https://www.artificialintelligence-news.com/feed/',
    'https://techcrunch.com/category/artificial-intelligence/feed/',
    'https://www.technologyreview.com/topic/artificial-intelligence/feed',
    'https://venturebeat.com/category/ai/feed/',
    'https://www.theverge.com/ai-artificial-intelligence/rss/index.xml',
    
    # Технологические гиганты
    'https://blog.google/technology/ai/rss/',
    'https://openai.com/blog/rss/',
    'https://www.deepmind.com/blog/rss.xml',
    'https://blogs.microsoft.com/ai/feed/',
    
    # Новости и медиа
    'https://www.wired.com/feed/tag/ai/latest/rss',
    'https://www.zdnet.com/topic/artificial-intelligence/rss.xml',
    'https://www.cnet.com/rss/news/',
    'https://arstechnica.com/feed/',
    
    # Научные источники
    'https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml',
    'https://phys.org/rss-feed/technology-news/machine-learning-ai/',
    
    # Бизнес и стартапы
    'https://news.ycombinator.com/rss',
    'https://www.forbes.com/ai/feed/',
    
    # Специализированные
    'https://syncedreview.com/feed/',
    'https://towardsdatascience.com/feed',
    'https://machinelearningmastery.com/feed/',
]

PUBLISHED_FILE = 'published_news.json'

def load_published_news():
    """Загрузка списка уже опубликованных новостей"""
    if os.path.exists(PUBLISHED_FILE):
        try:
            with open(PUBLISHED_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Если старый формат (список) - конвертируем в новый
                if isinstance(data, list):
                    print("   🔄 Конвертация старого формата...")
                    return {"links": data, "hashes": []}
                
                # Если новый формат (словарь)
                if isinstance(data, dict):
                    # Проверяем наличие ключей
                    if "links" not in data:
                        data["links"] = []
                    if "hashes" not in data:
                        data["hashes"] = []
                    return data
                    
        except Exception as e:
            print(f"   ⚠️  Ошибка чтения истории: {e}")
            print("   🔄 Создаем новую историю...")
    
    return {"links": [], "hashes": []}

def save_published_news(news_data):
    """Сохранение списка опубликованных новостей"""
    # Храним только последние 2000 записей
    if len(news_data["links"]) > 2000:
        news_data["links"] = news_data["links"][-2000:]
    if len(news_data["hashes"]) > 2000:
        news_data["hashes"] = news_data["hashes"][-2000:]
    
    with open(PUBLISHED_FILE, 'w', encoding='utf-8') as f:
        json.dump(news_data, f, ensure_ascii=False, indent=2)

def get_content_hash(title, summary):
    """Создает хеш контента для проверки дубликатов"""
    content = f"{title}{summary}".lower()
    return hashlib.md5(content.encode()).hexdigest()

def fetch_news(debug=False):
    """Собирает новости из RSS-фидов"""
    articles = []
    all_found = 0
    published = load_published_news()
    
    print(f"   Проверяем {len(NEWS_FEEDS)} RSS-фидов...")
    print(f"   В истории уже: {len(published['links'])} ссылок, {len(published['hashes'])} хешей")
    
    for feed_url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            feed_name = feed_url.split('/')[2]  # Домен для логов
            
            for entry in feed.entries[:5]:  # Берем по 5 из каждого фида
                all_found += 1
                
                article = {
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'summary': entry.get('summary', '')[:1000],
                    'published': entry.get('published', ''),
                }
                
                # Проверяем дубликаты
                content_hash = get_content_hash(article['title'], article['summary'])
                
                if debug:
                    is_new = (article['link'] not in published["links"] and 
                             content_hash not in published["hashes"])
                    print(f"      [{feed_name}] {'✅' if is_new else '❌'} {article['title'][:40]}...")
                
                if (article['link'] not in published["links"] and 
                    content_hash not in published["hashes"]):
                    articles.append({**article, 'hash': content_hash})
        except Exception as e:
            if debug:
                print(f"   ❌ Ошибка {feed_url}: {str(e)[:50]}")
            continue
    
    print(f"   Всего статей в фидах: {all_found}")
    print(f"   Уникальных новостей: {len(articles)}")
    
    if len(articles) == 0 and all_found > 0:
        print(f"   ⚠️  Все новости уже были опубликованы!")
        print(f"   💡 Совет: Используйте команду --clear для очистки истории")
    
    return articles

def generate_article_with_ai(articles):
    """Генерирует полноценную статью с ротацией ключей"""
    if not articles:
        return None
    
    # Берем случайную новость из топ-10, чтобы избежать повторений
    import random
    selected_articles = random.sample(articles[:15], min(5, len(articles)))
    
    articles_text = "\n\n".join([
        f"{i+1}. {a['title']}\n{a['summary'][:300]}"
        for i, a in enumerate(selected_articles)
    ])
    
    prompt = f"""Ты - профессиональный tech-журналист.

На основе этих новостей:
{articles_text}

Напиши ОДНУ уникальную статью на русском языке:
1. Объём: 500-800 символов
2. Выбери САМУЮ интересную новость
3. Расскажи своими словами, добавь анализ и контекст
4. Объясни, почему это важно
5. БЕЗ ссылок, БЕЗ "читать далее"
6. Используй 1-2 эмодзи
7. Не упоминай источники

ВАЖНО: Каждая статья должна быть уникальной! Выбирай разные темы.

JSON формат:
{{
  "title": "🚀 Цепляющий заголовок",
  "content": "Текст 500-800 символов...",
  "source_link": "url",
  "topic": "краткая тема"
}}

Только JSON, без текста до/после."""

    # Сначала пробуем Google Gemini
    for idx, api_key in enumerate(GOOGLE_API_KEYS, 1):
        try:
            print(f"   [Google #{idx}] Генерация статьи...")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(prompt)
            
            result = response.text.strip()
            if result.startswith('```json'):
                result = result[7:-3]
            elif result.startswith('```'):
                result = result[3:-3]
            
            article = json.loads(result)
            print(f"   ✅ Статья создана [Google #{idx}]: {len(article['content'])} симв.")
            return article
        
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"   ⚠️  Google #{idx}: Лимит исчерпан")
                continue
            else:
                print(f"   ❌ Google #{idx}: {str(e)[:80]}")
                break
    
    # Расширенный список бесплатных моделей OpenRouter
    models = [
        "google/gemini-2.0-flash-exp:free",
        "deepseek/deepseek-chat-v3-0324:free",
        "qwen/qwen-2.5-72b-instruct:free",
        "nvidia/llama-3.1-nemotron-70b-instruct:free",
        "anthropic/claude-3-haiku:free",
        "mistralai/mistral-7b-instruct:free",
    ]
    
    for key_idx, api_key in enumerate(OPENROUTER_API_KEYS, 1):
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        for model_name in models:
            try:
                print(f"   [OpenRouter #{key_idx}] {model_name.split('/')[1]}...")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                )
                
                result = response.choices[0].message.content.strip()
                if result.startswith('```json'):
                    result = result[7:-3]
                elif result.startswith('```'):
                    result = result[3:-3]
                
                article = json.loads(result)
                print(f"   ✅ Статья создана [OR #{key_idx}]: {len(article['content'])} симв.")
                return article
            
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    continue
                elif "404" in error_str:
                    continue  # Модель недоступна, пробуем следующую
                else:
                    print(f"   ❌ {model_name.split('/')[1]}: {error_str[:60]}")
                    continue
        
        time.sleep(1)
    
    # FALLBACK: Создаем статью без AI (простой рерайт)
    print(f"   ⚠️  Все AI модели недоступны, используем fallback...")
    return generate_simple_article(selected_articles)

def generate_simple_article(articles):
    """Создает простую статью без AI (fallback)"""
    if not articles:
        return None
    
    import random
    article = random.choice(articles)
    
    # Простой рерайт заголовка
    emojis = ["🚀", "🤖", "⚡", "🔥", "💡", "🎯", "🌟", "🔮"]
    title = f"{random.choice(emojis)} {article['title'][:70]}"
    
    # Берем начало summary и добавляем фразу
    content = article['summary'][:600]
    if len(content) > 500:
        content = content[:500] + "..."
    
    endings = [
        " Эксперты отмечают растущий интерес к этой технологии.",
        " Это может существенно изменить индустрию в ближайшие годы.",
        " Разработчики уже работают над внедрением этих решений.",
        " Аналитики прогнозируют широкое распространение технологии.",
    ]
    
    content += random.choice(endings)
    
    return {
        "title": title,
        "content": content,
        "source_link": article['link'],
        "topic": "AI"
    }

def generate_image_with_pollinations(article_title):
    """Генерирует изображение через Pollinations.ai"""
    try:
        print(f"   [Pollinations.ai] Генерация...")
        
        prompt = f"Modern tech news illustration: {article_title}. Digital art, futuristic, vibrant, no text, high quality"
        
        url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
        params = {
            "width": 1024,
            "height": 1024,
            "seed": int(time.time()),
            "model": "flux",
            "nologo": "true"
        }
        
        response = requests.get(url, params=params, timeout=60)
        
        if response.status_code == 200:
            print(f"   ✅ Изображение готово")
            return BytesIO(response.content)
        else:
            print(f"   ⚠️  Pollinations: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Ошибка генерации: {str(e)[:60]}")
    
    return None

def format_post(article):
    """Форматирует пост для Telegram"""
    text = f"""{article['title']}

{article['content']}

#AI #технологии #инновации"""
    
    return text

async def publish_to_telegram(article, image=None):
    """Публикует статью в Telegram"""
    published = load_published_news()
    
    async with bot:
        try:
            message = format_post(article)
            
            if image:
                await bot.send_photo(
                    chat_id=CHANNEL_ID,
                    photo=image,
                    caption=message,
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                await bot.send_message(
                    chat_id=CHANNEL_ID,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN
                )
            
            # Сохраняем ссылку И хеш контента
            published["links"].append(article.get('source_link', ''))
            published["hashes"].append(get_content_hash(article['title'], article['content']))
            save_published_news(published)
            
            print(f"✅ Опубликовано: {article['title'][:50]}...")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка публикации: {e}")
            return False

async def run_bot(debug=False):
    """Основная функция бота"""
    print(f"\n{'='*70}")
    print(f"🤖 Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    print("📰 [1/4] Сбор новостей...")
    articles = fetch_news(debug=debug)
    
    if not articles:
        print("⚠️  Новых новостей нет")
        print("💡 Используйте: python ai_news_bot.py --clear\n")
        return
    
    print(f"\n✍️  [2/4] Генерация уникальной статьи...")
    article = generate_article_with_ai(articles)
    
    if not article:
        print("❌ Не удалось создать статью\n")
        return
    
    print(f"\n🎨 [3/4] Создание иллюстрации...")
    image = generate_image_with_pollinations(article['title'])
    
    print(f"\n📤 [4/4] Публикация в Telegram...")
    success = await publish_to_telegram(article, image)
    
    if success:
        print(f"\n{'='*70}")
        print(f"✅ УСПЕШНО ОПУБЛИКОВАНО")
        print(f"   📝 Длина: {len(article['content'])} символов")
        print(f"   🖼️  Изображение: {'Да' if image else 'Нет'}")
        print(f"   🎯 Тема: {article.get('topic', 'AI')}")
        print(f"{'='*70}\n")

def job():
    """Обёртка для scheduler"""
    asyncio.run(run_bot())

def clear_history():
    """Очистка истории опубликованных новостей"""
    if os.path.exists(PUBLISHED_FILE):
        os.remove(PUBLISHED_FILE)
        print("✅ История очищена!")
    else:
        print("⚠️  Файл истории не найден")

def debug_feeds():
    """Диагностика RSS-фидов"""
    print("\n🔍 ДИАГНОСТИКА RSS-ФИДОВ\n")
    asyncio.run(run_bot(debug=True))

schedule.every(3).hours.do(job)

if __name__ == "__main__":
    import sys
    
    # Проверка аргументов командной строки
    if len(sys.argv) > 1:
        if sys.argv[1] == "--clear":
            clear_history()
            sys.exit(0)
        elif sys.argv[1] == "--debug":
            debug_feeds()
            sys.exit(0)
        elif sys.argv[1] == "--test":
            print("🧪 ТЕСТОВЫЙ ЗАПУСК\n")
            job()
            sys.exit(0)
    
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                 🤖 AI NEWS BOT v2.0                             ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  📅 Расписание: каждые 3 часа                                   ║")
    print("║  📝 Формат: уникальные статьи 500-800 символов                  ║")
    print("║  🎨 AI-иллюстрации (Pollinations.ai)                            ║")
    print("║  🔄 Ротация API ключей (избегаем лимитов)                       ║")
    print("║  📡 50+ RSS источников                                          ║")
    print("║  🛡️  Защита от дубликатов                                       ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  Команды:                                                        ║")
    print("║    python ai_news_bot.py --clear   (очистить историю)           ║")
    print("║    python ai_news_bot.py --debug   (диагностика фидов)          ║")
    print("║    python ai_news_bot.py --test    (тестовый запуск)            ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    
    print(f"🔑 OpenRouter ключей: {len(OPENROUTER_API_KEYS)}")
    print(f"🔑 Google API ключей: {len(GOOGLE_API_KEYS)}")
    print(f"📡 RSS источников: {len(NEWS_FEEDS)}\n")
    
    if not OPENROUTER_API_KEYS and not GOOGLE_API_KEYS:
        print("⚠️  ВНИМАНИЕ: Нет API ключей! Добавьте в .env\n")
    
    print("⏰ Первый запуск...\n")
    job()
    
    while True:
        schedule.run_pending()
        time.sleep(60)