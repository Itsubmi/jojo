import asyncio
import sys

# ЭТО ДОЛЖНО БЫТЬ В САМОМ НАЧАЛЕ ФАЙЛА, ДО ЛЮБЫХ ДРУГИХ ИМПОРТОВ
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
import re

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
)
from aiogram.utils.markdown import hlink

# Импорт для работы с Gigachat
from langchain_gigachat import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage
from config import *
# ========== Логирование ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== Инициализация ==========
bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Инициализация Gigachat
giga = GigaChat(
    credentials=GIGACHAT_CREDENTIALS,
    scope=GIGACHAT_SCOPE,
    verify_ssl_certs=False,
    model="GigaChat"
)

# ========== Определение состояний ==========
class SurveyStates(StatesGroup):
    waiting_for_consent = State()
    goal = State()
    weight = State()
    height = State()
    activity = State()
    brelok = State()
    restrictions = State()
    diet_type = State()
    deficiencies = State()
    finish = State()
    waiting_for_question = State()

# ========== Структура товара ==========
@dataclass
class Product:
    name: str
    category: str
    goal: str
    contraindications: str
    diet_type: str
    url_wb: str
    url_kultlab: str
    article: str
    usage_method: str
    description: str
    
    def get_primary_url(self) -> str:
        if self.url_kultlab and self.url_kultlab != 'nan':
            return self.url_kultlab
        elif self.url_wb and self.url_wb != 'nan':
            return self.url_wb
        return "#"
    
    def get_wb_url(self) -> str:
        return self.url_wb if self.url_wb and self.url_wb != 'nan' else None
    
    def get_kultlab_url(self) -> str:
        return self.url_kultlab if self.url_kultlab and self.url_kultlab != 'nan' else None

# ========== Загрузка из CSV ==========
def load_products_from_csv(file_path: str) -> List[Product]:
    """Загрузка товаров из CSV файла"""
    try:
        if not os.path.exists(file_path):
            print(f"❌ Файл {file_path} не найден!")
            return []
            
        df = pd.read_csv(file_path, encoding='cp1251', sep=';')
        products = []
        
        for _, row in df.iterrows():
            if pd.isna(row.get('Название')):
                continue
                
            product = Product(
                name=str(row.get('Название', '')).strip(),
                category=str(row.get('Категория', '')).strip() if pd.notna(row.get('Категория')) else '',
                goal=str(row.get('Цель', '')).strip() if pd.notna(row.get('Цель')) else '',
                contraindications=str(row.get('Противопоказания', '')).strip() if pd.notna(row.get('Противопоказания')) else '',
                diet_type=str(row.get('Тип питания', '')).strip() if pd.notna(row.get('Тип питания')) else '',
                url_wb=str(row.get('Ссылка на WB', '')).strip() if pd.notna(row.get('Ссылка на WB')) else '',
                url_kultlab=str(row.get('Ссылка на kultlab.ru', '')).strip() if pd.notna(row.get('Ссылка на kultlab.ru')) else '',
                article=str(row.get('артикулы', '')).strip() if pd.notna(row.get('артикулы')) else '',
                usage_method=str(row.get('способ применения', '')).strip() if pd.notna(row.get('способ применения')) else '',
                description=str(row.get('описание', '')).strip() if pd.notna(row.get('описание')) else ''
            )
            products.append(product)
            
        print(f"✅ Загружено {len(products)} товаров из CSV")
        return products
        
    except Exception as e:
        print(f"❌ Ошибка загрузки CSV: {e}")
        return []

def load_deficiency_products(file_path: str = "products_csv/deficiencies.csv") -> List[Product]:
    return load_products_from_csv(file_path)

# ========== Основной класс рекомендаций через GigaChat ==========
class GigaProductRecommender:
    def __init__(self, products: List[Product], giga_instance):
        self.products = products
        self.giga = giga_instance
    
    async def recommend_products(self, user_data: Dict) -> Tuple[List[Tuple[Product, str]], str]:
        """
        Основной метод: GigaChat сам выбирает 3 лучших товара
        Возвращает: (список товаров с причинами, raw_response от Giga)
        """
        
        # Формируем промпт для GigaChat
        prompt = self._build_selection_prompt(user_data)
        
        try:
            messages = [
                SystemMessage(content="""Ты — AI-ассистент бренда спортивного питания KULTLAB. 
Твоя задача — выбрать 3 наиболее подходящих товара из предоставленного списка, 
строго учитывая все ограничения пользователя. Отвечай ТОЛЬКО в формате JSON."""),
                HumanMessage(content=prompt)
            ]
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.giga.invoke(messages))
            
            # Парсим ответ Giga
            selected_products = self._parse_giga_response(response.content, user_data)
            
            return selected_products, response.content
            
        except Exception as e:
            print(f"❌ Ошибка GigaChat: {e}")
            return [], ""
    
    def _build_selection_prompt(self, user_data: Dict) -> str:
        """Строит промпт для GigaChat"""
        
        # Формируем список товаров в читаемом виде
        products_text = []
        for idx, p in enumerate(self.products):
            products_text.append(f"""
Товар {idx + 1}:
- Название: {p.name}
- Категория: {p.category}
- Цель по таблице: {p.goal}
- Противопоказания: {p.contraindications}
- Тип питания (из таблицы): {p.diet_type}
- Способ применения: {p.usage_method}
- Описание: {p.description[:200]}...
- Ссылка Kultlab: {p.get_kultlab_url() or 'нет'}
- Ссылка WB: {p.get_wb_url() or 'нет'}
""")
        
        prompt = f"""
Твоя задача — выбрать 3 (ТРИ) наиболее подходящих товара из списка ниже для пользователя.

=== ДАННЫЕ ПОЛЬЗОВАТЕЛЯ ===
- Цель: {user_data.get('goal', 'не указана')}
- Уровень активности: {user_data.get('activity', 'не указан')}
- Белок из еды: {user_data.get('brelok', 'не указано')}  (варианты: "достаточно", "не хватает", "частично")
- Ограничения/аллергии: {user_data.get('restrictions', 'нет')}
- Тип питания: {user_data.get('diet_type', 'не указан')}  (варианты: "всеядный", "веганство", "лакто-ово-вегетарианство", "пескетарианство")
- Дефициты: {user_data.get('deficiencies', 'нет дефицитов')}
- Проблемы с ЖКТ: {user_data.get('gut_issues', 'нет')}
- Проблемы с сердцем/давлением: {user_data.get('heart_issues', 'нет')}
- Тревожность: {user_data.get('anxiety', 'нет')}

=== ПРАВИЛА ПОДБОРА (ОБЯЗАТЕЛЬНЫ К ИСПОЛНЕНИЮ) ===

1. **Корректировка приоритетов:**
   - Если белка из еды НЕ хватает (brelok = "не хватает" или "частично") → добавь в рекомендацию протеин
   - Если активность высокая → можно добавить BCAA, креатин, изотоник
   - Если активность низкая → НЕ предлагай предтреники, жиросжигатели, стимуляторы

2. **Абсолютные исключения (НЕ ПРЕДЛАГАТЬ):**
   - Лактоза: исключи сывороточный концентрат, молочный протеин
   - ЖКТ: исключи жиросжигатели, предтреники, высокий кофеин, с осторожностью креатин
   - Сердце/давление: исключи кофеин, гуарану, йохимбин, аргинин, стимуляторы
   - Тревожность: исключи кофеин, стимуляторы
   - Аллергия на орехи: исключи батончики с орехами, протеин с орехами
   - Аллергия на рыбу: исключи рыбий жир, морской коллаген

3. **Диетические ограничения:**
   - Веганство: только растительный протеин, без желатина/коллагена, без рыбьего жира
   - Лакто-ово-вегетарианство: можно молочное, НЕЛЬЗЯ рыбий жир, коллаген животный
   - Пескетарианство: можно рыбу, НЕЛЬЗЯ сывороточный протеин, казеин, говяжий коллаген

4. **Дефициты:**
   - Если указаны дефициты (например: железо, витамин D, B12, магний, цинк, кальций, йод, омега-3)
   - Ты должен найти и предложить товары, которые содержат эти элементы
   - Смотри на название товара, категорию, описание
   - Пример: если дефицит железа → ищи "железо", "феррум", "Fe" в названии или описании
   - Пример: если дефицит витамина D → ищи "витамин D", "D3"
   - Пример: если дефицит магния → ищи "магний", "Mg", "магний B6"
   - Можешь предложить 1 товар, который закрывает несколько дефицитов сразу

5. **Важно:**
   - Выбери 3 РАЗНЫХ товара (разные категории/функции)
   - Если есть дефициты - товар для их восполнения ДОЛЖЕН быть в рекомендации
   - Не придумывай товары, которых нет в списке
   - Учитывай цель пользователя как основной ориентир

=== СПИСОК ТОВАРОВ ===
{''.join(products_text)}

=== ФОРМАТ ОТВЕТА (ТОЛЬКО JSON, БЕЗ ЛИШНИХ СЛОВ) ===
{{
  "selected_products": [
    {{
      "product_index": 1,
      "reason": "почему этот товар подходит пользователю (2-3 предложения)",
      "how_to_use": "как принимать (из способа применения товара или описание)",
      "warnings": "если есть предупреждения для этого пользователя"
    }}
  ]
}}

Верни ТОЛЬКО JSON, без пояснений перед ним.
"""
        return prompt
    
    def _parse_giga_response(self, response_text: str, user_data: Dict) -> List[Tuple[Product, str]]:
        """Парсит JSON ответ от Giga и возвращает список товаров с причинами"""
        
        try:
            # Пытаемся найти JSON в ответе
            import json
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                print("❌ Не найден JSON в ответе Giga")
                return []
            
            result = []
            for item in data.get('selected_products', []):
                idx = item.get('product_index', 0) - 1
                if 0 <= idx < len(self.products):
                    reason = item.get('reason', 'Подобран по вашим параметрам')
                    result.append((self.products[idx], reason))
            
            return result[:3]  # максимум 3
            
        except Exception as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
            print(f"Ответ Giga: {response_text[:500]}")
            return []

# ========== Генерация финального ответа ==========
async def generate_recommendation_response(
    products_with_reasons: List[Tuple[Product, str]], 
    user_data: Dict,
    giga_instance
) -> str:
    """Генерирует красивый ответ пользователю через GigaChat"""
    
    if not products_with_reasons:
        return "😔 К сожалению, не удалось подобрать продукты под ваши параметры. Попробуйте обратиться к консультанту KULTLAB: https://kultlab.ru/"
    
    # Формируем данные о подобранных товарах
    products_info = []
    for product, reason in products_with_reasons:
        products_info.append(f"""
Название: {product.name}
Категория: {product.category}
Причина подбора: {reason}
Способ применения: {product.usage_method}
Ссылка Kultlab: {product.get_kultlab_url() or 'нет'}
Ссылка WB: {product.get_wb_url() or 'нет'}
""")
    
    prompt = f"""
Ты — эксперт KULTLAB. Составь дружелюбный ответ пользователю по шаблону.

=== ДАННЫЕ ПОЛЬЗОВАТЕЛЯ ===
Цель: {user_data.get('goal')}
Активность: {user_data.get('activity')}
Ограничения: {user_data.get('restrictions')}
Тип питания: {user_data.get('diet_type')}
Дефициты: {user_data.get('deficiencies')}

=== ПОДОБРАННЫЕ ТОВАРЫ ===
{''.join(products_info)}

=== ШАБЛОН ОТВЕТА ===
🎯 *Ваш персональный подбор KULTLAB*

(1-2 предложения, почему эти продукты подходят именно вам, учитывая дефициты, тип питания и остальные данные пользователя)

*Подбор:* 
— *Название 1* → причина подбора → как принимать
— *Название 2* → причина подбора → как принимать
— *Название 3* → причина подбора → как принимать

*Где купить:*
— Название 1: [ссылка]
— Название 2: [ссылка]
— Название 3: [ссылка]

(Если есть ограничения/аллергии, добавь этот блок)
⚠️ *Важно:* Перед применением проконсультируйтесь со специалистом, учитывая ваши хронические заболевания.

👉 Это базовый набор, который уже даст результат без перегрузки.

*Важно:* Используй ТОЛЬКО реальные товары из списка. Будь краток и понятен.
"""
    
    try:
        messages = [
            SystemMessage(content="Ты эксперт KULTLAB. Отвечай по шаблону, используя только реальные товары из списка."),
            HumanMessage(content=prompt)
        ]
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: giga_instance.invoke(messages))
        
        return response.content.strip()
        
    except Exception as e:
        print(f"❌ Ошибка генерации ответа: {e}")
        return _fallback_response(products_with_reasons)

def _fallback_response(products_with_reasons: List[Tuple[Product, str]]) -> str:
    """Простой fallback ответ"""
    response = "🎯 *Ваш персональный подбор KULTLAB*\n\n*Подбор:*\n"
    
    for i, (product, reason) in enumerate(products_with_reasons, 1):
        url = product.get_kultlab_url() or product.get_wb_url()
        response += f"\n*{i}. {product.name}*\n→ {reason}\n→ {product.usage_method}\n"
        if url:
            response += f"→ Купить: {url}\n"
    
    response += "\n👉 Базовый набор для старта без перегрузки."
    return response

# ========== Упрощенная загрузка CSV по цели ==========
def load_products_by_goal_csv(goal_keyword: str, base_path: str = "products_csv") -> List[Product]:
    """
    Загружает CSV файл, соответствующий цели пользователя
    goal_keyword: 'снижение веса', 'набор массы', 'выносливость', 'здоровье', 'красота', 'others'
    """
    csv_files = {
        'снижение веса': 'weight_loss.csv',
        'набор мышечной массы': 'muscle_gain.csv',
        'выносливость': 'endurance.csv',
        'здоровье': 'health.csv',
        'красота': 'beauty.csv',
    }
    
    # Поиск по ключевым словам
    goal_lower = goal_keyword.lower()
    for key, filename in csv_files.items():
        if key in goal_lower:
            file_path = os.path.join(base_path, filename)
            return load_products_from_csv(file_path)
    
    # Если не нашли, грузим others.csv
    return load_products_from_csv(os.path.join(base_path, 'others.csv'))

# ========== ИСПОЛЬЗОВАНИЕ ==========
async def main_recommendation_flow(user_data: Dict, giga_instance):
    """
    Главный флоу рекомендации
    """
    # 1. Загружаем CSV по цели пользователя
    products = load_products_by_goal_csv(user_data.get('goal', ''))
    
    if not products:
        return "❌ Нет товаров для вашей цели. Обратитесь к консультанту."
    
    # 2. Создаем рекомендателя
    recommender = GigaProductRecommender(products, giga_instance)
    
    # 3. Получаем топ-3 товара от Giga
    selected_products, _ = await recommender.recommend_products(user_data)
    
    if not selected_products:
        return "😔 Не удалось подобрать продукты. Попробуйте изменить параметры или обратитесь к консультанту."
    
    # 4. Генерируем финальный ответ
    response = await generate_recommendation_response(selected_products, user_data, giga_instance)
    
    return response

# Пример использования:
# result = await main_recommendation_flow(user_data, giga)

# ========== Клавиатуры ==========
consent_keyboard = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="Да"), KeyboardButton(text="Нет")]],
    resize_keyboard=True,
    one_time_keyboard=True
)

goal_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="снижение веса / рельеф", callback_data="goal_loss")],
    [InlineKeyboardButton(text="набор мышечной массы", callback_data="goal_gain")],
    [InlineKeyboardButton(text="красота: кожа, волосы, ногти", callback_data="goal_beauty")],
    [InlineKeyboardButton(text="выносливость / силовые", callback_data="goal_endurance")],
    [InlineKeyboardButton(text="здоровье и восстановление", callback_data="goal_health")]
])

activity_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="Низкий (сидячий образ жизни)", callback_data="activity_low")],
    [InlineKeyboardButton(text="Средний (тренировки 3-5 раз/нед)", callback_data="activity_medium")],
    [InlineKeyboardButton(text="Высокий (ежедневные тренировки)", callback_data="activity_high")]
])

brelok_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="Да, хватает", callback_data="brelok_yes")],
    [InlineKeyboardButton(text="Нет, не хватает", callback_data="brelok_no")],
    [InlineKeyboardButton(text="Частично", callback_data="brelok_sometimes")]
])

RESTRICTIONS = {
    "restriction_none": "Нет ограничений",
    "restriction_lactose": "Лактоза",
    "restriction_nuts": "Орехи",
    "restriction_fish": "Рыба/морепродукты",
    "restriction_stomach": "Проблемы с ЖКТ",
    "restriction_heart": "Проблемы с сердцем/давлением",
    "restriction_anxiety": "Тревожность"
}

diet_type_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="Обычное питание", callback_data="diet_regular")],
    [InlineKeyboardButton(text="Лакто-ово-вегетарианство", callback_data="diet_lacto")],
    [InlineKeyboardButton(text="Веганство", callback_data="diet_vegan")],
    [InlineKeyboardButton(text="Пескетарианство", callback_data="diet_pescatarian")]
])

deficiencies_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="Нет дефицитов", callback_data="deficiencies_none")],
    [InlineKeyboardButton(text="Железо", callback_data="deficiencies_iron")],
    [InlineKeyboardButton(text="Витамин D", callback_data="deficiencies_vitd")],
    [InlineKeyboardButton(text="Витамин B12", callback_data="deficiencies_b12")],
    [InlineKeyboardButton(text="Йод", callback_data="deficiencies_iodine")],
    [InlineKeyboardButton(text="Магний", callback_data="deficiencies_magnesium")],
    [InlineKeyboardButton(text="Цинк", callback_data="deficiencies_zinc")],
    [InlineKeyboardButton(text="Напишу сам", callback_data="deficiencies_custom")]
])

temp_restrictions: Dict[int, List[str]] = {}

def get_restrictions_keyboard(user_id: int) -> InlineKeyboardMarkup:
    selected = temp_restrictions.get(user_id, [])
    buttons = []
    for cb, text in RESTRICTIONS.items():
        display_text = f"✔️ {text}" if cb in selected else text
        buttons.append([InlineKeyboardButton(text=display_text, callback_data=cb)])
    buttons.append([InlineKeyboardButton(text="✅ Готово", callback_data="restrictions_done")])
    buttons.append([InlineKeyboardButton(text="🗑️ Очистить всё", callback_data="restrictions_clear")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)

# Клавиатура для продолжения диалога
continue_keyboard = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="Пройти опрос заново", callback_data="restart")],
    [InlineKeyboardButton(text="Связаться с консультантом", callback_data="consultant")],
    [InlineKeyboardButton(text="Задать вопрос", callback_data="ask_question")]
])

'''
# Или Reply-клавиатура
reply_continue_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Пройти заново")],
        [KeyboardButton(text="Консультант"), KeyboardButton(text="Вопрос")]
    ],
    resize_keyboard=True
)
@dp.message(F.text == "🔄 Пройти заново")
async def restart_survey(message: Message, state: FSMContext):
    await cmd_start(message, state)

@dp.callback_query(F.data == "restart")
async def restart_callback(callback: CallbackQuery, state: FSMContext):
    await cmd_start(callback.message, state)
    await callback.answer()

@dp.message(F.text == "Консультант")
@dp.callback_query(F.data == "consultant")
async def contact_consultant(callback_or_message, state: FSMContext = None):
    # Определяем тип объекта
    if hasattr(callback_or_message, 'message'):  # это CallbackQuery
        message = callback_or_message.message
        await callback_or_message.answer()
    else:  # это Message
        message = callback_or_message
    
    await message.answer(
        "👋 *Связь с консультантом KULTLAB*\n\n"
        "Наши специалисты помогут с выбором:\n\n"
        "📱 *Telegram:* @kultlab_support\n"
        "📞 *Телефон:* 8 (800) 123-45-67\n"
        "✉️ *Email:* shop@kultlab.ru\n\n"
        "Или напишите свой вопрос прямо сейчас — я передам консультанту!",
        parse_mode="Markdown",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text="◀️ Назад в меню")]],
            resize_keyboard=True
        )
    )
    await state.set_state(None) if state else None

@dp.message(F.text == "Вопрос")
@dp.callback_query(F.data == "ask_question")
async def ask_question(callback_or_message, state: FSMContext):
    if hasattr(callback_or_message, 'message'):
        message = callback_or_message.message
        await callback_or_message.answer()
    else:
        message = callback_or_message
    
    await message.answer(
        "💬 *Задайте свой вопрос*\n\n"
        "Напишите, что вас интересует:\n"
        "— про дозировки\n"
        "— про совместимость продуктов\n"
        "— про противопоказания\n"
        "— или что-то ещё\n\n"
        "Я отвечу как эксперт по KULTLAB!",
        parse_mode="Markdown",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[[KeyboardButton(text="◀️ Назад в меню")]],
            resize_keyboard=True
        )
    )
    await state.set_state(SurveyStates.waiting_for_question)


@dp.message(SurveyStates.waiting_for_question)
async def process_question(message: Message, state: FSMContext):
    user_question = message.text
    
    if user_question == "◀️ Назад в меню":
        await cmd_start(message, state)
        return
    
    # Отправляем вопрос в GigaChat для ответа
    await message.answer("*Анализирую ваш вопрос...*", parse_mode="Markdown")
    
    prompt = f"""
    Ты - эксперт по спортивному питанию бренда KULTLAB.
    
    Вопрос пользователя: {user_question}
    
    Ответь кратко, понятно и по делу. Если вопрос про конкретный продукт - уточни, что его можно купить на сайте или WB.
    Если не знаешь ответа - предложи связаться с консультантом.
    
    Ответь на русском, живым языком.
    """
    
    try:
        messages = [
            SystemMessage(content="Ты эксперт KULTLAB. Отвечай полезно, кратко, с ссылкой на продукты бренда."),
            HumanMessage(content=prompt)
        ]
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: giga.invoke(messages))
        
        await message.answer(
            f"💡 *Ответ эксперта:*\n\n{response.content}\n\n"
            "Остались вопросы? Просто напиши!\n"
            "Или выбери действие ниже 👇",
            parse_mode="Markdown",
            reply_markup=reply_continue_keyboard
        )
    except Exception as e:
        logger.error(f"Ошибка ответа на вопрос: {e}")
        await message.answer(
            "🙏 *Не могу точно ответить на этот вопрос*\n\n"
            "Рекомендую обратиться к официальному консультанту KULTLAB:\n"
            "📱 Telegram: @kultlab_support\n\n"
            "Или задайте вопрос иначе - я постараюсь помочь!",
            parse_mode="Markdown",
            reply_markup=reply_continue_keyboard
        )
    
    await state.set_state(None)

@dp.message(F.text == "◀️ Назад в меню")
async def back_to_menu(message: Message, state: FSMContext):
    await cmd_start(message, state)'''

# ========== ОБНОВЛЕННАЯ ФУНКЦИЯ finalize_survey ==========
async def finalize_survey(message: Message, state: FSMContext, deficiencies: str):
    """Завершение опроса с генерацией рекомендаций"""
    data = await state.get_data()
    data['deficiencies'] = deficiencies

    health_issues = extract_health_issues(data.get('restrictions', ''))
    data.update(health_issues)
    
    # Отправляем сообщение о генерации
    waiting_msg = await message.answer("🔄 *Подбираю лучшие продукты KULTLAB специально для вас...*\n\nЭто займёт несколько секунд ⏳", parse_mode="Markdown")
    
    goal_products = load_products_by_goal_csv(data.get('goal', ''))
    
    # Загружаем товары для восполнения дефицитов
    deficiency_products = load_deficiency_products()
    
    # Объединяем товары (убираем дубликаты по названию)
    all_products = goal_products.copy()
    existing_names = {p.name for p in all_products}
    for p in deficiency_products:
        if p.name not in existing_names:
            all_products.append(p)
            existing_names.add(p.name)
    
    if not all_products:
        response_text = "❌ Нет товаров для вашей цели. Обратитесь к консультанту."
    else:
        recommender = GigaProductRecommender(all_products, giga)
        selected_products, _ = await recommender.recommend_products(data)
        
        if selected_products:
            response_text = await generate_recommendation_response(selected_products, data, giga)
        else:
            response_text = "😔 Не удалось подобрать продукты. Попробуйте изменить параметры или обратитесь к консультанту."
    
    await waiting_msg.delete()
    

    await message.answer(response_text, parse_mode="Markdown")
    # НОВЫЙ ФОРМАТ ОТВЕТА С МАГАЗИНАМИ
    shop_info = """
    *👉
    Купить можно здесь:*

    — Wildberries: (https://www.wildberries.ru/brands/kultlab)
    — Ozon: (https://www.ozon.ru/brand/kultlab-100155610/)
    — Официальный сайт: (https://kultlab.ru/)

    *
    📍 Адрес магазина в Новосибирске:*
       просп. Карла Маркса, 43
    """
    await message.answer(shop_info, parse_mode="Markdown")
    await state.clear()

# ========== Обработчики ==========
@dp.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    
    await state.clear()
    await message.answer(
        "👋 *Привет! Я помогу подобрать спортивное питание и БАДы KULTLAB специально для тебя!*\n\n"
        "Отвечай на несколько вопросов — и я соберу тебе готовую корзину с рекомендациями 👇\n\n"
        "_Перед началом важно:_\n\n"
        "Нажимая *«да»*, ты даёшь согласие на обработку персональных данных и принимаешь политику конфиденциальности: https://kultlab.ru/confidential/\n\n"
        "*Начинаем?*",
        reply_markup=consent_keyboard,
        parse_mode="Markdown"
    )
    await state.set_state(SurveyStates.waiting_for_consent)

@dp.message(SurveyStates.waiting_for_consent, F.text.lower().in_(["да", "нет"]))
async def process_consent(message: Message, state: FSMContext):
    if message.text.lower() == "да":
        await message.answer(
            "Отлично, начинаем! 🚀\n\n"
            "*1/8 Какая у тебя цель на ближайшие 2-3 месяца?*\n\n",
            reply_markup=goal_keyboard,
            parse_mode="Markdown"
        )
        await state.set_state(SurveyStates.goal)
    else:
        await message.answer(
            "Жаль 😔\n\n"
            "Если передумаешь — просто напиши /start",
            reply_markup=ReplyKeyboardRemove()
        )
        await state.clear()

@dp.message(SurveyStates.waiting_for_consent)
async def invalid_consent(message: Message):
    await message.answer("Пожалуйста, ответьте 'Да' или 'Нет' с помощью кнопок.")

@dp.callback_query(SurveyStates.goal, F.data.startswith("goal_"))
async def process_goal(callback: CallbackQuery, state: FSMContext):
    goal_map = {
        "goal_loss": "снижение веса / рельеф",
        "goal_gain": "набор мышечной массы",
        "goal_beauty": "красота: кожа, волосы, ногти",
        "goal_endurance": "выносливость / силовые",
        "goal_health": "здоровье и восстановление"
    }
    goal = goal_map.get(callback.data, "Неизвестно")
    await state.update_data(goal=goal)
    await callback.message.answer(
        "*2/8 Какой у тебя вес?*\n(в кг, например: 70.5)\n\n",
        parse_mode="Markdown"
    )
    await state.set_state(SurveyStates.weight)
    await callback.answer()

@dp.message(SurveyStates.weight)
async def process_weight(message: Message, state: FSMContext):
    try:
        weight = float(message.text.replace(",", "."))
        await state.update_data(weight=weight)
        await message.answer(
            "*3. Какой у тебя рост?*\n(в см, например: 175)\n\n",
            parse_mode="Markdown"
        )
        await state.set_state(SurveyStates.height)
    except ValueError:
        await message.answer("❌ Пожалуйста, введите число (например: 70.5)")

@dp.message(SurveyStates.height)
async def process_height(message: Message, state: FSMContext):
    try:
        height = float(message.text.replace(",", "."))
        await state.update_data(height=height)
        await message.answer(
            "*4. Какой у тебя уровень активности?*\n\n",
            reply_markup=activity_keyboard,
            parse_mode="Markdown"
        )
        await state.set_state(SurveyStates.activity)
    except ValueError:
        await message.answer("❌ Пожалуйста, введите число (например: 175)")

@dp.callback_query(SurveyStates.activity, F.data.startswith("activity_"))
async def process_activity(callback: CallbackQuery, state: FSMContext):
    activity_map = {
        "activity_low": "Низкий (сидячий образ жизни)",
        "activity_medium": "Средний (тренировки 3-5 раз/нед)",
        "activity_high": "Высокий (ежедневные тренировки)"
    }
    activity = activity_map.get(callback.data, "Неизвестно")
    await state.update_data(activity=activity)
    await callback.message.answer(
        "*5/8 Ты добираешь норму белка из обычной еды?*\n\n",
        reply_markup=brelok_keyboard,
        parse_mode="Markdown"
    )
    await state.set_state(SurveyStates.brelok)
    await callback.answer()

@dp.callback_query(SurveyStates.brelok, F.data.startswith("brelok_"))
async def process_brelok(callback: CallbackQuery, state: FSMContext):
    brelok_map = {
        "brelok_yes": "Да, хватает",
        "brelok_no": "Нет, не хватает",
        "brelok_sometimes": "Частично"
    }
    brelok = brelok_map.get(callback.data, "Неизвестно")
    await state.update_data(brelok=brelok)
    
    user_id = callback.from_user.id
    temp_restrictions[user_id] = []
    
    await callback.message.answer(
        "*6/8 Есть ли у тебя аллергии, хронические заболевания или особенности здоровья?*\n\n"
        "Можно выбрать *несколько* вариантов.\n"
        "После выбора нажмите 'Готово'.",
        reply_markup=get_restrictions_keyboard(user_id),
        parse_mode="Markdown"
    )
    await state.set_state(SurveyStates.restrictions)
    await callback.answer()

@dp.callback_query(SurveyStates.restrictions, F.data.startswith("restriction_"))
async def process_restriction_toggle(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    restriction_cb = callback.data
    
    if restriction_cb not in RESTRICTIONS:
        await callback.answer("Неизвестное ограничение")
        return
    
    selected = temp_restrictions.get(user_id, [])
    
    if restriction_cb in selected:
        selected.remove(restriction_cb)
    else:
        if restriction_cb == "restriction_none":
            selected = ["restriction_none"]
        else:
            if "restriction_none" in selected:
                selected.remove("restriction_none")
            selected.append(restriction_cb)
    
    temp_restrictions[user_id] = selected
    await callback.message.edit_reply_markup(reply_markup=get_restrictions_keyboard(user_id))
    await callback.answer()

@dp.callback_query(SurveyStates.restrictions, F.data == "restrictions_clear")
async def process_restrictions_clear(callback: CallbackQuery):
    user_id = callback.from_user.id
    temp_restrictions[user_id] = []
    await callback.message.edit_reply_markup(reply_markup=get_restrictions_keyboard(user_id))
    await callback.answer("✅ Все ограничения очищены")

@dp.callback_query(SurveyStates.restrictions, F.data == "restrictions_done")
async def process_restrictions_done(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    selected_codes = temp_restrictions.get(user_id, [])
    
    if not selected_codes or (len(selected_codes) == 1 and "restriction_none" in selected_codes):
        restrictions_text = "Нет ограничений"
    else:
        restrictions_list = [RESTRICTIONS[code] for code in selected_codes if code != "restriction_none"]
        restrictions_text = ", ".join(restrictions_list)
    
    await state.update_data(restrictions=restrictions_text)
    
    if user_id in temp_restrictions:
        del temp_restrictions[user_id]
    
    await callback.message.answer(
        "*7/8 Какой у тебя тип питания?*\n\n",
        reply_markup=diet_type_keyboard,
        parse_mode="Markdown"
    )
    await state.set_state(SurveyStates.diet_type)
    await callback.answer()

@dp.callback_query(SurveyStates.diet_type, F.data.startswith("diet_"))
async def process_diet_type(callback: CallbackQuery, state: FSMContext):
    diet_map = {
        "diet_regular": "Обычное",
        "diet_lacto": "Лакто-ово-вегетарианство",
        "diet_vegan": "Веганство",
        "diet_pescatarian": "Пескетарианство"
    }
    diet = diet_map.get(callback.data, "Неизвестно")
    await state.update_data(diet_type=diet)
    
    await callback.message.answer(
        "*8/8. Есть ли у тебя подтверждённые дефициты по анализам?*\n\n"
        "Можно выбрать из списка или написать самому 👇\n",
        reply_markup=deficiencies_keyboard,
        parse_mode="Markdown"
    )
    await state.set_state(SurveyStates.deficiencies)
    await callback.answer()

@dp.callback_query(SurveyStates.deficiencies, F.data.startswith("deficiencies_"))
async def process_deficiencies_choice(callback: CallbackQuery, state: FSMContext):
    if callback.data == "deficiencies_none":
        await finalize_survey(callback.message, state, "Нет дефицитов")
        await callback.answer()
    elif callback.data == "deficiencies_custom":
        await callback.message.answer(
            "*8/8 Напиши свои дефициты*\n\n"
            "Перечисли через запятую, например:\n"
            "железо, витамин D, магний\n\n"
            "Или напиши «нет», если дефицитов нет",
            parse_mode="Markdown"
        )
        await callback.answer()
    else:
        deficiency_map = {
            "deficiencies_iron": "железо",
            "deficiencies_vitd": "витамин D",
            "deficiencies_b12": "витамин B12",
            "deficiencies_iodine": "йод",
            "deficiencies_magnesium": "магний",
            "deficiencies_zinc": "цинк"
        }
        deficiencies_text = deficiency_map.get(callback.data, "")
        await finalize_survey(callback.message, state, deficiencies_text)
        await callback.answer()

@dp.message(SurveyStates.deficiencies)
async def process_deficiencies_text(message: Message, state: FSMContext):
    text = message.text.strip()
    if text.lower() == "нет":
        deficiencies_text = "Нет дефицитов"
    else:
        deficiencies_text = text
    await finalize_survey(message, state, deficiencies_text)

# Добавьте эту функцию
def extract_health_issues(restrictions_text: str) -> Dict[str, str]:
    """Извлекает специфические проблемы здоровья из текста ограничений"""
    restrictions_lower = restrictions_text.lower()
    return {
        'gut_issues': 'есть' if 'жкт' in restrictions_lower else 'нет',
        'heart_issues': 'есть' if any(x in restrictions_lower for x in ['сердце', 'давление']) else 'нет',
        'anxiety': 'есть' if 'тревожность' in restrictions_lower else 'нет'
    }

# ========== Запуск бота ==========
async def main():
    global products_cache
    
    print("\n🚀 Бот готов к работе!")
    print("💡 Нажми /start в Telegram для начала")
    
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
