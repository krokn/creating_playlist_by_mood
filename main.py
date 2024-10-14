import os
import re
import tempfile
import uvicorn
import requests
import hashlib
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from deepface import DeepFace
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from PIL import Image

from config import FOLDER_ID, API_KEY, YANDEXGPT_API_URL


def setup_logging():
    logger = logging.getLogger("image_analyzer")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler("image_analysis.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

logger = setup_logging()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    expose_headers=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class ErrorResponse(BaseModel):
    detail: str


class SuccessResponse(BaseModel):
    playlist: list
    affirmation: str


def format_response(api_response):
    if 'result' not in api_response or not api_response['result'].get('alternatives'):
        logger.error("Нет данных в ответе YandexGPT API.")
        raise HTTPException(status_code=500, detail="Нет данных в ответе YandexGPT API.")

    songs_text = api_response['result']['alternatives'][0]['message']['text']

    logger.info(f'songs_text = {songs_text}')

    if not songs_text.strip():
        logger.error("Текст ответа от YandexGPT API пуст.")
        raise HTTPException(status_code=500, detail="Текст ответа от YandexGPT API пуст.")

    songs_text = songs_text.replace('«', '').replace('»', '').replace(';', '')

    songs_list = songs_text.split("\n")
    formatted_songs = {}

    pattern = re.compile(r"(\d+)\.\s*([^,–—-]+)\s*[,–—-]\s*(.+)")
    for song in songs_list:
        song = song.strip()
        if not song:
            continue

        match = pattern.match(song)
        if match:
            song_number = match.group(1)
            artist = match.group(2).strip()
            song_title = match.group(3).strip().rstrip('.')
            formatted_songs[song_number] = f"{song_title}, {artist}"
        else:
            logger.warning(f"Не удалось распознать строку: {song}")

    if not formatted_songs:
        logger.error("Список песен пуст после форматирования.")
        raise HTTPException(status_code=405, detail="Список песен пуст.")

    return list(formatted_songs.values())


@app.post("/generate_text/", response_model=SuccessResponse, responses={
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    405: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
})
async def generate_text(file: UploadFile = File(...)):
    contents = await file.read()

    if not contents:
        logger.warning("Получен пустой файл для анализа.")
        raise HTTPException(status_code=400, detail="Загруженный файл пуст.")

    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(contents)
            temp_file_path = tmp_file.name
        logger.info(f"Временный файл создан: {temp_file_path}")

        file_size = os.path.getsize(temp_file_path)
        logger.info(f"Размер файла: {file_size} байт")

        with Image.open(temp_file_path) as img:
            width, height = img.size
            format = img.format
            mode = img.mode
            logger.info(f"Размер изображения: {width}x{height} пикселей")
            logger.info(f"Формат изображения: {format}")
            logger.info(f"Цветовой режим: {mode}")

            try:
                exif_data = img._getexif()
                if exif_data:
                    logger.info(f"EXIF-данные: {exif_data}")
                else:
                    logger.info("EXIF-данные отсутствуют.")
            except AttributeError:
                logger.info("EXIF-данные недоступны для этого изображения.")

            megapixels = (width * height) / 1_000_000
            logger.info(f"Разрешение: {megapixels:.2f} мегапикселей")

            try:
                analysis_result = DeepFace.analyze(
                    img_path=temp_file_path,
                    actions=['emotion', 'age', 'gender'],
                    enforce_detection=True
                )
                logger.info(f"DeepFace Analysis Result: {analysis_result}")
            except ValueError:
                raise HTTPException(status_code=404, detail="Лицо не найдено")

        gender_translation = {
            'Man': 'муж',
            'Woman': 'жен'
        }

        emotion_translation = {
            'angry': 'злость',
            'disgust': 'отвращение',
            'fear': 'страх',
            'happy': 'счастье',
            'sad': 'грусть',
            'surprise': 'удивление',
            'neutral': 'нейтральное'
        }

        age = analysis_result.get('age')
        gender = gender_translation.get(analysis_result.get('gender'), 'Неизвестно')
        emotion = emotion_translation.get(analysis_result.get('dominant_emotion'), 'Неизвестно')

        logger.info(f"age: {age}")
        logger.info(f"gender: {gender}")
        logger.info(f"emotion: {emotion}")

        headers = {
            'Authorization': f'Api-Key {API_KEY}',
            'x-folder-id': FOLDER_ID,
            'Content-Type': 'application/json'
        }

        data_for_playlist = {
            'modelUri': f'gpt://{FOLDER_ID}/yandexgpt/latest',
            'completionOptions': {
                'stream': False,
                'temperature': 0.7,
                'maxTokens': 2000
            },
            "messages": [
                {
                    "role": "system",
                    "text": (
                        "Ты система, которая должна составить плейлист из 20 песен, которые подходят под пол, "
                        "возраст и настроение человека. Представь результат в форме текста, например 1 - автор песни, "
                        "название песни. и так далее. Нужны только данные без вводных фраз и "
                        "объяснений. Не используй разметку Markdown!"
                    )
                },
                {
                    "role": "user",
                    "text": f"Возраст: {age}, Пол: {gender}, Эмоции: {emotion}"
                }
            ]
        }

        data_for_affirmation = {
            'modelUri': f'gpt://{FOLDER_ID}/yandexgpt/latest',
            'completionOptions': {
                'stream': False,
                'temperature': 0.7,
                'maxTokens': 4000
            },
            "messages": [
                {
                    "role": "system",
                    "text": (
                        "ты человек, которая желает напутственные слова на хороший день исходя из его "
                        "возраста, пола и эмоций. Минимум 7 предложений. Нужна только строка без вводных фраз и ничего кроме напутственных слов. "
                        "Не используй разметку Markdown!"
                    )
                },
                {
                    "role": "user",
                    "text": f"Возраст: {age}, Пол: {gender}, Эмоции: {emotion}"
                }
            ]
        }

        try:
            response_for_playlist = requests.post(YANDEXGPT_API_URL, json=data_for_playlist, headers=headers)
            logger.info(f"YandexGPT Playlist API Response Status: {response_for_playlist.status_code}")
            logger.info(f"YandexGPT Playlist API Response Body: {response_for_playlist.text}")

            response_for_affirmation = requests.post(YANDEXGPT_API_URL, json=data_for_affirmation, headers=headers)
            logger.info(f"YandexGPT Affirmation API Response Status: {response_for_affirmation.status_code}")
            logger.info(f"YandexGPT Affirmation API Response Body: {response_for_affirmation.text}")
        except requests.RequestException as e:
            logger.error(f"Ошибка при обращении к YandexGPT API: {e}")
            raise HTTPException(status_code=502, detail=f"Ошибка при обращении к YandexGPT API: {str(e)}")

        if response_for_playlist.status_code == 200 and response_for_affirmation.status_code == 200:
            api_response = response_for_playlist.json()
            formatted_result = format_response(api_response)

            affirmation_response = response_for_affirmation.json()
            try:
                text = affirmation_response["result"]["alternatives"][0]["message"]["text"]
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Ошибка при разборе JSON аффирмации: {e}")
                raise HTTPException(status_code=500, detail="Неправильная структура JSON в ответе аффирмации")
            text = text.replace('\n', ' ')
            text = text.replace('\n', '*')

            return JSONResponse(content={
                "playlist": formatted_result,
                "affirmation": text,
                "age": age,
                "gender": gender,
                "emotion": emotion

            }, status_code=200)
        else:
            error_message = f"Ошибка при обращении к YandexGPT API: Playlist Status {response_for_playlist.status_code}, Affirmation Status {response_for_affirmation.status_code}"
            logger.error(error_message)
            raise HTTPException(
                status_code=502,
                detail=error_message
            )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception("Произошла непредвиденная ошибка при обработке запроса.")
        raise HTTPException(status_code=500, detail=f"Непредвиденная ошибка: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Временный файл удален: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл {temp_file_path}: {e}")
