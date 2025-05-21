import io
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import torch
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForImageClassification
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_API")

print("Загрузка модели...")
processor = AutoProcessor.from_pretrained("Falconsai/nsfw_image_detection", use_fast=True)
model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Модель загружена на {device.upper()}")

executor = ThreadPoolExecutor()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Отправьте изображение, и я проверю его на наличие 18+ контента.")


def analyze_image_sync(image: Image.Image) -> dict:
    try:
        image = image.resize((224, 224))

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = outputs.logits.softmax(dim=1).cpu().numpy()[0]
        nsfw_score = float(probs[1])
        safe_score = float(probs[0])

        return {
            "nsfw_score": nsfw_score,
            "is_nsfw": nsfw_score > 0.8,
            "safe_score": safe_score
        }
    except Exception as e:
        print(f"Ошибка при анализе: {e}")
        return {"error": str(e)}


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        image = Image.open(io.BytesIO(photo_bytes)).convert("RGB")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, analyze_image_sync, image)

        if "error" in result:
            await update.message.reply_text("❌ Ошибка при анализе изображения")
            return

        if result["is_nsfw"]:
            await update.message.reply_text("❌ NSFW-контент удален.")
            await update.message.delete()

        else:
            await update.message.reply_text(
                f"✅ Контент безопасен (уверенность: {result['safe_score']:.2%})"
            )

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        await update.message.reply_text("❌ Произошла критическая ошибка")


if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))

    print("Бот запущен...")
    app.run_polling()
