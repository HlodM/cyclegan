import os
import telebot
from telebot import types
from PIL import Image
from change_image import change_image, sr_image
import train_params
import io


bot = telebot.TeleBot(train_params.TOKEN)
dir_to_save = train_params.dir_to_save


@bot.message_handler(commands=['start'])
def start(message):
    mess = f'HELLO, {message.from_user.first_name}, this bot can make photo in Vangogh style. Just send one to try'
    bot.send_message(message.chat.id, mess)


@bot.message_handler(content_types=['photo'])
def get_user_photo(message):
    bot.send_message(message.chat.id, 'cool photo!')
    bot.send_message(message.chat.id, 'what about this one?')
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    if not os.path.exists(f"{dir_to_save}/images/"):
        os.mkdir(f"{dir_to_save}/images")
    with open(f"{train_params.dir_to_save}/images/bot_image.jpg", "wb") as file:
        file.write(downloaded_file)
    photo = Image.open(io.BytesIO(downloaded_file))
    if len(photo.getbands()) != 3:
        bot.send_message(message.chat.id, 'incorrect image channels number, must be 3 (RGB)')
    else:
        im_path = change_image(photo)
        if train_params.sr:
            im_path = sr_image(im_path)
        photo = Image.open(im_path)
        bot.send_photo(message.chat.id, photo)
        bot.send_message(message.chat.id, 'want to do the same or better - send "/website" message')


@bot.message_handler(commands=['website'])
def website(message):
    markup = types.InlineKeyboardMarkup()
    btn1 = types.InlineKeyboardButton('stepik dls', url='https://stepik.org/course/109539/promo')
    btn2 = types.InlineKeyboardButton('github', url='https://github.com/HlodM/cyclegan')
    markup.add(btn1, btn2)
    bot.send_message(message.chat.id, 'course - stepik dls, project repository - github', reply_markup=markup)


@bot.message_handler(commands=['help'])
def commands_list(message):
    bot.send_message(message.chat.id, 'available commands:')
    bot.send_message(message.chat.id, '/start')
    bot.send_message(message.chat.id, '/website')
    bot.send_message(message.chat.id, '/help')


# @bot.message_handler(content_types=['text'])
# def get_user_text(message):
#     bot.send_message(message.chat.id, 'this bot can make photo in Vangogh style. Just send one to try')
#     bot.send_message(message.chat.id, 'available commands:')
#     bot.send_message(message.chat.id, '/start')
#     bot.send_message(message.chat.id, '/website')
#     bot.send_message(message.chat.id, '/help')


bot.polling(none_stop=True)
